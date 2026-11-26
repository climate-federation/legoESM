#!/usr/bin/env python
"""Monthly climatology (12 records) of a monthly forcing file over a year span.

Every variable with a ``time`` dimension is averaged per calendar month over
``--years FIRST LAST``; everything else is copied.  The 12 records keep the
FIRST year's time stamps in the source's own units, so the forcing loader reads
the file as cyclic (<= 12 records), phase-anchored at the first record.

``--optics``: aerosol optics are averaged extinction/scattering-consistently
(``aod`` plain mean, ``ssa`` aod-weighted, ``asy`` aod*ssa-weighted).

Usage::

  python scripts/data/build_monthly_climatology.py --in SRC.nc --out OUT.nc \\
      --years 2005 2014 [--optics]
"""
from __future__ import annotations

import argparse

import numpy as np


def record_year_month(time_values, units: str, calendar: str):
    """(year, month) of each monthly record."""
    t = np.asarray(time_values, dtype=np.float64)
    if units.startswith("months since"):
        y0, m0, d0 = ([int(float(x)) for x in units.split()[2].split("-")] + [1])[:3]
        clock = units.split()[3:]
        if d0 != 1 or any(float(x) != 0.0 for c in clock for x in c.split(":")):
            # floor(t) is the month index only from 00:00 on day 1 of a month
            raise ValueError(f"'{units}': months-since time needs a reference at "
                             "00:00 on day 1 of a month")
        k = (m0 - 1) + np.floor(t).astype(int)
        return y0 + k // 12, k % 12 + 1
    import cftime
    d = cftime.num2date(t, units, calendar=calendar)
    return np.array([x.year for x in d]), np.array([x.month for x in d])


def climatology(ds, first: int, last: int, optics: bool = False):
    import xarray as xr
    tv = ds["time"]
    bounds = tv.attrs.get("bounds")
    if bounds and bounds not in ds.variables:
        raise ValueError(f"time:bounds names {bounds!r}, which the file does not carry")
    # the month a record covers: its bounds' midpoint when it has bounds (a
    # stamp at a month's end or the next month's start would mislabel it)
    t_month = (ds[bounds].values.astype(np.float64).mean(-1) if bounds in ds.variables
               else tv.values)
    years, months = record_year_month(t_month, tv.attrs["units"],
                                      tv.attrs.get("calendar", "standard"))
    sel = (years >= first) & (years <= last)
    # every (year, month) exactly once (VOTED a/a/a): a per-month count would
    # let a duplicate in one year hide a gap in another
    for y in range(first, last + 1):
        for m in range(1, 13):
            n = int(np.sum((years == y) & (months == m)))
            if n != 1:
                raise ValueError(f"{y}-{m:02d}: {n} records, expected exactly 1")
    # the stamps (and bounds) of the FIRST YEAR's records, whatever the file order
    first_idx = [int(np.flatnonzero((years == first) & (months == m))[0])
                 for m in range(1, 13)]
    aux = [c for c in ds.coords if c != "time" and "time" in ds[c].dims]
    if aux:     # they would keep the file's record order beside sorted months
        raise ValueError(f"time-dependent coordinates {aux} are not supported")
    out = ds.drop_vars([v for v in ds.data_vars if "time" in ds[v].dims] + ["time"])
    timed = [v for v in ds.data_vars if "time" in ds[v].dims]
    if optics and not {"aod", "ssa", "asy"} <= set(timed):
        raise ValueError("--optics needs time-varying aod, ssa and asy")
    means = {v: [] for v in timed}
    for m in range(1, 13):
        idx = np.flatnonzero(sel & (months == m))
        sub = {v: ds[v].isel(time=idx).astype(np.float64) for v in timed}
        for v in timed:
            if optics and v in ("ssa", "asy"):
                # aod/ssa/asy must describe the same records: a missing value in
                # any of them would weight the others on a different subset
                if not all(bool(np.isfinite(sub[o].values).all()) for o in ("aod", "ssa", "asy")):
                    raise ValueError(f"--optics: month {m} has missing or non-finite aod/ssa/asy "
                                     "values in the span; incomplete optical records "
                                     "are refused")
                w = sub["aod"] * (sub["ssa"] if v == "asy" else 1.0)
                mv = (w * sub[v]).sum("time") / w.sum("time")
                means[v].append(mv.where(w.sum("time") > 0, sub[v].mean("time")))
            else:
                means[v].append(sub[v].mean("time"))
    # stamped at the covered month's middle when bounds say which month (the
    # forcing readers interpolate on the stamp and ignore bounds)
    stamp = (np.asarray(t_month, dtype=np.float64) if bounds   # a midpoint stays fractional
             else tv.values)
    time = xr.DataArray(np.asarray(stamp)[first_idx], dims="time", attrs=tv.attrs)
    for v in timed:
        da = xr.concat(means[v], dim="time")
        if np.issubdtype(ds[v].dtype, np.floating):   # an integer field keeps its float mean
            da = da.astype(ds[v].dtype)
        da = da.transpose(*ds[v].dims)
        da.attrs = ds[v].attrs
        out[v] = da.assign_coords(time=time)
    out = out.assign_coords(time=time)
    if bounds in ds.variables:     # the stamps' own first-year bounds
        out[bounds] = ds[bounds].isel(time=first_idx).assign_coords(time=time)
    out.attrs = dict(ds.attrs)
    out.attrs["history"] = (f"monthly climatology {first}-{last}"
                            f"{' (optics-weighted ssa/asy)' if optics else ''} by "
                            "scripts/data/build_monthly_climatology.py; "
                            + str(ds.attrs.get("history", "")))
    return out


def main(argv=None) -> int:
    import xarray as xr
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--years", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    p.add_argument("--optics", action="store_true")
    a = p.parse_args(argv)
    if a.years[0] > a.years[1]:
        raise SystemExit("--years: first after last")
    with xr.open_dataset(a.src, decode_times=False) as ds:
        out = climatology(ds, *a.years, optics=a.optics)
        out.to_netcdf(a.out + ".partial")
    import os
    os.replace(a.out + ".partial", a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
