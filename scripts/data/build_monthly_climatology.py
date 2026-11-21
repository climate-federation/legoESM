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
        y0, m0 = (int(x) for x in units.split()[2].split("-")[:2])
        k = (m0 - 1) + np.floor(t).astype(int)
        return y0 + k // 12, k % 12 + 1
    import cftime
    d = cftime.num2date(t, units, calendar=calendar)
    return np.array([x.year for x in d]), np.array([x.month for x in d])


def climatology(ds, first: int, last: int, optics: bool = False):
    import xarray as xr
    tv = ds["time"]
    years, months = record_year_month(tv.values, tv.attrs["units"],
                                      tv.attrs.get("calendar", "standard"))
    sel = (years >= first) & (years <= last)
    n_years = last - first + 1
    for m in range(1, 13):
        if int(np.sum(sel & (months == m))) != n_years:
            raise ValueError(f"month {m}: {int(np.sum(sel & (months == m)))} records in "
                             f"{first}-{last}, expected {n_years}")
    first_idx = [int(np.flatnonzero(sel & (months == m))[0]) for m in range(1, 13)]
    out = ds.drop_vars([v for v in ds.data_vars if "time" in ds[v].dims] + ["time"])
    bounds = tv.attrs.get("bounds")
    timed = [v for v in ds.data_vars if "time" in ds[v].dims and v != bounds]
    means = {v: [] for v in timed}
    for m in range(1, 13):
        idx = np.flatnonzero(sel & (months == m))
        sub = {v: ds[v].isel(time=idx).astype(np.float64) for v in timed}
        for v in timed:
            if optics and v in ("ssa", "asy"):
                w = sub["aod"] * (sub["ssa"] if v == "asy" else 1.0)
                mv = (w * sub[v]).sum("time") / w.sum("time")
                means[v].append(mv.where(w.sum("time") > 0, sub[v].mean("time")))
            else:
                means[v].append(sub[v].mean("time"))
    time = xr.DataArray(tv.values[first_idx], dims="time", attrs=tv.attrs)
    for v in timed:
        da = xr.concat(means[v], dim="time").astype(ds[v].dtype)
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
