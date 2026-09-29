#!/usr/bin/env python
"""Adapt CESM2/CAM6 inputdata forcing files to the legoESM AMIP loaders.

Two CESM layouts the loaders in ``legoesm.forcing.external`` cannot read as-is:

``volcanic``  ``atm/cam/volc/CMIP_DOE-ACME_radiation_1850-2014_v3_c20171205.nc``
    (CMIP6 IACETH stratospheric aerosol optics, RRTMG 14 SW / 16 LW bands).
    Its record axis is a bare ``month`` dimension ("month starting from 1850
    01") with no CF units, so the loader would fall back to CYCLIC 12-month
    sampling and put the 1850-2014 record in the wrong months.  This writes a
    CF ``time`` axis (days since 1850-01-01, noleap, mid-month) built from the
    file's own ``date`` (YYYYMMDD) variable, and keeps only the years asked for.

``ozone``  ``atm/cam/ozone_strataero/ozone_strataero_WACCM_L70_zm5day_*.nc``
    (CAM6 prescribed ozone, 5-day zonal means on the WACCM L70 HYBRID grid).
    Its ``lev`` coordinate is the REFERENCE pressure ``1000*(A+B)`` hPa, not the
    pressure the ozone sits at.  This interpolates each (time, lat) column in
    log-pressure from its true pressure ``hyam*P0 + hybm*PS`` onto the fixed
    levels ``hyam*P0 + hybm*1e5`` Pa, and writes ``O3(time, plev, lat)`` with
    ``plev`` in Pa.  The 5-day record spacing is kept (the loader interpolates
    any monotonic multi-record axis).

Usage::

    python scripts/data/adapt_cesm_forcing.py volcanic --in <volc.nc> \\
        --out volcanic_cmip6_from_cesm.nc --years 1978 2014
    python scripts/data/adapt_cesm_forcing.py ozone --in <ozone_strataero.nc> \\
        --out ozone_cam6_plev.nc --years 1978 2014
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Cumulative days BEFORE each month in a 365-day (noleap) year.
_NOLEAP_CUM = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
_NOLEAP_LEN = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_TIME_UNITS = "days since 1850-01-01 00:00:00"


def _split_date(date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(date).astype(np.int64)
    return d // 10000, (d // 100) % 100, d % 100


def _year_mask(years: np.ndarray, first: int, last: int) -> np.ndarray:
    keep = (years >= first) & (years <= last)
    if not keep.any():
        raise ValueError(f"no records in years {first}-{last}")
    return keep


def adapt_volcanic(ds, first_year: int, last_year: int):
    """CF-time the CESM volcanic optics file and keep ``[first_year, last_year]``.

    Each monthly record is placed at the middle of its month on a noleap
    ``days since 1850-01-01`` axis, from the file's own ``date`` variable (the
    ``month`` coordinate is a bare index).  Records of other years are dropped;
    the file duplicates 1850 as "1849" and 2014 as "2015", so those labels are
    never inside a 1850-2014 window.
    """
    if "month" not in ds.dims or "date" not in ds:
        raise ValueError("expected the CESM volcanic layout: a 'month' record "
                         "dimension and a YYYYMMDD 'date' variable")
    years, months, _ = _split_date(ds["date"].values)
    keep = _year_mask(years, first_year, last_year)
    idx = np.nonzero(keep)[0]
    out = ds.isel(month=idx)
    y, m = years[idx], months[idx]
    if np.any(np.diff(y * 12 + m) != 1):
        raise ValueError("volcanic records are not consecutive months")
    mid = (y - 1850) * 365.0 + _NOLEAP_CUM[m - 1] + 0.5 * _NOLEAP_LEN[m - 1]
    # ``date`` holds the file's own labels; the CF ``time`` below replaces it,
    # and a stale copy would invite a reader to anchor on it instead.
    out = out.drop_vars(["date", "datesec"], errors="ignore")
    out = out.rename({"month": "time"}).drop_vars("time", errors="ignore")
    out = out.assign_coords(time=("time", mid, {"units": _TIME_UNITS,
                                                "calendar": "noleap"}))
    out.attrs["legoesm_adapted"] = (
        "scripts/data/adapt_cesm_forcing.py volcanic: bare month axis -> CF "
        f"noleap mid-month time, years {first_year}-{last_year}")
    return out


def _interp_logp(p_src: np.ndarray, q_src: np.ndarray,
                 p_tgt: np.ndarray) -> np.ndarray:
    """Interpolate ``q`` along its LAST axis from ``p_src`` to ``p_tgt`` in
    log-pressure; constant extrapolation beyond the source column."""
    lp_src = np.log(p_src)
    lp_tgt = np.log(p_tgt)
    lead = q_src.shape[:-1]
    flat_p = lp_src.reshape(-1, lp_src.shape[-1])
    flat_q = q_src.reshape(-1, q_src.shape[-1])
    out = np.empty((flat_q.shape[0], lp_tgt.size), dtype=np.float64)
    for i in range(flat_q.shape[0]):
        order = np.argsort(flat_p[i])
        out[i] = np.interp(lp_tgt, flat_p[i][order], flat_q[i][order])
    return out.reshape(lead + (lp_tgt.size,))


def adapt_ozone(ds, first_year: int, last_year: int, ps_ref_pa: float = 1.0e5):
    """True-pressure CESM hybrid-level ozone -> fixed pressure levels [Pa].

    Target levels are ``hyam*P0 + hybm*ps_ref_pa`` (the file's own reference
    column); each (time, lat) column is interpolated in log-p from its actual
    ``hyam*P0 + hybm*PS``.  Each record is time-stamped at the midpoint of
    its ``time_bnds`` interval (CAM's own ``time`` is the interval end), and
    the records whose MIDPOINT falls in ``[first_year, last_year]`` are kept.
    """
    import cftime
    for v in ("O3", "PS", "hyam", "hybm", "P0", "time_bnds"):
        if v not in ds:
            raise ValueError(f"expected CESM ozone variable {v!r}")
    tunits = ds["time"].attrs.get("units")
    if not tunits:
        raise ValueError("ozone 'time' has no units attribute; refusing to "
                         "guess its epoch")
    tcal = ds["time"].attrs.get("calendar", "noleap")
    t_mid_all = np.asarray(ds["time_bnds"].values, dtype=np.float64).mean(axis=-1)
    years = np.array([d.year for d in cftime.num2date(t_mid_all, tunits, tcal)])
    idx = np.nonzero(_year_mask(years, first_year, last_year))[0]
    sub = ds.isel(time=idx)
    t_mid = t_mid_all[idx]
    hyam = np.asarray(sub["hyam"].values, dtype=np.float64)
    hybm = np.asarray(sub["hybm"].values, dtype=np.float64)
    p0 = float(sub["P0"].values)
    ps = np.asarray(sub["PS"].transpose("time", "lat").values, dtype=np.float64)
    o3 = np.asarray(sub["O3"].transpose("time", "lat", "lev").values,
                    dtype=np.float64)
    p_src = hyam[None, None, :] * p0 + hybm[None, None, :] * ps[:, :, None]
    p_tgt = hyam * p0 + hybm * ps_ref_pa
    order = np.argsort(p_tgt)
    p_tgt = p_tgt[order]
    o3_tgt = _interp_logp(p_src, o3, p_tgt)                # (time, lat, plev)

    # CAM stamps each 5-day mean at its interval END ("time"); the record's
    # centre is the midpoint of time_bnds (== date+datesec, what CAM
    # interpolates at).  Anchor there, not 2.5 days late.

    import xarray as xr
    return xr.Dataset(
        data_vars={
            "O3": (("time", "plev", "lat"), np.transpose(o3_tgt, (0, 2, 1)),
                   {"units": "mol/mol",
                    "long_name": "O3 volume mixing ratio on fixed pressure "
                                 "levels (from CAM6 hybrid levels)"}),
        },
        coords={
            "time": ("time", t_mid,
                     {"units": tunits, "calendar": tcal}),
            "plev": ("plev", p_tgt, {"units": "Pa"}),
            "lat": ("lat", np.asarray(sub["lat"].values, dtype=np.float64),
                    {"units": "degrees_north"}),
        },
        attrs={"legoesm_adapted": (
            "scripts/data/adapt_cesm_forcing.py ozone: hybrid levels -> fixed "
            f"plev (hyam*P0+hybm*{ps_ref_pa:g} Pa) by log-p interpolation from "
            f"hyam*P0+hybm*PS, years {first_year}-{last_year}")},
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=("volcanic", "ozone"))
    p.add_argument("--in", dest="in_file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--years", type=int, nargs=2, required=True,
                   metavar=("FIRST", "LAST"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    import xarray as xr
    a = parse_args(argv)
    ds = xr.open_dataset(a.in_file, decode_times=False)
    try:
        fn = adapt_volcanic if a.kind == "volcanic" else adapt_ozone
        out = fn(ds, a.years[0], a.years[1])
        out.to_netcdf(a.out)
    finally:
        ds.close()
    print(f"[adapt-cesm] {a.kind}: wrote {a.out} ({dict(out.sizes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
