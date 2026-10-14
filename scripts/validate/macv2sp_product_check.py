"""Sanity numbers for the staged MACv2-SP aerosol products
(scripts/data/adapt_cmip6_aerosol.py): area-weighted global-mean 550 nm AOD per
year (day-weighted annual mean) and the time coverage, plus the MACv1 anthropogenic AOD of a
year for comparison.

    python scripts/validate/macv2sp_product_check.py FIN TOT --years 1850 1979 2005 2014 \\
        [--macv1-ant raw_macv1/g30_ant_2005.nc]
"""
from __future__ import annotations

import argparse
import calendar
import subprocess
from pathlib import Path

import numpy as np

_VIS_UM = 0.55
_PI_YEAR = 1850     # MACv2-SP year_weight is zero here: the background year


def _band_550(nc) -> int:
    lo, hi = np.asarray(nc["wl1_sun"][:]), np.asarray(nc["wl2_sun"][:])
    return int(np.flatnonzero((lo <= _VIS_UM) & (hi > _VIS_UM))[0])


def global_mean(field2d: np.ndarray, lat_deg: np.ndarray) -> float:
    """cos(lat)-weighted mean of a (lat, lon) field on a regular grid."""
    w = np.cos(np.deg2rad(lat_deg))[:, None] * np.ones_like(field2d)
    return float(np.sum(field2d * w) / np.sum(w))


def annual_gm_550(path: Path, years) -> dict[int, float]:
    import cftime
    import netCDF4

    with netCDF4.Dataset(path) as nc:
        t = nc["time"]
        when = cftime.num2date(t[:], t.units, t.calendar)
        yrs = np.array([d.year for d in when])
        ib, lat = _band_550(nc), np.asarray(nc["lat"][:], dtype=np.float64)
        out = {}
        for y in years:
            idx = np.flatnonzero(yrs == y)
            if idx.size != 12:
                raise ValueError(f"{path}: year {y} has {idx.size} months")
            days = np.array([calendar.monthrange(y, m)[1] for m in range(1, 13)], float)
            aod = np.tensordot(days / days.sum(),
                               np.asarray(nc["aod"][idx, ib], dtype=np.float64), axes=1)
            out[y] = global_mean(aod, lat)
        return out, (int(yrs[0]), int(yrs[-1]), len(yrs))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fin", type=Path)
    ap.add_argument("tot", type=Path)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--macv1-ant", type=Path, help="MACv1 g30_ant_YYYY.nc (record 4 = 550 nm)")
    a = ap.parse_args(argv)
    sha = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"git {sha}")
    gm = {}
    years = sorted(set(a.years) | {_PI_YEAR})
    for kind, path in (("fin", a.fin), ("tot", a.tot)):
        gm[kind], cover = annual_gm_550(path, years)
        print(f"{kind} {path.name}: {cover[0]}-{cover[1]}, {cover[2]} months")
    for y in a.years:
        print(f"{y}: global-mean AOD550 fin {gm['fin'][y]:.4f} tot {gm['tot'][y]:.4f} "
              f"anthropogenic (fin - fin[{_PI_YEAR}]) {gm['fin'][y] - gm['fin'][_PI_YEAR]:.4f}")
    if a.macv1_ant:
        import netCDF4
        with netCDF4.Dataset(a.macv1_ant) as nc:
            days = np.array([calendar.monthrange(2001, m)[1] for m in range(1, 13)], float)
            aod = np.tensordot(days / days.sum(), np.asarray(nc["aod"][3], dtype=np.float64),
                               axes=1)                                   # (record, time, lat, lon)
            print(f"MACv1 anthropogenic {a.macv1_ant.name}: global-mean AOD550 "
                  f"{global_mean(aod, np.asarray(nc['lat'][:], dtype=np.float64)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
