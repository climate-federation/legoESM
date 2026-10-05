"""Window temperature bias by pressure level and region vs ERA5 (same days).

  t_profile_window_bias.py ERA5_TA_WINDOW.nc RUN_WINDOW_DIR [RUN2 ...]

ERA5_TA_WINDOW.nc: ERA5 'ta'/'t' window mean on a few pressure levels
(cdo timmean of the daily pool, levels in Pa).  RUN_WINDOW_DIR: cmor_window.py
tree (Amon/ta_*.nc plev19, Amon/tas_*.nc, fx/sftlf).  ERA5 box-averaged into
model cells (t2m_window_metrics.era5_on_model).  Cells where either side is
missing at a level (below ground) are excluded at that level; the count of
used cells is printed.  Also prints the 2-m row (ERA5 2-m needs a separate
file, so the 2-m row uses the tas window metrics probe; here only levels).
"""
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2m_window_metrics import _bounds, _one, load_run  # noqa: E402


def box_mean(v, la, lo, lat, lon):
    lb, ob = _bounds(lat), _bounds(lon % 360.0)
    i = np.clip(np.searchsorted(lb, la) - 1, 0, lat.size - 1)
    j = np.searchsorted(ob, lo % 360.0) - 1
    j = np.where(j < 0, lon.size - 1, np.where(j >= lon.size, 0, j))
    w = np.cos(np.deg2rad(la))[:, None] * np.ones((1, lo.size))
    idx = (i[:, None] * lon.size + j[None, :]).ravel()
    s = np.bincount(idx, weights=(w * v).ravel(), minlength=lat.size * lon.size)
    n = np.bincount(idx, weights=w.ravel(), minlength=lat.size * lon.size)
    return (s / n).reshape(lat.size, lon.size)


def main(e5file, runs):
    e5 = xr.open_dataset(e5file)
    var = [v for v in e5.data_vars if e5[v].ndim >= 3][0]
    t5 = e5[var].squeeze(drop=True)
    levdim = [d for d in t5.dims if "lev" in d or "plev" in d][0]
    la = t5["lat"].values
    lo = t5["lon"].values
    for d in runs:
        lat, lon, _tas, lf = load_run(d)
        ta = xr.open_dataset(_one(f"{d}/Amon/ta_*.nc"))["ta"].squeeze(drop=True)
        LA = lat[:, None] * np.ones((1, lon.size))
        land = lf >= 50.0
        regions = {"land45-70N": land & (LA >= 45) & (LA <= 70),
                   "ocean45-70N": ~land & (LA >= 45) & (LA <= 70),
                   "land30-45N": land & (LA >= 30) & (LA < 45),
                   "ocean30-45N": ~land & (LA >= 30) & (LA < 45),
                   "tropics30": np.abs(LA) <= 30}
        w = np.cos(np.deg2rad(LA))
        print(f"\n{d}\n  {'plev hPa':>8s} " + " ".join(f"{k:>14s}" for k in regions))
        for p in t5[levdim].values:
            r = box_mean(t5.sel({levdim: p}).values, la, lo, lat, lon)
            m = ta.sel(plev=p, method="nearest")
            if abs(float(m["plev"]) - p) > 1.0:
                raise SystemExit(f"model has no level {p}")
            m = np.where(np.abs(m.values) > 1e10, np.nan, m.values)
            out = []
            for mk in regions.values():
                ok = mk & np.isfinite(m) & np.isfinite(r)
                out.append(f"{np.average((m - r)[ok], weights=w[ok]):+7.2f} n{ok.sum():5d}")
            print(f"  {p / 100:8.0f} " + " ".join(f"{s:>14s}" for s in out))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
