"""Window T2m skill vs ERA5 on the SAME days, by region (experiment log metric).

  t2m_window_metrics.py ERA5_WINDOW.nc RUN_WINDOW_DIR [RUN_WINDOW_DIR ...]

ERA5_WINDOW.nc : one-time-step ERA5 tas window mean on its 0.25-degree grid
                 (climateeval_era5_window_ref.sh output).
RUN_WINDOW_DIR : cmor tree from cmor_window.py (<dir>/Amon/tas_*.nc and
                 <dir>/fx/sftlf_*.nc), model lat-lon grid.

ERA5 is box-averaged into each model cell (cos-lat weighted mean of the
0.25-degree points whose centres fall inside the model cell bounds), then
per region: cos-lat weighted bias (model - ERA5), RMSE, and R2 = 1 - SSres/SStot
(ClimateEval's area-weighted definition).  Land = sftlf >= 50 %.
Regions: global, land, ocean, tropics 30S-30N, NH extratropical land 30-90N,
45-70N land.  For two or more runs, also prints run - first run.
"""
import glob
import sys

import numpy as np
import xarray as xr


def _one(pattern):
    f = sorted(glob.glob(pattern))
    if len(f) != 1:
        raise SystemExit(f"expected one file for {pattern}, got {f}")
    return f[0]


def _bounds(c):
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid,
                           [c[-1] + (c[-1] - mid[-1])]])


def era5_on_model(era5_file, lat, lon, var="tas"):
    ds = xr.open_dataset(era5_file)
    t = ds[var].squeeze(drop=True)
    la = t[[d for d in t.dims if "lat" in d][0]].values
    lo = t[[d for d in t.dims if "lon" in d][0]].values % 360.0
    v = t.values
    if v.shape != (la.size, lo.size):
        v = v.T
    if not np.isfinite(v).all():
        raise SystemExit("ERA5 window has non-finite values")
    lb, ob = _bounds(lat), _bounds(lon % 360.0)
    i = np.clip(np.searchsorted(lb, la) - 1, 0, lat.size - 1)
    # longitude bins on the circle
    j = np.searchsorted(ob, lo) - 1
    j = np.where(j < 0, lon.size - 1, np.where(j >= lon.size, 0, j))
    w = np.cos(np.deg2rad(la))[:, None] * np.ones((1, lo.size))
    idx = (i[:, None] * lon.size + j[None, :]).ravel()
    s = np.bincount(idx, weights=(w * v).ravel(), minlength=lat.size * lon.size)
    n = np.bincount(idx, weights=w.ravel(), minlength=lat.size * lon.size)
    if (n == 0).any():
        raise SystemExit("a model cell received no ERA5 point")
    return (s / n).reshape(lat.size, lon.size)


def load_run(d):
    t = xr.open_dataset(_one(f"{d}/Amon/tas_*.nc"))["tas"].squeeze(drop=True)
    f = xr.open_dataset(_one(f"{d}/fx/sftlf_*.nc"))["sftlf"].squeeze(drop=True)
    return t["lat"].values, t["lon"].values, t.values, f.values


def metrics(m, r, wgt, mask):
    w = wgt[mask]
    b = (m - r)[mask]
    bias = np.average(b, weights=w)
    rmse = np.sqrt(np.average(b ** 2, weights=w))
    rr = r[mask]
    sst = np.average((rr - np.average(rr, weights=w)) ** 2, weights=w)
    return bias, rmse, 1.0 - np.average(b ** 2, weights=w) / sst


def main(era5_file, runs):
    lat, lon, m0, lf = load_run(runs[0])
    ref = era5_on_model(era5_file, lat, lon)
    wgt = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    LA = lat[:, None] * np.ones((1, lon.size))
    land = lf >= 50.0
    regions = {"global": np.ones_like(land), "land": land, "ocean": ~land,
               "tropics30": np.abs(LA) <= 30.0,
               "NHext_land30-90": land & (LA >= 30.0),
               "land45-70N": land & (LA >= 45.0) & (LA <= 70.0)}
    first = None
    print(f"ERA5 window: {era5_file}")
    for d in runs:
        la2, lo2, m, lf2 = load_run(d)
        if not (np.array_equal(la2, lat) and np.array_equal(lo2, lon)
                and np.array_equal(lf2, lf)):
            raise SystemExit(f"{d}: grid or land fraction differs from {runs[0]}")
        if not np.isfinite(m).all():
            raise SystemExit(f"{d}: non-finite model tas")
        print(f"\n{d}")
        print(f"  {'region':16s} {'bias K':>8s} {'RMSE K':>8s} {'R2':>6s}"
              + ("  d(bias) d(RMSE) vs first" if first is not None else ""))
        for name, mk in regions.items():
            b, e, r2 = metrics(m, ref, wgt, mk)
            extra = ""
            if first is not None:
                b0, e0, _ = metrics(first, ref, wgt, mk)
                extra = f"  {b - b0:+7.2f} {e - e0:+7.2f}"
            print(f"  {name:16s} {b:+8.2f} {e:8.2f} {r2:6.3f}{extra}")
        if first is None:
            first = m


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
