#!/usr/bin/env python3
"""Write a CMOR Amon tree holding a run's mean over a DAY WINDOW of its last month.

ClimateEval scores calendar-month CMOR files.  A restart arm's December file is
the mean from Dec 1 (or its restart day) to the run's end; to score a window
(e.g. Dec 11-31) the window's sum is recovered from the monthly-accumulator
sidecars the run writes every day (same method as window_diff.py):

    S_end = mean_file * n_end,  n_end = c_start + per_day * (end - start)
    window = (S_end - S_start) / (n_end - c_start)

``per_day`` (24 for hourly-sampled state, 1 for daily flux means) is MEASURED
per variable from two consecutive sidecars, and asserted constant over the
window.  The 3-D sidecar's level order is matched to the file by spatial
correlation (refuses below 0.8).  Only windowed variables are written; fx is
copied.  Files get a ``window`` global attribute.

    cmor_window.py <run_dir> <start_day> <end_day> <out_dir>
"""
from __future__ import annotations

import glob
import json
import pathlib
import shutil
import sys

import numpy as np
import xarray as xr

MIN_CORR = 0.8


def bucket(path, month):
    z = np.load(path, allow_pickle=True)
    man = json.loads(str(z["monthly.__manifest__"]))
    out = {}
    for kind in ("data_2d", "data_3d"):
        for y, m, var, count, key in man[kind]:
            if m == month:
                out[var] = (np.asarray(z[f"monthly.{key}"], np.float64), int(count))
    return out


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main(run, start, end, out):
    run, out = pathlib.Path(run), pathlib.Path(out)
    files = sorted(glob.glob(str(run / "cmor" / "Amon" / "*_Amon_*_gn_*12.nc")))
    if not files:
        raise SystemExit(f"{run}: no Amon file ending in December")
    s0 = bucket(run / f"cmor_accum_day_{start:04d}.npz", 12)
    s1 = bucket(run / f"cmor_accum_day_{start + 1:04d}.npz", 12)
    last = end - 1                                   # terminal day has no sidecar
    sl = bucket(run / f"cmor_accum_day_{last:04d}.npz", 12)
    (out / "cmor" / "Amon").mkdir(parents=True, exist_ok=True)
    for f in files:
        var = pathlib.Path(f).name.split("_")[0]
        if var not in s0:
            print(f"skip {var}: not in sidecar")
            continue
        sum0, c0 = s0[var]
        per_day = s1[var][1] - c0
        if per_day <= 0 or sl[var][1] - c0 != per_day * (last - start):
            raise SystemExit(f"{var}: sample cadence not constant "
                             f"({c0}, {s1[var][1]}, {sl[var][1]})")
        n_end = c0 + per_day * (end - start)
        ds = xr.open_dataset(f, decode_times=False)
        mf = np.asarray(ds[var].isel(time=-1).values, np.float64)
        if sum0.ndim == 3:
            # sidecar is (lat, lon, lev); the file is (lev, lat, lon), and the
            # sidecar's level order is matched to the file's by correlation.
            sum0 = np.moveaxis(sum0, -1, 0)
            if corr(np.nan_to_num(sum0[::-1] / c0), np.nan_to_num(mf)) > corr(
                    np.nan_to_num(sum0 / c0), np.nan_to_num(mf)):
                sum0 = sum0[::-1]
        if sum0.shape != mf.shape:
            raise SystemExit(f"{var}: sidecar {sum0.shape} vs file {mf.shape}")
        r = corr(np.nan_to_num(sum0 / c0), np.nan_to_num(mf))
        if not r > MIN_CORR:
            raise SystemExit(f"{var}: sidecar/file correlation {r:.3f}")
        win = (mf * n_end - sum0) / (n_end - c0)
        data = ds[var].values.copy()
        data[-1] = win.astype(data.dtype)
        ds2 = ds.copy()
        ds2[var].values[...] = data
        ds2.attrs["window"] = (f"elapsed days {start}..{end} (mean of the last "
                               f"{end - start} days of the month), cmor_window.py")
        ds2.to_netcdf(out / "cmor" / "Amon" / pathlib.Path(f).name)
        print(f"{var}: per_day {per_day}, n {n_end - c0}, corr {r:.3f}")
    fx = run / "cmor" / "fx"
    if fx.is_dir():
        shutil.copytree(fx, out / "cmor" / "fx", dirs_exist_ok=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) != 4:
        raise SystemExit(__doc__)
    main(a[0], int(a[1]), int(a[2]), a[3])
