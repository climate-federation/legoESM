#!/usr/bin/env python
"""Score an ACE2-ERA5 (fme) inference run as the AIMIP SFNO baseline.

fme writes ``autoregressive_predictions.nc`` + ``autoregressive_target.nc``
(ACE's own held-out ERA5 target) into the run dir.  This computes the
area-weighted RMSE/bias for the mid-atmosphere temperature
(``air_temperature_4`` — ACE's 5th of 8 model levels, the ~500 hPa analog),
2 m temperature, and 850 hPa temperature, at the FIRST forward step (the
6 h lead that matches the AIMIP eval) — so ACE's skill is reported on the
same footing as the legoESM classical/column_nn/sfno families.

Usage: python score_ace_baseline.py results/ace_baseline/run_<jobid>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# xarray is a heavy import (only needed for the .nc read in score()); keep it
# function-scoped so the area-weighting math stays importable/testable cheaply.

# ACE model-level temperatures + named diagnostics worth scoring.  air_*_4 is
# the mid-column (~500 hPa analog); TMP850/TMP2m are named pressure/surface
# fields ACE emits directly.
_VARS = ["air_temperature_4", "TMP850", "TMP2m", "air_temperature_0",
         "air_temperature_7"]


def _area_weighted_rmse_bias(pred, tgt, lat):
    """cos(lat)-weighted RMSE + bias over (lat, lon)."""
    w = np.cos(np.deg2rad(np.asarray(lat)))
    w = w / w.sum()
    d = np.asarray(pred) - np.asarray(tgt)
    # weight along the lat axis; uniform along lon
    wcol = w[:, None]
    rmse = float(np.sqrt((wcol * d**2).sum() / (wcol * np.ones_like(d)).sum()))
    bias = float((wcol * d).sum() / (wcol * np.ones_like(d)).sum())
    return rmse, bias


def score(run_dir: str) -> dict:
    import xarray as xr
    run = Path(run_dir)
    pred = xr.open_dataset(run / "autoregressive_predictions.nc")
    tgt = xr.open_dataset(run / "autoregressive_target.nc")
    # Identify lat coordinate name.
    latname = next((c for c in ("lat", "latitude", "grid_yt") if c in pred.coords),
                   None)
    if latname is None:
        raise SystemExit(f"no lat coord in {list(pred.coords)}")
    lat = pred[latname].values
    # First forecast step (the 6 h lead) — dim is usually 'time'/'timestep'.
    tdim = next((d for d in ("time", "timestep", "forecast_step", "sample")
                 if d in pred.dims), None)

    out = {}
    for v in _VARS:
        if v not in pred or v not in tgt:
            continue
        p, t = pred[v], tgt[v]
        if tdim is not None and tdim in p.dims:
            # step index 1 = first PREDICTED step (0 is the IC); fall back to 0.
            idx = 1 if p.sizes[tdim] > 1 else 0
            p = p.isel({tdim: idx})
            t = t.isel({tdim: idx})
        # collapse any leftover singleton (ensemble/sample) dims
        p = p.squeeze()
        t = t.squeeze()
        rmse, bias = _area_weighted_rmse_bias(p.values, t.values, lat)
        out[v] = {"rmse": rmse, "bias": bias}
        print(f"  {v:20s} rmse={rmse:8.3f}  bias={bias:+8.3f}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: score_ace_baseline.py <ace_run_dir>")
    print(f"ACE2-ERA5 baseline skill (first 6h step), area-weighted:")
    res = score(sys.argv[1])
    import json
    (Path(sys.argv[1]) / "ace_baseline_score.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {sys.argv[1]}/ace_baseline_score.json")
