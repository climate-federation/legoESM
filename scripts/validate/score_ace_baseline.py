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

# ACE emits TMP500/TMP850/TMP2m as DIRECT pressure/surface temperatures (the
# training_validation truth carries them too) — TMP500 is the exact T@500hPa,
# the apples-to-apples match for the legoESM AIMIP families' headline metric.
_VARS = ["TMP500", "TMP850", "TMP2m", "air_temperature_4"]


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


def score(run_dir: str, truth_path: str, step: int = 1) -> dict:
    """Score ACE predictions vs an EXTERNAL ERA5 truth file.

    The fme inference run writes ``autoregressive_predictions.nc`` but its
    ``autoregressive_target.nc`` carries no verification fields, so the
    truth is taken from the shipped ``training_validation`` ERA5 dataset
    (which DOES hold TMP500/TMP850/...), aligned by the prediction's
    ``valid_time``.  ``step`` is the forecast-step index to score (1 = the
    first 6 h lead, matching the AIMIP families).
    """
    import xarray as xr
    run = Path(run_dir)
    pred = xr.open_dataset(run / "autoregressive_predictions.nc")
    truth = xr.open_dataset(truth_path)
    latname = next((c for c in ("lat", "latitude", "grid_yt") if c in pred.coords),
                   None)
    if latname is None:
        raise SystemExit(f"no lat coord in {list(pred.coords)}")
    lat = pred[latname].values
    tdim = next((d for d in ("time", "timestep", "forecast_step", "sample")
                 if d in pred.dims), None)
    if tdim is None:
        raise SystemExit(f"no time-like dim in predictions {list(pred.dims)}")
    idx = step if pred.sizes[tdim] > step else pred.sizes[tdim] - 1

    out = {}
    for v in _VARS:
        if v not in pred or v not in truth:
            continue
        p = pred[v].isel({tdim: idx}).squeeze()
        # Align the truth to this prediction's valid time.
        vt = p["valid_time"].values if "valid_time" in p.coords else None
        ttdim = next((d for d in ("time", "valid_time") if d in truth[v].dims), None)
        if vt is not None and ttdim is not None:
            t = truth[v].sel({ttdim: vt}, method="nearest").squeeze()
        elif ttdim is not None:
            t = truth[v].isel({ttdim: idx}).squeeze()
        else:
            t = truth[v].squeeze()
        rmse, bias = _area_weighted_rmse_bias(p.values, t.values, lat)
        out[v] = {"rmse": rmse, "bias": bias}
        print(f"  {v:20s} rmse={rmse:8.3f}  bias={bias:+8.3f}")
    if not out:
        print("  (no scored vars — check var names match between pred + truth)")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: score_ace_baseline.py <ace_run_dir> <era5_truth.nc>")
    print("ACE2-ERA5 baseline skill (first 6h step), area-weighted:")
    res = score(sys.argv[1], sys.argv[2])
    import json
    (Path(sys.argv[1]) / "ace_baseline_score.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {sys.argv[1]}/ace_baseline_score.json")
