#!/usr/bin/env python
"""Which sea-level reduction reproduces ERA5 mean sea-level pressure?

Feeds ERA5's OWN surface pressure, surface geopotential and 2 m temperature
(the surface-temperature stand-in) to candidate reductions and scores each
against ERA5 ``mean_sea_level_pressure`` on land / ocean, cos-lat weighted.
No model state, no T63 truncation: a mismatch here is the formula, not the
model or the spectral round-trip.

Candidates: the ECMWF/Trenberth reduction with hard limits, and smoothed at
each ``--widths`` value. The previous formula (column-mean T in the
hypsometric term) scored 604 Pa land RMSE here at git 316b7d6434 and was
removed.

Usage:
    python scripts/validate/wb_mslp_reduction_probe.py --year 2020 --n-times 24 \
        --out logs/wb_floor/mslp_probe.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

_ZARR = ("gs://weatherbench2/datasets/era5/"
         "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")


def score(pred, truth, land, lat_deg):
    w = np.broadcast_to(np.cos(np.deg2rad(lat_deg))[:, None], pred.shape[-2:])
    out = {}
    for name, m in (("all", np.ones_like(land, bool)), ("land", land), ("ocean", ~land)):
        ww = np.where(m, w, 0.0)
        err = pred - truth
        out[name] = {
            "rmse": float(np.sqrt((ww * err ** 2).sum(axis=(-2, -1)).mean() / ww.sum())),
            "bias": float((ww * err).sum(axis=(-2, -1)).mean() / ww.sum()),
        }
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--n-times", type=int, default=24, dest="n_times")
    p.add_argument("--stride-hours", type=int, default=360, dest="stride_hours")
    p.add_argument("--coarsen", type=int, default=6,
                   help="take every Nth 0.25 deg point (6 = 1.5 deg, the WB2 grid)")
    p.add_argument("--widths", type=float, nargs="*", default=[],
                   help="also score the smoothed ECMWF limits at these widths [K]")
    p.add_argument("--out", default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import xarray as xr
    from evaluations.headline_diagnostics import (
        mean_sea_level_pressure_ecmwf)

    ds = xr.open_zarr(_ZARR, storage_options={"token": "anon"})
    times = np.datetime64(f"{a.year}-01-01T00") + np.arange(a.n_times) * np.timedelta64(a.stride_hours, "h")
    sl = dict(latitude=slice(None, None, a.coarsen), longitude=slice(None, None, a.coarsen))
    sub = ds[["surface_pressure", "2m_temperature", "mean_sea_level_pressure"]].sel(time=times).isel(**sl).load()
    phis = ds["geopotential_at_surface"].isel(**sl).load().values
    land = ds["land_sea_mask"].isel(**sl).load().values >= 0.5
    lat = sub["latitude"].values

    p_s = jnp.asarray(sub["surface_pressure"].values)
    t_s = jnp.asarray(sub["2m_temperature"].values)
    truth = sub["mean_sea_level_pressure"].values
    phis_j = jnp.asarray(phis)

    cands = {
        "ecmwf": np.asarray(mean_sea_level_pressure_ecmwf(p_s, t_s, phis_j, limit_width_k=0.0)),
    }
    for w in a.widths:
        cands[f"ecmwf_w{w:g}K"] = np.asarray(
            mean_sea_level_pressure_ecmwf(p_s, t_s, phis_j, limit_width_k=w))
    out = {
        "meta": {"zarr": _ZARR, "year": a.year, "n_times": a.n_times,
                 "stride_hours": a.stride_hours, "coarsen": a.coarsen,
                 "surface_T": "ERA5 2m_temperature",
                 "land": "ERA5 land_sea_mask >= 0.5", "weights": "cos(lat)",
                 "git": subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True, cwd=_REPO).stdout.strip()},
        "scores": {k: score(v, truth, land, lat) for k, v in cands.items()},
    }
    text = json.dumps(out, indent=2)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
