#!/usr/bin/env python
"""Score a legoESM tripole snapshot against the NEMO ORCA1 reference grid_T.

Both fields live on (different) curvilinear eORCA1 grids, so each is regridded
to a common regular lat-lon grid by cKDTree inverse-distance weighting over its
ocean cells (same pattern as ``legoesm.grids.regridding``, but for a *curvilinear*
2-D source rather than a regular 1-D-axis one). On the common grid we report
area-weighted (cos-lat) global means, bias, RMSE and pattern correlation for SST
and SSS, plus zonal means, and score against provisional tolerances.

Faithfulness caveats (see OMIP_faithful.md): runoff=0 in the legoESM forcing, so
SSS is gated (non-faithful) — reported for information, not pass/fail. Runs are
short spinups (a few years), not 40-yr equilibrium.

Usage:
    python scripts/compare_omip_nemo.py \
        --legoesm-snapshot results/omip_nemo/legoesm_tripole/snapshot_year004.npz \
        --nemo-gridt /…/RUN_REF/ORCA1_1m_..._grid_T.nc --nemo-time-idx -1 \
        --output-dir results/omip_nemo/compare_year4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Provisional "match" tolerances for a few-year spinup, model-vs-model
# (NEMO-vs-observation SST RMSE is ~0.5-1 K; we allow more for coarse legoESM).
_TOL = {
    "sst_rmse_excellent_C": 1.5, "sst_rmse_good_C": 2.5,
    "sss_rmse_excellent": 0.5, "sss_rmse_good": 1.0,   # gated (runoff=0)
}


def _xyz(lat_rad, lon_rad):
    """Unit-sphere Cartesian (inline; trivial, avoids a private cross-module import)."""
    cl = np.cos(lat_rad)
    return np.stack([cl * np.cos(lon_rad), cl * np.sin(lon_rad),
                     np.sin(lat_rad)], axis=-1)


def regrid_curv_to_latlon(field2d, src_lat_deg, src_lon_deg, ocean_mask,
                          tgt_lat_deg, tgt_lon_deg, k=4, max_deg=2.5):
    """IDW-regrid a curvilinear 2-D field (ocean cells only) onto a regular
    lat-lon grid. Returns (regridded 2-D, ocean_flag 2-D) where ocean_flag=0
    for target cells farther than ``max_deg`` from any source ocean cell."""
    from scipy.spatial import cKDTree
    m = np.asarray(ocean_mask).ravel() > 0.5
    if not m.any():
        raise ValueError("no ocean source cells")
    src_xyz = _xyz(np.deg2rad(np.asarray(src_lat_deg).ravel()[m]),
                   np.deg2rad(np.asarray(src_lon_deg).ravel()[m]))
    vals = np.asarray(field2d, dtype=np.float64).ravel()[m]
    tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lon_deg, tgt_lat_deg)
    tgt_xyz = _xyz(np.deg2rad(tgt_lat2d.ravel()), np.deg2rad(tgt_lon2d.ravel()))
    tree = cKDTree(src_xyz)
    d, idx = tree.query(tgt_xyz, k=k)
    d = np.maximum(d, 1e-12)
    w = (1.0 / d) / (1.0 / d).sum(axis=1, keepdims=True)
    out = (vals[idx] * w).sum(axis=1).reshape(tgt_lat2d.shape)
    # chord distance for max_deg separation on the unit sphere
    chord = 2.0 * np.sin(np.deg2rad(max_deg) / 2.0)
    ocean = (d[:, 0].reshape(tgt_lat2d.shape) < chord).astype(np.float64)
    return out, ocean


def _wstats(a, b, area):
    """Area-weighted bias, RMSE, pattern correlation of a vs b over area>0."""
    w = np.asarray(area, dtype=np.float64)
    sel = w > 0
    w = w[sel]; x = np.asarray(a)[sel]; y = np.asarray(b)[sel]
    W = w.sum()
    bias = float((w * (x - y)).sum() / W)
    rmse = float(np.sqrt((w * (x - y) ** 2).sum() / W))
    xm = (w * x).sum() / W; ym = (w * y).sum() / W
    cov = (w * (x - xm) * (y - ym)).sum() / W
    sx = np.sqrt((w * (x - xm) ** 2).sum() / W)
    sy = np.sqrt((w * (y - ym) ** 2).sum() / W)
    corr = float(cov / (sx * sy)) if sx > 0 and sy > 0 else float("nan")
    return {"lego_mean": float(xm), "nemo_mean": float(ym),
            "bias": bias, "rmse": rmse, "corr": corr}


def _load_legoesm(path):
    s = np.load(path)
    # lat_T/lon_T are written by run_omip_core2._save_snapshot ALREADY IN DEGREES
    # (via _grid_lat2d_deg, which applies np.rad2deg at save time). Applying
    # rad2deg AGAIN here corrupted the coordinates (45 deg -> 2578) -> the regrid
    # mapped every cell to nonsense lat-lon, scrambling the SST/SSS pattern
    # (corr ~0.1) and inflating the bias. Use the stored degrees as-is.
    return {
        "sst": np.asarray(s["T"])[..., 0], "sss": np.asarray(s["S"])[..., 0],
        "lat": np.asarray(s["lat_T"]),
        "lon": np.asarray(s["lon_T"]),
        "mask": np.asarray(s["land_mask"]),
    }


def _load_nemo(path, tidx):
    import xarray as xr
    ds = xr.open_dataset(path, decode_times=False)
    tdim = "time_counter" if "time_counter" in ds["tos"].dims else None
    sel = (lambda v: np.asarray(ds[v].isel({tdim: tidx})) if tdim
           else np.asarray(ds[v]))
    sst = sel("tos"); sss = sel("sos")
    lat = np.asarray(ds["nav_lat"]); lon = np.asarray(ds["nav_lon"])
    mask = (np.isfinite(sst) & (np.abs(sst) > 1e-6)).astype(np.float64)
    return {"sst": np.nan_to_num(sst), "sss": np.nan_to_num(sss),
            "lat": lat, "lon": lon % 360.0, "mask": mask,
            "n_time": int(ds.sizes.get("time_counter", 1))}


def _plot(out_dir, tgt_lat, tgt_lon, fields, ocean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, (L, N) in fields.items():
        Lm = np.where(ocean, L, np.nan); Nm = np.where(ocean, N, np.nan)
        vmin = np.nanmin([Lm, Nm]); vmax = np.nanmax([Lm, Nm])
        fig, ax = plt.subplots(1, 3, figsize=(18, 4))
        for a, dat, ttl in [(ax[0], Lm, f"legoESM {name}"),
                            (ax[1], Nm, f"NEMO {name}")]:
            im = a.pcolormesh(tgt_lon, tgt_lat, dat, vmin=vmin, vmax=vmax,
                              cmap="RdYlBu_r", shading="auto")
            a.set_title(ttl); plt.colorbar(im, ax=a, shrink=0.8)
        dmax = np.nanmax(np.abs(Lm - Nm))
        im = ax[2].pcolormesh(tgt_lon, tgt_lat, Lm - Nm, vmin=-dmax, vmax=dmax,
                              cmap="RdBu_r", shading="auto")
        ax[2].set_title(f"{name} diff (lego-NEMO)"); plt.colorbar(im, ax=ax[2], shrink=0.8)
        fig.tight_layout(); fig.savefig(out_dir / f"{name}_maps.png", dpi=90)
        plt.close(fig)
    # zonal means
    fig, ax = plt.subplots(1, len(fields), figsize=(6 * len(fields), 4))
    if len(fields) == 1:
        ax = [ax]
    for a, (name, (L, N)) in zip(ax, fields.items()):
        Lz = np.nanmean(np.where(ocean, L, np.nan), axis=1)
        Nz = np.nanmean(np.where(ocean, N, np.nan), axis=1)
        a.plot(Lz, tgt_lat, label="legoESM"); a.plot(Nz, tgt_lat, label="NEMO")
        a.set_title(f"zonal-mean {name}"); a.set_ylabel("lat"); a.legend()
    fig.tight_layout(); fig.savefig(out_dir / "zonal_means.png", dpi=90)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legoesm-snapshot", type=Path, required=True)
    p.add_argument("--nemo-gridt", type=Path, required=True)
    p.add_argument("--nemo-time-idx", type=int, default=-1,
                   help="NEMO grid_T time record (default last).")
    p.add_argument("--res-deg", type=float, default=1.0)
    p.add_argument("--output-dir", type=Path, default=Path("results/omip_nemo/compare"))
    p.add_argument("--freeze-clamp-C", type=float, default=None,
                   help="Floor legoESM SST at this temperature [deg C] before "
                        "scoring, to mimic NEMO's sea-ice-capped surface "
                        "(~-1.9 C). legoESM has no sea ice so high-lat cells "
                        "cool below freezing; this tests how much that inflates "
                        "the SST RMSE/bias vs NEMO.")
    args = p.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)

    L = _load_legoesm(args.legoesm_snapshot)
    N = _load_nemo(args.nemo_gridt, args.nemo_time_idx)
    print(f"[load] legoESM {L['sst'].shape}, NEMO {N['sst'].shape} "
          f"({N['n_time']} time records)")

    r = args.res_deg
    tgt_lat = np.arange(-89.5, 90.0, r)
    tgt_lon = np.arange(0.5, 360.0, r)

    if args.freeze_clamp_C is not None:
        n_below = int((L["sst"] < args.freeze_clamp_C).sum())
        L["sst"] = np.maximum(L["sst"], args.freeze_clamp_C)
        print(f"[freeze-clamp] floored legoESM SST at {args.freeze_clamp_C} C "
              f"({n_below} cells were below)")
    sstL, ocL = regrid_curv_to_latlon(L["sst"], L["lat"], L["lon"], L["mask"], tgt_lat, tgt_lon)
    sstN, ocN = regrid_curv_to_latlon(N["sst"], N["lat"], N["lon"], N["mask"], tgt_lat, tgt_lon)
    sssL, _ = regrid_curv_to_latlon(L["sss"], L["lat"], L["lon"], L["mask"], tgt_lat, tgt_lon)
    sssN, _ = regrid_curv_to_latlon(N["sss"], N["lat"], N["lon"], N["mask"], tgt_lat, tgt_lon)
    ocean = (ocL > 0.5) & (ocN > 0.5)
    area = (np.cos(np.deg2rad(tgt_lat))[:, None]
            * np.ones_like(tgt_lon)[None, :]) * ocean

    sst = _wstats(sstL, sstN, area)
    sss = _wstats(sssL, sssN, area)
    print(f"[SST] {sst}")
    print(f"[SSS] {sss}  (GATED: runoff=0, informational)")

    def _verdict(rmse, exc, good):
        return ("excellent" if rmse < exc else "good" if rmse < good else "poor")
    sst_v = _verdict(sst["rmse"], _TOL["sst_rmse_excellent_C"], _TOL["sst_rmse_good_C"])

    report = {
        "legoesm_snapshot": str(args.legoesm_snapshot),
        "nemo_gridt": str(args.nemo_gridt), "nemo_time_idx": args.nemo_time_idx,
        "n_ocean_cells": int(ocean.sum()),
        "SST": sst, "SST_verdict": sst_v,
        "SSS_gated_runoff0": sss,
        "tolerances": _TOL,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    _plot(out, tgt_lat, tgt_lon, {"SST": (sstL, sstN), "SSS": (sssL, sssN)}, ocean)
    print(f"[verdict] SST match = {sst_v} (RMSE {sst['rmse']:.3f} C, "
          f"bias {sst['bias']:.3f} C, corr {sst['corr']:.3f})")
    print(f"[done] report + plots -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
