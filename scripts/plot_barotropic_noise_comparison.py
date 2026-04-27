#!/usr/bin/env python
"""Plot V_baro time-mean spatial structure for baseline / divdamp / implicit runs.

Reproduces the smoking-gun figure
``results/ocean/momentum_budget_online/Vbaro_spatial_structure.png`` for
each of the three 1-yr GO+GM/Redi runs (baseline, Stage 0 divdamp,
Stage 3 implicit), for visual side-by-side comparison.

Usage:
    JAX_ENABLE_X64=1 python scripts/plot_barotropic_noise_comparison.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig


RUNS = [
    ("baseline (cosine filter)",
     "results/ocean/momentum_budget_online/tendency_3d_means.npz"),
    ("Stage 0: divdamp = 0.1",
     "results/ocean/momentum_budget_online_divdamp/tendency_3d_means.npz"),
    ("Stage 3: implicit CN",
     "results/ocean/momentum_budget_online_implicit/tendency_3d_means.npz"),
]

J_DRAKE = np.arange(2, 7)


def _depth_mean(field_3d, dz):
    H = float(dz.sum())
    return np.sum(field_3d * dz[None, None, :], axis=-1) / H


def main():
    cfg = GlobalOverturningConfig(use_gm_redi=True)
    z_coord = create_ocean_z_star(
        n_levels=cfg.n_levels, H_max=cfg.H_max,
        dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep,
    )
    grid = create_latlon_grid(36, 72)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)

    restart = np.load(
        "results/ocean/global_overturning_50yr_gmredi/restart_day018250.npz",
        allow_pickle=False,
    )
    v_mask = restart["v_mask"].astype(np.float64)

    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    lat_v_deg = np.concatenate([
        [lat_deg[0] - 0.5 * float(grid.dlat) * 180.0 / np.pi],
        0.5 * (lat_deg[:-1] + lat_deg[1:]),
        [lat_deg[-1] + 0.5 * float(grid.dlat) * 180.0 / np.pi],
    ])
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi

    available = [(label, path) for label, path in RUNS if Path(path).is_file()]
    if not available:
        print("No tendency_3d_means.npz files found — run the integrators first.",
              file=sys.stderr)
        sys.exit(1)

    n = len(available)
    fig, axes = plt.subplots(
        2, n, figsize=(6 * n, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    vmax = 0.05

    for col, (label, path) in enumerate(available):
        d = np.load(path, allow_pickle=False)
        v3 = d["state_v_mean"]
        V = _depth_mean(v3, dz) * v_mask  # (n_lat+1, n_lon)

        ax_top = axes[0, col]
        im = ax_top.pcolormesh(
            lon_deg, lat_v_deg, V * 100.0,
            cmap="RdBu_r", vmin=-vmax * 100.0, vmax=vmax * 100.0,
            shading="auto",
        )
        # Drake band
        ax_top.axhline(-77.5, color="k", linestyle=":", linewidth=0.7)
        ax_top.axhline(-57.5, color="k", linestyle=":", linewidth=0.7)
        ax_top.set_title(f"V_baro time-mean — {label}", fontsize=10)
        ax_top.set_xlabel("Longitude (deg)")
        ax_top.set_ylabel("Latitude (deg)")
        plt.colorbar(im, ax=ax_top, label="V_baro (cm/s)")

        ax_bot = axes[1, col]
        # Zonal mean over wet faces
        den = v_mask.sum(1)
        zm = np.where(den > 0, np.sum(V * v_mask, axis=1) / np.maximum(den, 1.0),
                      0.0)
        ax_bot.plot(lat_v_deg, zm * 100.0, marker="o", markersize=3, lw=1)
        ax_bot.axhline(0, color="k", lw=0.5)
        ax_bot.axvspan(-80, -55, color="green", alpha=0.1, label="Drake band")
        ax_bot.set_xlabel("Latitude (deg)")
        ax_bot.set_ylabel("⟨V_baro⟩_zonal (cm/s)")
        ax_bot.set_title(f"Zonal-mean V_baro — {label}", fontsize=10)
        ax_bot.grid(alpha=0.3)
        ax_bot.legend(loc="best", fontsize=8)

    plt.tight_layout()
    out = Path("results/ocean/Vbaro_three_way_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
