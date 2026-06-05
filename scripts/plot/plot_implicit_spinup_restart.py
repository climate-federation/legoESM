#!/usr/bin/env python
"""Quick visualization of an implicit-spinup restart file.

Plots:
  - SSH (eta)
  - SST (T at surface)
  - Surface speed (|u| at top level)
  - Zonal-mean u(lat, depth)
  - Zonal-mean T(lat, depth)
  - Bottom-cell V_baro grid-noise diagnostic (3-pt Laplacian over latitude)

Usage:
    python scripts/plot_implicit_spinup_restart.py <day>
    python scripts/plot_implicit_spinup_restart.py        # plots latest
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

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig


SPINUP_DIR = Path("results/ocean/global_overturning_implicit_spinup")


def main():
    if len(sys.argv) > 1:
        day = int(sys.argv[1])
        restart = SPINUP_DIR / f"restart_day{day:06d}.npz"
    else:
        files = sorted(SPINUP_DIR.glob("restart_day*.npz"))
        if not files:
            print(f"No restart files in {SPINUP_DIR}")
            return
        restart = files[-1]
        day = int(restart.stem.removeprefix("restart_day"))

    if not restart.exists():
        print(f"Not found: {restart}")
        return

    print(f"Plotting {restart.name} (sim day {day} = year {day/365:.2f})")
    d = np.load(restart, allow_pickle=False)

    cfg = GlobalOverturningConfig(use_gm_redi=True)
    z_coord = create_ocean_z_star(
        n_levels=cfg.n_levels, H_max=cfg.H_max,
        dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep,
    )
    grid = create_latlon_grid(36, 72)
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    H_max = cfg.H_max
    lat = np.asarray(grid.lat) * 180 / np.pi
    lon = np.asarray(grid.lon) * 180 / np.pi

    eta = d["eta"]
    T = d["T"]
    u = d["u"]
    v = d["v"]
    H_bathy = d["H_bathy"]
    mask = d["land_mask"]
    u_mask = d["u_mask"]
    v_mask = d["v_mask"]

    # Cell-centred surface speed
    u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
    v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
    speed_sfc = np.sqrt(u_c**2 + v_c**2)
    speed_sfc = np.where(mask > 0.5, speed_sfc, np.nan)

    eta_plot = np.where(mask > 0.5, eta, np.nan)
    sst_plot = np.where(mask > 0.5, T[:, :, 0], np.nan)

    # Zonal-mean u(lat, depth) — wet-cell average over u-faces
    m = u_mask[:, :, None]
    den = m.sum(axis=1)
    zm_u = np.where(den > 0,
                    (u * m).sum(axis=1) / np.where(den > 0, den, 1.0),
                    np.nan)

    # Zonal-mean T(lat, depth)
    zm_T = np.where(mask[:, :, None] > 0.5,
                     T, np.nan)
    T_zonal = np.nanmean(zm_T, axis=1)

    # V_baro grid-noise indicator
    H_total_cell = (eta + H_bathy) * mask
    H_v_int = 0.5 * (H_total_cell[:-1] + H_total_cell[1:])
    H_v = np.concatenate([np.zeros((1, 72)), H_v_int, np.zeros((1, 72))], axis=0)
    h_v = dz[None, None, :] * (H_v / H_max)[..., None]
    V_bar = np.where(H_v > 1e-3,
                     np.sum(v * h_v, axis=-1) / np.maximum(H_v, 1e-3), 0)
    V_bar_masked = np.where(v_mask > 0.5, V_bar, np.nan)
    # 3-pt meridional Laplacian
    Lap_V = (V_bar_masked[:-2] - 2 * V_bar_masked[1:-1] + V_bar_masked[2:])
    Lap_V_padded = np.concatenate(
        [np.full((1, 72), np.nan), Lap_V, np.full((1, 72), np.nan)], axis=0,
    )

    # ---- Plot ----
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # SSH
    ax = fig.add_subplot(gs[0, 0])
    vmax = float(np.nanmax(np.abs(eta_plot)))
    im = ax.pcolormesh(lon, lat, eta_plot, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto")
    plt.colorbar(im, ax=ax, label="η (m)")
    ax.set_title("SSH"); ax.set_xlabel("Lon"); ax.set_ylabel("Lat")

    # SST
    ax = fig.add_subplot(gs[0, 1])
    im = ax.pcolormesh(lon, lat, sst_plot, cmap="RdYlBu_r", shading="auto")
    plt.colorbar(im, ax=ax, label="T (°C)")
    ax.set_title("SST"); ax.set_xlabel("Lon"); ax.set_ylabel("Lat")

    # Surface speed
    ax = fig.add_subplot(gs[0, 2])
    vmax_speed = float(np.nanmax(speed_sfc))
    im = ax.pcolormesh(lon, lat, speed_sfc, cmap="magma",
                       vmin=0, vmax=vmax_speed, shading="auto")
    plt.colorbar(im, ax=ax, label="|u| (m/s)")
    ax.set_title(f"Surface speed (max {vmax_speed:.3f} m/s)")
    ax.set_xlabel("Lon"); ax.set_ylabel("Lat")

    # Zonal-mean u
    ax = fig.add_subplot(gs[1, :])
    vmax_u = float(np.nanmax(np.abs(zm_u)))
    if vmax_u == 0 or not np.isfinite(vmax_u):
        vmax_u = 0.01
    im = ax.pcolormesh(lat, z_full, zm_u.T, cmap="RdBu_r",
                       vmin=-vmax_u, vmax=vmax_u, shading="auto")
    plt.colorbar(im, ax=ax, label="u (m/s)")
    ax.set_title(f"Zonal-mean u(lat, depth) — max |u| = {vmax_u:.3f} m/s")
    ax.set_xlabel("Lat"); ax.set_ylabel("Depth (m)")

    # Zonal-mean T
    ax = fig.add_subplot(gs[2, 0:2])
    im = ax.pcolormesh(lat, z_full, T_zonal.T, cmap="RdYlBu_r",
                       shading="auto")
    plt.colorbar(im, ax=ax, label="T (°C)")
    ax.set_title("Zonal-mean T(lat, depth)")
    ax.set_xlabel("Lat"); ax.set_ylabel("Depth (m)")

    # V_baro Laplacian (grid-noise indicator)
    ax = fig.add_subplot(gs[2, 2])
    vmax_lap = float(np.nanmax(np.abs(Lap_V_padded))) if np.any(np.isfinite(Lap_V_padded)) else 1e-3
    if vmax_lap == 0:
        vmax_lap = 1e-4
    sigma = float(np.nanstd(Lap_V_padded))
    # V_baro lives at v-face latitudes (n_lat+1 points)
    dlat = float(grid.dlat)
    lat_v = np.concatenate([
        [lat[0] - 0.5 * dlat * 180 / np.pi],
        0.5 * (lat[:-1] + lat[1:]),
        [lat[-1] + 0.5 * dlat * 180 / np.pi],
    ])
    im = ax.pcolormesh(lon, lat_v, Lap_V_padded, cmap="RdBu_r",
                       vmin=-vmax_lap, vmax=vmax_lap, shading="auto")
    plt.colorbar(im, ax=ax, label="∇²_lat V (m/s)")
    ax.set_title(f"V_baro grid noise\nσ = {sigma:.2e}  (target < 1e-2)")
    ax.set_xlabel("Lon"); ax.set_ylabel("Lat")

    plt.suptitle(
        f"Implicit-solver spinup — day {day} (year {day/365:.2f})",
        fontsize=12,
    )
    out = SPINUP_DIR / f"snapshot_day{day:06d}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # Also print quick health summary
    print(f"\nState summary at day {day}:")
    print(f"  η range: [{float(np.nanmin(eta_plot)):+.4f}, "
          f"{float(np.nanmax(eta_plot)):+.4f}] m")
    print(f"  SST range: [{float(np.nanmin(sst_plot)):+.2f}, "
          f"{float(np.nanmax(sst_plot)):+.2f}] °C")
    print(f"  Max surface speed: {float(np.nanmax(speed_sfc)):.4f} m/s")
    print(f"  Zonal-mean u extrema: [{float(np.nanmin(zm_u)):+.4f}, "
          f"{float(np.nanmax(zm_u)):+.4f}] m/s")
    print(f"  σ(∇²_lat V_baro): {sigma:.3e} m/s  (target < 1e-2 by Crit 1.3)")


if __name__ == "__main__":
    main()
