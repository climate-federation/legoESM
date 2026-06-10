#!/usr/bin/env python
"""Diagnostic plots for the 5° realistic-geometry GO 50-yr rerun.

Produces 5 plots using the final restart:
  1. SSH map
  2. SST map
  3. Surface speed map
  4. Zonal-mean T(lat, depth)
  5. MOC streamfunction

Usage:
    python scripts/run/global_overturning/plot_5deg_rerun_diagnostics.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig
from legoesm.ocean.diagnostics_streamfunction import moc_streamfunction

RUN_DIR = Path("results/ocean/global_overturning_realistic_50yr_polar_cap")
ETOPO_FILE = Path("data/bathymetry/etopo_1deg.nc")


def main():
    # Find the last restart
    restarts = sorted(RUN_DIR.glob("restart_day*.npz"))
    if not restarts:
        print(f"No restarts in {RUN_DIR}")
        return
    last = restarts[-1]
    day = int(last.stem.removeprefix("restart_day"))
    yr = day / 365.0
    print(f"Using restart: {last.name} (year {yr:.1f})")
    d = np.load(last, allow_pickle=False)

    # Reconstruct grid
    cfg = GlobalOverturningConfig(H_max=5000.0, dz_surface=20.0)
    grid = create_latlon_grid(36, 72)
    z_base = create_ocean_z_star(
        n_levels=cfg.n_levels, H_max=cfg.H_max,
        dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep,
    )
    bathy_cfg = BathymetryConfig(
        source="file", path=str(ETOPO_FILE),
        H_max=cfg.H_max, H_min=50.0, smoothing_passes=5,
        enforce_straits=True, fill_isolated_basins=True,
        depth_is_negative=True, r_factor_max=0.2, north_cap_lat=80.0,
    )
    H_bathy_jax, _ = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy = np.asarray(H_bathy_jax)
    z_coord = create_partial_cell_coordinate(
        z_base, jnp.asarray(H_bathy, dtype=jnp.float32),
    )
    h_partial = np.asarray(z_coord.h_partial)
    z_full = -np.abs(np.asarray(z_coord.z_full_ref))
    lat = np.asarray(grid.lat) * 180 / np.pi
    lon = np.asarray(grid.lon) * 180 / np.pi
    lat_v = np.linspace(-90, 90, len(lat) + 1)

    eta = d["eta"]
    T = d["T"]
    u = d["u"]
    v = d["v"]
    mask = d["land_mask"]
    ocean = mask > 0.5

    u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
    v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
    speed = np.sqrt(u_c**2 + v_c**2)

    # --- Figure 1: SSH ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ssh = np.where(ocean, eta, np.nan)
    vmax = float(np.nanmax(np.abs(ssh)))
    vmax = max(vmax, 0.1)
    im = ax.pcolormesh(lon, lat, ssh, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto")
    plt.colorbar(im, ax=ax, label="SSH (m)", fraction=0.025)
    ax.set_title(f"Sea Surface Height — Year {yr:.0f}")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "ssh_final.png", dpi=150)
    plt.close()
    print(f"Saved {RUN_DIR / 'ssh_final.png'}")

    # --- Figure 2: SST ---
    fig, ax = plt.subplots(figsize=(12, 5))
    sst = np.where(ocean, T[:, :, 0], np.nan)
    im = ax.pcolormesh(lon, lat, sst, cmap="RdYlBu_r", shading="auto")
    plt.colorbar(im, ax=ax, label="SST (°C)", fraction=0.025)
    ax.set_title(f"Sea Surface Temperature — Year {yr:.0f}")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "sst_final.png", dpi=150)
    plt.close()
    print(f"Saved {RUN_DIR / 'sst_final.png'}")

    # --- Figure 3: Surface speed ---
    fig, ax = plt.subplots(figsize=(12, 5))
    spd = np.where(ocean, speed, np.nan)
    im = ax.pcolormesh(lon, lat, spd, cmap="magma", shading="auto",
                       vmin=0, vmax=float(np.nanpercentile(spd, 99)))
    plt.colorbar(im, ax=ax, label="Speed (m/s)", fraction=0.025)
    ax.set_title(f"Surface Current Speed — Year {yr:.0f}")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "speed_final.png", dpi=150)
    plt.close()
    print(f"Saved {RUN_DIR / 'speed_final.png'}")

    # --- Figure 4: Zonal-mean T(lat, depth) ---
    T_masked = np.where(ocean[:, :, None], T, np.nan)
    T_zm = np.nanmean(T_masked, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    T_levels = np.arange(0, 26, 2)
    im = ax.pcolormesh(lat, z_full, T_zm.T, cmap="RdYlBu_r",
                       shading="auto", vmin=0, vmax=25)
    cs = ax.contour(lat, z_full, T_zm.T, levels=T_levels,
                    colors="k", linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%g")
    plt.colorbar(im, ax=ax, label="T (°C)", fraction=0.025)
    ax.set_title(f"Zonal-Mean Temperature — Year {yr:.0f}")
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Depth (m)")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "T_zonal_mean_final.png", dpi=150)
    plt.close()
    print(f"Saved {RUN_DIR / 'T_zonal_mean_final.png'}")

    # --- Figure 5: MOC streamfunction ---
    psi_moc = moc_streamfunction(v, h_partial, eta, H_bathy, mask, grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    psi_max = max(float(np.nanmax(np.abs(psi_moc))), 1.0)
    psi_levels = np.linspace(-psi_max, psi_max, 21)
    im = ax.contourf(lat_v, z_full, psi_moc.T,
                     levels=psi_levels, cmap="RdBu_r", extend="both")
    cs = ax.contour(lat_v, z_full, psi_moc.T, levels=psi_levels[::4],
                    colors="k", linewidths=0.4, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    plt.colorbar(im, ax=ax, label="ψ (Sv)", fraction=0.025)
    ax.set_title(f"Meridional Overturning Streamfunction — Year {yr:.0f}")
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Depth (m)")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "moc_final.png", dpi=150)
    plt.close()
    print(f"Saved {RUN_DIR / 'moc_final.png'}")

    # Summary
    print(f"\nFinal state (year {yr:.0f}):")
    print(f"  SST:   [{np.nanmin(sst):.1f}, {np.nanmax(sst):.1f}] °C, "
          f"mean = {np.nanmean(sst):.1f} °C")
    print(f"  SSH:   [{np.nanmin(ssh):.2f}, {np.nanmax(ssh):.2f}] m")
    print(f"  Speed: max = {np.nanmax(spd):.3f} m/s")
    print(f"  MOC:   [{psi_moc.min():.1f}, {psi_moc.max():.1f}] Sv")


if __name__ == "__main__":
    main()
