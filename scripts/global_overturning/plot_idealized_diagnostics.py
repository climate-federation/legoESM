#!/usr/bin/env python
"""Diagnostic plots for idealized (flat-bottom) Wolfe-Cessi runs.

Reads the final restart from each run directory and produces:
  1. SSH, SST, surface speed maps
  2. Zonal-mean T(lat, depth)
  3. MOC streamfunction

Generates separate plots for 5° and 1°, plus a side-by-side comparison.

Usage:
    python scripts/global_overturning/plot_idealized_diagnostics.py
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig
from legoesm.ocean.diagnostics_streamfunction import moc_streamfunction

DIR_5DEG = Path("results/ocean/global_overturning_idealized_5deg")
DIR_1DEG = Path("results/ocean/global_overturning_idealized_1deg")
OUT_DIR = Path("results/ocean/idealized_comparison")


def _last_restart(d):
    restarts = sorted(d.glob("restart_day*.npz"))
    if not restarts:
        return None, None
    last = restarts[-1]
    day = int(last.stem.removeprefix("restart_day"))
    return last, day / 365.0


def _load_and_diagnose(restart_path, n_lat, n_lon):
    d = np.load(restart_path, allow_pickle=False)
    cfg = GlobalOverturningConfig()
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(
        n_levels=cfg.n_levels, H_max=cfg.H_max,
        dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep,
    )

    eta = d["eta"]
    T = d["T"]
    u = d["u"]
    v = d["v"]
    mask = d["land_mask"]
    ocean = mask > 0.5

    lat = np.asarray(grid.lat) * 180 / np.pi
    lon = np.asarray(grid.lon) * 180 / np.pi
    z_full = -np.abs(np.asarray(z_coord.z_full_ref))
    lat_v = np.linspace(-90, 90, len(lat) + 1)

    u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
    v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
    speed = np.sqrt(u_c**2 + v_c**2)

    h_ref = np.asarray(z_coord.dz_ref)
    h_3d = np.broadcast_to(h_ref[None, None, :], T.shape)
    psi_moc = moc_streamfunction(v, h_3d, eta,
                                  np.full_like(eta, cfg.H_max),
                                  mask, grid)

    T_masked = np.where(ocean[:, :, None], T, np.nan)
    T_zm = np.nanmean(T_masked, axis=1)

    return {
        "lat": lat, "lon": lon, "z_full": z_full, "lat_v": lat_v,
        "ssh": np.where(ocean, eta, np.nan),
        "sst": np.where(ocean, T[:, :, 0], np.nan),
        "speed": np.where(ocean, speed, np.nan),
        "T_zm": T_zm,
        "psi_moc": psi_moc,
    }


def _plot_single(data, yr, label, out_dir):
    """Generate 5 individual plots for one run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = data["lat"], data["lon"]

    # SSH
    fig, ax = plt.subplots(figsize=(12, 5))
    vmax = max(float(np.nanmax(np.abs(data["ssh"]))), 0.1)
    im = ax.pcolormesh(lon, lat, data["ssh"], cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto")
    plt.colorbar(im, ax=ax, label="SSH (m)", fraction=0.025)
    ax.set_title(f"SSH — {label}, Year {yr:.0f}")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal"); plt.tight_layout()
    plt.savefig(out_dir / "ssh.png", dpi=150); plt.close()

    # SST
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(lon, lat, data["sst"], cmap="RdYlBu_r", shading="auto")
    plt.colorbar(im, ax=ax, label="SST (°C)", fraction=0.025)
    ax.set_title(f"SST — {label}, Year {yr:.0f}")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal"); plt.tight_layout()
    plt.savefig(out_dir / "sst.png", dpi=150); plt.close()

    # Speed
    fig, ax = plt.subplots(figsize=(12, 5))
    vmax_spd = float(np.nanpercentile(data["speed"], 99))
    im = ax.pcolormesh(lon, lat, data["speed"], cmap="magma", shading="auto",
                       vmin=0, vmax=vmax_spd)
    plt.colorbar(im, ax=ax, label="Speed (m/s)", fraction=0.025)
    ax.set_title(f"Surface Speed — {label}, Year {yr:.0f}")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal"); plt.tight_layout()
    plt.savefig(out_dir / "speed.png", dpi=150); plt.close()

    # Zonal-mean T
    fig, ax = plt.subplots(figsize=(8, 5))
    T_levels = np.arange(0, 26, 2)
    im = ax.pcolormesh(lat, data["z_full"], data["T_zm"].T,
                       cmap="RdYlBu_r", shading="auto", vmin=0, vmax=25)
    cs = ax.contour(lat, data["z_full"], data["T_zm"].T,
                    levels=T_levels, colors="k", linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%g")
    plt.colorbar(im, ax=ax, label="T (°C)", fraction=0.025)
    ax.set_title(f"Zonal-Mean T — {label}, Year {yr:.0f}")
    ax.set_xlabel("Latitude (°)"); ax.set_ylabel("Depth (m)")
    plt.tight_layout()
    plt.savefig(out_dir / "T_zonal_mean.png", dpi=150); plt.close()

    # MOC
    fig, ax = plt.subplots(figsize=(8, 5))
    psi_max = max(float(np.nanmax(np.abs(data["psi_moc"]))), 1.0)
    psi_levels = np.linspace(-psi_max, psi_max, 21)
    im = ax.contourf(data["lat_v"], data["z_full"], data["psi_moc"].T,
                     levels=psi_levels, cmap="RdBu_r", extend="both")
    cs = ax.contour(data["lat_v"], data["z_full"], data["psi_moc"].T,
                    levels=psi_levels[::4], colors="k", linewidths=0.4, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    plt.colorbar(im, ax=ax, label="ψ (Sv)", fraction=0.025)
    ax.set_title(f"MOC — {label}, Year {yr:.0f}")
    ax.set_xlabel("Latitude (°)"); ax.set_ylabel("Depth (m)")
    plt.tight_layout()
    plt.savefig(out_dir / "moc.png", dpi=150); plt.close()

    print(f"  Saved 5 plots to {out_dir}/")


def _plot_comparison(data_5, yr_5, data_1, yr_1, out_dir):
    """Side-by-side comparison of 5° and 1°."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for row, (data, yr, label) in enumerate([
        (data_5, yr_5, f"5° (yr {yr_5:.0f})"),
        (data_1, yr_1, f"1° (yr {yr_1:.0f})"),
    ]):
        lat, lon = data["lat"], data["lon"]

        # SSH
        vmax = max(float(np.nanmax(np.abs(data["ssh"]))), 0.1)
        im = axes[row, 0].pcolormesh(lon, lat, data["ssh"], cmap="RdBu_r",
                                      vmin=-vmax, vmax=vmax, shading="auto")
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046)
        axes[row, 0].set_title(f"SSH — {label}")
        axes[row, 0].set_ylabel("Lat (°)")

        # SST
        im = axes[row, 1].pcolormesh(lon, lat, data["sst"], cmap="RdYlBu_r",
                                      shading="auto", vmin=0, vmax=25)
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)
        axes[row, 1].set_title(f"SST — {label}")

        # Speed
        vmax_spd = max(float(np.nanpercentile(data["speed"], 99)), 0.01)
        im = axes[row, 2].pcolormesh(lon, lat, data["speed"], cmap="magma",
                                      shading="auto", vmin=0, vmax=vmax_spd)
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)
        axes[row, 2].set_title(f"Speed — {label}")

    axes[1, 0].set_xlabel("Lon (°)")
    axes[1, 1].set_xlabel("Lon (°)")
    axes[1, 2].set_xlabel("Lon (°)")
    plt.suptitle("Idealized Wolfe-Cessi: 5° vs 1° (flat bottom)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_ssh_sst_speed.png", dpi=130)
    plt.close()

    # T zonal-mean comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    T_levels = np.arange(0, 26, 2)
    for i, (data, yr, label) in enumerate([
        (data_5, yr_5, f"5° (yr {yr_5:.0f})"),
        (data_1, yr_1, f"1° (yr {yr_1:.0f})"),
    ]):
        im = axes[i].pcolormesh(data["lat"], data["z_full"], data["T_zm"].T,
                                cmap="RdYlBu_r", shading="auto", vmin=0, vmax=25)
        cs = axes[i].contour(data["lat"], data["z_full"], data["T_zm"].T,
                             levels=T_levels, colors="k", linewidths=0.5, alpha=0.6)
        axes[i].clabel(cs, inline=True, fontsize=7, fmt="%g")
        plt.colorbar(im, ax=axes[i], label="T (°C)" if i == 1 else None,
                     fraction=0.046)
        axes[i].set_title(label)
        axes[i].set_xlabel("Latitude (°)")
    axes[0].set_ylabel("Depth (m)")
    plt.suptitle("Zonal-Mean Temperature: 5° vs 1°", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_T_zonal_mean.png", dpi=130)
    plt.close()

    # MOC comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    psi_max = max(
        float(np.nanmax(np.abs(data_5["psi_moc"]))),
        float(np.nanmax(np.abs(data_1["psi_moc"]))),
        1.0,
    )
    psi_levels = np.linspace(-psi_max, psi_max, 21)
    for i, (data, yr, label) in enumerate([
        (data_5, yr_5, f"5° (yr {yr_5:.0f})"),
        (data_1, yr_1, f"1° (yr {yr_1:.0f})"),
    ]):
        im = axes[i].contourf(data["lat_v"], data["z_full"], data["psi_moc"].T,
                              levels=psi_levels, cmap="RdBu_r", extend="both")
        cs = axes[i].contour(data["lat_v"], data["z_full"], data["psi_moc"].T,
                             levels=psi_levels[::4], colors="k", linewidths=0.4)
        axes[i].clabel(cs, inline=True, fontsize=7, fmt="%.1f")
        plt.colorbar(im, ax=axes[i], label="ψ (Sv)" if i == 1 else None,
                     fraction=0.046)
        axes[i].set_title(label)
        axes[i].set_xlabel("Latitude (°)")
    axes[0].set_ylabel("Depth (m)")
    plt.suptitle("MOC Streamfunction: 5° vs 1°", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_moc.png", dpi=130)
    plt.close()

    print(f"  Saved comparison plots to {out_dir}/")


def main():
    have_5 = DIR_5DEG.exists()
    have_1 = DIR_1DEG.exists()

    if have_5:
        path_5, yr_5 = _last_restart(DIR_5DEG)
        if path_5:
            print(f"5°: {path_5.name} (year {yr_5:.0f})")
            data_5 = _load_and_diagnose(path_5, 36, 72)
            _plot_single(data_5, yr_5, "5° idealized", DIR_5DEG)
        else:
            have_5 = False

    if have_1:
        path_1, yr_1 = _last_restart(DIR_1DEG)
        if path_1:
            print(f"1°: {path_1.name} (year {yr_1:.0f})")
            data_1 = _load_and_diagnose(path_1, 180, 360)
            _plot_single(data_1, yr_1, "1° idealized", DIR_1DEG)
        else:
            have_1 = False

    if have_5 and have_1:
        _plot_comparison(data_5, yr_5, data_1, yr_1, OUT_DIR)

    if not have_5 and not have_1:
        print("No results found in either directory.")


if __name__ == "__main__":
    main()
