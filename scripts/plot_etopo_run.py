#!/usr/bin/env python
"""Plot the ETOPO 1-month run snapshots.

Reads restart_dayNNNNNN.npz files and timeseries.csv from
``results/etopo_1month_snap/latlon/180x360`` and produces:

- bathymetry.png            — H_bathy + land mask
- timeseries.png            — globally-averaged scalars over time
- snapshot_<field>_dayXX.png — surface and column maps at each restart day
- snapshot_T_zonal_dayXX.png — zonal-mean T(lat, depth) at each restart day
- snapshot_S_zonal_dayXX.png — zonal-mean S(lat, depth)
- evolution_summary.png     — multi-panel showing SST/SSH/speed at days 10/20/30
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/etopo_1month_eqsmooth/latlon/180x360")
TIMESERIES_CSV = RESULTS_DIR / "timeseries.csv"
OUT_DIR = RESULTS_DIR

# Geometry of the run
N_LAT, N_LON, N_LEV = 180, 360, 20
DZ_SURFACE, DZ_DEEP, H_MAX = 20.0, 500.0, 5000.0


def _build_z_full():
    """Re-derive z_full from create_ocean_z_star (tanh stretch)."""
    n = N_LEV
    sigma = np.linspace(0.0, 1.0, n + 1)
    A = (DZ_DEEP - DZ_SURFACE) / 2
    B = (DZ_DEEP + DZ_SURFACE) / 2
    dz = B + A * np.tanh((sigma[:-1] - 0.5) * 4)
    dz = dz / dz.sum() * H_MAX
    z_half = np.concatenate(([0.0], -np.cumsum(dz)))
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    return z_full, z_half


def _make_lonlat():
    lat = np.linspace(-90.0 + 0.5, 90.0 - 0.5, N_LAT)
    lon = np.linspace(0.5, 360.0 - 0.5, N_LON)
    return lat, lon


def _save(fig, name):
    out = OUT_DIR / name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _load_state(day):
    fn = RESULTS_DIR / f"restart_day{int(day):06d}.npz"
    if not fn.exists():
        return None
    return dict(np.load(fn))


def _maps_for_day(state, day, z_full, lat, lon):
    """Plot SST/SSS/SSH/speed maps for a given restart day."""
    T = state["T"]; S = state["S"]; eta = state["eta"]
    u = state["u"]; v = state["v"]; H = state["H_bathy"]
    land = state["land_mask"]
    ocean = land > 0.5

    def _msk2d(f):
        return np.where(ocean, f, np.nan)

    u_c = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
    v_c = 0.5 * (v[:-1, :, :] + v[1:, :, :])
    speed = np.sqrt(u_c**2 + v_c**2)
    surf_speed = _msk2d(speed[..., 0])
    speed_max_for_plot = max(0.1, float(np.nanpercentile(surf_speed, 99)))

    # SST
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.pcolormesh(lon, lat, _msk2d(T[..., 0]), cmap="RdYlBu_r",
                        shading="auto", vmin=0, vmax=30)
    ax.set_title(f"SST at day {int(day)}  [°C]")
    ax.set_xlabel("Longitude [°E]"); ax.set_ylabel("Latitude [°N]")
    plt.colorbar(im, ax=ax, label="SST [°C]")
    _save(fig, f"snapshot_sst_day{int(day):02d}.png")

    # SSS
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.pcolormesh(lon, lat, _msk2d(S[..., 0]), cmap="viridis",
                        shading="auto", vmin=33, vmax=37)
    ax.set_title(f"SSS at day {int(day)}  [PSU]")
    ax.set_xlabel("Longitude [°E]"); ax.set_ylabel("Latitude [°N]")
    plt.colorbar(im, ax=ax, label="SSS [PSU]")
    _save(fig, f"snapshot_sss_day{int(day):02d}.png")

    # SSH
    eta_lim = max(0.05, float(np.nanpercentile(np.abs(_msk2d(eta)), 99)))
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.pcolormesh(lon, lat, _msk2d(eta), cmap="RdBu_r",
                        shading="auto", vmin=-eta_lim, vmax=eta_lim)
    ax.set_title(f"Sea surface height (η) at day {int(day)}  [m]")
    ax.set_xlabel("Longitude [°E]"); ax.set_ylabel("Latitude [°N]")
    plt.colorbar(im, ax=ax, label="η [m]")
    _save(fig, f"snapshot_ssh_day{int(day):02d}.png")

    # Surface speed
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.pcolormesh(lon, lat, surf_speed, cmap="magma",
                        shading="auto", vmin=0, vmax=speed_max_for_plot)
    ax.set_title(f"Surface |u| at day {int(day)}  [m/s]   "
                 f"(max plotted: {speed_max_for_plot:.2f})")
    ax.set_xlabel("Longitude [°E]"); ax.set_ylabel("Latitude [°N]")
    plt.colorbar(im, ax=ax, label="|u| [m/s]")
    _save(fig, f"snapshot_speed_surface_day{int(day):02d}.png")

    # Zonal-mean T
    T_mask = np.where(ocean[..., None], T, np.nan)
    T_z = np.nanmean(T_mask, axis=1)
    LAT, DEPTH = np.meshgrid(lat, z_full, indexing="ij")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(LAT, DEPTH, T_z, cmap="RdYlBu_r",
                        shading="auto", vmin=0, vmax=30)
    ax.set_title(f"Zonal-mean T at day {int(day)}  [°C]")
    ax.set_xlabel("Latitude [°N]"); ax.set_ylabel("Depth [m]")
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label="T [°C]")
    _save(fig, f"snapshot_T_zonal_day{int(day):02d}.png")

    # Zonal-mean S
    S_mask = np.where(ocean[..., None], S, np.nan)
    S_z = np.nanmean(S_mask, axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(LAT, DEPTH, S_z, cmap="viridis",
                        shading="auto", vmin=33, vmax=37)
    ax.set_title(f"Zonal-mean S at day {int(day)}  [PSU]")
    ax.set_xlabel("Latitude [°N]"); ax.set_ylabel("Depth [m]")
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label="S [PSU]")
    _save(fig, f"snapshot_S_zonal_day{int(day):02d}.png")


def _evolution_summary(states, days, lat, lon):
    """Multi-panel: SST and surface speed at days 10, 20, 30."""
    n_panels = len(states)
    fig, axes = plt.subplots(2, n_panels, figsize=(5 * n_panels, 8))

    for col, (day, st) in enumerate(zip(days, states)):
        ocean = st["land_mask"] > 0.5
        sst = np.where(ocean, st["T"][..., 0], np.nan)
        u = st["u"]; v = st["v"]
        u_c = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
        v_c = 0.5 * (v[:-1, :, :] + v[1:, :, :])
        sp = np.sqrt(u_c[..., 0]**2 + v_c[..., 0]**2)
        sp = np.where(ocean, sp, np.nan)
        sp_max = max(0.1, float(np.nanpercentile(sp, 99)))

        ax = axes[0, col]
        im = ax.pcolormesh(lon, lat, sst, cmap="RdYlBu_r",
                           shading="auto", vmin=18, vmax=28)
        ax.set_title(f"SST  •  Day {int(day)}")
        if col == 0:
            ax.set_ylabel("Latitude [°N]")
        plt.colorbar(im, ax=ax, label="°C", fraction=0.04)

        ax = axes[1, col]
        im = ax.pcolormesh(lon, lat, sp, cmap="magma",
                           shading="auto", vmin=0, vmax=sp_max)
        ax.set_title(f"Surface |u|  •  Day {int(day)}  (max plotted: {sp_max:.2f})")
        ax.set_xlabel("Longitude [°E]")
        if col == 0:
            ax.set_ylabel("Latitude [°N]")
        plt.colorbar(im, ax=ax, label="m/s", fraction=0.04)

    fig.suptitle(
        "ETOPO 1° + JRA55-do RYF: 30-day evolution\n"
        "(KPP + GM/Redi + enhanced-diffusion convection, "
        "partial-cell thickness ≥ 30% of dz_ref, dt=300s)",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "evolution_summary.png")


def main():
    z_full, z_half = _build_z_full()
    lat, lon = _make_lonlat()

    # Bathymetry from any restart (it's static)
    base = _load_state(10)
    if base is None:
        print(f"!! No restart_day000010.npz in {RESULTS_DIR}")
        return 1

    print(f"Loading bathymetry from day-10 restart…")
    H = base["H_bathy"]; land = base["land_mask"]
    ocean = land > 0.5
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.pcolormesh(lon, lat, np.where(ocean, H, np.nan),
                        cmap="ocean_r", shading="auto", vmin=0, vmax=5000)
    ax.set_title("ETOPO bathymetry [m] (1° regridded, H_min=50, "
                 "5 smoothing passes, partial-cell snap 30%)")
    ax.set_xlabel("Longitude [°E]"); ax.set_ylabel("Latitude [°N]")
    plt.colorbar(im, ax=ax, label="depth [m]")
    _save(fig, "bathymetry.png")

    # Per-restart-day plots
    days = []
    states = []
    for d in [10, 20, 30]:
        st = _load_state(d)
        if st is None:
            continue
        print(f"\nDay {d}:")
        print(f"  T range: [{np.nanmin(st['T']):.2f}, {np.nanmax(st['T']):.2f}]")
        print(f"  max|u|: {float(np.max(np.abs(st['u']))):.4f} m/s")
        _maps_for_day(st, d, z_full, lat, lon)
        days.append(d)
        states.append(st)

    if states:
        _evolution_summary(states, days, lat, lon)

    # Timeseries
    if TIMESERIES_CSV.exists():
        ts = np.genfromtxt(TIMESERIES_CSV, delimiter=",", names=True)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

        ax = axes[0, 0]
        ax.plot(ts["day"], ts["SST"], "-o", color="C3", lw=1, ms=2.5)
        ax.set_ylabel("SST [°C]"); ax.grid(True, alpha=0.3)
        ax.set_title("Surface temperature")

        ax = axes[0, 1]
        ax.plot(ts["day"], ts["SSS"], "-o", color="C0", lw=1, ms=2.5)
        ax.set_ylabel("SSS [PSU]"); ax.grid(True, alpha=0.3)
        ax.set_title("Surface salinity")

        ax = axes[1, 0]
        ax.plot(ts["day"], ts["SSH"], "-o", color="C2", lw=1, ms=2.5)
        ax.set_ylabel("SSH [m]"); ax.set_xlabel("Day")
        ax.grid(True, alpha=0.3)
        ax.set_title("Sea-surface height (global mean)")

        ax = axes[1, 1]
        ax.plot(ts["day"], ts["max_speed"], "-o", color="C1", lw=1, ms=2.5)
        ax.set_ylabel("max|u| [m/s]"); ax.set_xlabel("Day")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title("Maximum current speed (log scale)")

        fig.suptitle(
            "ETOPO 1° + JRA55-do RYF: 30-day spinup with partial-cell snap (30%)\n"
            "(KPP + enhanced-diffusion convection + GM/Redi, "
            "A_h=2e5 cos²(lat), B_h=5e9, dt=300s)",
            fontsize=11,
        )
        fig.tight_layout()
        _save(fig, "timeseries.png")

    print(f"\nAll plots written to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
