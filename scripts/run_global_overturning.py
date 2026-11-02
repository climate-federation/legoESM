#!/usr/bin/env python
"""Global overturning circulation — standalone run + diagnostics.

Idealized Wolfe-Cessi (2010) setup on a 5° lat-lon grid (36×72):
- Global ocean with simplified continent (20–60°E, from north cap to 55°S)
- Open Drake Passage south of 55°S → circumpolar current develops
- 3-belt zonal wind stress (trades, westerlies, polar easterlies)
- SST restoring: warm equator → cold poles (cosine profile)
- Linear EOS (temperature-only buoyancy)
- 20 z* levels, flat bottom at 4000 m

Quick start (30 days, ~2 min on CPU)::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python scripts/run_global_overturning.py
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python scripts/run_global_overturning.py --days 365

Output (results/global_overturning/ by default):
  - snapshots/snapshot_NNNNN.npz
  - run_metadata.json
  - diagnostics.png     — SST, SSH, zonal wind stress
  - drake_budget.png    — Drake-band zonal momentum budget bar chart
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--days", type=float, default=30.0,
                   help="Simulation length in days (default 30).")
    p.add_argument("--dt", type=float, default=600.0,
                   help="Baroclinic timestep in seconds (default 600).")
    p.add_argument("--snapshot-every-days", type=float, default=5.0,
                   help="Snapshot interval in days (default 5).")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/global_overturning"),
                   help="Output directory.")
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Snapshot / diagnostics
# ---------------------------------------------------------------------------

def _save_snapshot(state, t_seconds: float, snap_dir: Path, idx: int) -> dict:
    d = {
        "time_days": t_seconds / 86400.0,
        "T": np.asarray(state.T.data),
        "eta": np.asarray(state.eta.data),
        "u": np.asarray(state.u.data),
        "v": np.asarray(state.v.data),
        "land_mask": np.asarray(state.land_mask.data),
    }
    np.savez_compressed(snap_dir / f"snapshot_{idx:05d}.npz", **d)
    return d


def _diagnose(state) -> dict:
    mask = np.asarray(state.land_mask.data) > 0.5
    u = np.asarray(state.u.data)
    T = np.asarray(state.T.data)
    eta = np.asarray(state.eta.data)
    return {
        "u_max": float(np.max(np.abs(u))),
        "eta_max": float(np.max(np.abs(eta))),
        "T_sfc_mean": float(np.mean(T[:, :, 0][mask])) if mask.any() else float("nan"),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_diagnostics_figure(snaps: list[dict], grid, output_dir: Path) -> None:
    """3-row figure: SST, SSH, surface zonal velocity at 3 times."""
    finite = [s for s in snaps if np.all(np.isfinite(s["T"][:, :, 0]))]
    if not finite:
        print("  All snapshots non-finite — skipping figure.")
        return

    n_show = min(3, len(finite))
    show = [finite[int(i)] for i in np.linspace(0, len(finite) - 1, n_show)]

    lat = np.degrees(np.asarray(grid.lat, dtype=np.float64))
    lon = np.degrees(np.asarray(grid.lon, dtype=np.float64))
    lat_1d = lat[:, 0] if lat.ndim == 2 else lat
    lon_1d = lon[0, :] if lon.ndim == 2 else lon

    fig, axes = plt.subplots(3, n_show, figsize=(5 * n_show, 12),
                             constrained_layout=True)
    if n_show == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle("Global Overturning (Wolfe-Cessi 2010, 5°, flat bottom)\n"
                 "Continent: 20–60°E; Drake Passage open south of 55°S",
                 fontsize=11, fontweight="bold")

    for col, snap in enumerate(show):
        day = float(snap["time_days"])
        mask = snap["land_mask"] > 0.5

        # SST
        T_sfc = snap["T"][:, :, 0].copy(); T_sfc[~mask] = np.nan
        ax = axes[0, col]
        im = ax.pcolormesh(lon_1d, lat_1d, T_sfc, cmap="RdYlBu_r",
                           vmin=0, vmax=25, shading="nearest")
        fig.colorbar(im, ax=ax, label="°C")
        ax.set_title(f"Day {day:.0f}  —  SST (°C)")
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")

        # SSH
        eta = snap["eta"].copy(); eta[~mask] = np.nan
        em = max(float(np.nanmax(np.abs(eta))), 1e-3)
        ax = axes[1, col]
        im = ax.pcolormesh(lon_1d, lat_1d, eta, cmap="RdBu_r",
                           vmin=-em, vmax=em, shading="nearest")
        fig.colorbar(im, ax=ax, label="m")
        ax.set_title(f"Day {day:.0f}  —  SSH (m)")
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")

        # Surface zonal velocity
        u_cc = 0.5 * (snap["u"][:, :-1, 0] + snap["u"][:, 1:, 0])
        u_cc[~mask] = np.nan
        um = max(float(np.nanmax(np.abs(u_cc))), 1e-4)
        ax = axes[2, col]
        im = ax.pcolormesh(lon_1d, lat_1d, u_cc, cmap="RdBu_r",
                           vmin=-um, vmax=um, shading="nearest")
        fig.colorbar(im, ax=ax, label="m/s")
        # Mark Drake latitude band
        ax.axhspan(-77.5, -57.5, color="yellow", alpha=0.15, label="Drake band")
        ax.set_title(f"Day {day:.0f}  —  Surface u (m/s)")
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")

    out = output_dir / "diagnostics.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Diagnostics → {out}")


def _make_drake_budget(state, grid, z_coord, config, output_dir: Path) -> None:
    """Estimate Drake-band zonal momentum budget from final state.

    Components (depth-integrated, zonally and meridionally averaged
    over the Drake band ~55°S–75°S):
      F_wind     — applied surface wind stress (from config)
      F_bot_drag — linear bottom drag on depth-mean flow
      F_visc     — lateral viscous diffusion of zonal momentum
    Residual = F_wind + F_bot_drag + F_visc  (should close if model is steady)
    All units: Pa (equivalent stress).
    """
    from legoesm import constants

    lat = np.degrees(np.asarray(grid.lat, dtype=np.float64))
    lon = np.degrees(np.asarray(grid.lon, dtype=np.float64))
    lat_1d = lat[:, 0] if lat.ndim == 2 else lat
    lon_1d = lon[0, :] if lon.ndim == 2 else lon
    n_lat, n_lon = len(lat_1d), len(lon_1d)

    # Drake band row indices (~55°S–75°S)
    drake_j = np.where((lat_1d >= -77.5) & (lat_1d <= -55.0))[0]
    if len(drake_j) == 0:
        print("  No Drake band rows found — skipping budget figure.")
        return

    mask = np.asarray(state.land_mask.data, dtype=np.float64)    # (n_lat, n_lon)
    u3d  = np.asarray(state.u.data, dtype=np.float64)            # (n_lat, n_lon+1, nlev)
    dz   = np.asarray(z_coord.dz_ref, dtype=np.float64)          # (nlev,)
    area = np.asarray(grid.area, dtype=np.float64)                # (n_lat, n_lon)

    # Depth-mean zonal velocity at cell centres
    u_cc = 0.5 * (u3d[:, :-1, :] + u3d[:, 1:, :])               # (n_lat, n_lon, nlev)
    H_cell = np.sum(dz) * mask                                    # (n_lat, n_lon)
    U_bar = np.sum(u_cc * dz[np.newaxis, np.newaxis, :], axis=2) / np.where(H_cell > 0, H_cell, 1.0)

    # --- F_wind: 3-belt zonal wind stress (same formula as GlobalOverturningConfig) ---
    lat_rad = np.radians(lat_1d)
    # Westerlies peak ~45°, trades peak ~15°, polar easterlies at poles
    tau_wind = config.tau_max * (
        np.cos(np.pi * lat_rad / np.radians(70.0)) * 0.8
        - 0.2 * np.cos(3.0 * np.pi * lat_rad / np.radians(70.0))
    )
    tau_wind_drake = float(np.mean(tau_wind[drake_j]))   # Pa (positive = westward)

    # --- F_bot_drag: ρ₀ · r · U_bar (depth-mean, Drake band, ocean cells only) ---
    ocean_drake = mask[np.ix_(drake_j, np.arange(n_lon))] > 0.5
    U_bar_drake = U_bar[np.ix_(drake_j, np.arange(n_lon))]
    if ocean_drake.any():
        U_bar_mean = float(np.mean(U_bar_drake[ocean_drake]))
    else:
        U_bar_mean = 0.0
    rho0 = constants.rho_ocean
    F_bot_drag = -rho0 * config.bottom_drag_coeff * U_bar_mean   # Pa (opposing flow)

    # --- F_visc: A_h * ∂²U_bar/∂y² * H_mean (meridional viscous diffusion) ---
    dy = float(np.mean(np.diff(lat_1d))) * np.pi / 180.0 * constants.R_earth
    U_bar_zonal = np.nanmean(np.where(mask > 0.5, U_bar, np.nan), axis=1)  # (n_lat,)
    d2U = np.gradient(np.gradient(U_bar_zonal, dy), dy)
    H_mean = float(np.mean(H_cell[np.ix_(drake_j, np.arange(n_lon))][ocean_drake])) if ocean_drake.any() else config.H_max
    F_visc = float(np.mean(d2U[drake_j])) * config.A_h * H_mean

    # --- Budget summary ---
    residual = tau_wind_drake + F_bot_drag + F_visc

    labels  = ["Wind stress\n(F_wind)", "Bottom drag\n(F_bot)", "Lat. visc.\n(F_visc)", "Residual"]
    values  = [tau_wind_drake, F_bot_drag, F_visc, residual]
    colors  = ["steelblue", "tomato", "goldenrod", "grey"]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, edgecolor="k", width=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Equivalent stress (Pa)")
    ax.set_title(
        f"Drake-band zonal momentum budget  (lat {lat_1d[drake_j[0]]:.1f}°–{lat_1d[drake_j[-1]]:.1f}°)\n"
        f"Final model state — idealized 5° Wolfe-Cessi",
        fontsize=10)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + np.sign(val) * 3e-4,
                f"{val:.3e}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8)
    out = output_dir / "drake_budget.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Drake budget → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    import jax
    if not jax.config.x64_enabled:
        import warnings
        warnings.warn("JAX_ENABLE_X64 is OFF — re-run with JAX_ENABLE_X64=1.",
                      stacklevel=2)

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.experiments.global_overturning import (
        GlobalOverturningConfig,
        create_initial_conditions,
        create_forcings,
        create_eos_config,
    )

    cfg = GlobalOverturningConfig()
    dt  = args.dt

    print("Building 5° global lat-lon grid (36×72, 20 levels)…")
    grid    = create_latlon_grid(36, 72)
    z_coord = create_ocean_z_star(
        n_levels=cfg.n_levels, H_max=cfg.H_max,
        dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep,
    )

    physics    = create_forcings("latlon", grid, cfg)
    eos_config = create_eos_config(cfg)
    model_cfg  = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,
        physics=physics,
        A_h=cfg.A_h,
        A_v=cfg.A_v,
        K_v=cfg.K_v,
        bottom_drag_r=cfg.bottom_drag_coeff,
        eos="linear",
        eos_linear=eos_config,
    )
    model = LatLonCGridOceanModel(grid, z_coord, model_cfg)

    print("Initialising global overturning state…")
    state = create_initial_conditions("latlon", grid, z_coord, cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = args.output_dir / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    with open(args.output_dir / "run_metadata.json", "w") as f:
        json.dump({"days": args.days, "dt": dt, "grid": "36x72",
                   "n_levels": cfg.n_levels, "H_max": cfg.H_max}, f, indent=2)

    n_steps    = int(round(args.days * 86400.0 / dt))
    snap_every = max(1, int(round(args.snapshot_every_days * 86400.0 / dt)))

    print(f"\nGlobal Overturning: 36×72 lat-lon, {cfg.n_levels} levels, dt={dt}s")
    print(f"Run: {n_steps} steps = {args.days:.0f} days, "
          f"snapshot every {snap_every*dt/86400:.1f} days")
    print(f"Continent: {cfg.continent_lon_west}–{cfg.continent_lon_east}°E  "
          f"(Drake open south of {cfg.continent_lat_south}°S)")
    print(f"Wind: τmax={cfg.tau_max} Pa, SST restoring τ={cfg.tau_T_days} d")
    print(f"Output: {args.output_dir}\n")
    print(f"{'step':>6} {'day':>7} {'u_max':>10} {'|eta|_max':>10} {'T_sfc':>8}")

    snapshots: list[dict] = []
    snap_idx = 0
    s0 = _save_snapshot(state, 0.0, snap_dir, snap_idx)
    snapshots.append(s0)
    t_wall = time.time()

    for k in range(n_steps):
        state = model.step(state, dt=dt)
        is_last = k == n_steps - 1
        if (k + 1) % snap_every == 0 or is_last:
            snap_idx += 1
            t_sec = (k + 1) * dt
            d = _save_snapshot(state, t_sec, snap_dir, snap_idx)
            snapshots.append(d)
            diag = _diagnose(state)
            print(f"{k+1:6d} {t_sec/86400.0:7.1f} "
                  f"{diag['u_max']:10.4e} {diag['eta_max']:10.4e} "
                  f"{diag['T_sfc_mean']:8.3f}")

    wall = time.time() - t_wall
    print(f"\nDone. Wall time: {wall:.1f}s  ({wall/n_steps*1000:.1f} ms/step). "
          f"Snapshots: {snap_idx+1}")

    if not args.no_figure:
        print("\nGenerating figures…")
        _make_diagnostics_figure(snapshots, grid, args.output_dir)
        _make_drake_budget(state, grid, z_coord, cfg, args.output_dir)


if __name__ == "__main__":
    main()
