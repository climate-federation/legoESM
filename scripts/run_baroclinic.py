#!/usr/bin/env python
"""Eady baroclinic instability — standalone run + diagnostic figure.

A meridional temperature front in thermal wind balance is released into
a periodic channel.  A small SSH perturbation seeds baroclinic instability.
No wind forcing — pure APE → KE conversion (Eady 1949).

Expected behaviour:
  - Days 0–5:  geostrophic adjustment (jet sharpens)
  - Days 5–30: exponential growth of the most-unstable Eady mode (~10–20 day e-fold)
  - Days 30+:  meanders and eddies visible along the front at ~45°N

Quick start::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python scripts/run_baroclinic.py
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python scripts/run_baroclinic.py --days 40

Output (results/baroclinic/ by default):
  - snapshots/snapshot_NNNNN.npz  — state at each output step
  - run_metadata.json
  - diagnostics.png               — SST, SSH, surface speed at 3 times
"""

from __future__ import annotations

import argparse
import dataclasses
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
    p.add_argument("--dt", type=float, default=300.0,
                   help="Baroclinic timestep in seconds (default 300).")
    p.add_argument("--n-lat", type=int, default=24,
                   help="Number of interior latitude cells (default 24).")
    p.add_argument("--n-lon", type=int, default=72,
                   help="Number of longitude cells (default 72).")
    p.add_argument("--n-levels", type=int, default=20,
                   help="Number of vertical levels (default 20).")
    p.add_argument("--snapshot-every-days", type=float, default=2.0,
                   help="Snapshot interval in days (default 2).")
    p.add_argument("--output-dir", type=Path, default=Path("results/baroclinic"),
                   help="Output directory (default results/baroclinic).")
    p.add_argument("--no-figure", action="store_true",
                   help="Skip the final diagnostic figure.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _diagnose(state) -> dict:
    mask = np.asarray(state.land_mask.data) > 0.5
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    T = np.asarray(state.T.data)
    eta = np.asarray(state.eta.data)
    u_cc = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
    v_cc = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
    return {
        "u_max": float(np.max(np.abs(u))),
        "speed_sfc_max": float(np.max(np.sqrt(u_cc**2 + v_cc**2))),
        "eta_max": float(np.max(np.abs(eta))),
        "T_sfc_mean": float(np.mean(T[:, :, 0][mask])) if mask.any() else float("nan"),
        "KE_sfc": float(np.mean(0.5 * (u_cc**2 + v_cc**2)[mask])) if mask.any() else float("nan"),
    }


def _save_snapshot(state, t_seconds: float, snap_dir: Path, idx: int) -> dict:
    d = {
        "time_seconds": t_seconds,
        "time_days": t_seconds / 86400.0,
        "u": np.asarray(state.u.data),
        "v": np.asarray(state.v.data),
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        "eta": np.asarray(state.eta.data),
        "land_mask": np.asarray(state.land_mask.data),
    }
    np.savez_compressed(snap_dir / f"snapshot_{idx:05d}.npz", **d)
    return d


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_figure(snaps: list[dict], lat_s: float, lat_n: float,
                 lon_w: float, lon_e: float, output_dir: Path) -> None:
    """3-row summary: SST, SSH, surface speed at 3 evenly-spaced times."""
    finite = [s for s in snaps if np.all(np.isfinite(s["T"][:, :, 0]))]
    if not finite:
        print("  All snapshots non-finite — skipping figure.")
        return

    n_show = min(3, len(finite))
    show = [finite[int(i)] for i in np.linspace(0, len(finite) - 1, n_show)]

    data_nlat, data_nlon = show[0]["T"].shape[:2]
    lat_1d = np.linspace(lat_s, lat_n, data_nlat)
    lon_1d = np.linspace(lon_w, lon_e, data_nlon)

    fig, axes = plt.subplots(3, n_show, figsize=(5 * n_show, 12),
                             constrained_layout=True)
    if n_show == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle("Eady Baroclinic Instability — lat-lon channel\n"
                 "tanh front @ 45°N in thermal-wind balance, no wind forcing",
                 fontsize=12, fontweight="bold")

    for col, snap in enumerate(show):
        day = float(snap["time_days"])
        mask = snap["land_mask"] > 0.5

        # SST
        T_sfc = snap["T"][:, :, 0].copy(); T_sfc[~mask] = np.nan
        ax = axes[0, col]
        vmin = float(np.nanmin(T_sfc)); vmax = float(np.nanmax(T_sfc))
        im = ax.pcolormesh(lon_1d, lat_1d, T_sfc, cmap="RdYlBu_r",
                           vmin=vmin, vmax=vmax, shading="nearest")
        fig.colorbar(im, ax=ax, label="°C")
        ax.axhline(45.0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"Day {day:.1f}  —  SST (°C)")
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")

        # SSH
        eta = snap["eta"].copy(); eta[~mask] = np.nan
        em = float(np.nanmax(np.abs(eta))) or 1e-3
        ax = axes[1, col]
        im = ax.pcolormesh(lon_1d, lat_1d, eta, cmap="RdBu_r",
                           vmin=-em, vmax=em, shading="nearest")
        fig.colorbar(im, ax=ax, label="m")
        ax.axhline(45.0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"Day {day:.1f}  —  SSH (m, ±{em:.3f})")
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")

        # Surface speed
        u_cc = 0.5 * (snap["u"][:, :-1, 0] + snap["u"][:, 1:, 0])
        v_cc = 0.5 * (snap["v"][:-1, :, 0] + snap["v"][1:, :, 0])
        spd = np.sqrt(u_cc**2 + v_cc**2); spd[~mask] = np.nan
        ax = axes[2, col]
        im = ax.pcolormesh(lon_1d, lat_1d, spd, cmap="plasma",
                           vmin=0, vmax=max(float(np.nanmax(spd)), 1e-4),
                           shading="nearest")
        fig.colorbar(im, ax=ax, label="m/s")
        ax.axhline(45.0, color="w", lw=0.8, ls="--", alpha=0.7)
        ax.set_title(f"Day {day:.1f}  —  Surface speed (m/s)")
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")

    out = output_dir / "diagnostics.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {out}")


def _make_growth_figure(diag_log: list[dict], output_dir: Path) -> None:
    """Time-series of surface KE and max speed — shows exponential growth."""
    if len(diag_log) < 2:
        return
    days = np.array([d["day"] for d in diag_log])
    ke   = np.array([d["KE_sfc"] for d in diag_log])
    spd  = np.array([d["speed_sfc_max"] for d in diag_log])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    fig.suptitle("Baroclinic growth: surface KE & max speed", fontsize=12)

    ax1.semilogy(days, ke, "b-o", ms=3)
    ax1.set_xlabel("Day"); ax1.set_ylabel("Mean surface KE (m²/s²)")
    ax1.set_title("Exponential growth → saturation")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(days, spd, "r-o", ms=3)
    ax2.set_xlabel("Day"); ax2.set_ylabel("Max surface speed (m/s)")
    ax2.set_title("Peak surface speed")
    ax2.grid(True, alpha=0.3)

    out = output_dir / "growth.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Growth figure → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    import jax
    if not jax.config.x64_enabled:
        import warnings
        warnings.warn(
            "JAX_ENABLE_X64 is OFF. Re-run with JAX_ENABLE_X64=1.",
            stacklevel=2,
        )

    from legoesm.grids.latlon import create_regional_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.experiments.eady_instability import (
        EadyInstabilityConfig,
        create_initial_conditions as eady_ic,
        create_forcings as eady_forcings,
    )

    cfg = EadyInstabilityConfig()
    dt = args.dt
    nlev = args.n_levels

    print("Building lat-lon channel grid "
          f"({args.n_lat} × {args.n_lon}, {nlev} levels)…")
    grid, _wall_mask = create_regional_latlon_grid(
        args.n_lat, args.n_lon,
        cfg.lat_south, cfg.lat_north,
        lon_west=cfg.lon_west, lon_east=cfg.lon_east,
        periodic_x=True,
    )
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=cfg.H_max)

    physics = eady_forcings("latlon_channel", None, cfg)
    model_cfg = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,
        physics=physics,
        A_h=cfg.A_h,
        bottom_drag_r=cfg.bottom_drag_coeff,
    )
    model = LatLonCGridOceanModel(grid, z_coord, model_cfg)

    print("Initialising Eady instability state…")
    state = eady_ic("latlon_channel", grid, z_coord, cfg)

    # Output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = args.output_dir / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    with open(args.output_dir / "run_metadata.json", "w") as f:
        json.dump({
            "days": args.days, "dt": dt,
            "n_lat": args.n_lat, "n_lon": args.n_lon, "n_levels": nlev,
            "lat_south": cfg.lat_south, "lat_north": cfg.lat_north,
            "lon_west": cfg.lon_west, "lon_east": cfg.lon_east,
            "eady_config": dataclasses.asdict(cfg),
        }, f, indent=2)

    n_steps = int(round(args.days * 86400.0 / dt))
    snap_every = max(1, int(round(args.snapshot_every_days * 86400.0 / dt)))

    print(f"\nEady Instability: {args.n_lat}×{args.n_lon} lat-lon, "
          f"{nlev} levels, dt={dt}s")
    print(f"Run: {n_steps} steps = {args.days:.1f} days, "
          f"snapshot every {snap_every} steps = {snap_every*dt/86400.0:.1f} days")
    print(f"Front: ΔT={cfg.front_delta_T}°C tanh @ {cfg.front_lat_center}°N, "
          f"wavenumber-{cfg.perturbation_wavenumber} SSH seed")
    print(f"Output: {args.output_dir}\n")
    print(f"{'step':>6} {'day':>7} {'u_max':>10} {'spd_max':>10} "
          f"{'|eta|_max':>10} {'KE_sfc':>12}")

    snapshots: list[dict] = []
    diag_log: list[dict] = []
    snap_idx = 0
    d0 = _save_snapshot(state, 0.0, snap_dir, snap_idx)
    snapshots.append(d0)

    t_wall_start = time.time()

    for k in range(n_steps):
        state = model.step(state, dt=dt)

        is_last = k == n_steps - 1
        if (k + 1) % snap_every == 0 or is_last:
            snap_idx += 1
            t_seconds = (k + 1) * dt
            d = _save_snapshot(state, t_seconds, snap_dir, snap_idx)
            snapshots.append(d)
            diag = _diagnose(state)
            diag_log.append({"day": t_seconds / 86400.0, **diag})
            print(f"{k+1:6d} {t_seconds/86400.0:7.2f} "
                  f"{diag['u_max']:10.4e} {diag['speed_sfc_max']:10.4e} "
                  f"{diag['eta_max']:10.4e} {diag['KE_sfc']:12.4e}")

    wall = time.time() - t_wall_start
    print(f"\nDone. Wall time: {wall:.1f}s  "
          f"({wall/n_steps*1000:.1f} ms/step).  Snapshots: {snap_idx+1}")

    if not args.no_figure:
        print("\nGenerating figures…")
        _make_figure(snapshots, cfg.lat_south, cfg.lat_north,
                     cfg.lon_west, cfg.lon_east, args.output_dir)
        _make_growth_figure(diag_log, args.output_dir)


if __name__ == "__main__":
    main()
