#!/usr/bin/env python
"""ACC-like channel experiment — standalone run + diagnostic figure.

Runs a wind-driven stratified channel with a Gaussian ridge (based on
Zhang et al. 2024, JPO) and saves snapshots + a PNG summary figure.

Quick start::

    JAX_ENABLE_X64=1 python scripts/run_acc.py
    JAX_ENABLE_X64=1 python scripts/run_acc.py --days 10 --n-lat 24 --n-lon 72

The default resolution (24 x 72 lat-lon channel) runs on CPU in a few
minutes for a 5-day simulation.  For a production/GPU run use --days 30.

Output (in results/acc_channel/ by default):
  - snapshots/snapshot_NNNNN.npz  — model state at each output time
  - run_metadata.json             — config + grid info
  - diagnostics.png               — surface T, SSH, surface speed summary
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--days", type=float, default=5.0,
                   help="Simulation length in days (default 5).")
    p.add_argument("--dt", type=float, default=300.0,
                   help="Baroclinic timestep in seconds (default 300).")
    p.add_argument("--n-lat", type=int, default=24,
                   help="Number of latitude cells (default 24, ~0.8° spacing).")
    p.add_argument("--n-lon", type=int, default=72,
                   help="Number of longitude cells (default 72, ~0.25° zonal).")
    p.add_argument("--n-levels", type=int, default=20,
                   help="Number of vertical levels (default 20).")
    p.add_argument("--snapshot-every-days", type=float, default=1.0,
                   help="Snapshot interval in days (default 1).")
    p.add_argument("--output-dir", type=Path, default=Path("results/acc_channel"),
                   help="Output directory (default results/acc_channel).")
    p.add_argument("--no-sponge", action="store_true",
                   help="Disable the northern-boundary sponge layer.")
    p.add_argument("--no-figure", action="store_true",
                   help="Skip the final diagnostic figure.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def _diagnose(state) -> dict:
    """Extract scalar diagnostics from the current model state."""
    mask = np.asarray(state.land_mask.data) > 0.5
    u = np.asarray(state.u.data)    # (n_lat, n_lon+1, nlev)
    v = np.asarray(state.v.data)    # (n_lat+1, n_lon, nlev)
    T = np.asarray(state.T.data)
    eta = np.asarray(state.eta.data)

    u_sfc_cc = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
    v_sfc_cc = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
    speed_sfc = np.sqrt(u_sfc_cc ** 2 + v_sfc_cc ** 2)

    return {
        "u_max": float(np.max(np.abs(u))),
        "speed_sfc_max": float(np.max(speed_sfc)),
        "eta_max": float(np.max(np.abs(eta))),
        "T_sfc_mean": float(np.mean(T[:, :, 0][mask])) if mask.any() else float("nan"),
        "T_min": float(np.min(T[mask])) if mask.any() else float("nan"),
        "T_max": float(np.max(T[mask])) if mask.any() else float("nan"),
    }


def _save_snapshot(state, t_seconds: float, snap_dir: Path, idx: int) -> None:
    np.savez_compressed(
        snap_dir / f"snapshot_{idx:05d}.npz",
        time_seconds=t_seconds,
        time_days=t_seconds / 86400.0,
        u=np.asarray(state.u.data),
        v=np.asarray(state.v.data),
        T=np.asarray(state.T.data),
        S=np.asarray(state.S.data),
        eta=np.asarray(state.eta.data),
        land_mask=np.asarray(state.land_mask.data),
        H_bathy=np.asarray(state.H_bathy.data),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _grid_coords_deg(grid, data_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-centre (lon_deg, lat_deg) 1-D arrays matching data_shape."""
    n_lat_data, n_lon_data = data_shape
    lon = np.degrees(np.asarray(grid.lon, dtype=np.float64))
    lat = np.degrees(np.asarray(grid.lat, dtype=np.float64))
    if lon.ndim == 2:
        lon_1d = lon[0, :]
        lat_1d = lat[:, 0]
    else:
        lon_1d = lon
        lat_1d = lat
    # If the grid coords don't match the data (e.g. wall rows added),
    # rebuild uniform coords spanning the same domain with the actual cell count.
    if len(lat_1d) != n_lat_data:
        lat_1d = np.linspace(float(lat_1d[0]), float(lat_1d[-1]), n_lat_data)
    if len(lon_1d) != n_lon_data:
        lon_1d = np.linspace(float(lon_1d[0]), float(lon_1d[-1]), n_lon_data)
    return lon_1d, lat_1d


def _make_figure(snaps: list[dict], grid, output_dir: Path) -> None:
    """3-row summary: surface T, SSH, surface speed at 3 evenly-spaced times."""
    finite = [s for s in snaps if np.all(np.isfinite(s["T"][:, :, 0]))]
    if not finite:
        print("  All snapshots non-finite — skipping figure.")
        return

    n_show = min(3, len(finite))
    show = [finite[int(i)] for i in np.linspace(0, len(finite) - 1, n_show)]

    # Derive coords from actual data shape (wall rows may inflate n_lat)
    data_nlat, data_nlon = show[0]["T"].shape[:2]
    lon_1d, lat_1d = _grid_coords_deg(grid, (data_nlat, data_nlon))

    fig, axes = plt.subplots(3, n_show, figsize=(5 * n_show, 12),
                             constrained_layout=True)
    if n_show == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("ACC Channel (lat-lon, wind-driven, Gaussian ridge)",
                 fontsize=13, fontweight="bold")

    for col, snap in enumerate(show):
        day = float(snap["time_days"])
        mask = snap["land_mask"] > 0.5

        # --- Row 0: surface temperature ---
        T_sfc = snap["T"][:, :, 0].copy()
        T_sfc[~mask] = np.nan
        ax = axes[0, col]
        im = ax.pcolormesh(lon_1d, lat_1d, T_sfc, cmap="RdYlBu_r",
                           vmin=0, vmax=8, shading="nearest")
        fig.colorbar(im, ax=ax, label="°C")
        ax.set_title(f"Day {day:.1f}  —  Surface T (°C)")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")

        # --- Row 1: sea-surface height ---
        eta = snap["eta"].copy()
        eta[~mask] = np.nan
        eta_abs = float(np.nanmax(np.abs(eta))) or 1e-3
        ax = axes[1, col]
        im = ax.pcolormesh(lon_1d, lat_1d, eta, cmap="RdBu_r",
                           vmin=-eta_abs, vmax=eta_abs, shading="nearest")
        fig.colorbar(im, ax=ax, label="m")
        ax.set_title(f"Day {day:.1f}  —  SSH (m, ±{eta_abs:.3f})")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")

        # --- Row 2: surface speed ---
        u_cc = 0.5 * (snap["u"][:, :-1, 0] + snap["u"][:, 1:, 0])
        v_cc = 0.5 * (snap["v"][:-1, :, 0] + snap["v"][1:, :, 0])
        spd = np.sqrt(u_cc ** 2 + v_cc ** 2)
        spd[~mask] = np.nan
        ax = axes[2, col]
        im = ax.pcolormesh(lon_1d, lat_1d, spd, cmap="plasma",
                           vmin=0, vmax=max(float(np.nanmax(spd)), 1e-4),
                           shading="nearest")
        fig.colorbar(im, ax=ax, label="m/s")
        ax.set_title(f"Day {day:.1f}  —  Surface speed (m/s)")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")

    out = output_dir / "diagnostics.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    import jax
    if not jax.config.x64_enabled:
        import warnings
        warnings.warn(
            "JAX_ENABLE_X64 is OFF. The ACC channel uses a linear EOS and "
            "barotropic-baroclinic split that benefit from 64-bit precision. "
            "Re-run with `JAX_ENABLE_X64=1 python scripts/run_acc.py`.",
            stacklevel=2,
        )

    # ---- Imports (after JAX init) ----------------------------------------
    from legoesm.grids.latlon import create_regional_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.eos import LinearEOSConfig
    from legoesm.ocean.experiments.acc_channel import (
        ACCChannelConfig,
        create_initial_conditions as acc_ic,
        create_forcings as acc_forcings,
        create_sponge as acc_sponge,
    )

    cfg = ACCChannelConfig()
    dt = args.dt
    nlev = args.n_levels

    # ---- Grid ---------------------------------------------------------------
    print("Building lat-lon channel grid "
          f"({args.n_lat} × {args.n_lon}, {nlev} levels)…")
    grid, _wall_mask = create_regional_latlon_grid(
        args.n_lat, args.n_lon,
        cfg.lat_south, cfg.lat_north,
        lon_west=cfg.lon_west, lon_east=cfg.lon_east,
        periodic_x=True,
    )
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=cfg.H_max)

    # ---- Physics / model config -------------------------------------------
    physics = acc_forcings("latlon_channel", None, cfg)
    eos_cfg = LinearEOSConfig(
        alpha_T=cfg.alpha_T,
        rho_ref=cfg.rho_0,
        T_ref=cfg.T_ref,
        S_ref=cfg.S_uniform,
    )
    model_cfg = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,
        physics=physics,
        A_h=cfg.A_h,
        B_h=cfg.B_h,
        C_smag=cfg.C_smag,
        K_h=cfg.K_h,
        A_v=cfg.A_v,
        K_v=cfg.K_v,
        bottom_drag_r=cfg.bottom_drag_coeff,
        eos="linear",
        eos_linear=eos_cfg,
        barotropic_diffusion_alpha=cfg.barotropic_diffusion_alpha,
        barotropic_div_damp=cfg.barotropic_div_damp,
    )
    model = LatLonCGridOceanModel(grid, z_coord, model_cfg)

    # ---- Initial conditions -----------------------------------------------
    print("Initialising ACC channel state…")
    state = acc_ic("latlon_channel", grid, z_coord, cfg)

    # ---- Sponge -------------------------------------------------------------
    sponge = None if args.no_sponge else acc_sponge(
        "latlon_channel", grid, z_coord, cfg)

    # ---- Output setup -------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = args.output_dir / "snapshots"
    snap_dir.mkdir(exist_ok=True)

    run_meta = {
        "days": args.days,
        "dt": dt,
        "n_lat": args.n_lat,
        "n_lon": args.n_lon,
        "n_levels": nlev,
        "lat_south": cfg.lat_south,
        "lat_north": cfg.lat_north,
        "lon_west": cfg.lon_west,
        "lon_east": cfg.lon_east,
        "sponge": not args.no_sponge,
        "acc_config": dataclasses.asdict(cfg),
    }
    with open(args.output_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # ---- Time loop ----------------------------------------------------------
    n_steps = int(round(args.days * 86400.0 / dt))
    snap_every = max(1, int(round(args.snapshot_every_days * 86400.0 / dt)))

    print(f"\nACC Channel: {args.n_lat}×{args.n_lon} lat-lon, {nlev} levels, "
          f"dt={dt}s")
    print(f"Run: {n_steps} steps = {args.days:.2f} days, "
          f"snapshot every {snap_every} steps "
          f"= {snap_every * dt / 86400.0:.2f} days")
    print(f"Sponge: {'OFF' if args.no_sponge else 'northern boundary restoring'}")
    print(f"Output: {args.output_dir}\n")
    print(f"{'step':>6} {'day':>7} {'u_max':>10} {'spd_sfc':>10} "
          f"{'|eta|_max':>10} {'T_sfc':>8}")

    snapshots: list[dict] = []
    snap_idx = 0
    _save_snapshot(state, 0.0, snap_dir, snap_idx)
    snapshots.append({
        "time_days": 0.0,
        "T": np.asarray(state.T.data),
        "eta": np.asarray(state.eta.data),
        "u": np.asarray(state.u.data),
        "v": np.asarray(state.v.data),
        "land_mask": np.asarray(state.land_mask.data),
    })

    t_wall_start = time.time()

    for k in range(n_steps):
        state = model.step(state, dt=dt, sponge=sponge)

        is_last = k == n_steps - 1
        if (k + 1) % snap_every == 0 or is_last:
            snap_idx += 1
            t_seconds = (k + 1) * dt
            _save_snapshot(state, t_seconds, snap_dir, snap_idx)
            d = _diagnose(state)
            snapshots.append({
                "time_days": t_seconds / 86400.0,
                "T": np.asarray(state.T.data),
                "eta": np.asarray(state.eta.data),
                "u": np.asarray(state.u.data),
                "v": np.asarray(state.v.data),
                "land_mask": np.asarray(state.land_mask.data),
            })
            print(f"{k+1:6d} {t_seconds/86400.0:7.2f} "
                  f"{d['u_max']:10.4e} {d['speed_sfc_max']:10.4e} "
                  f"{d['eta_max']:10.4e} {d['T_sfc_mean']:8.3f}")

    wall = time.time() - t_wall_start
    print(f"\nDone. Wall time: {wall:.1f}s  "
          f"({wall / n_steps * 1000:.1f} ms/step).  "
          f"Snapshots: {snap_idx + 1}")

    # ---- Diagnostic figure --------------------------------------------------
    if not args.no_figure:
        print("\nGenerating diagnostic figure…")
        _make_figure(snapshots, grid, args.output_dir)


if __name__ == "__main__":
    main()
