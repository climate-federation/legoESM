#!/usr/bin/env python
"""200-year global overturning circulation experiment.

Runs the global baroclinic overturning experiment on a lat-lon 36x72
(5-deg) grid for 200 years with periodic restart saves and diagnostics.

Usage:
    JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_global_overturning_200yr.py

Restart files are saved every 10 years to allow continuation.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the scripts directory is on the path for ocean_test_matrix imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_initial_conditions, create_forcings,
    create_eos_config,
)


def main():
    # ---- Configuration ----
    total_years = 200
    restart_every_years = 10      # save restart every N years
    diag_every_days = 36.5        # ~10 diagnostics per year
    snapshot_every_years = 5      # save full 2D snapshots every N years
    dt = 600.0                    # 10-min timestep (convective CFL safe)
    n_lat, n_lon = 36, 72         # 5-degree resolution
    n_barotropic_substeps = 30

    output_dir = Path("results/ocean/global_overturning_200yr")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GlobalOverturningConfig()
    total_days = total_years * 365.0
    n_steps = int(total_days * 86400 / dt)
    diag_every = max(1, int(diag_every_days * 86400 / dt))
    restart_every = int(restart_every_years * 365 * 86400 / dt)
    snapshot_every = int(snapshot_every_years * 365 * 86400 / dt)

    print(f"Global Overturning 200-year run")
    print(f"  Grid: lat-lon {n_lat}x{n_lon} (5-deg)")
    print(f"  Vertical: {config.n_levels} levels, H_max={config.H_max}m")
    print(f"  dt={dt}s, n_steps={n_steps:,}")
    print(f"  A_h={config.A_h:.0e}, K_v={config.K_v:.0e}, A_v={config.A_v:.0e}")
    print(f"  bottom_drag_r={config.bottom_drag_coeff}")
    print(f"  Restart save every {restart_every_years} years")
    print(f"  Diagnostics every {diag_every_days} days")
    print(f"  Output: {output_dir}")
    print()

    # ---- Setup ----
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels,
        H_max=config.H_max,
        dz_surface=config.dz_surface,
        dz_deep=config.dz_deep,
    )
    grid = create_latlon_grid(n_lat, n_lon)
    physics = create_forcings("latlon", grid, config)
    eos_config = create_eos_config(config)

    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=n_barotropic_substeps,
        physics=physics,
        A_h=config.A_h,
        A_v=config.A_v,
        K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        eos="linear",
        eos_linear=eos_config,
    )
    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)
    state = create_initial_conditions("latlon", grid, z_coord, config)

    # ---- Coordinate arrays for diagnostics ----
    lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
    lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    # ---- Diagnostic storage ----
    diag_times = []
    diag_max_speed = []
    diag_mean_sst = []
    diag_mean_T = []
    diag_T_deep = []
    diag_mean_eta = []
    snapshots = {}

    def record_diagnostics(state, day):
        """Extract and store scalar diagnostics."""
        eta = np.asarray(state.eta.data)
        T = np.asarray(state.T.data)
        u = np.asarray(state.u.data)    # (nlat, nlon+1, nlev)
        v = np.asarray(state.v.data)    # (nlat+1, nlon, nlev)
        mask = np.asarray(state.land_mask.data)

        # Interpolate staggered velocities to cell centers for speed
        u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])   # (nlat, nlon)
        v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])   # (nlat, nlon)
        speed = np.sqrt(u_c**2 + v_c**2)
        max_spd = float(np.max(speed))

        ocean_mask = mask > 0.5
        sst = T[:, :, 0]
        mean_sst = float(np.mean(sst[ocean_mask])) if ocean_mask.any() else 0.0
        mean_T_all = float(np.mean(T[ocean_mask])) if ocean_mask.any() else 0.0
        T_bot = T[:, :, -1]
        mean_T_deep = float(np.mean(T_bot[ocean_mask])) if ocean_mask.any() else 0.0
        mean_eta = float(np.mean(eta[ocean_mask])) if ocean_mask.any() else 0.0

        diag_times.append(day)
        diag_max_speed.append(max_spd)
        diag_mean_sst.append(mean_sst)
        diag_mean_T.append(mean_T_all)
        diag_T_deep.append(mean_T_deep)
        diag_mean_eta.append(mean_eta)

    def save_snapshot(state, step, day):
        """Save 2D fields for later plotting."""
        eta = np.asarray(state.eta.data)
        T = np.asarray(state.T.data)
        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data)
        u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
        v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
        speed_sfc = np.sqrt(u_c**2 + v_c**2)
        snapshots[day] = {
            "eta": eta.copy(),
            "SST": T[:, :, 0].copy(),
            "speed_sfc": speed_sfc.copy(),
            "T_zonal_mean": np.mean(T, axis=1).copy(),  # lat x depth
        }

    def save_restart_file(state, step, day):
        """Save restart NPZ."""
        restart = {
            "step": step,
            "time_days": day,
            "grid_type": "latlon",
        }
        for field_name in state._fields:
            field_obj = getattr(state, field_name)
            if hasattr(field_obj, 'data'):
                restart[field_name] = np.asarray(field_obj.data)
        fname = output_dir / f"restart_day{int(day):06d}.npz"
        np.savez_compressed(fname, **restart)
        print(f"  Restart saved: {fname}")

    # ---- Initial diagnostics ----
    record_diagnostics(state, 0.0)
    save_snapshot(state, 0, 0.0)

    # ---- Time integration ----
    print(f"\nStarting {total_years}-year integration ({n_steps:,} steps)...")
    t0 = time.time()
    last_print = t0

    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1
        day = step * dt / 86400.0

        # Blowup check every 100 steps
        if step % 100 == 0:
            eta_max = float(jnp.max(jnp.abs(state.eta.data)))
            if not np.isfinite(eta_max) or eta_max > 100.0:
                print(f"\n  BLOWUP at step {step} (day {day:.1f}), "
                      f"eta_max={eta_max}")
                save_restart_file(state, step, day)
                break

        # Diagnostics
        if step % diag_every == 0:
            record_diagnostics(state, day)
            now = time.time()
            if now - last_print > 30:
                yr = day / 365.0
                elapsed = now - t0
                rate = day / elapsed * 365  # years per wall-second * 365
                eta_s = elapsed / yr * total_years - elapsed if yr > 0 else 0
                print(f"  Year {yr:6.1f}/{total_years} | "
                      f"spd={diag_max_speed[-1]:.3f} | "
                      f"SST={diag_mean_sst[-1]:.2f} | "
                      f"T_deep={diag_T_deep[-1]:.2f} | "
                      f"eta={diag_mean_eta[-1]:.2e} | "
                      f"ETA {eta_s/3600:.1f}h")
                last_print = now

        # Snapshots
        if step % snapshot_every == 0:
            save_snapshot(state, step, day)

        # Restart saves
        if step % restart_every == 0:
            save_restart_file(state, step, day)

    jax.block_until_ready(state.eta.data)
    wall = time.time() - t0
    print(f"\nCompleted in {wall:.0f}s ({wall/3600:.1f}h)")

    # ---- Save final restart ----
    final_day = n_steps * dt / 86400.0
    save_restart_file(state, n_steps, final_day)

    # ---- Save diagnostics ----
    np.savez_compressed(
        output_dir / "diagnostics.npz",
        times=np.array(diag_times),
        max_speed=np.array(diag_max_speed),
        mean_sst=np.array(diag_mean_sst),
        mean_T=np.array(diag_mean_T),
        T_deep=np.array(diag_T_deep),
        mean_eta=np.array(diag_mean_eta),
    )
    print(f"  Diagnostics saved: {output_dir / 'diagnostics.npz'}")

    # ---- Save snapshots ----
    snap_data = {}
    for day_key, fields in snapshots.items():
        for fname, arr in fields.items():
            snap_data[f"{fname}_day{int(day_key):06d}"] = arr
    snap_data["snapshot_days"] = np.array(sorted(snapshots.keys()))
    np.savez_compressed(output_dir / "snapshots.npz", **snap_data)
    print(f"  Snapshots saved: {output_dir / 'snapshots.npz'}")

    # ---- Plot diagnostics ----
    _plot_diagnostics(output_dir, diag_times, diag_max_speed,
                      diag_mean_sst, diag_mean_T, diag_T_deep,
                      diag_mean_eta, snapshots, lat_deg)
    print(f"\nAll output in: {output_dir}")


def _plot_diagnostics(output_dir, times, max_speed, mean_sst,
                      mean_T, T_deep, mean_eta, snapshots, lat_deg):
    """Generate summary plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = np.array(times) / 365.0

    # ---- Timeseries ----
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(years, max_speed)
    axes[0].set_ylabel("Max speed (m/s)")
    axes[0].set_title("Global Overturning — 200-year timeseries")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(years, mean_sst, label="SST")
    axes[1].set_ylabel("Mean SST (degC)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(years, mean_T, label="All-depth")
    axes[2].plot(years, T_deep, label="Bottom layer")
    axes[2].set_ylabel("Mean T (degC)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(years, mean_eta)
    axes[3].set_ylabel("Mean SSH (m)")
    axes[3].set_xlabel("Year")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "timeseries_200yr.png", dpi=150)
    plt.close()

    # ---- Snapshot evolution (SSH, SST, speed) ----
    snap_days = sorted(snapshots.keys())
    n_snaps = len(snap_days)
    if n_snaps > 0:
        fig, axes = plt.subplots(n_snaps, 3, figsize=(15, 3 * n_snaps))
        if n_snaps == 1:
            axes = axes[np.newaxis, :]
        for i, day in enumerate(snap_days):
            yr = day / 365.0
            s = snapshots[day]
            for j, (key, label, cmap) in enumerate([
                ("eta", "SSH (m)", "RdBu_r"),
                ("SST", "SST (degC)", "RdYlBu_r"),
                ("speed_sfc", "Speed (m/s)", "magma"),
            ]):
                im = axes[i, j].imshow(
                    s[key], aspect="auto", origin="lower", cmap=cmap)
                plt.colorbar(im, ax=axes[i, j], fraction=0.046)
                if i == 0:
                    axes[i, j].set_title(label)
                axes[i, j].set_ylabel(f"Year {yr:.0f}")
        plt.suptitle("Global Overturning — snapshot evolution", y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / "snapshots_200yr.png", dpi=120)
        plt.close()

    # ---- Zonal-mean T sections ----
    if n_snaps > 0:
        fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 5))
        if n_snaps == 1:
            axes = [axes]
        for i, day in enumerate(snap_days):
            yr = day / 365.0
            T_zm = snapshots[day]["T_zonal_mean"]  # (nlat, nlev)
            im = axes[i].imshow(
                T_zm.T, aspect="auto", origin="upper",
                cmap="RdYlBu_r", extent=[lat_deg[0], lat_deg[-1],
                                          T_zm.shape[1], 0])
            plt.colorbar(im, ax=axes[i], fraction=0.046, label="T (degC)")
            axes[i].set_title(f"Year {yr:.0f}")
            axes[i].set_xlabel("Latitude (deg)")
            if i == 0:
                axes[i].set_ylabel("Level index")
        plt.suptitle("Zonal-mean temperature sections", y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "T_zonal_mean_200yr.png", dpi=150)
        plt.close()

    print(f"  Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
