#!/usr/bin/env python3
"""Sweep A_h for global barotropic wind at 36x72 (5 deg), 60 days.

Finds the minimum stable A_h that preserves gyre structure.
Saves SSH snapshots for each value to results/ocean/Ah_sweep/.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"

import sys
import time
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))

from legoesm.ocean.experiments.global_barotropic_wind import (
    GlobalBarotropicWindConfig, create_initial_conditions, create_forcings,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig

# Sweep values
A_h_values = [1e4, 2e4, 5e4, 1e5, 2e5, 5e5]
days = 60.0
dt = 300.0
n_steps = int(days * 86400 / dt)
diag_every = max(1, n_steps // 20)
nlev = 10

out_dir = Path("results/ocean/Ah_sweep")
out_dir.mkdir(parents=True, exist_ok=True)

results = {}

for A_h in A_h_values:
    label = f"A_h={A_h:.0e}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    gbw_config = GlobalBarotropicWindConfig(A_h=A_h)
    physics = create_forcings("latlon", None, gbw_config)

    grid = create_latlon_grid(n_lat=36, n_lon=72)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=5500.0)

    config_ = LatLonCGridOceanConfig.from_flat(
        n_barotropic_substeps=30,
        physics=physics,
        A_h=A_h,
    )
    model = LatLonCGridOceanModel(grid, z_coord, config_)
    state = create_initial_conditions("latlon", grid, z_coord, gbw_config)

    t0 = time.time()
    ok = True
    max_speeds = []
    eta_snapshots = []
    snap_times = []

    for step in range(1, n_steps + 1):
        state = model.step(state, dt)

        if step % diag_every == 0 or step == n_steps:
            eta = np.asarray(state.eta.data)
            mask = np.asarray(state.land_mask.data)

            if not np.all(np.isfinite(eta)):
                print(f"    Step {step}: NaN/Inf detected — UNSTABLE")
                ok = False
                break

            # Compute max speed from u,v at surface
            u_sfc = np.asarray(state.u.data[..., 0])
            v_sfc = np.asarray(state.v.data[..., 0])
            # u is on u-faces (nlat, nlon+1), v on v-faces (nlat+1, nlon)
            u_cell = 0.5 * (u_sfc[:, :-1] + u_sfc[:, 1:])
            v_cell = 0.5 * (v_sfc[:-1, :] + v_sfc[1:, :])
            speed = np.sqrt(u_cell**2 + v_cell**2)
            max_spd = float(np.max(speed * mask))
            max_speeds.append(max_spd)

            day = step * dt / 86400
            max_eta = float(np.max(np.abs(eta * mask)))
            print(f"    Day {day:6.1f}/{days:.0f} | max_eta={max_eta:.4f} | max_spd={max_spd:.4f}")

            eta_snapshots.append(eta.copy())
            snap_times.append(day)

    wall = time.time() - t0
    status = "PASS" if ok else "FAIL"
    print(f"  {status} in {wall:.1f}s")

    results[A_h] = {
        "status": status,
        "wall": wall,
        "max_speeds": max_speeds,
        "eta_snapshots": eta_snapshots,
        "snap_times": snap_times,
        "mask": np.asarray(state.land_mask.data),
    }

# --- Plot comparison ---
lon = np.asarray(grid.lon) * 180 / np.pi
lat = np.asarray(grid.lat) * 180 / np.pi
lon_2d, lat_2d = np.meshgrid(lon, lat)

stable_values = [a for a in A_h_values if results[a]["status"] == "PASS"]
n_stable = len(stable_values)

if n_stable > 0:
    fig, axes = plt.subplots(2, min(n_stable, 3), figsize=(5*min(n_stable, 3), 8))
    if n_stable == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif n_stable <= 3:
        axes = axes.reshape(2, -1)
    else:
        # More than 3: just show first 3 and last 3
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, A_h in enumerate(stable_values[:6]):
        row = i // 3
        col = i % 3
        if n_stable <= 3:
            ax = axes[0, i] if i < n_stable else None
        else:
            ax = axes[row, col]
        if ax is None:
            continue

        snaps = results[A_h]["eta_snapshots"]
        mask = results[A_h]["mask"]
        eta_final = snaps[-1]
        eta_masked = np.where(mask > 0.5, eta_final, np.nan)

        im = ax.pcolormesh(lon, lat, eta_masked, cmap="RdBu_r",
                          vmin=-0.08, vmax=0.08, shading="auto")
        ax.set_title(f"A_h = {A_h:.0e}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax, label="SSH (m)", shrink=0.8)

    # Time series of max speed
    if n_stable <= 3:
        for i, A_h in enumerate(stable_values):
            ax = axes[1, i]
            ax.plot(results[A_h]["snap_times"], results[A_h]["max_speeds"])
            ax.set_xlabel("Day")
            ax.set_ylabel("Max speed (m/s)")
            ax.set_title(f"A_h = {A_h:.0e}")
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"A_h sweep — global barotropic wind, latlon 36x72, {days:.0f} days",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "Ah_sweep_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_dir / 'Ah_sweep_comparison.png'}")

# Also make a single figure with ALL stable runs final SSH
if n_stable > 0:
    ncols = min(n_stable, 3)
    nrows = (n_stable + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n_stable == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, A_h in enumerate(stable_values):
        ax = axes[i // ncols, i % ncols]
        snaps = results[A_h]["eta_snapshots"]
        mask = results[A_h]["mask"]
        eta_final = snaps[-1]
        eta_masked = np.where(mask > 0.5, eta_final, np.nan)
        im = ax.pcolormesh(lon, lat, eta_masked, cmap="RdBu_r",
                          vmin=-0.08, vmax=0.08, shading="auto")
        ax.set_title(f"A_h = {A_h:.0e} m²/s")
        ax.set_aspect("auto")
        plt.colorbar(im, ax=ax, label="SSH (m)", shrink=0.8)

    # Hide unused axes
    for i in range(n_stable, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle(f"Final SSH (day {days:.0f}) — A_h sweep, latlon 36x72",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "Ah_sweep_final_SSH.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'Ah_sweep_final_SSH.png'}")

# Summary
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"  {'A_h':>10}  {'Status':>6}  {'Max speed':>10}  {'Wall':>8}")
print(f"  {'-'*42}")
for A_h in A_h_values:
    r = results[A_h]
    spd = r["max_speeds"][-1] if r["max_speeds"] else 0
    print(f"  {A_h:10.0e}  {r['status']:>6}  {spd:10.4f}  {r['wall']:7.1f}s")
