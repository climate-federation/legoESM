#!/usr/bin/env python3
"""Long run at A_h=5e4 for global barotropic wind (36x72, 180 days).

Check if the slowly growing speeds stabilize or blow up.
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

sys.path.insert(0, str(Path(__file__).parent))

from legoesm.ocean.experiments.global_barotropic_wind import (
    GlobalBarotropicWindConfig, create_initial_conditions, create_forcings,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig

A_h = 5e4
days = 360.0
dt = 300.0
n_steps = int(days * 86400 / dt)
diag_every = max(1, n_steps // 60)
nlev = 10

out_dir = Path("results/ocean/Ah_sweep")
out_dir.mkdir(parents=True, exist_ok=True)

gbw_config = GlobalBarotropicWindConfig(A_h=A_h)
physics = create_forcings("latlon", None, gbw_config)
grid = create_latlon_grid(n_lat=36, n_lon=72)
z_coord = create_ocean_z_star(n_levels=nlev, H_max=5500.0)
config_ = LatLonCGridOceanConfig.from_flat(
    n_barotropic_substeps=30, physics=physics, A_h=A_h,
)
model = LatLonCGridOceanModel(grid, z_coord, config_)
state = create_initial_conditions("latlon", grid, z_coord, gbw_config)

lon = np.asarray(grid.lon) * 180 / np.pi
lat = np.asarray(grid.lat) * 180 / np.pi

t0 = time.time()
max_speeds = []
max_etas = []
snap_days = []
eta_snapshots = []
# Save snapshots at specific days for the evolution plot
snap_targets = [5, 15, 30, 60, 90, 120, 180, 240, 360]
snap_idx = 0

for step in range(1, n_steps + 1):
    state = model.step(state, dt)
    day = step * dt / 86400

    if step % diag_every == 0 or step == n_steps:
        eta = np.asarray(state.eta.data)
        mask = np.asarray(state.land_mask.data)

        if not np.all(np.isfinite(eta)):
            print(f"  Day {day:.1f}: NaN/Inf — UNSTABLE")
            break

        u_sfc = np.asarray(state.u.data[..., 0])
        v_sfc = np.asarray(state.v.data[..., 0])
        u_cell = 0.5 * (u_sfc[:, :-1] + u_sfc[:, 1:])
        v_cell = 0.5 * (v_sfc[:-1, :] + v_sfc[1:, :])
        speed = np.sqrt(u_cell**2 + v_cell**2)
        max_spd = float(np.max(speed * mask))
        max_eta = float(np.max(np.abs(eta * mask)))
        max_speeds.append(max_spd)
        max_etas.append(max_eta)
        snap_days.append(day)

        print(f"  Day {day:6.1f}/{days:.0f} | max_eta={max_eta:.4f} | max_spd={max_spd:.4f}")

    # Capture snapshots at target days
    if snap_idx < len(snap_targets) and day >= snap_targets[snap_idx] - 0.01:
        eta = np.asarray(state.eta.data)
        mask = np.asarray(state.land_mask.data)
        eta_snapshots.append((snap_targets[snap_idx], np.where(mask > 0.5, eta, np.nan)))
        snap_idx += 1

wall = time.time() - t0
print(f"\nCompleted in {wall:.1f}s ({wall/60:.1f} min)")

# --- Plot 1: SSH evolution ---
n_snaps = len(eta_snapshots)
ncols = min(n_snaps, 3)
nrows = (n_snaps + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = np.atleast_2d(axes)

for i, (d, eta_masked) in enumerate(eta_snapshots):
    ax = axes[i // ncols, i % ncols]
    im = ax.pcolormesh(lon, lat, eta_masked, cmap="RdBu_r",
                      vmin=-0.08, vmax=0.08, shading="auto")
    ax.set_title(f"Day {d:.0f}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="SSH (m)", shrink=0.8)

for i in range(n_snaps, nrows * ncols):
    axes[i // ncols, i % ncols].set_visible(False)

fig.suptitle(f"SSH evolution — A_h = {A_h:.0e} m²/s, latlon 36x72",
            fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(out_dir / "Ah_5e4_long_SSH_evolution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_dir / 'Ah_5e4_long_SSH_evolution.png'}")

# --- Plot 2: Time series ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(snap_days, max_speeds, "b-")
ax1.set_ylabel("Max speed (m/s)")
ax1.set_title(f"A_h = {A_h:.0e} m²/s — {days:.0f} day run")
ax1.grid(True, alpha=0.3)
ax2.plot(snap_days, max_etas, "r-")
ax2.set_ylabel("Max |η| (m)")
ax2.set_xlabel("Day")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(out_dir / "Ah_5e4_long_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_dir / 'Ah_5e4_long_timeseries.png'}")
