"""Extend the A_biharm_1e14 DINO run from year 105 → year 125 (20 more years).

Loads the year-105 snapshot from results/dino_sensitivity/A_biharm_1e14/
and continues with B_h = 1e14 m^4/s biharmonic viscosity.

Goal: see whether the large-scale Drake-Passage v power saturates or
keeps growing under pure biharmonic dissipation.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time
from pathlib import Path

import numpy as np

from legoesm.ocean.experiments.dino import (
    DINOConfig, create_dino_z_star, dino_lat_lon_grid,
    dino_lat_lon_state, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, apply_dino_lat_lon_surface_forcing,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel


# --- Restart input ---
restart_path = Path("results/dino_sensitivity/A_biharm_1e14/snapshots/snapshot_005.npz")
out_dir = Path("results/dino_sensitivity/A_biharm_1e14_ext")
snap_dir = out_dir / "snapshots"
snap_dir.mkdir(parents=True, exist_ok=True)

# --- Config ---
cfg = DINOConfig()  # paper-exact base (A_h_floor=0, A_h_eq_boost=1)
z = create_dino_z_star(cfg)
grid = dino_lat_lon_grid(cfg, n_lon=50)
forcing = dino_lat_lon_surface_forcing_arrays(grid, cfg)
dt = cfg.dt

# Build model config with B_h = 1e14 override
model_cfg_base, _ = dino_lat_lon_model_config(grid, cfg, physics=True)
model_cfg = model_cfg_base._replace(B_h=1e14)
model = LatLonCGridOceanModel(grid, z, model_cfg)

# --- Load restart ---
print(f"Loading restart: {restart_path}")
restart = np.load(restart_path)
restart_day = float(restart["time_days"])
print(f"  Restart at time_days = {restart_day:.0f}  (year {restart_day/365:.2f})")

# Rebuild state from snapshot
from legoesm.core.field import Field
from legoesm.ocean.state import LatLonCGridOceanState
import jax.numpy as jnp

fresh = dino_lat_lon_state(grid, z, cfg)
state = fresh._replace(
    eta=Field(jnp.array(restart["eta"], dtype=jnp.float64)),
    T=Field(jnp.array(restart["T"], dtype=jnp.float64)),
    S=Field(jnp.array(restart["S"], dtype=jnp.float64)),
    u=Field(jnp.array(restart["u"], dtype=jnp.float64)),
    v=Field(jnp.array(restart["v"], dtype=jnp.float64)),
    land_mask=Field(jnp.array(restart["land_mask"], dtype=jnp.float64)),
    H_bathy=Field(jnp.array(restart["H_bathy"], dtype=jnp.float64)),
)

# --- Run plan: 20 more years, yearly snapshots ---
EXTRA_YEARS = 20
run_days = EXTRA_YEARS * 365.0
n_steps = int(round(run_days * 86400.0 / dt))
snap_every = int(round(365.0 * 86400.0 / dt))

print(f"\nRunning {EXTRA_YEARS} more years ({n_steps} steps), "
      f"snapshot every {snap_every} steps = 1 year")
print(f"Config: B_h={model_cfg.B_h:.1e} m^4/s, "
      f"A_h_floor={model_cfg.A_h_floor}, A_h_eq_boost={model_cfg.A_h_eq_boost}")
print(f"Grid: {grid.n_lat}x{grid.n_lon}, dt={dt}s")
print(f"Output: {snap_dir}\n")


def compute_diagnostics(state, mask):
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    eta = np.asarray(state.eta.data)
    T = np.asarray(state.T.data)
    return {
        "u_max": float(np.max(np.abs(u))),
        "v_max": float(np.max(np.abs(v))),
        "eta_max": float(np.max(np.abs(eta))),
        "T_max": float(np.max(T[mask, :])),
        "T_min": float(np.min(T[mask, :])),
        "eta_mean": float((eta * mask).sum() / mask.sum()),
    }


# Save initial diagnostics
mask = np.asarray(state.land_mask.data) > 0.5
d0 = compute_diagnostics(state, mask)
print(f"  step     0: day {restart_day:7.0f} "
      f"|u|={d0['u_max']:.3f} |v|={d0['v_max']:.3f} "
      f"T=[{d0['T_min']:.2f},{d0['T_max']:.2f}] eta_mean={d0['eta_mean']:.3f}")

t0 = time.time()
# Continue snapshot numbering from where the original run left off (005 → 006...)
snap_idx = 5

for k in range(n_steps):
    state = apply_dino_lat_lon_surface_forcing(state, forcing, z, cfg, dt)
    state = model.step(state, dt=dt)

    if (k + 1) % snap_every == 0 or k == n_steps - 1:
        snap_idx += 1
        t_days = restart_day + (k + 1) * dt / 86400.0
        d = compute_diagnostics(state, mask)
        elapsed = time.time() - t0
        eta_total = elapsed / (k + 1) * n_steps
        print(f"  step {k+1:6d}: day {t_days:7.0f} (yr {t_days/365:.1f}) "
              f"|u|={d['u_max']:.3f} |v|={d['v_max']:.3f} "
              f"T=[{d['T_min']:.2f},{d['T_max']:.2f}] eta_mean={d['eta_mean']:.3f} "
              f"[{elapsed/60:.1f}/{eta_total/60:.1f}min]")

        np.savez_compressed(
            snap_dir / f"snapshot_{snap_idx:03d}.npz",
            time_days=t_days,
            eta=np.asarray(state.eta.data),
            T=np.asarray(state.T.data),
            S=np.asarray(state.S.data),
            u=np.asarray(state.u.data),
            v=np.asarray(state.v.data),
            land_mask=np.asarray(state.land_mask.data),
            H_bathy=np.asarray(state.H_bathy.data),
        )

wall = time.time() - t0
print(f"\nDone. Wall time: {wall:.1f}s = {wall/60:.1f} min "
      f"({wall/n_steps*1000:.1f} ms/step)")
print(f"Final state: year {(restart_day + run_days)/365:.1f}")
