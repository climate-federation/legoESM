"""DINO sensitivity: B_h = 5e14 m^4/s biharmonic viscosity (5× the round 2 value).

Restart from year-100, run 5 years with yearly snapshots.
Same format as run_dino_dissipation_sensitivity.py round 2 experiments.

Run on CPU (both GPUs occupied):
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
        python scripts/run_dino_biharm_5e14.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import time
from pathlib import Path
import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.experiments.dino import (
    DINOConfig, create_dino_z_star, dino_lat_lon_grid,
    dino_lat_lon_state, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, apply_dino_lat_lon_surface_forcing,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.core.field import Field


def to_f64_state(s):
    """Cast all Field members to float64 (CPU barotropic CG workaround)."""
    kw = {}
    for k in s._fields:
        v = getattr(s, k)
        if hasattr(v, 'data'):
            kw[k] = Field(jnp.asarray(v.data, dtype=jnp.float64))
    return s._replace(**kw)


# --- Config ---
B_H = 5e14
RUN_YEARS = 5
EXP_NAME = "F_biharm_5e14"

restart_path = Path("results/dino_100yr_latlon/snapshots/snapshot_00200.npz")
out_dir = Path(f"results/dino_sensitivity/{EXP_NAME}")
snap_dir = out_dir / "snapshots"
snap_dir.mkdir(parents=True, exist_ok=True)

cfg = DINOConfig()
z = create_dino_z_star(cfg)
grid = dino_lat_lon_grid(cfg, n_lon=50)
forcing = dino_lat_lon_surface_forcing_arrays(grid, cfg)
dt = cfg.dt

model_cfg_base, _ = dino_lat_lon_model_config(grid, cfg, physics=True)
model_cfg = model_cfg_base._replace(B_h=B_H)
model = LatLonCGridOceanModel(grid, z, model_cfg)

# --- Load restart ---
print(f"Loading restart: {restart_path}")
restart = np.load(restart_path)
restart_day = float(restart["time_days"])
print(f"  Restart at day {restart_day:.0f} (year {restart_day/365:.1f})")
print(f"  B_h = {B_H:.1e} m^4/s")
print(f"  Platform: {jax.devices()}")

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
state = to_f64_state(state)

# --- Run ---
run_days = RUN_YEARS * 365.0
n_steps = int(round(run_days * 86400.0 / dt))
snap_every = int(round(365.0 * 86400.0 / dt))

print(f"\n{n_steps} steps = {RUN_YEARS} years, snapshot every {snap_every} steps")

mask = np.asarray(state.land_mask.data) > 0.5
timeseries = []

t0 = time.time()
snap_idx = 0

for k in range(n_steps):
    state = apply_dino_lat_lon_surface_forcing(state, forcing, z, cfg, dt)
    state = model.step(state, dt=dt)
    state = to_f64_state(state)

    if (k + 1) % snap_every == 0 or k == n_steps - 1:
        snap_idx += 1
        t_days = (k + 1) * dt / 86400.0
        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data)
        eta = np.asarray(state.eta.data)
        T = np.asarray(state.T.data)

        d = {
            "step": k + 1,
            "day": float(t_days),
            "u_max": float(np.max(np.abs(u))),
            "v_max": float(np.max(np.abs(v))),
            "eta_max": float(np.max(np.abs(eta))),
            "T_max": float(np.max(T[mask, :])),
            "T_min": float(np.min(T[mask, :])),
            "eta_mean": float((eta * mask).sum() / mask.sum()),
        }
        timeseries.append(d)

        elapsed = time.time() - t0
        eta_total = elapsed / (k + 1) * n_steps
        print(f"  yr {t_days/365:5.1f}: |u|={d['u_max']:.3f} |v|={d['v_max']:.3f} "
              f"T=[{d['T_min']:.2f},{d['T_max']:.2f}] eta_mean={d['eta_mean']:.3f} "
              f"[{elapsed/60:.1f}/{eta_total/60:.1f}min]")

        np.savez_compressed(
            snap_dir / f"snapshot_{snap_idx:03d}.npz",
            time_days=t_days,
            eta=eta, T=T, S=np.asarray(state.S.data),
            u=u, v=v,
            land_mask=np.asarray(state.land_mask.data),
            H_bathy=np.asarray(state.H_bathy.data),
        )

wall = time.time() - t0
print(f"\nDone. {wall:.1f}s = {wall/60:.1f} min ({wall/n_steps*1000:.1f} ms/step)")

with open(out_dir / "timeseries.json", "w") as f:
    json.dump(timeseries, f, indent=2)
print(f"Wrote timeseries to {out_dir / 'timeseries.json'}")
