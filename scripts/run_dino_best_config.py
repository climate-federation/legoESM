"""DINO 20-year run from proper ICs with best dissipation config.

Config I: B_h=5e14, A_h=75k constant (no cos scaling), K_h=0 (GM/Redi only).
Start from analytical DINO initial conditions (rest state).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
    """Cast all Field members to float64 (barotropic CG dtype workaround)."""
    kw = {}
    for k in s._fields:
        v = getattr(s, k)
        if hasattr(v, 'data'):
            kw[k] = Field(jnp.asarray(v.data, dtype=jnp.float64))
    return s._replace(**kw)


# --- Config ---
RUN_YEARS = 20
SNAP_EVERY_YEARS = 1

out_dir = Path("results/dino_20yr_best_config")
snap_dir = out_dir / "snapshots"
snap_dir.mkdir(parents=True, exist_ok=True)

cfg = DINOConfig()
z = create_dino_z_star(cfg)
grid = dino_lat_lon_grid(cfg, n_lon=50)
forcing = dino_lat_lon_surface_forcing_arrays(grid, cfg)
dt = cfg.dt

# Best config (experiment I)
model_cfg_base, _ = dino_lat_lon_model_config(grid, cfg, physics=True)
model_cfg = model_cfg_base._replace(
    B_h=5e14,
    A_h=75000.0,
    A_h_lat_scaling=False,
    A_h_floor=0.0,
    A_h_eq_boost=1.0,
    K_h=0.0,
)
model = LatLonCGridOceanModel(grid, z, model_cfg)

# --- Initial conditions from analytical DINO profiles ---
state = dino_lat_lon_state(grid, z, cfg)
state = to_f64_state(state)

print(f"DINO 20-year run from rest — best config (I)")
print(f"  B_h = {model_cfg.B_h:.1e} m^4/s")
print(f"  A_h = {model_cfg.A_h:.0f} m^2/s (constant)")
print(f"  K_h = {model_cfg.K_h}")
print(f"  Grid: {grid.n_lat}x{grid.n_lon}, dt={dt}s")
print(f"  Platform: {jax.devices()}", flush=True)

# Save metadata
meta = {
    "config": {
        "B_h": model_cfg.B_h,
        "A_h": model_cfg.A_h,
        "A_h_lat_scaling": model_cfg.A_h_lat_scaling,
        "A_h_floor": model_cfg.A_h_floor,
        "A_h_eq_boost": model_cfg.A_h_eq_boost,
        "K_h": model_cfg.K_h,
        "dt": dt,
        "run_years": RUN_YEARS,
    },
    "grid": {
        "n_lat": grid.n_lat,
        "n_lon": grid.n_lon,
    },
}
with open(out_dir / "run_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# --- Run ---
run_days = RUN_YEARS * 365.0
n_steps = int(round(run_days * 86400.0 / dt))
snap_every = int(round(SNAP_EVERY_YEARS * 365.0 * 86400.0 / dt))

print(f"\n{n_steps} steps = {RUN_YEARS} years, snapshot every {snap_every} steps ({SNAP_EVERY_YEARS} yr)")
print(flush=True)

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
        S = np.asarray(state.S.data)

        d = {
            "step": k + 1,
            "day": float(t_days),
            "u_max": float(np.max(np.abs(u))),
            "v_max": float(np.max(np.abs(v))),
            "eta_max": float(np.max(np.abs(eta))),
            "T_max": float(np.max(T[mask, :])),
            "T_min": float(np.min(T[mask, :])),
            "S_max": float(np.max(S[mask, :])),
            "S_min": float(np.min(S[mask, :])),
            "eta_mean": float((eta * mask).sum() / mask.sum()),
        }
        timeseries.append(d)

        elapsed = time.time() - t0
        eta_total = elapsed / (k + 1) * n_steps
        print(f"  yr {t_days/365:6.1f}: |u|={d['u_max']:.3f} |v|={d['v_max']:.3f} "
              f"T=[{d['T_min']:.2f},{d['T_max']:.2f}] "
              f"S=[{d['S_min']:.2f},{d['S_max']:.2f}] "
              f"eta_mean={d['eta_mean']:.3f} "
              f"[{elapsed/60:.1f}/{eta_total/60:.1f}min]", flush=True)

        np.savez_compressed(
            snap_dir / f"snapshot_{snap_idx:03d}.npz",
            time_days=t_days,
            eta=eta, T=T, S=S,
            u=u, v=v,
            land_mask=np.asarray(state.land_mask.data),
            H_bathy=np.asarray(state.H_bathy.data),
        )

wall = time.time() - t0
print(f"\nDone. {wall:.1f}s = {wall/60:.1f} min ({wall/n_steps*1000:.1f} ms/step)")

with open(out_dir / "timeseries.json", "w") as f:
    json.dump(timeseries, f, indent=2)
print(f"Wrote timeseries to {out_dir / 'timeseries.json'}")
