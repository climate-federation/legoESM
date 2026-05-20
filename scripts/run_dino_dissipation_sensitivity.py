"""DINO dissipation sensitivity tests.

Restart from the year-100 state (which has accumulated grid-scale noise)
and run 5 years with different dissipation schemes. Compare which ones
damp the near-grid-scale modes.

Experiments:
  0. CONTROL     — same as the 100-year run (paper-exact, no extras)
  1. BIHARM      — biharmonic viscosity B_h = 1e10 m^4/s
  2. SMAG_LAP    — Laplacian Smagorinsky C_smag_lap = 0.15 (MOM6 OM4)
  3. SMAG_BIH    — biharmonic Smagorinsky C_smag = 3.0
  4. LEITH       — modified Leith C_leith = 2.0
  5. AH_FLOOR    — restore A_h_floor = 1000 m^2/s
  6. BIHARM+FLOOR — B_h = 1e10 + A_h_floor = 1000
"""
import dataclasses
import time
from pathlib import Path

import numpy as np

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.ocean.experiments.dino import (
    DINOConfig, create_dino_z_star, dino_lat_lon_grid,
    dino_lat_lon_state, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, apply_dino_lat_lon_surface_forcing,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig

# --- Load year-100 restart ---
restart_path = Path("results/dino_100yr_latlon/snapshots/snapshot_00200.npz")
print(f"Loading restart: {restart_path}")
restart = np.load(restart_path)
print(f"Restart at day {restart['time_days']:.0f} (year {restart['time_days']/365:.1f})")

# --- Build grid and z-coordinate ---
cfg = DINOConfig()
z = create_dino_z_star(cfg)
grid = dino_lat_lon_grid(cfg, n_lon=50)
forcing = dino_lat_lon_surface_forcing_arrays(grid, cfg)
dt = cfg.dt

# --- Define experiments ---
# Each is a dict of overrides to apply to the model config
experiments = {
    # Round 2: calibrated parameters based on single-step tendency test
    # Base A_h = 15012 m^2/s, so floors < 15000 are no-ops.
    "A_biharm_1e14": {"B_h": 1e14},
    "B_smag_lap_1": {"C_smag_lap": 1.0},
    "C_smag_lap_2": {"C_smag_lap": 2.0},
    "D_ah_2x": {"A_h_floor": 30000.0},  # double the base
    "E_biharm_smag": {"B_h": 1e14, "C_smag_lap": 1.0},  # combo
}

run_days = 5 * 365  # 5 years
n_steps = int(round(run_days * 86400.0 / dt))
snap_every = int(round(365.0 * 86400.0 / dt))  # yearly snapshots

print(f"\n{n_steps} steps = {run_days/365:.0f} years, snapshot every {snap_every} steps")
print(f"Running {len(experiments)} experiments\n")


def rebuild_state(grid, z_coord, restart_data):
    """Reconstruct a LatLonCGridOceanState from NPZ snapshot."""
    from legoesm.core.field import Field
    from legoesm.ocean.state import LatLonCGridOceanState
    import jax.numpy as jnp

    eta = Field(jnp.array(restart_data["eta"], dtype=jnp.float64))
    T = Field(jnp.array(restart_data["T"], dtype=jnp.float64))
    S = Field(jnp.array(restart_data["S"], dtype=jnp.float64))
    u = Field(jnp.array(restart_data["u"], dtype=jnp.float64))
    v = Field(jnp.array(restart_data["v"], dtype=jnp.float64))
    land_mask = Field(jnp.array(restart_data["land_mask"], dtype=jnp.float64))
    H_bathy = Field(jnp.array(restart_data["H_bathy"], dtype=jnp.float64))

    # Build a fresh state to get the right masks, then replace fields
    fresh = dino_lat_lon_state(grid, z_coord, cfg)
    state = fresh._replace(
        eta=eta, T=T, S=S, u=u, v=v,
        land_mask=land_mask, H_bathy=H_bathy,
    )
    return state


def compute_diagnostics(state, mask):
    """Quick diagnostics dict."""
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    eta = np.asarray(state.eta.data)
    T = np.asarray(state.T.data)
    S = np.asarray(state.S.data)

    u_sfc = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
    v_sfc = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])

    # 5-pt Laplacian (checkerboard) for u
    u_m = np.where(mask, u_sfc, 0.0)
    lap = np.zeros_like(u_m)
    lap[1:-1, 1:-1] = (u_m[1:-1, 1:-1]
        - 0.25 * (u_m[:-2, 1:-1] + u_m[2:, 1:-1]
                   + u_m[1:-1, :-2] + u_m[1:-1, 2:]))
    checker_u = np.sqrt(np.mean(lap[1:-1, 1:-1][mask[1:-1, 1:-1]]**2))

    # Same for v
    v_m = np.where(mask, v_sfc, 0.0)
    lap_v = np.zeros_like(v_m)
    lap_v[1:-1, 1:-1] = (v_m[1:-1, 1:-1]
        - 0.25 * (v_m[:-2, 1:-1] + v_m[2:, 1:-1]
                   + v_m[1:-1, :-2] + v_m[1:-1, 2:]))
    checker_v = np.sqrt(np.mean(lap_v[1:-1, 1:-1][mask[1:-1, 1:-1]]**2))

    return {
        "u_max": float(np.max(np.abs(u))),
        "v_max": float(np.max(np.abs(v))),
        "eta_max": float(np.max(np.abs(eta))),
        "T_max": float(np.max(T[mask, :])),
        "T_min": float(np.min(T[mask, :])),
        "checker_u": float(checker_u),
        "checker_v": float(checker_v),
        "eta_mean": float((eta * mask).sum() / mask.sum()),
    }


# --- Run experiments ---
results = {}

for name, overrides in experiments.items():
  try:
    print(f"=== {name} ===")
    print(f"  Overrides: {overrides}")

    # Build model config with overrides
    model_cfg_base, _ = dino_lat_lon_model_config(grid, cfg, physics=True)
    if overrides:
        model_cfg = model_cfg_base._replace(**overrides)
    else:
        model_cfg = model_cfg_base

    model = LatLonCGridOceanModel(grid, z, model_cfg)
    state = rebuild_state(grid, z, restart)

    # Output dir
    out_dir = Path(f"results/dino_sensitivity/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(exist_ok=True)

    # Save initial state diagnostics
    mask = np.asarray(state.land_mask.data) > 0.5
    diag0 = compute_diagnostics(state, mask)

    timeseries = [{"step": 0, "day": 0, **diag0}]
    print(f"  step     0: checker_u={diag0['checker_u']:.4e} checker_v={diag0['checker_v']:.4e}")

    t0 = time.time()
    snap_idx = 0
    for k in range(n_steps):
        state = apply_dino_lat_lon_surface_forcing(state, forcing, z, cfg, dt)
        state = model.step(state, dt=dt)

        if (k + 1) % snap_every == 0 or k == n_steps - 1:
            snap_idx += 1
            d = compute_diagnostics(state, mask)
            t_days = (k + 1) * dt / 86400.0
            timeseries.append({"step": k + 1, "day": t_days, **d})
            print(f"  step {k+1:5d}: day {t_days:7.0f} "
                  f"checker_u={d['checker_u']:.4e} checker_v={d['checker_v']:.4e} "
                  f"|u|={d['u_max']:.3f} |v|={d['v_max']:.3f} "
                  f"eta_mean={d['eta_mean']:.3f}")

            # Save snapshot
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
    print(f"  Done: {wall:.1f}s ({wall/n_steps*1000:.1f} ms/step)\n")

    # Save timeseries
    import json
    with open(out_dir / "timeseries.json", "w") as f:
        json.dump(timeseries, f, indent=2)

    results[name] = timeseries
  except Exception as e:
    print(f"  FAILED: {e}\n")

# --- Summary comparison ---
print("\n=== SUMMARY: Checkerboard noise after 5 years ===")
print(f"{'Experiment':>20} {'checker_u_init':>14} {'checker_u_final':>15} {'ratio':>8} {'checker_v_final':>15}")
for name, ts in results.items():
    c0 = ts[0]["checker_u"]
    cf = ts[-1]["checker_u"]
    cv = ts[-1]["checker_v"]
    ratio = cf / c0 if c0 > 0 else float("nan")
    print(f"{name:>20} {c0:14.4e} {cf:15.4e} {ratio:8.3f} {cv:15.4e}")
