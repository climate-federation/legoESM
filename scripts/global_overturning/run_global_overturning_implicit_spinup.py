#!/usr/bin/env python
"""Follow-up A: 10-year fresh spinup of GO+GM/Redi with the implicit
Crank-Nicolson barotropic solver.

Starts from rest (no flow, the experiment's default stratified IC) and
integrates for 10 sim-years using ``barotropic_solver = 'implicit_cn'``
from step 1 — so the resulting state has no chequerboard inherited
from the old explicit-substep solver.  This restart is then the clean
IC for the 1-yr verification run that decides whether Crit 1.2/1.3 of
``docs/issues/barotropic_mode_noise.md`` pass.

Inner stepping uses the same lax.scan + JIT pattern as
``_drake_momentum_budget_runner.py`` (Follow-up D), without the
per-term momentum diagnostic — gives ~10x speedup over a pure Python
loop for a model.step-only run.

Outputs (results/ocean/global_overturning_implicit_spinup/):
  - restart_day{000000,000730,001460,...,003650}.npz
    every 2 sim-years; days are integer count from day 0
  - run.log

Usage:
    JAX_ENABLE_X64=1 python scripts/global_overturning/run_global_overturning_implicit_spinup.py
"""

from __future__ import annotations

import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_initial_conditions, create_forcings,
    create_eos_config, create_gm_redi_config,
)


OUTPUT_DIR = Path("results/ocean/global_overturning_implicit_spinup")


def _save_restart(state, day, output_dir):
    npz = {"step": int(round(day * 86400 / 600)), "time_days": float(day),
           "grid_type": "latlon"}
    for f in state._fields:
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        npz[f] = np.asarray(obj.data)
    fname = output_dir / f"restart_day{int(round(day)):06d}.npz"
    np.savez_compressed(fname, **npz)
    print(f"    Restart saved: {fname.name}")


def _make_step_block(model, dt):
    """JIT-compiled n-step scan of model.step (no diagnostic capture)."""
    def scan_body(state, _):
        return model.step(state, dt), None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    return block_fn


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Configuration ----
    total_years = 10.0
    dt = 600.0
    n_steps = int(total_years * 365.0 * 86400 / dt)
    block_size = 1000
    restart_every_years = 2.0
    n_steps_per_restart = int(restart_every_years * 365.0 * 86400 / dt)

    config = GlobalOverturningConfig(use_gm_redi=True)
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels, H_max=config.H_max,
        dz_surface=config.dz_surface, dz_deep=config.dz_deep,
    )
    grid = create_latlon_grid(36, 72)
    physics = create_forcings("latlon", grid, config)
    eos_config = create_eos_config(config)
    gm_redi_cfg = create_gm_redi_config(config)
    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,                 # unused by implicit solver
        physics=physics,
        A_h=config.A_h, A_v=config.A_v, K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        eos="linear", eos_linear=eos_config,
        gm_redi=gm_redi_cfg,
        barotropic_solver="implicit_cn",
    )

    print(f"=== Follow-up A: 10-year fresh spinup with implicit solver ===")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  barotropic_solver = {ocean_config.barotropic_solver}")
    print(f"  Grid: 36×72 (5°), 20 levels, H_max={config.H_max} m")
    print(f"  dt = {dt} s, n_steps = {n_steps:,} ({total_years} sim-yr)")
    print(f"  Block size: {block_size} steps  ({n_steps // block_size} blocks)")
    print(f"  Restart cadence: every {restart_every_years} yr "
          f"({n_steps // n_steps_per_restart} mid-run + 1 final restart)")
    print()

    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)
    state = create_initial_conditions("latlon", grid, z_coord, config)
    print("Initial state: rest with stratified T (T_water_init_C=20°C, T_deep=2°C)")
    print(f"  T range: [{float(jnp.min(state.T.data)):.2f}, "
          f"{float(jnp.max(state.T.data)):.2f}] °C")
    print(f"  S uniform: {float(jnp.mean(state.S.data)):.2f} PSU")
    print(f"  η: {float(jnp.max(jnp.abs(state.eta.data))):.3e} m  (rest)")
    print()

    block_fn = _make_step_block(model, dt)

    # Save day-0 restart for reference
    _save_restart(state, 0.0, OUTPUT_DIR)

    n_blocks = n_steps // block_size
    n_remainder = n_steps - n_blocks * block_size
    print(f"Starting integration ({n_blocks} blocks × {block_size} steps "
          f"+ {n_remainder} remainder)")
    t0 = time.time()
    last_print = t0

    steps_done = 0
    last_restart_step = 0
    progress_every = max(1, n_blocks // 30)

    for b in range(n_blocks):
        state = block_fn(state, block_size)
        steps_done += block_size

        # Periodic restarts at multiples of ~n_steps_per_restart
        if (steps_done - last_restart_step) >= n_steps_per_restart:
            jax.block_until_ready(state.eta.data)
            day = steps_done * dt / 86400.0
            _save_restart(state, day, OUTPUT_DIR)
            last_restart_step = steps_done

        if (b + 1) % progress_every == 0 or (b + 1) == n_blocks:
            now = time.time()
            if now - last_print > 30 or (b + 1) == n_blocks:
                jax.block_until_ready(state.eta.data)
                yr = steps_done * dt / 86400.0 / 365.0
                eta_max = float(jnp.max(jnp.abs(state.eta.data)))
                T_max = float(jnp.max(state.T.data))
                T_min = float(jnp.min(state.T.data))
                u_max = float(jnp.max(jnp.abs(state.u.data)))
                elapsed = now - t0
                eta_s = elapsed / yr * total_years - elapsed if yr > 0 else 0
                if eta_s < 3600:
                    eta_str = f"{eta_s/60:.1f} min"
                else:
                    eta_str = f"{eta_s/3600:.2f} h"
                print(f"  Yr {yr:5.2f}/{total_years:.1f} | "
                      f"|η|max={eta_max:.2e} | "
                      f"T∈[{T_min:.1f},{T_max:.1f}] | "
                      f"|u|max={u_max:.3f} | "
                      f"ETA {eta_str}", flush=True)
                last_print = now

    if n_remainder > 0:
        state = block_fn(state, n_remainder)
        steps_done += n_remainder

    jax.block_until_ready(state.eta.data)
    wall = time.time() - t0
    final_day = steps_done * dt / 86400.0
    print(f"\nSpinup complete in {wall:.0f}s ({wall/60:.1f} min)")
    print(f"  Final state at sim day {final_day:.0f}")
    print(f"  |η|max = {float(jnp.max(jnp.abs(state.eta.data))):.4e} m")
    print(f"  T range: [{float(jnp.min(state.T.data)):.2f}, "
          f"{float(jnp.max(state.T.data)):.2f}] °C")
    print(f"  |u|max = {float(jnp.max(jnp.abs(state.u.data))):.4e} m/s")

    # Final restart
    _save_restart(state, final_day, OUTPUT_DIR)
    print(f"\nReady for verification: re-run "
          f"scripts/run/run_drake_momentum_budget_implicit.py with "
          f"RESTART_PATH = {OUTPUT_DIR}/restart_day{int(round(final_day)):06d}.npz")


if __name__ == "__main__":
    main()
