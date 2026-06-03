#!/usr/bin/env python
"""Continuation of GO+GM/Redi from the implicit-solver 10-yr spinup
endpoint to sim-year 50 (40 more years).

Loads ``restart_day003650.npz`` from the spinup, integrates 40 more
sim-yr with ``barotropic_solver = 'implicit_cn'``, saving restarts
every 5 sim-yr to match the original 50yr-run cadence (days 5475,
7300, 9125, 10950, 12775, 14600, 16425, 18250).

Output: ``results/ocean/global_overturning_50yr_implicit/``

Companion to the original (broken-solver) run at
``results/ocean/global_overturning_50yr_gmredi/`` for direct comparison
of the Drake-band momentum budget, MOC, and zonal-mean fields after
the chequerboard noise is removed.
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


SPINUP_RESTART = Path(
    "results/ocean/global_overturning_implicit_spinup/restart_day003650.npz"
)
OUTPUT_DIR = Path("results/ocean/global_overturning_50yr_implicit")


def _restore_state(template, restart_path):
    npz = np.load(restart_path, allow_pickle=False)
    new = {}
    for f in template._fields:
        obj = getattr(template, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        if f not in npz.files:
            raise KeyError(f"Restart {restart_path} missing {f!r}")
        new[f] = obj.replace(data=jnp.asarray(npz[f], dtype=obj.data.dtype))
    return template._replace(**new), float(npz["time_days"])


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
    def scan_body(state, _):
        return model.step(state, dt), None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    return block_fn


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extension_years = 40.0
    dt = 600.0
    n_steps = int(extension_years * 365.0 * 86400 / dt)
    block_size = 1000
    restart_every_years = 5.0
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
        n_barotropic_substeps=30,
        physics=physics,
        A_h=config.A_h, A_v=config.A_v, K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        eos="linear", eos_linear=eos_config,
        gm_redi=gm_redi_cfg,
        barotropic_solver="implicit_cn",
    )

    print(f"=== 40-yr continuation: spinup yr 10 → sim-yr 50 ===")
    print(f"  Restart from: {SPINUP_RESTART}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  barotropic_solver = {ocean_config.barotropic_solver}")
    print(f"  dt = {dt} s, n_steps = {n_steps:,} ({extension_years} more sim-yr)")
    print(f"  Block size: {block_size}  (~{n_steps // block_size} blocks)")
    print(f"  Restart cadence: every {restart_every_years} yr "
          f"({n_steps // n_steps_per_restart} mid-run + 1 final)")

    if not SPINUP_RESTART.exists():
        raise FileNotFoundError(f"Spinup restart not found: {SPINUP_RESTART}")

    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)
    template = create_initial_conditions("latlon", grid, z_coord, config)
    state, day_offset = _restore_state(template, SPINUP_RESTART)
    print(f"\nLoaded spinup state at day {day_offset:.1f} (year {day_offset/365:.2f})")
    print(f"  |η|max = {float(jnp.max(jnp.abs(state.eta.data))):.4e} m")
    print(f"  T range: [{float(jnp.min(state.T.data)):.2f}, "
          f"{float(jnp.max(state.T.data)):.2f}] °C")
    print(f"  |u|max = {float(jnp.max(jnp.abs(state.u.data))):.4e} m/s")
    print()

    block_fn = _make_step_block(model, dt)

    print(f"Starting integration ({n_steps:,} steps)")
    t0 = time.time()
    last_print = t0
    steps_done = 0
    last_restart_step = 0
    n_blocks = n_steps // block_size
    n_remainder = n_steps - n_blocks * block_size
    progress_every = max(1, n_blocks // 50)

    for b in range(n_blocks):
        state = block_fn(state, block_size)
        steps_done += block_size

        if (steps_done - last_restart_step) >= n_steps_per_restart:
            jax.block_until_ready(state.eta.data)
            day = day_offset + steps_done * dt / 86400.0
            _save_restart(state, day, OUTPUT_DIR)
            last_restart_step = steps_done

        if (b + 1) % progress_every == 0 or (b + 1) == n_blocks:
            now = time.time()
            if now - last_print > 30 or (b + 1) == n_blocks:
                jax.block_until_ready(state.eta.data)
                yr = day_offset / 365.0 + steps_done * dt / 86400.0 / 365.0
                eta_max = float(jnp.max(jnp.abs(state.eta.data)))
                T_max = float(jnp.max(state.T.data))
                T_min = float(jnp.min(state.T.data))
                u_max = float(jnp.max(jnp.abs(state.u.data)))
                elapsed = now - t0
                yr_done = steps_done * dt / 86400.0 / 365.0
                eta_s = (
                    elapsed / yr_done * extension_years - elapsed
                    if yr_done > 0 else 0
                )
                eta_str = (f"{eta_s/60:.1f} min" if eta_s < 3600
                           else f"{eta_s/3600:.2f} h")
                print(f"  Yr {yr:5.1f}/50 (extension {yr_done:5.2f}/{extension_years}) | "
                      f"|η|max={eta_max:.2e} | "
                      f"T∈[{T_min:.1f},{T_max:.1f}] | "
                      f"|u|max={u_max:.3f} | ETA {eta_str}", flush=True)
                last_print = now

    if n_remainder > 0:
        state = block_fn(state, n_remainder)
        steps_done += n_remainder

    jax.block_until_ready(state.eta.data)
    wall = time.time() - t0
    final_day = day_offset + steps_done * dt / 86400.0
    print(f"\nContinuation done in {wall:.0f}s ({wall/60:.1f} min, "
          f"{wall/3600:.2f} h)")
    print(f"  Final state: sim day {final_day:.0f} (year {final_day/365:.2f})")
    print(f"  |η|max = {float(jnp.max(jnp.abs(state.eta.data))):.4e} m")
    print(f"  T range: [{float(jnp.min(state.T.data)):.2f}, "
          f"{float(jnp.max(state.T.data)):.2f}] °C")
    print(f"  |u|max = {float(jnp.max(jnp.abs(state.u.data))):.4e} m/s")

    _save_restart(state, final_day, OUTPUT_DIR)


if __name__ == "__main__":
    main()
