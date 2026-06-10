#!/usr/bin/env python
"""5-year idealized Wolfe-Cessi on flat bottom, 1° (180×360).

Same physics as the 5° idealized baseline but at 1° resolution.
Flat bottom, simple continent — no ETOPO bathymetry.
This isolates whether 1° grid spacing alone causes instability,
separate from topographic effects.

Usage:
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_global_overturning_idealized_1deg.py
"""
from __future__ import annotations

import os, sys, time
from functools import partial
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_initial_conditions,
    create_forcings, create_eos_config,
)

OUTPUT_DIR = Path("results/ocean/global_overturning_idealized_1deg")


def _save_restart(state, day, output_dir):
    npz = {"time_days": float(day), "grid_type": "latlon"}
    for f in state._fields:
        obj = getattr(state, f)
        if obj is not None and hasattr(obj, "data"):
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

    total_years = float(os.environ.get("GO_YEARS", "5.0"))
    dt = 300.0          # tighter CFL at 1°
    n_steps = int(total_years * 365.0 * 86400 / dt)
    block_size = 288    # = 1 day
    restart_every_years = 1.0
    n_steps_per_restart = int(restart_every_years * 365.0 * 86400 / dt)

    config = GlobalOverturningConfig()  # flat bottom, simple continent
    grid = create_latlon_grid(180, 360)
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels, H_max=config.H_max,
        dz_surface=config.dz_surface, dz_deep=config.dz_deep,
    )
    physics = create_forcings("latlon", grid, config)
    eos_config = create_eos_config(config)

    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,
        physics=physics,
        A_h=config.A_h,
        A_v=config.A_v,
        K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        eos="linear",
        eos_linear=eos_config,
        barotropic_solver="implicit_cn",
    )
    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)
    state = create_initial_conditions("latlon", grid, z_coord, config)

    print(f"=== 1° idealized Wolfe-Cessi (flat bottom) ===")
    print(f"  Grid: 180×360, {config.n_levels} levels, H_max={config.H_max}")
    print(f"  A_h={config.A_h:.0e}, dt={dt}s, {total_years} yr")
    print(f"  {n_steps:,} steps, block_size={block_size}")
    print(flush=True)

    block_fn = _make_step_block(model, dt)
    _save_restart(state, 0.0, OUTPUT_DIR)

    n_blocks = n_steps // block_size
    t0 = time.time()
    s = state
    last_restart_step = 0
    steps_done = 0
    progress_every = max(1, n_blocks // 100)

    for b in range(n_blocks):
        s = block_fn(s, block_size)
        steps_done += block_size

        if (steps_done - last_restart_step) >= n_steps_per_restart:
            jax.block_until_ready(s.eta.data)
            day = steps_done * dt / 86400.0
            _save_restart(s, day, OUTPUT_DIR)
            last_restart_step = steps_done

        if (b + 1) % progress_every == 0 or (b + 1) == n_blocks:
            jax.block_until_ready(s.eta.data)
            yr = steps_done * dt / (365.0 * 86400.0)
            u_max = float(jnp.max(jnp.abs(s.u.data)))
            finite = bool(jnp.all(jnp.isfinite(s.u.data)))
            eta_remaining = (time.time() - t0) / max(yr, 1e-3) * max(total_years - yr, 0) / 60
            print(f"  Yr {yr:5.2f}/{total_years:.0f} | |u|max={u_max:.3f} | "
                  f"finite={finite} | ETA {eta_remaining:.1f} min", flush=True)
            if not finite:
                print("  BLEW UP", flush=True)
                break

    wall = time.time() - t0
    print(f"\nDone in {wall:.0f}s ({wall/60:.1f} min)")
    _save_restart(s, steps_done * dt / 86400.0, OUTPUT_DIR)


if __name__ == "__main__":
    main()
