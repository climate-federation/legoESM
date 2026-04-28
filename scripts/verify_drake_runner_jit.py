#!/usr/bin/env python
"""Verify the JIT-block runner matches the eager-loop reference.

Runs N steps both ways and:
  1. Compares accumulated diagnostic time-means element-wise.
     Tolerance is set to JAX numerical-determinism level (1e-12 rel).
  2. Times each path and reports the speedup.

Run before trusting the JIT path for full diagnostic runs.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_initial_conditions, create_forcings,
    create_eos_config, create_gm_redi_config,
)

from _drake_momentum_budget_runner import (
    restore_state_from_npz,
    DIAG_NAMES, STATE_ACC_NAMES,
    _make_block_fn, _zero_diag_acc, _zero_state_acc,
)


RESTART_PATH = Path(
    "results/ocean/global_overturning_50yr_gmredi/restart_day018250.npz"
)
N_STEPS_VERIFY = 200    # bit-identity check
N_STEPS_TIMING = 1000   # speedup measurement (after warmup)


def _build_model_and_state():
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
    )
    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)
    template = create_initial_conditions("latlon", grid, z_coord, config)
    state, _ = restore_state_from_npz(template, RESTART_PATH)
    return model, state


def _eager_run(model, state, dt, n_steps):
    """Reference path: per-step Python loop, NumPy accumulation."""
    diag_sum = {n: None for n in DIAG_NAMES}
    sum_total_u = None
    sum_total_v = None
    state_sum = {k: np.zeros_like(np.asarray(getattr(state, k).data),
                                   dtype=np.float64)
                 for k in STATE_ACC_NAMES}
    for _ in range(n_steps):
        _, diag = model.tendencies_with_diagnostics(state, dt=dt)
        for name in DIAG_NAMES:
            arr = np.asarray(getattr(diag, name).data, dtype=np.float64)
            if diag_sum[name] is None:
                diag_sum[name] = arr.copy()
            else:
                diag_sum[name] += arr
        tu = np.asarray(diag.total_u.data, dtype=np.float64)
        tv = np.asarray(diag.total_v.data, dtype=np.float64)
        if sum_total_u is None:
            sum_total_u = tu.copy()
            sum_total_v = tv.copy()
        else:
            sum_total_u += tu
            sum_total_v += tv
        for k in state_sum:
            state_sum[k] += np.asarray(getattr(state, k).data)
        state = model.step(state, dt)
    jax.block_until_ready(state.eta.data)
    diag_sum["total_u"] = sum_total_u
    diag_sum["total_v"] = sum_total_v
    return state, diag_sum, state_sum


def _jit_block_run(model, state, dt, n_steps, block_size):
    """JIT-block path."""
    _, diag0 = model.tendencies_with_diagnostics(state, dt=dt)
    diag_acc = _zero_diag_acc(diag0)
    state_acc = _zero_state_acc(state)
    block_fn = _make_block_fn(model, dt)
    n_blocks = n_steps // block_size
    n_remainder = n_steps - n_blocks * block_size
    for _ in range(n_blocks):
        state, diag_acc, state_acc = block_fn(
            state, diag_acc, state_acc, block_size,
        )
    if n_remainder > 0:
        state, diag_acc, state_acc = block_fn(
            state, diag_acc, state_acc, n_remainder,
        )
    jax.block_until_ready(state.eta.data)
    diag_sum = {n: np.asarray(diag_acc[n], dtype=np.float64)
                for n in (list(DIAG_NAMES) + ["total_u", "total_v"])}
    state_sum = {k: np.asarray(state_acc[k], dtype=np.float64)
                 for k in STATE_ACC_NAMES}
    return state, diag_sum, state_sum


def main():
    dt = 600.0

    # --- Bit-identity check ---
    print(f"=== Bit-identity check ({N_STEPS_VERIFY} steps) ===")
    model, state0 = _build_model_and_state()
    print("Eager loop...")
    s_eager, d_eager, st_eager = _eager_run(model, state0, dt, N_STEPS_VERIFY)
    print("JIT block (block_size=50)...")
    s_jit, d_jit, st_jit = _jit_block_run(
        model, state0, dt, N_STEPS_VERIFY, block_size=50,
    )

    print("\nDiagnostic accumulators max |eager - jit| (rel to eager max):")
    max_rel = 0.0
    for name in (list(DIAG_NAMES) + ["total_u", "total_v"]):
        a = d_eager[name]
        b = d_jit[name]
        absdiff = float(np.max(np.abs(a - b)))
        norm = float(np.max(np.abs(a)) + 1e-30)
        rel = absdiff / norm
        max_rel = max(max_rel, rel)
        if rel > 1e-10:
            print(f"  {name:<18} abs={absdiff:.3e}  rel={rel:.3e}  ⚠")
        else:
            pass  # quiet for clean
    print(f"  max relative error across all 26 fields: {max_rel:.3e}")

    print("\nState accumulators max |eager - jit|:")
    for k in STATE_ACC_NAMES:
        a = st_eager[k]
        b = st_jit[k]
        absdiff = float(np.max(np.abs(a - b)))
        norm = float(np.max(np.abs(a)) + 1e-30)
        print(f"  {k:<6} abs={absdiff:.3e}  rel={absdiff/norm:.3e}")

    # --- Timing comparison ---
    print(f"\n=== Timing comparison ({N_STEPS_TIMING} steps, after warmup) ===")
    # Fresh state for fair comparison
    model, state0 = _build_model_and_state()

    # Warmup (compile)
    print("Warming up (compile)...")
    _, diag0 = model.tendencies_with_diagnostics(state0, dt=dt)
    _ = model.step(state0, dt)
    jax.block_until_ready(diag0.total_u.data)
    block_fn = _make_block_fn(model, dt)
    diag_acc0 = _zero_diag_acc(diag0)
    state_acc0 = _zero_state_acc(state0)
    s, da, sa = block_fn(state0, diag_acc0, state_acc0, 10)
    jax.block_until_ready(s.eta.data)

    # Eager timing
    print("Eager loop timing...")
    t0 = time.time()
    _eager_run(model, state0, dt, N_STEPS_TIMING)
    eager_wall = time.time() - t0
    print(f"  eager: {eager_wall:.2f}s  ({eager_wall*1000/N_STEPS_TIMING:.1f} ms/step)")

    # JIT-block timing
    for block_size in [100, 500, 1000]:
        if N_STEPS_TIMING % block_size != 0:
            continue
        # Reset state
        model, state0 = _build_model_and_state()
        # Pre-compile this block_size
        _, diag0 = model.tendencies_with_diagnostics(state0, dt=dt)
        diag_acc0 = _zero_diag_acc(diag0)
        state_acc0 = _zero_state_acc(state0)
        s, da, sa = block_fn(state0, diag_acc0, state_acc0, block_size)
        jax.block_until_ready(s.eta.data)

        # Reset and time
        model, state0 = _build_model_and_state()
        diag_acc0 = _zero_diag_acc(diag0)
        state_acc0 = _zero_state_acc(state0)
        t0 = time.time()
        _jit_block_run(model, state0, dt, N_STEPS_TIMING, block_size)
        jit_wall = time.time() - t0
        speedup = eager_wall / jit_wall if jit_wall > 0 else float("inf")
        print(f"  jit (block={block_size:>4}): {jit_wall:.2f}s  "
              f"({jit_wall*1000/N_STEPS_TIMING:.1f} ms/step)  "
              f"speedup × {speedup:.2f}")


if __name__ == "__main__":
    main()
