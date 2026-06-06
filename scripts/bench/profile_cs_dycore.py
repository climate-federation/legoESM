"""Profile the cubed-sphere C48/L26 dycore step on a local GPU.

Builds the C48/L26 model exactly as ``scripts/bench/run_levante_gpu_scaling.py``
does (``run_benchmark(..., grid_type='cubed-sphere', physics_level='gray_sbm')``
through ``_build_segment_benchmark``), then breaks per-step wall time into
sub-stages using:

* ``time.perf_counter()`` + ``jax.block_until_ready`` around isolated
  closures for each stage (full segment, bare dycore step, hyperdiffusion
  on a 3D tracer field, PPM tracer advection, cubed-sphere halo
  exchange).
* ``jax.profiler.start_trace`` / ``stop_trace`` over a small window of
  full-segment steps so the resulting TensorBoard trace can be inspected
  in ``results/scaling_gpu_profile/``.

Read-only investigation — all heavy lifting reuses
``scripts.run_levante_gpu_scaling._build_segment_benchmark`` and the
production dycore step (``model.step``).

Invocation::

    PYTHONPATH=. JAX_ENABLE_X64=1 .venv/bin/python scripts/profile_cs_dycore.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

# Reuse the segment benchmark builder from the production scaling driver.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent  # scripts/bench/ -> repo root
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Force x64 BEFORE importing jax (segment benchmark uses fp64 by default).
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from scripts.bench.run_levante_gpu_scaling import _build_segment_benchmark

# Production modules used only for *isolated sub-stage* profiling.
from legoesm.core.operators_3d import (
    fv_scalar_advection_3d,
    hyperdiffusion_3d,
)
from legoesm.grids.halo import pad_halo_4d


# ===========================================================================
# Stage-level timing helpers
# ===========================================================================

def _time_stage(
    name: str,
    fn: Callable[[], object],
    n_warmup: int = 3,
    n_timing: int = 20,
) -> tuple[str, float]:
    """Warm + JIT-compile ``fn``, then return (name, ms_per_call_mean).

    ``fn`` is a thunk that returns one or more device arrays.  We force a
    device sync with ``jax.block_until_ready`` after every call so we are
    measuring real device wall-time, not async submission latency.
    """
    # Warmup (also triggers JIT compile of any wrapped jax.jit).
    for _ in range(n_warmup):
        out = fn()
        jax.block_until_ready(jax.tree.leaves(out))

    t0 = time.perf_counter()
    for _ in range(n_timing):
        out = fn()
    jax.block_until_ready(jax.tree.leaves(out))
    t1 = time.perf_counter()
    ms = (t1 - t0) / n_timing * 1000.0
    print(f"    [{name:<28s}] {ms:8.3f} ms/call")
    return name, ms


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    # --- C48/L26 baseline configuration, single GPU. -----------------------
    n_grid = 48
    n_levels = 26
    n_gpus = 1
    precision = "float64"
    # _build_segment_benchmark always wires radiation/convection on top of
    # the dycore.  Pick the cheapest moist tier (gray + sbm, no kessler)
    # so dycore cost is the largest fraction of segment time but the
    # build still goes through the same ModelDriver path as
    # run_levante_gpu_scaling.
    physics_level = "gray_sbm"

    profile_dir = _REPO / "results" / "scaling_gpu_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"profile_cs_dycore: C{n_grid}/L{n_levels} cubed-sphere, {precision},"
          f" physics={physics_level}, n_gpus={n_gpus}")
    print(f"device: {jax.devices()}  jax={jax.__version__}")
    print("=" * 72)

    # --- Build segment step (reuses run_levante_gpu_scaling builder). ------
    step_fn, carry, dt, total_cells, cells_per_gpu = _build_segment_benchmark(
        physics_level=physics_level,
        grid_type="cubed-sphere",
        n_grid=n_grid,
        n_levels=n_levels,
        n_gpus=n_gpus,
        precision=precision,
    )
    print(f"  dt={dt:.0f}s  cells={total_cells:,}  cells/GPU={cells_per_gpu:,}")

    # --- Independently reconstruct the production dycore + grid handles so
    #     we can also profile bare dycore.step() and individual operators
    #     against the same C48/L26 configuration.  No production code is
    #     mutated; we re-use the public ModelDriver path that
    #     _build_segment_benchmark uses internally.
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.model_driver import ModelDriver

    rad_scheme = "gray" if physics_level == "gray_sbm" else "rrtmgp"
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=n_grid, nlev=n_levels),
        dycore=DycoreConfig(
            model_type="hydrostatic",
            discretization="cdgrid",
            dt=dt,
            fix_mass=True,
        ),
        output=OutputConfig(diag_days=999, checkpoint_days=0),
        dataset="analytical",
        radiation=rad_scheme,
        convection="sbm",
        microphysics="none",
        topography="flat",
        days=1,
        precision="fp64" if precision == "float64" else "fp32",
        fix_moisture=False,
        n_devices=n_gpus,
    )
    driver = ModelDriver(config)
    driver.setup()
    dyn_model = driver.model            # CDGridPrimitiveEquationModel
    grid = driver.grid                  # CubedSphereGrid
    # The segment driver carries cell-centred HydrostaticState (see
    # ``_rebuild_state`` in legoesm.driver.compiled_segments); the dycore
    # routes to ``_step_cell_centre`` for this shape.  Use the same form
    # for our bare-dycore probe.
    state0 = driver.state               # HydrostaticState

    # ----------------------------------------------------------------------
    # Stage 1: full segment step (dycore + physics + fixers + hyperdiff_qv)
    # ----------------------------------------------------------------------
    # ``step_fn`` from _build_segment_benchmark is non-jit (wraps the
    # non-donating ``.raw`` variant).  Wrap in jit ourselves so the first
    # warmup call triggers compilation, mirroring run_levante's path.
    @jax.jit
    def _seg_step(c):
        return step_fn(c, dt)

    # ----------------------------------------------------------------------
    # Stage 2: bare cubed-sphere FV3 dycore step (no physics, no fixers).
    #   This is exactly model.step(state, dt) — the inner SSP-RK3 +
    #   sponge + dgrid sync + del6 vorticity damp.
    # ----------------------------------------------------------------------
    _dyn_step = dyn_model.step  # already JITted internally on _step_cell_centre
    def _dycore_only():
        return _dyn_step(state0, dt)

    # ----------------------------------------------------------------------
    # Stage 3: hyperdiffusion on a 3D scalar field (q_v-style smoothing).
    #   Matches the per-step hyperdiff call in
    #   legoesm.driver.compiled_segments._single_step.
    # ----------------------------------------------------------------------
    qv0 = jnp.asarray(driver.q_v)
    hd_coeff = float(getattr(dyn_model.config, "hyperdiff_coeff", 1.0e15))
    _hd_jit = jax.jit(lambda q: hyperdiffusion_3d(q, grid, hd_coeff))
    def _hyperdiff_qv():
        return _hd_jit(qv0)

    # ----------------------------------------------------------------------
    # Stage 4: PPM tracer advection (fv_scalar_advection_3d) on a moisture
    #   field, using the cell-centre winds reconstructed from the FV3 D-grid
    #   state.  This isolates the dominant cubed-sphere flux-divergence
    #   kernel that is exercised every RK3 sub-stage inside the dycore.
    # ----------------------------------------------------------------------
    u_cc = state0.u.data
    v_cc = state0.v.data
    _adv_jit = jax.jit(
        lambda q, u, v: fv_scalar_advection_3d(q, u, v, grid, limiter=True),
    )
    def _tracer_adv():
        return _adv_jit(qv0, u_cc, v_cc)

    # ----------------------------------------------------------------------
    # Stage 5: cubed-sphere halo exchange (4D, all levels at once).
    #   Per-step every halo-bearing operator (gradients, Laplacians, PPM,
    #   hyperdiff) routes through ``pad_halo_4d``.  Isolating this number
    #   tells us whether the cube-face stitching is the bottleneck.
    # ----------------------------------------------------------------------
    T0 = state0.T.data
    # pad_halo_4d takes (data, halo_width=1) — single-node path uses
    # the duogrid stored on the cubed-sphere grid for cross-face interp.
    _halo_duogrid = getattr(grid, "duogrid", None)
    _halo_jit = jax.jit(lambda f: pad_halo_4d(f, halo=1, duogrid=_halo_duogrid))
    def _halo_exchange():
        return _halo_jit(T0)

    # ----------------------------------------------------------------------
    # Run all stages once (warmup) before grouping into the table so JIT
    # compile time does not pollute the per-stage numbers.
    # ----------------------------------------------------------------------
    print("\nWarming up + timing per-stage closures:")
    stages = []
    stages.append(_time_stage("segment_step (full)", lambda: _seg_step(carry),
                              n_warmup=3, n_timing=30))
    stages.append(_time_stage("bare_dycore_step", _dycore_only,
                              n_warmup=3, n_timing=30))
    stages.append(_time_stage("hyperdiff_qv_3d", _hyperdiff_qv,
                              n_warmup=5, n_timing=100))
    stages.append(_time_stage("tracer_adv_ppm_3d", _tracer_adv,
                              n_warmup=5, n_timing=100))
    stages.append(_time_stage("halo_exchange_4d", _halo_exchange,
                              n_warmup=5, n_timing=200))

    # ----------------------------------------------------------------------
    # JAX profiler trace over a small window of full-segment steps.  The
    # resulting xplane.pb / trace.json in ``results/scaling_gpu_profile/``
    # can be opened with ``tensorboard --logdir results/scaling_gpu_profile``.
    # ----------------------------------------------------------------------
    n_trace_steps = 5
    print(f"\nRecording JAX profiler trace ({n_trace_steps} segment steps) to:"
          f"\n  {profile_dir}")
    # Block once to make sure all prior async work is flushed.
    jax.block_until_ready(jax.tree.leaves(carry))
    jax.profiler.start_trace(str(profile_dir))
    try:
        c = carry
        for _ in range(n_trace_steps):
            c = _seg_step(c)
        jax.block_until_ready(jax.tree.leaves(c))
    finally:
        jax.profiler.stop_trace()
    print("  trace written.")

    # ----------------------------------------------------------------------
    # Pretty-print breakdown table.
    # ----------------------------------------------------------------------
    # The "segment_step (full)" line is the reference 100% — every other
    # stage is reported as a fraction of it (with the bare dycore step the
    # most meaningful sub-component; hyperdiff/adv/halo are individual
    # operator costs and are *also* counted inside dycore + segment, so
    # they are deliberately reported as overlapping sub-stage costs, not
    # as a partition).
    total_ms = stages[0][1]
    print("\n" + "=" * 72)
    print(f"{'stage':<30s} {'ms/step':>10s}  {'% of segment':>14s}")
    print("-" * 72)
    for name, ms in stages:
        pct = 100.0 * ms / total_ms if total_ms > 0 else float("nan")
        print(f"{name:<30s} {ms:10.3f}  {pct:13.1f}%")
    print("=" * 72)

    # Residual = segment - bare_dycore (i.e. physics + fixers + qv smoothing).
    bare = stages[1][1]
    residual = total_ms - bare
    print(f"{'physics+fixers (residual)':<30s} {residual:10.3f}  "
          f"{100.0 * residual / total_ms:13.1f}%")
    print("=" * 72)
    print("Note: hyperdiff/tracer_adv/halo lines measure isolated operator")
    print("      costs.  They are already INCLUDED in dycore_step and")
    print("      segment_step totals (overlap, not partition).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
