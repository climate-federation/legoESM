"""Profile the plane CRM (f-plane compressible Euler) step on GPU.

Mirrors ``scripts/bench/profile_cs_dycore.py`` but for the plane non-hydrostatic
dycore at N=96 / nlev=30 (276 480 horizontal cells x 30 levels), the same
configuration used by ``scripts/bench/bench_crm_gpu_scaling.py``.

For each stage we report ``ms/step`` measured via
``time.perf_counter()`` + ``jax.block_until_ready`` around isolated
closures::

  * full_step               — ``model.step(state, dt)`` (RK3 + acoustic
                              substep loop + hyperdiff + Smag + sponge
                              + Coriolis). Reference for the % column.
  * slow_tend               — single call to
                              ``plane_compressible_euler_slow_tendencies``
                              (PG, Coriolis, mass continuity, advection,
                              vertical advection, sponge, hyperdiff,
                              Smag).
  * acoustic_substeps_x12   — one call to
                              ``plane_acoustic_substeps_semi_implicit``
                              with ``n_substeps=12`` (the per-stage cost
                              of the 12-iter fori_loop the production
                              path runs three times per RK3 step).
  * acoustic_substeps_x1    — one substep only (sanity check; the
                              fori_loop should be 12 x this).
  * hyperdiff_biharmonic    — isolated ``laplacian_vlast(laplacian_vlast)``
                              on theta_p — the kernel applied to u, v,
                              theta', rho', w inside slow_tend.
  * sponge_profile_only     — multiplication by the precomputed Rayleigh
                              sponge taper (sanity check; trivially fast
                              compared to substeps/hyperdiff).

Then ``jax.profiler.start_trace`` / ``stop_trace`` records a window of
five full-step calls into ``results/scaling_crm_gpu_profile/`` for
TensorBoard inspection.

Read-only investigation. Every stage uses an existing leaf/public symbol
from the dynamics module — no production code is modified.

Invocation::

    PYTHONPATH=. JAX_ENABLE_X64=1 .venv/bin/python scripts/profile_crm_step.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent  # scripts/bench/ -> repo root
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Force x64 BEFORE importing jax (bench script uses fp64 by default).
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Mirror run_levante_gpu_scaling._configure_jax defaults so the profiled
# step runs with the same XLA scheduling tweaks production uses.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (  # noqa: E402
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (  # noqa: E402
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
    make_rest_state,
    plane_acoustic_substeps_semi_implicit,
    plane_compressible_euler_slow_tendencies,
    laplacian_vlast,
)
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (  # noqa: E402
    sponge_profile,
)
from legoesm.grids.plane import create_plane_grid  # noqa: E402
from legoesm.grids.vertical import (  # noqa: E402
    create_stretched_height_coordinate,
)
from legoesm.timestepping.split_explicit import SplitExplicitConfig  # noqa: E402


# ===========================================================================
# Builders — replicate scripts/bench/bench_crm_gpu_scaling.py:_build_model exactly
# ===========================================================================

def _build_model(nx: int, ny: int, nlev: int, dx: float, dt: float,
                 dtype_x64: bool = True):
    dtype = jnp.float64 if dtype_x64 else jnp.float32
    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx,
                             dtype=dtype)
    hc = create_stretched_height_coordinate(nlev, H=20_000.0, dz_sfc=100.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        n_acoustic_substeps=12, semi_implicit_acoustic=True,
        sponge_coeff=0.05, sponge_width=5000.0,
        hyperdiff_coeff=1e6, hyperdiff_rho_coeff=1e6, hyperdiff_w_coeff=1e6,
        smagorinsky_cs=0.0, use_coriolis=True,
        fix_mass=False, anchor_mass_to_initial=False,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    state = make_rest_state(grid, hc, dtype=dtype)
    rng = jax.random.PRNGKey(0)
    kick = 0.001 * jax.random.normal(
        rng, state.theta_prime.data.shape, dtype=dtype,
    )
    state = state._replace(theta_prime=state.theta_prime.replace(data=kick))
    return model, state, grid, hc, tm, cfg, dtype


# ===========================================================================
# Stage timer — same pattern as profile_cs_dycore._time_stage
# ===========================================================================

def _time_stage(
    name: str,
    fn: Callable[[], object],
    n_warmup: int = 3,
    n_timing: int = 30,
) -> tuple[str, float]:
    """Warm + compile, then return ``(name, ms_per_call_mean)``."""
    out = None
    for _ in range(n_warmup):
        out = fn()
        jax.block_until_ready(jax.tree.leaves(out))

    t0 = time.perf_counter()
    for _ in range(n_timing):
        out = fn()
    jax.block_until_ready(jax.tree.leaves(out))
    t1 = time.perf_counter()
    ms = (t1 - t0) / n_timing * 1000.0
    print(f"    [{name:<32s}] {ms:9.3f} ms/call")
    return name, ms


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    nx = ny = 96
    nlev = 30
    dx = 2000.0
    dt = 2.0

    profile_dir = _REPO / "results" / "scaling_crm_gpu_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"profile_crm_step: plane CRM N{nx} / nlev={nlev}, "
          f"dx={dx:.0f} m, dt={dt:.0f} s")
    print(f"device: {jax.devices()}  jax={jax.__version__}  "
          f"x64={jax.config.read('jax_enable_x64')}")
    print("=" * 78)

    model, state, grid, hc, tm, cfg, dtype = _build_model(
        nx, ny, nlev, dx, dt, dtype_x64=True,
    )
    total_cells = nx * ny * nlev
    print(f"  cells = {nx}x{ny}x{nlev} = {total_cells:,}  "
          f"dtype={jnp.dtype(dtype).name}")
    print(f"  n_acoustic_substeps={cfg.n_acoustic_substeps}  "
          f"semi_implicit={cfg.semi_implicit_acoustic}  "
          f"smag_cs={cfg.smagorinsky_cs}  coriolis={cfg.use_coriolis}")

    # ----------------------------------------------------------------------
    # Stage 1: full step (the production path — calls _step_jit which wires
    # split_explicit_step around slow_tend + acoustic_update_fn).
    # ----------------------------------------------------------------------
    _step = model.step  # already jit'd internally on _step_jit.

    def _full_step():
        return _step(state, dt)

    # ----------------------------------------------------------------------
    # Stage 2: single slow_tendency call. The plane RK3 path evaluates this
    # three times per outer step (Strang/Shu-Osher RK3 stages).
    # ----------------------------------------------------------------------
    _slow_jit = jax.jit(
        lambda s: plane_compressible_euler_slow_tendencies(
            s, grid, hc, tm, cfg,
        ),
    )

    def _slow_tend():
        return _slow_jit(state)

    # ----------------------------------------------------------------------
    # Stage 3: acoustic substep loop with the full 12 iterations. This is
    # the per-RK-stage cost of the inner fori_loop. Production runs it
    # three times per outer step.
    # ----------------------------------------------------------------------
    se_cfg = SplitExplicitConfig(
        n_substeps=cfg.n_acoustic_substeps,
        outer_integrator=cfg.outer_integrator,
    )
    # Build a zero "slow tendency" pytree (acoustic substep only uses it
    # for signature parity; the column kernel never reads it).
    zero_tend = jax.tree.map(
        lambda x: jnp.zeros_like(x) if hasattr(x, "shape") else x, state,
    )
    dt_substep = float(dt) / cfg.n_acoustic_substeps
    n_sub_12 = int(cfg.n_acoustic_substeps)
    n_sub_1 = 1

    _acoustic_jit_12 = jax.jit(
        lambda s: plane_acoustic_substeps_semi_implicit(
            s, zero_tend, dt_substep, n_sub_12, se_cfg, hc, tm, cfg,
        ),
    )

    def _acoustic_substeps_12():
        return _acoustic_jit_12(state)

    _acoustic_jit_1 = jax.jit(
        lambda s: plane_acoustic_substeps_semi_implicit(
            s, zero_tend, dt_substep, n_sub_1, se_cfg, hc, tm, cfg,
        ),
    )

    def _acoustic_substeps_1():
        return _acoustic_jit_1(state)

    # ----------------------------------------------------------------------
    # Stage 4: isolated biharmonic hyperdiffusion on a single 3D field.
    # The slow_tend applies this exact kernel to {u, v, theta', rho', w}.
    # ----------------------------------------------------------------------
    theta_p0 = state.theta_prime.data

    @jax.jit
    def _hd(f):
        return laplacian_vlast(laplacian_vlast(f, grid), grid)

    def _hyperdiff_biharmonic():
        return _hd(theta_p0)

    # ----------------------------------------------------------------------
    # Stage 5: sponge taper multiply. The same Rayleigh sponge profile is
    # multiplied by u, v, theta', rho', w inside slow_tend. Isolating it
    # tells us how cheap it is relative to the substep loop.
    # ----------------------------------------------------------------------
    sponge_full = sponge_profile(
        hc.z_full, hc.H, cfg.sponge_width, cfg.sponge_coeff,
    )

    @jax.jit
    def _sponge_mul(f):
        return sponge_full * f

    def _sponge_only():
        return _sponge_mul(theta_p0)

    # ----------------------------------------------------------------------
    # Run + collect.
    # ----------------------------------------------------------------------
    print("\nWarming up + timing per-stage closures:")
    stages: list[tuple[str, float]] = []
    stages.append(_time_stage(
        "full_step (model.step)", _full_step, n_warmup=3, n_timing=30,
    ))
    stages.append(_time_stage(
        "slow_tend (1 RK3 stage)", _slow_tend, n_warmup=3, n_timing=30,
    ))
    stages.append(_time_stage(
        "acoustic_substeps x12", _acoustic_substeps_12,
        n_warmup=3, n_timing=30,
    ))
    stages.append(_time_stage(
        "acoustic_substeps x1", _acoustic_substeps_1,
        n_warmup=3, n_timing=50,
    ))
    stages.append(_time_stage(
        "hyperdiff_biharmonic_3d", _hyperdiff_biharmonic,
        n_warmup=5, n_timing=100,
    ))
    stages.append(_time_stage(
        "sponge_taper_multiply", _sponge_only,
        n_warmup=5, n_timing=200,
    ))

    # ----------------------------------------------------------------------
    # JAX profiler trace.
    # ----------------------------------------------------------------------
    n_trace_steps = 5
    print(f"\nRecording JAX profiler trace ({n_trace_steps} full steps) to:"
          f"\n  {profile_dir}")
    jax.block_until_ready(jax.tree.leaves(state))
    jax.profiler.start_trace(str(profile_dir))
    try:
        s = state
        for _ in range(n_trace_steps):
            s = _step(s, dt)
        jax.block_until_ready(jax.tree.leaves(s))
    finally:
        jax.profiler.stop_trace()
    print("  trace written.")

    # ----------------------------------------------------------------------
    # Breakdown table.
    # ----------------------------------------------------------------------
    full_ms = stages[0][1]
    print("\n" + "=" * 78)
    print(f"{'stage':<32s} {'ms/step':>10s}  {'% of full_step':>16s}")
    print("-" * 78)
    for name, ms in stages:
        pct = 100.0 * ms / full_ms if full_ms > 0 else float("nan")
        print(f"{name:<32s} {ms:10.3f}  {pct:15.1f}%")
    print("-" * 78)

    # Useful derived quantities.
    slow_ms = stages[1][1]
    acoustic12_ms = stages[2][1]
    acoustic1_ms = stages[3][1]
    rk3_slow_ms = 3.0 * slow_ms
    rk3_acoustic_ms = 3.0 * acoustic12_ms
    print(f"{'3 x slow_tend (RK3 outer)':<32s} {rk3_slow_ms:10.3f}  "
          f"{100.0 * rk3_slow_ms / full_ms:15.1f}%")
    print(f"{'3 x acoustic_substeps x12':<32s} {rk3_acoustic_ms:10.3f}  "
          f"{100.0 * rk3_acoustic_ms / full_ms:15.1f}%")
    print(f"{'acoustic_x12 / (12 * x1)':<32s} "
          f"{acoustic12_ms / max(acoustic1_ms * 12.0, 1e-12):10.3f}x  "
          f"{'(fori_loop overhead)':>16s}")
    print("=" * 78)
    print("Note: hyperdiff_biharmonic / sponge_taper_multiply / single-substep")
    print("      rows measure ISOLATED leaf operator cost — they are already")
    print("      INCLUDED in slow_tend / acoustic_substeps x12, so the table")
    print("      is overlapping (not a partition).")
    print(f"      Halo-pad: single-rank plane path uses doubly-periodic")
    print(f"      jnp.roll inside plane_operators (no MPI sendrecv) so no")
    print(f"      separate halo stage is exposed at single-rank.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
