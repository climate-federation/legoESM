"""Profile the MPAS-ocean I6/L20 step on a local GPU.

Builds the MPAS Voronoi ocean model exactly as
``scripts/bench/bench_ocean_gpu_scaling.py`` does (subdivision_level=6,
lloyd_iterations=5, 20 z* levels, default ``MPASOceanConfig``), then
breaks per-step wall time into sub-stages using:

* ``time.perf_counter()`` + ``jax.block_until_ready`` around isolated
  closures for each stage (full segment step via ``lax.scan``, bare
  ``model._step_impl``, MPAS baroclinic tendency, implicit vertical
  mixing, barotropic substeps, EOS + pressure anomaly, tracer
  advection horizontal flux divergence, and the indirect-addressing
  primitive operators).
* ``jax.profiler.start_trace`` / ``stop_trace`` over a small window of
  full steps so the resulting TensorBoard trace can be inspected in
  ``results/scaling_gpu_profile_mpas/``.

Read-only investigation — reuses the production ``MPASOceanModel`` and
its public step. No production code is mutated; sub-stage closures
either call public ``model.tendencies(...)``/``model._step_impl(...)``
or assemble inputs from the public mesh + state pytree.

Aim is to verify the bottleneck for ``I6/L20 fp64`` GPU bench is the
Voronoi indirect-addressing gather (``cellsOnEdge``, ``edgesOnCell``,
``edgesOnEdge``) rather than something more tunable.

Invocation::

    PYTHONPATH=. .venv/bin/python scripts/profile_mpas_ocean.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

# --- Environment setup BEFORE jax import ----------------------------------
# Match scaling bench exactly (fp64 + CUDA-graphs command buffers).
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent  # scripts/bench/ -> repo root
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("JAX_ENABLE_X64", "1")

_CUDA_GRAPH_FLAG = (
    "--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,CUBLAS,CUDNN"
)


def _has_xla_token(existing: str, token: str) -> bool:
    """Whitespace-tokenized flag check (mirrors bench_ocean_gpu_scaling)."""
    for part in existing.split():
        name = part.split("=", 1)[0]
        if name == token:
            return True
    return False


def _enable_cuda_graphs():
    existing = os.environ.get("XLA_FLAGS", "")
    if _has_xla_token(existing, "--xla_gpu_enable_command_buffer"):
        return
    os.environ["XLA_FLAGS"] = f"{existing} {_CUDA_GRAPH_FLAG}".strip()


_enable_cuda_graphs()

import jax
import jax.numpy as jnp

# Production modules.
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.dynamics.ocean_model_mpas import (
    MPASOceanModel,
    MPASOceanConfig,
)
from legoesm.core.operators_voronoi import (
    divergence_cell_3d,
    gradient_edge_3d,
    kinetic_energy_cell_3d,
    cell_to_edge_avg_3d,
    tangential_velocity,
)
from legoesm.ocean.dynamics.barotropic_mpas import barotropic_substeps_mpas


# ===========================================================================
# Stage-level timing helper
# ===========================================================================

def _time_stage(
    name: str,
    fn: Callable[[], object],
    n_warmup: int = 3,
    n_timing: int = 20,
) -> tuple[str, float]:
    """Warm + JIT-compile ``fn``, then return ``(name, ms_per_call_mean)``.

    ``fn`` is a thunk returning one or more device arrays.  A device
    sync via ``jax.block_until_ready`` after every call ensures we
    measure real device wall-time rather than async-submission cost.
    """
    for _ in range(n_warmup):
        out = fn()
        jax.block_until_ready(jax.tree_util.tree_leaves(out))

    t0 = time.perf_counter()
    for _ in range(n_timing):
        out = fn()
    jax.block_until_ready(jax.tree_util.tree_leaves(out))
    t1 = time.perf_counter()
    ms = (t1 - t0) / n_timing * 1000.0
    print(f"    [{name:<32s}] {ms:9.3f} ms/call")
    return name, ms


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    # --- I6/L20 baseline (matches bench_ocean_gpu_scaling). ----------------
    subdivision_level = 6
    lloyd_iterations = 5
    n_levels = 20
    dt = 600.0
    precision = "fp64" if os.environ.get("JAX_ENABLE_X64", "0") == "1" else "fp32"

    profile_dir = _REPO / "results" / "scaling_gpu_profile_mpas"
    profile_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(
        f"profile_mpas_ocean: I{subdivision_level}/L{n_levels}, "
        f"precision={precision}, dt={dt}s"
    )
    print(f"device: {jax.devices()}  jax={jax.__version__}")
    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")
    print("=" * 78)

    # --- Build mesh + state + model ----------------------------------------
    t0 = time.perf_counter()
    mesh = create_voronoi_mesh(
        subdivision_level=subdivision_level,
        lloyd_iterations=lloyd_iterations,
    )
    z = create_ocean_z_star(n_levels=n_levels)
    cfg = MPASOceanConfig()
    model = MPASOceanModel(mesh, z, cfg)
    state0 = rest_state_mpas_ocean(mesh, z)
    build_s = time.perf_counter() - t0

    n_cells = int(mesh.nCells)
    n_edges = int(mesh.nEdges)
    n_vertices = int(mesh.nVertices)
    n_baro_sub = int(cfg.barotropic.n_barotropic_substeps)
    cell_lev = n_cells * n_levels
    edge_lev = n_edges * n_levels

    print(
        f"mesh: nCells={n_cells:,}  nEdges={n_edges:,}  "
        f"nVertices={n_vertices:,}  maxEdges={int(mesh.maxEdges)}\n"
        f"levels: {n_levels}  barotropic substeps/step: {n_baro_sub}\n"
        f"cell-lev d.o.f.: {cell_lev:,}  edge-lev d.o.f.: {edge_lev:,}\n"
        f"tracer_advection={cfg.tracer_advection}  "
        f"implicit_vertical_mixing={cfg.implicit_vertical_mixing}  "
        f"barotropic_solver={cfg.barotropic.barotropic_solver}\n"
        f"build wall: {build_s:.2f}s"
    )

    # --- Stage 1: full step via lax.scan (matches the bench timing path). --
    # ``model.step`` is JIT-decorated; wrapping in ``lax.scan`` mirrors
    # exactly the path bench_ocean_gpu_scaling._time_step uses for
    # ``time_per_step_ms``.
    input_dtypes = jax.tree.map(
        lambda x: x.dtype if hasattr(x, "dtype") else None, state0,
    )

    def _scan_step(carry, _):
        new = model.step(carry, dt)
        new = jax.tree.map(
            lambda x, d: x.astype(d)
            if d is not None and hasattr(x, "astype") else x,
            new, input_dtypes,
        )
        return new, None

    @jax.jit
    def _scan_N(state, n):
        return jax.lax.scan(_scan_step, state, None, length=n)[0]

    # ``length`` must be static for ``lax.scan``; use a fixed N=20 batch.
    N_SCAN = 20

    @jax.jit
    def _scan20(state):
        return jax.lax.scan(_scan_step, state, None, length=N_SCAN)[0]

    def _full_step_via_scan():
        return _scan20(state0)

    # --- Stage 2: bare _step_impl (avoid double-JIT inside outer scan). ----
    # ``model._step_impl`` is the un-jitted core.  Wrap once in our own jit
    # so the warmup triggers compile, then time it standalone (this
    # mirrors what the production scan body executes per iteration).
    _step_impl_jit = jax.jit(lambda s: model._step_impl(s, dt))

    def _bare_step_impl():
        return _step_impl_jit(state0)

    # --- Stage 3: baroclinic tendency only (no barotropic, no advection). --
    # This is the 3D PE tendency function — issues all the major Voronoi
    # gather ops (gradient_edge, divergence_cell, kinetic_energy, PV flux,
    # vector laplacian) on (nEdges, nlev) / (nCells, nlev).
    _tend_jit = jax.jit(lambda s: model.tendencies(s))

    def _baroclinic_tendency():
        return _tend_jit(state0)

    # --- Stage 4: barotropic substep loop alone --------------------------
    # The barotropic substep does 30 iterations of a single-level (2D)
    # forward-backward scheme.  Each iteration calls ``divergence_cell``,
    # ``gradient_edge``, ``tangential_velocity`` (and tangential needs
    # the ``edgesOnEdge`` gather — the biggest indirect-addressing
    # stencil in the code base).
    F_slow_eta = jnp.zeros_like(state0.eta.data)
    F_slow_u = jnp.zeros_like(state0.u.data[:, 0])
    dt_baro = dt / n_baro_sub

    @jax.jit
    def _baro_only(s):
        return barotropic_substeps_mpas(
            s, mesh, z, cfg, dt_baro, n_baro_sub,
            F_slow_eta=F_slow_eta, F_slow_u=F_slow_u,
        )

    def _barotropic_substeps():
        return _baro_only(state0)

    # --- Stage 5: EOS + baroclinic pressure anomaly (no PGF gather) -------
    # Replicates the ``iterate_eos_and_pressure_anomaly`` call inside
    # ``mpas_ocean_baroclinic_tendencies`` (ocean_pe_mpas.py:174).  This
    # has NO Voronoi indirect-addressing at all (works on (nCells, nlev)
    # only) — it isolates the pure-compute EOS + cumsum cost so we can
    # compare against the indirect-addressing-heavy stages.
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        iterate_eos_and_pressure_anomaly,
    )
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas

    eos_fn = make_eos_fn(cfg.eos, getattr(cfg, "eos_linear", None))
    c1_m = mesh.cellsOnEdge[0]
    c2_m = mesh.cellsOnEdge[1]
    _mask = state0.land_mask.data

    def _fill_fn(field):
        return fill_land_cells_mpas(field, _mask, c1_m, c2_m,
                                    mesh.edgesOnCell, mesh.nEdgesOnCell)

    @jax.jit
    def _eos_press(T, S):
        return iterate_eos_and_pressure_anomaly(
            T, S, _mask, _fill_fn, eos_fn, z.dz_ref, cfg.rho_0, cfg.g,
            n_iter=2,
        )

    T0 = state0.T.data
    S0 = state0.S.data

    def _eos_pressure():
        return _eos_press(T0, S0)

    # --- Stage 6: pure indirect-addressing primitives (no math, no fill) --
    # These four are the recurring indirect-gather kernels every Voronoi
    # operator routes through.  Isolating each tells us whether the
    # indirect addressing alone is the floor we cannot get below — even
    # in a pure-XLA setting.
    u0 = state0.u.data           # (nEdges, nlev)
    phi0 = T0                    # (nCells, nlev) cell-centred scalar

    _div_jit = jax.jit(lambda u: divergence_cell_3d(u, mesh))
    _grad_jit = jax.jit(lambda p: gradient_edge_3d(p, mesh))
    _ke_jit = jax.jit(lambda u: kinetic_energy_cell_3d(u, mesh))
    _c2e_jit = jax.jit(lambda p: cell_to_edge_avg_3d(p, mesh))
    _tang_jit = jax.jit(lambda u: tangential_velocity(u, mesh))

    def _op_div_3d():
        return _div_jit(u0)

    def _op_grad_3d():
        return _grad_jit(phi0)

    def _op_ke_3d():
        return _ke_jit(u0)

    def _op_c2e_3d():
        return _c2e_jit(phi0)

    def _op_tang_2d():
        return _tang_jit(u0[:, 0])

    # --- Stage 7: gather-bound vs compute-bound reference scans -----------
    # To discriminate "indirect addressing" from "pure compute" cost
    # inside the barotropic loop, run two synthetic 30-iter scans on
    # 2D edge arrays:
    #
    #   * ``_baro_compute_ref``: 30 iters of pure-elementwise FMA on
    #     (nEdges,).  No gather/scatter.  Lower bound on what 30
    #     substeps of *any* arithmetic on this shape can cost.
    #   * ``_baro_gather_ref``: 30 iters of pure ``tangential_velocity``
    #     (the biggest gather in the barotropic loop: ``edgesOnEdge``
    #     has shape (maxEdges2=10, nEdges)).  Upper bound on
    #     gather-only cost.
    u_2d = u0[:, 0]
    _n_iter_ref = n_baro_sub

    @jax.jit
    def _baro_compute_ref(u):
        def _b(c, _):
            # Six FMAs per iter — matches what the barotropic substep
            # does elementwise on the resulting fields (PGF + Coriolis +
            # drag + clipping + damping + filter accumulation).
            c = c * 0.999 + 1.0e-6
            c = c + 1.0e-7 * c * c
            c = c * 0.9999 + 1.0e-8
            c = c + 2.0e-9 * c
            c = jnp.minimum(c, 1.0e3)
            c = c * 0.99999
            return c, None
        return jax.lax.scan(_b, u, None, length=_n_iter_ref)[0]

    @jax.jit
    def _baro_gather_ref(u):
        def _b(c, _):
            return tangential_velocity(c, mesh), None
        return jax.lax.scan(_b, u, None, length=_n_iter_ref)[0]

    # ``_baro_gather_stack``: 30 iterations of the *full* indirect-
    # addressing stencil set used by one barotropic substep body, but
    # with all arithmetic shrunk to identity multiplications.  This
    # isolates the cost of {grad + div + tangential×2 + cell-to-edge
    # gather} × 30 substeps on (nEdges,) / (nCells,) arrays.
    from legoesm.core.operators_voronoi import divergence_cell, gradient_edge

    @jax.jit
    def _baro_gather_stack(eta_in, u_in):
        def _b(carry, _):
            eta_c, u_bar_c = carry
            # Full mass-flux divergence (eta update)
            div_u = divergence_cell(u_bar_c, mesh)
            eta_n = eta_c - 1e-3 * div_u
            # PGF gradient (u_bar update, first half)
            grad_eta = gradient_edge(eta_n, mesh)
            u_star = u_bar_c - 1e-3 * grad_eta
            # Two tangential_velocity calls for semi-implicit Coriolis
            v_t_old = tangential_velocity(u_bar_c, mesh)
            v_t_star = tangential_velocity(u_star, mesh)
            u_n = u_bar_c + 1e-3 * (v_t_old + v_t_star) - 1e-3 * grad_eta
            return (eta_n, u_n), None
        return jax.lax.scan(
            _b, (eta_in, u_in), None, length=_n_iter_ref,
        )[0]

    def _gather_stack_ref():
        return _baro_gather_stack(state0.eta.data, u_2d)

    def _compute_ref():
        return _baro_compute_ref(u_2d)

    def _gather_ref():
        return _baro_gather_ref(u_2d)

    # ----------------------------------------------------------------------
    # Run all stages.  Top-level closures are warmed first; the bare step
    # and tendency benefit from extra warmup since they hit the longest
    # XLA-fusion paths.
    # ----------------------------------------------------------------------
    print("\nWarming up + timing per-stage closures:")

    stages: list[tuple[str, float]] = []

    # Top-line stages first (bench-equivalent).
    s_step20 = _time_stage(
        "full_step_x20 (lax.scan)", _full_step_via_scan,
        n_warmup=2, n_timing=5,
    )
    full_step_ms = s_step20[1] / N_SCAN
    stages.append(("full_step (per-step from scan)", full_step_ms))

    stages.append(_time_stage(
        "bare _step_impl (jit)", _bare_step_impl,
        n_warmup=3, n_timing=20,
    ))

    stages.append(_time_stage(
        "baroclinic_tendency", _baroclinic_tendency,
        n_warmup=3, n_timing=30,
    ))

    stages.append(_time_stage(
        "barotropic_substeps (30x)", _barotropic_substeps,
        n_warmup=3, n_timing=30,
    ))

    stages.append(_time_stage(
        "eos + p_prime (no gather)", _eos_pressure,
        n_warmup=3, n_timing=50,
    ))

    print("\n  -- Indirect-addressing primitive kernels --")
    stages.append(_time_stage(
        "divergence_cell_3d", _op_div_3d,
        n_warmup=5, n_timing=200,
    ))
    stages.append(_time_stage(
        "gradient_edge_3d", _op_grad_3d,
        n_warmup=5, n_timing=200,
    ))
    stages.append(_time_stage(
        "kinetic_energy_cell_3d", _op_ke_3d,
        n_warmup=5, n_timing=200,
    ))
    stages.append(_time_stage(
        "cell_to_edge_avg_3d", _op_c2e_3d,
        n_warmup=5, n_timing=200,
    ))
    stages.append(_time_stage(
        "tangential_velocity (2D)", _op_tang_2d,
        n_warmup=5, n_timing=200,
    ))

    print("\n  -- Discriminator: 30-iter scan compute-only vs gather-only --")
    stages.append(_time_stage(
        "scan30 pure-FMA (no gather)", _compute_ref,
        n_warmup=3, n_timing=50,
    ))
    stages.append(_time_stage(
        "scan30 tangential (gather)", _gather_ref,
        n_warmup=3, n_timing=50,
    ))
    stages.append(_time_stage(
        "scan30 gather stack (div+grad+2tang)", _gather_stack_ref,
        n_warmup=3, n_timing=50,
    ))

    # ----------------------------------------------------------------------
    # JAX profiler trace over a small window of full steps.  The xplane.pb
    # / trace.json can be opened with ``tensorboard --logdir
    # results/scaling_gpu_profile_mpas``.
    # ----------------------------------------------------------------------
    n_trace_steps = 5
    print(
        f"\nRecording JAX profiler trace ({n_trace_steps} full steps) to:\n"
        f"  {profile_dir}"
    )
    jax.block_until_ready(jax.tree_util.tree_leaves(state0))
    jax.profiler.start_trace(str(profile_dir))
    try:
        s = state0
        for _ in range(n_trace_steps):
            s = model.step(s, dt)
        jax.block_until_ready(jax.tree_util.tree_leaves(s))
    finally:
        jax.profiler.stop_trace()
    print("  trace written.")

    # ----------------------------------------------------------------------
    # Pretty-print breakdown table + diagnosis hints.
    # ----------------------------------------------------------------------
    full_ms = full_step_ms  # reference 100%
    print("\n" + "=" * 78)
    print(f"{'stage':<36s} {'ms/call':>11s}  {'% of full step':>16s}")
    print("-" * 78)

    for name, ms in stages:
        pct = 100.0 * ms / full_ms if full_ms > 0 else float("nan")
        print(f"{name:<36s} {ms:11.3f}  {pct:15.1f}%")
    print("=" * 78)

    # Quick passes/cell-lev for the dominant operators.  These ratios are
    # computed against the LAX-scan-per-step number so they are
    # backend-agnostic.
    print(
        "\nPasses (cell·lev / ms / pass-equivalent):\n"
        f"  divergence_cell_3d : {cell_lev / (stages[5][1] * 1e3):.3f}  Mcell·lev/s\n"
        f"  gradient_edge_3d   : {edge_lev / (stages[6][1] * 1e3):.3f}  Medge·lev/s\n"
        f"  kinetic_energy_3d  : {cell_lev / (stages[7][1] * 1e3):.3f}  Mcell·lev/s\n"
        f"  cell_to_edge_3d    : {edge_lev / (stages[8][1] * 1e3):.3f}  Medge·lev/s\n"
        f"  tangential_velocity (2D): {n_edges / (stages[9][1] * 1e3):.3f}  Medge/s"
    )

    # Inferred indirect-addressing budget vs full step.
    indirect_kernel_ms = sum(ms for _, ms in stages[5:10])
    indirect_per_step_estimate = (
        # divergence_cell_3d + gradient_edge_3d appear ~3-4× each per step
        # (PE tendency hits each of {div, grad, KE, c2e} once for the
        # tendency, the barotropic substep iterates div+grad+tangential
        # 30× on 2D, the tracer advection issues another div_3d).
        # Conservative lower bound = sum of bare kernel costs ×
        # average issue count.
        4 * stages[5][1]    # 4 × div_3d issues per step (tend, tracer-h, tracer-flux, etc.)
        + 4 * stages[6][1]  # 4 × grad_3d issues
        + 1 * stages[7][1]  # 1 × KE_3d
        + 2 * stages[8][1]  # 2 × c2e_3d
        + n_baro_sub * stages[9][1] * 2  # 2 × tang per substep × 30 substeps (semi-implicit)
    )

    print(
        "\nIndirect-addressing kernel budget (best-effort estimate):\n"
        f"  raw kernel sum (bare ops, 1× each)      : {indirect_kernel_ms:8.3f} ms\n"
        f"  scaled-by-issue-count lower-bound       : {indirect_per_step_estimate:8.3f} ms\n"
        f"  full step ms (reference)                : {full_ms:8.3f} ms\n"
        f"  EOS + p_prime (no-gather, pure compute) : {stages[4][1]:8.3f} ms\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
