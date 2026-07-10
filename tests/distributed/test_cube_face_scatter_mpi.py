"""MPI replicated-vs-scattered equivalence for cubed-sphere face-scatter.

Run under mpirun with a face-only rank count (1, 2, 3, or 6)::

    mpirun -np 3 .venv/bin/python -m pytest tests/distributed/test_cube_face_scatter_mpi.py

The gate: a multi-step cubed-sphere PE rollout (dynamics + hyperdiffusion +
divergence damping + anchored mass fixer) on the rank's OWNED faces must match
the single-process reference (full 6 faces, local halo backend) to ~machine
precision, and a gradient through the rollout must match the reference owned-
face gradient (the cross-rank halo is custom_vjp-differentiable, the mass fixer
reduces via global_sum_mpi).  Bit-faithful equivalence vs the already-validated
replicated path means face-scatter introduces no new halo/seam artifact.

The reference is computed FIRST, while the halo backend is still "local";
``initialize_distributed`` then switches the process to the MPI backend.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.dynamics.flux_form_tracer_transport import (
    flux_form_tracer_step,
)
from legoesm.core.field import Field
from legoesm.core.fv3_sw_core import d2a2c_vect
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import create_sigma_coordinate

from tests.test_cases.baroclinic_wave import baroclinic_wave_init

_N = 12
_NLEV = 5
_DT = 120.0
_K = 4
_FACE_ONLY_RANKS = (1, 2, 3, 6)


def _require_face_only():
    size = MPI.COMM_WORLD.Get_size()
    if size not in _FACE_ONLY_RANKS:
        pytest.skip(
            f"cube face-scatter is face-only; run with -np in {_FACE_ONLY_RANKS} "
            f"(got {size})"
        )


def _build(grid, sigma, *, moisture_flux_form=False):
    # Modest del2-style hyperdiff so the diffusion/halo-amplifier paths are
    # exercised; anchored mass fixer ON; per-stage zero-mean OFF (unsupported
    # under scatter — see make_rank_local_cube_model).
    hd = float(0.05 * (grid.dx.min()) ** 2 / _DT)
    cfg = CDGridPrimitiveEquationConfig(
        hyperdiff_coeff=hd,
        hyperdiff_ps_coeff=hd,
        fix_mass=True,
        use_conservation_fixer=True,
        anchor_mass_to_initial=True,
        zero_mean_ps_tendency=False,
        moisture_flux_form=moisture_flux_form,
    )
    cdgrid = create_cubed_sphere_cdgrid(grid)
    state0 = hydrostatic_to_fv3(
        baroclinic_wave_init(grid, sigma, perturbed=True), cdgrid
    )
    if moisture_flux_form:
        # A positive moisture blob on face 0 so transport carries q ACROSS a face
        # (and, at np=2/3, across a face-SCATTER seam) — the exact place the
        # 4D-halo flux-form transport + wind reconstruction must stay
        # replicated-vs-scattered identical (#811).
        nlev = state0.T.data.shape[-1]
        q = jnp.full((6, _N, _N, nlev), 1e-4).at[
            0, _N // 3:2 * _N // 3, _N // 3:2 * _N // 3, :].set(2e-2)
        state0 = state0._replace(tracers={"q_v": Field(q, name="q_v")})
    return cfg, state0


def _column_water_partial(model, sigma, cfg, state):
    """∫ area·δp·q over this rank's held faces (NO allreduce).  Full cube on the
    local-backend reference; owned-face partial on the scattered path."""
    area = model.cdgrid.base.area
    dp = sigma.layer_thickness_dp(
        jnp.clip(state.p_s.data, cfg.p_floor, cfg.p_ceil))
    return jnp.sum(area[..., None] * dp * state.tracers["q_v"].data)


def test_scattered_rollout_matches_replicated_reference():
    _require_face_only()
    rank = MPI.COMM_WORLD.Get_rank()
    nproc = MPI.COMM_WORLD.Get_size()

    # Fully reset distributed state BEFORE building anything.  The session-scoped
    # conftest fixture ``_init_mpi_layout`` pre-arms
    # ``initialize_distributed(global_n=max(np,2))`` (a SMALL grid), leaving the
    # "mpi" halo backend + a stale topology armed process-wide.  If we build the
    # IC under that backend, ``hydrostatic_to_fv3`` halo-pads through the stale
    # (wrong-grid) topology and produces a per-rank-CORRUPTED initial state — the
    # reference then diverges across ranks.  reset_distributed_topology() clears
    # topology/layout + sets the "local" backend, so the IC + reference are clean
    # and our initialize_distributed(global_n=_N) below is a FRESH init.
    from legoesm.parallel.distributed import reset_distributed_topology

    reset_distributed_topology()

    grid = create_cubed_sphere(_N)
    sigma = create_sigma_coordinate(_NLEV)
    cfg, state0 = _build(grid, sigma)

    # --- Reference: full 6 faces, local halo backend (validated physics) ---
    model_ref = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    s = state0
    for _ in range(_K):
        s = model_ref.step(s, _DT)
    ref = {f: getattr(s, f).data for f in ("u_d", "v_d", "T", "p_s")}

    # --- Scattered: owned faces, MPI backend, sliced metrics ---
    from legoesm.parallel.cube_face_scatter import make_rank_local_cube_model
    from legoesm.parallel.distributed import (
        get_active_topology,
        initialize_distributed,
    )
    from legoesm.parallel.layout import make_layout, scatter_pytree

    initialize_distributed(
        return_topology=True, global_n=_N, grid_type="cubed_sphere"
    )
    owned = list(get_active_topology().local_face_ids)
    model_sca = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    make_rank_local_cube_model(model_sca, owned)
    ssc = scatter_pytree(state0, make_layout(rank, nproc, _N))
    assert ssc.u_d.data.shape[0] == len(owned)
    assert model_sca.grid.halo_interp_offsets.shape[0] == 6  # kept full
    for _ in range(_K):
        ssc = model_sca.step(ssc, _DT)

    idx = jnp.asarray(owned)
    for f in ("u_d", "v_d", "T", "p_s"):
        a = getattr(ssc, f).data
        b = ref[f][idx]
        rel = float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))
        assert rel < 1e-10, f"{f} rel_err={rel:.3e} on rank {rank} (owns {owned})"


def test_scattered_rollout_gradient_matches_reference():
    _require_face_only()
    rank = MPI.COMM_WORLD.Get_rank()
    nproc = MPI.COMM_WORLD.Get_size()

    # Reset BEFORE building the IC (see the forward test: the conftest session
    # fixture leaves a stale "mpi" backend that would corrupt the halo-padded IC).
    from legoesm.parallel.distributed import reset_distributed_topology

    reset_distributed_topology()

    grid = create_cubed_sphere(_N)
    sigma = create_sigma_coordinate(4)
    cfg, state0 = _build(grid, sigma)

    from legoesm.parallel.reductions import global_sum_mpi

    def make_loss(model, s0, distributed):
        def loss(t_data):
            s = s0._replace(T=s0.T.replace(data=t_data))
            for _ in range(2):
                s = model.step(s, _DT)
            local = jnp.sum(s.T.data ** 2)
            return global_sum_mpi(local) if distributed else local

        return loss

    # Reference gradient (full, local backend) — IC + reference now clean.
    model_ref = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    g_ref = jax.grad(make_loss(model_ref, state0, False))(state0.T.data)

    # Scattered gradient (MPI).
    from legoesm.parallel.cube_face_scatter import make_rank_local_cube_model
    from legoesm.parallel.distributed import (
        get_active_topology,
        initialize_distributed,
    )
    from legoesm.parallel.layout import make_layout, scatter_pytree

    initialize_distributed(
        return_topology=True, global_n=_N, grid_type="cubed_sphere"
    )
    owned = list(get_active_topology().local_face_ids)
    model_sca = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    make_rank_local_cube_model(model_sca, owned)
    ssc = scatter_pytree(state0, make_layout(rank, nproc, _N))
    g_sca = jax.grad(make_loss(model_sca, ssc, True))(ssc.T.data)

    assert bool(jnp.all(jnp.isfinite(g_sca)))
    ref_owned = g_ref[jnp.asarray(owned)]
    rel = float(jnp.max(jnp.abs(g_sca - ref_owned)) / (jnp.max(jnp.abs(ref_owned)) + 1e-30))
    assert rel < 1e-8, f"grad rel_err={rel:.3e} on rank {rank} (owns {owned})"


def test_scattered_flux_form_moisture_matches_replicated():
    """#811: the flux-form moisture substep (``moisture_flux_form=True``) under MPI
    face-scatter matches the full-6-face reference to ~machine precision, and its
    GLOBAL column water (allreduce over owned-face partials) equals the
    reference's.  This certifies the 4D-halo transport (``transport_step_4d``) +
    4D wind reconstruction (``d2a2c_vect_4d``) + allreduce-aware rescale are all
    replicated-vs-scattered identical — the case #791 fail-closed and #811
    unblocks."""
    _require_face_only()
    rank = MPI.COMM_WORLD.Get_rank()
    nproc = MPI.COMM_WORLD.Get_size()

    from legoesm.parallel.distributed import reset_distributed_topology

    reset_distributed_topology()

    grid = create_cubed_sphere(_N)
    sigma = create_sigma_coordinate(_NLEV)
    cfg, state0 = _build(grid, sigma, moisture_flux_form=True)

    # --- Reference: full 6 faces, local halo backend. ---
    model_ref = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    s = state0
    for _ in range(_K):
        s = model_ref.step(s, _DT)
    ref_q = s.tracers["q_v"].data
    cw_ref = float(_column_water_partial(model_ref, sigma, cfg, s))  # full cube

    # --- Scattered: owned faces, MPI backend. ---
    from legoesm.parallel.cube_face_scatter import make_rank_local_cube_model
    from legoesm.parallel.distributed import (
        get_active_topology,
        initialize_distributed,
    )
    from legoesm.parallel.layout import make_layout, scatter_pytree
    from legoesm.parallel.reductions import global_sum_mpi

    initialize_distributed(
        return_topology=True, global_n=_N, grid_type="cubed_sphere"
    )
    owned = list(get_active_topology().local_face_ids)
    model_sca = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    make_rank_local_cube_model(model_sca, owned)
    ssc = scatter_pytree(state0, make_layout(rank, nproc, _N))
    assert ssc.tracers["q_v"].data.shape[0] == len(owned)
    for _ in range(_K):
        ssc = model_sca.step(ssc, _DT)

    idx = jnp.asarray(owned)
    a = ssc.tracers["q_v"].data
    b = ref_q[idx]
    rel = float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))
    assert rel < 1e-10, f"q_v rel_err={rel:.3e} on rank {rank} (owns {owned})"

    cw_sca = float(global_sum_mpi(_column_water_partial(model_sca, sigma, cfg, ssc)))
    rel_cw = abs(cw_sca - cw_ref) / (abs(cw_ref) + 1e-30)
    assert rel_cw < 1e-10, f"global column water {cw_sca:.6e} vs ref {cw_ref:.6e}"


def test_scattered_flux_form_moisture_gradient_matches_reference():
    """#811: a gradient through the 4D-halo flux-form moisture substep is finite
    and matches the reference owned-face gradient — certifying the 4D transport
    + wind halos + allreduce reductions are all reverse-mode differentiable and
    equivalent across face shards (end-to-end ``jax.grad``)."""
    _require_face_only()
    rank = MPI.COMM_WORLD.Get_rank()
    nproc = MPI.COMM_WORLD.Get_size()

    from legoesm.parallel.distributed import reset_distributed_topology

    reset_distributed_topology()

    grid = create_cubed_sphere(_N)
    sigma = create_sigma_coordinate(_NLEV)
    cfg, state0 = _build(grid, sigma, moisture_flux_form=True)

    from legoesm.parallel.reductions import global_sum_mpi

    def make_loss(model, s0, distributed):
        def loss(q_data):
            s = s0._replace(
                tracers={"q_v": s0.tracers["q_v"].replace(data=q_data)})
            for _ in range(2):
                s = model.step(s, _DT)
            local = jnp.sum(s.tracers["q_v"].data ** 2)
            return global_sum_mpi(local) if distributed else local

        return loss

    model_ref = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    g_ref = jax.grad(make_loss(model_ref, state0, False))(
        state0.tracers["q_v"].data)

    from legoesm.parallel.cube_face_scatter import make_rank_local_cube_model
    from legoesm.parallel.distributed import (
        get_active_topology,
        initialize_distributed,
    )
    from legoesm.parallel.layout import make_layout, scatter_pytree

    initialize_distributed(
        return_topology=True, global_n=_N, grid_type="cubed_sphere"
    )
    owned = list(get_active_topology().local_face_ids)
    model_sca = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    make_rank_local_cube_model(model_sca, owned)
    ssc = scatter_pytree(state0, make_layout(rank, nproc, _N))
    g_sca = jax.grad(make_loss(model_sca, ssc, True))(ssc.tracers["q_v"].data)

    assert bool(jnp.all(jnp.isfinite(g_sca)))
    ref_owned = g_ref[jnp.asarray(owned)]
    # Normalise by the GLOBAL gradient magnitude — ``g_ref`` is the full
    # replicated 6-face gradient held on every rank, so ``max|g_ref|`` is the
    # true gradient scale.  The rank-LOCAL ``max|g_ref[owned]|`` inflates the
    # relative error to meaninglessness on a rank whose owned faces are all far
    # from the moisture blob (tiny local gradient / tiny denominator).
    gscale = float(jnp.max(jnp.abs(g_ref)))
    rel = float(jnp.max(jnp.abs(g_sca - ref_owned)) / (gscale + 1e-30))
    # VJP EXACTNESS is certified separately by
    # ``test_scattered_flux_form_isolated_grad_matches_reference`` (a SINGLE
    # substep, machine precision ~1e-13).  This end-to-end gate runs TWO FULL
    # steps (RK3 + hyperdiff + p_s/tracer mass fixers + the moisture substep);
    # the scattered vs replicated allreduce + halo summation ORDERS differ at fp
    # round-off (~1e-10 forward — see the matches_replicated gate) and reverse-
    # mode AD over 2 steps + a sharp 200x moisture blob amplifies that to ~1e-6.
    # This is the algebraic-equality round-off floor, NOT a VJP defect: a genuine
    # cross-rank reduction bug shows O(1e-2 .. 1) here (the pre-#811-fix values)
    # and is still caught with wide margin.
    assert rel < 1e-5, f"flux-form grad rel_err={rel:.3e} on rank {rank}"


def test_scattered_flux_form_isolated_grad_matches_reference():
    """#811 VJP-EXACTNESS certificate: a gradient through ONE ``flux_form_tracer_
    step`` (the 4D-halo transport + the two allreduce-aware, broadcast-VJP mass
    rescales) is BIT-FAITHFUL scattered-vs-replicated to MACHINE PRECISION.

    Isolating the substep (winds + δp are CONSTANTS — no RK3, no per-step mass
    fixer, no 2-step compounding) removes the fp-round-off amplification that
    lifts the end-to-end gate (``..._gradient_matches_reference``) to ~1e-6, so
    this gate certifies the substep's OWN VJPs are exact:

    * the 4D ``pad_halo_mpi_4d`` cross-face ``sendrecv`` (``_sendrecv_vjp``);
    * ``_conserving_rescale``'s global mass reduction — which feeds the SHARED
      ``scale`` that rescales every face, so its reduction MUST use the broadcast
      VJP (``global_face_sum_if_scattered(..., differentiable_broadcast=True)``);
      the default identity VJP left a UNIFORM ~1e-3 cotangent leak (rel 1.1 on
      faces far from the transported blob) — the regression this locks.
    """
    _require_face_only()
    rank = MPI.COMM_WORLD.Get_rank()

    from legoesm.parallel.distributed import reset_distributed_topology
    reset_distributed_topology()

    n, nlev, dt = _N, _NLEV, 600.0
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    # Divergent contravariant winds (tiled over levels) + non-uniform δp + a
    # positive moisture blob on face 0 — all on the full 6-face cube.
    jx = jnp.arange(n + 1) / n
    u_d = 25.0 * jnp.sin(2 * jnp.pi * jx)[None, None, :] * jnp.ones((6, n, n + 1))
    v_d = 18.0 * jnp.cos(2 * jnp.pi * jx)[None, :, None] * jnp.ones((6, n + 1, n))
    _ua, _va, _uc, _vc, ut2d, vt2d = d2a2c_vect(
        u_d.astype(jnp.float64), v_d.astype(jnp.float64), cdgrid)
    ut = jnp.broadcast_to(ut2d[..., None], (*ut2d.shape, nlev))
    vt = jnp.broadcast_to(vt2d[..., None], (*vt2d.shape, nlev))
    base = 900.0 + 50.0 * jnp.cos(cdgrid.base.lat)
    delp = jnp.stack([base * (1.0 + 0.1 * k) for k in range(nlev)],
                     axis=-1).astype(jnp.float64)
    q = jnp.full((6, n, n, nlev, 1), 1e-4).at[
        0, n // 3:2 * n // 3, n // 3:2 * n // 3, :, 0].set(2e-2)

    from legoesm.parallel.reductions import global_sum_mpi

    def make_loss(cdg, ut_, vt_, delp_, distributed):
        def loss(q_data):
            q_new, _ = flux_form_tracer_step(q_data, delp_, ut_, vt_, dt, cdg)
            local = jnp.sum(q_new ** 2)
            return global_sum_mpi(local) if distributed else local
        return loss

    # Reference (full 6 faces, local backend) — BEFORE initialize_distributed.
    g_ref = jax.grad(make_loss(cdgrid, ut, vt, delp, False))(q)

    from legoesm.parallel.cube_face_scatter import slice_cubed_sphere_cdgrid
    from legoesm.parallel.distributed import (
        get_active_topology, initialize_distributed,
    )
    initialize_distributed(return_topology=True, global_n=_N,
                           grid_type="cubed_sphere")
    oi = jnp.asarray(list(get_active_topology().local_face_ids))
    cdg_l = slice_cubed_sphere_cdgrid(cdgrid, list(get_active_topology().local_face_ids))
    g_sca = jax.grad(make_loss(cdg_l, ut[oi], vt[oi], delp[oi], True))(q[oi])

    assert bool(jnp.all(jnp.isfinite(g_sca)))
    ref_owned = g_ref[oi]
    rel = float(jnp.max(jnp.abs(g_sca - ref_owned))
                / (jnp.max(jnp.abs(g_ref)) + 1e-30))
    assert rel < 1e-11, (
        f"isolated flux-form substep grad NOT machine-precision: "
        f"rel_err={rel:.3e} on rank {rank}")
