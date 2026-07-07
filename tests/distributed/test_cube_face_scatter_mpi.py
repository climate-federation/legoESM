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
from legoesm.core.field import Field
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
    rel = float(jnp.max(jnp.abs(g_sca - ref_owned)) / (jnp.max(jnp.abs(ref_owned)) + 1e-30))
    assert rel < 1e-8, f"flux-form grad rel_err={rel:.3e} on rank {rank}"
