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

from legoesm.atmosphere.dynamics.shared.flux_form_tracer_transport import (
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


def _column_water_partial(model, sigma, cfg, state):
    """∫ area·δp·q over this rank's held faces (NO allreduce).  Full cube on the
    local-backend reference; owned-face partial on the scattered path."""
    area = model.cdgrid.base.area
    dp = sigma.layer_thickness_dp(
        jnp.clip(state.p_s.data, cfg.p_floor, cfg.p_ceil))
    return jnp.sum(area[..., None] * dp * state.tracers["q_v"].data)


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
            return (global_sum_mpi(local, final_loss=True) if distributed
                    else local)
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
