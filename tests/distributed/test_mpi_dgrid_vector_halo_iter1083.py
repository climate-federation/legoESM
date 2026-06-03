"""FV3_3D iter-1083: MPI-aware DGRID vector halo bit-for-bit test."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

jax.config.update("jax_enable_x64", True)


def test_owned_face_halos_match_single_device_reference():
    """MPI dgrid halo per-owned-face matches single-device reference."""
    from legoesm.grids.dgrid_halo import (
        pad_halo_dgrid_vector_4d, pad_halo_dgrid_vector_4d_mpi,
    )
    from legoesm.parallel.comm import build_comm_topology

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if size > 6 or 6 % size != 0:
        pytest.skip("face-only requires np ∈ {1,2,3,6}")

    topology = build_comm_topology(rank, size)
    n, nlev = 4, 2
    key = jax.random.PRNGKey(2026 + 1083)
    ku, kv = jax.random.split(key)
    u_global = jax.random.normal(ku, (6, n, n + 1, nlev), dtype=jnp.float64)
    v_global = jax.random.normal(kv, (6, n + 1, n, nlev), dtype=jnp.float64)

    u_ref, v_ref = pad_halo_dgrid_vector_4d(u_global, v_global)

    idx = jnp.asarray(list(topology.local_face_ids), dtype=jnp.int32)
    u_local = u_global[idx]
    v_local = v_global[idx]

    u_mpi, v_mpi = pad_halo_dgrid_vector_4d_mpi(u_local, v_local, topology)

    for f_local, f_global in enumerate(topology.local_face_ids):
        np.testing.assert_allclose(
            np.asarray(u_mpi[f_local]), np.asarray(u_ref[f_global]),
            atol=1e-12,
            err_msg=f"rank {rank} u face {f_global}",
        )
        np.testing.assert_allclose(
            np.asarray(v_mpi[f_local]), np.asarray(v_ref[f_global]),
            atol=1e-12,
            err_msg=f"rank {rank} v face {f_global}",
        )


def test_grad_flows_through_mpi_dgrid_halo():
    """FV3_3D iter-1090: jax.grad must flow through the MPI dgrid
    halo.  Raw mpi4jax.sendrecv chokes on the symbolic Zero
    cotangent; the iter-1090 fix routes through ``_sendrecv_vjp``
    (the same custom_vjp wrapper the iter-1040+ MPI halo uses)."""
    from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d_replicated_mpi
    from legoesm.parallel.comm import build_comm_topology

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if size > 6 or 6 % size != 0:
        pytest.skip("face-only requires np ∈ {1,2,3,6}")

    topology = build_comm_topology(rank, size)
    n = 4
    u = jax.random.normal(jax.random.PRNGKey(20260101), (6, n, n + 1, 2),
                           dtype=jnp.float64)
    v = jax.random.normal(jax.random.PRNGKey(20260102), (6, n + 1, n, 2),
                           dtype=jnp.float64)

    def loss(u_in, v_in):
        up, vp = pad_halo_dgrid_vector_4d_replicated_mpi(u_in, v_in, topology)
        return jnp.sum(up ** 2) + jnp.sum(vp ** 2)

    g_u, g_v = jax.grad(loss, argnums=(0, 1))(u, v)
    assert g_u.shape == u.shape
    assert g_v.shape == v.shape
    assert bool(jnp.isfinite(g_u).all()), f"rank {rank} g_u has non-finite"
    assert bool(jnp.isfinite(g_v).all()), f"rank {rank} g_v has non-finite"
    assert not bool(jnp.allclose(g_u, 0.0)), f"rank {rank} g_u is all zeros"
    assert not bool(jnp.allclose(g_v, 0.0)), f"rank {rank} g_v is all zeros"


def test_jit_grad_composition():
    """FV3_3D iter-1091 (codex iter-1090 WARN-5): jax.jit(jax.grad(...))
    is the canonical training pipeline pattern.  iter-1090 verified
    plain jax.grad works; this test verifies the composition with JIT
    survives — both the forward primitive and the custom_vjp's bwd
    sendrecv must JIT-trace cleanly.
    """
    from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d_replicated_mpi
    from legoesm.parallel.comm import build_comm_topology

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if size > 6 or 6 % size != 0:
        pytest.skip("face-only requires np ∈ {1,2,3,6}")
    topology = build_comm_topology(rank, size)

    n = 4
    u = jax.random.normal(jax.random.PRNGKey(1091), (6, n, n + 1, 2),
                           dtype=jnp.float64)
    v = jax.random.normal(jax.random.PRNGKey(1092), (6, n + 1, n, 2),
                           dtype=jnp.float64)

    def loss(u_in, v_in):
        up, vp = pad_halo_dgrid_vector_4d_replicated_mpi(u_in, v_in, topology)
        return jnp.sum(up ** 2) + jnp.sum(vp ** 2)

    jit_grad = jax.jit(jax.grad(loss, argnums=(0, 1)))
    g_u, g_v = jit_grad(u, v)
    assert g_u.shape == u.shape
    assert bool(jnp.isfinite(g_u).all())
    assert bool(jnp.isfinite(g_v).all())


def test_grad_allreduce_recovers_single_device_reference():
    """FV3_3D iter-1091 (codex iter-1090 WARN-6): strict gradient
    correctness contract.  Per-rank MPI gradient, masked to owned
    faces, then allreduced (SUM), must equal the single-device
    reference gradient bit-for-bit.

    Mirrors the iter-1062 allreduce-recovery pattern from
    tests/distributed/test_mpi_differentiability.py.  A miswired
    DGRID cotangent that produces nonzero finite garbage on each
    rank but doesn't allreduce to the reference would fail this
    test while passing iter-1090's finite-nonzero check.
    """
    from legoesm.grids.dgrid_halo import (
        pad_halo_dgrid_vector_4d,
        pad_halo_dgrid_vector_4d_replicated_mpi,
    )
    from legoesm.parallel.comm import build_comm_topology
    from legoesm.parallel.reductions import global_sum_mpi

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if size > 6 or 6 % size != 0:
        pytest.skip("face-only requires np ∈ {1,2,3,6}")
    topology = build_comm_topology(rank, size)

    n = 4
    u = jax.random.normal(jax.random.PRNGKey(1091), (6, n, n + 1, 2),
                           dtype=jnp.float64)
    v = jax.random.normal(jax.random.PRNGKey(1092), (6, n + 1, n, 2),
                           dtype=jnp.float64)

    # Single-device reference gradient.
    def loss_ref(u_in, v_in):
        up, vp = pad_halo_dgrid_vector_4d(u_in, v_in)
        return jnp.sum(up ** 2) + jnp.sum(vp ** 2)

    g_u_ref, g_v_ref = jax.grad(loss_ref, argnums=(0, 1))(u, v)

    # Per-rank MPI gradient.
    def loss_mpi(u_in, v_in):
        up, vp = pad_halo_dgrid_vector_4d_replicated_mpi(u_in, v_in, topology)
        return jnp.sum(up ** 2) + jnp.sum(vp ** 2)

    g_u_mpi, g_v_mpi = jax.grad(loss_mpi, argnums=(0, 1))(u, v)

    # Mask to owned faces only.
    owned_mask_u = jnp.zeros((6, 1, 1, 1), dtype=g_u_mpi.dtype)
    owned_mask_v = jnp.zeros((6, 1, 1, 1), dtype=g_v_mpi.dtype)
    for f in topology.local_face_ids:
        owned_mask_u = owned_mask_u.at[f].set(1.0)
        owned_mask_v = owned_mask_v.at[f].set(1.0)

    # Allreduce to assemble full-grid gradient.
    g_u_full = global_sum_mpi(g_u_mpi * owned_mask_u)
    g_v_full = global_sum_mpi(g_v_mpi * owned_mask_v)

    np.testing.assert_allclose(
        np.asarray(g_u_full), np.asarray(g_u_ref),
        atol=1e-12,
        err_msg=(
            f"rank {rank} allreduced MPI g_u does not match single-"
            f"device reference.  Likely cause: miswired custom_vjp "
            f"in _sendrecv_vjp through DGRID halo."
        ),
    )
    np.testing.assert_allclose(
        np.asarray(g_v_full), np.asarray(g_v_ref),
        atol=1e-12,
        err_msg=(
            f"rank {rank} allreduced MPI g_v does not match single-"
            f"device reference."
        ),
    )
