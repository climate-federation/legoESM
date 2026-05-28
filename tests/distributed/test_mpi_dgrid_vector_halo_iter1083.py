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
