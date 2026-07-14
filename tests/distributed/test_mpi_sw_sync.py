"""MPI fidelity for ``CDGridShallowWaterModel._sync_dgrid_boundary``.

FV3_3D iter-1052 ported the SW corner-staggered edge + vertex sync
to MPI (iter-1050 audit found this as an open gap).  The MPI variant
uses batched-per-peer ``sendrecv`` for edge sync and
``allreduce(SUM)`` for the 8 cube-vertex broadcasts.

Run with::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 mpirun -np 2 --timeout 120 \\
        python -m pytest tests/distributed/test_mpi_sw_sync.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.grids.halo import get_halo_backend, set_halo_backend
from legoesm.parallel.distributed import initialize_distributed


@pytest.fixture(autouse=True)
def reset_halo_backend():
    yield
    set_halo_backend("local")


def _build_sw_model_and_state(n: int = 8):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        CDGridShallowWaterModel,
        CDGridShallowWaterState,
    )

    grid = create_cubed_sphere(n)
    config = CDGridShallowWaterConfig()
    ref_model = CDGridShallowWaterModel(grid, config)
    dist_model = CDGridShallowWaterModel(grid, config)

    # Deterministic init: h ~ 1000 m + small u_d perturbation
    key = jax.random.PRNGKey(20260528)
    h_init = jnp.full((6, n, n), 1000.0) + 10.0 * jax.random.normal(
        key, (6, n, n), dtype=jnp.float64,
    )
    u_d_init = jax.random.normal(
        jax.random.fold_in(key, 1), (6, n + 1, n + 1),
        dtype=jnp.float64,
    )
    v_d_init = jax.random.normal(
        jax.random.fold_in(key, 2), (6, n + 1, n + 1),
        dtype=jnp.float64,
    )
    state = CDGridShallowWaterState(
        h=h_init, u_d=u_d_init, v_d=v_d_init,
        h_s=jnp.zeros((6, n, n)),
    )
    return ref_model, dist_model, state


def _assert_mpi_matches_local(rank, size, ref_synced, dist_synced):
    """Owned-face bit-for-bit check helper."""
    from legoesm.parallel.comm import build_comm_topology
    topology = build_comm_topology(rank, size)

    assert ref_synced.u_d.dtype == jnp.float64
    assert dist_synced.u_d.dtype == jnp.float64

    for fname in ("u_d", "v_d"):
        ref_arr = np.asarray(getattr(ref_synced, fname))
        dist_arr = np.asarray(getattr(dist_synced, fname))
        for f in topology.local_face_ids:
            np.testing.assert_allclose(
                dist_arr[f], ref_arr[f],
                atol=1e-12, rtol=1e-12,
                err_msg=(
                    f"rank{rank} SW sync MPI ≠ local on owned face "
                    f"{f}, field '{fname}'"
                ),
            )


class TestSWSyncDgridBoundaryMPI:
    """MPI vs local fidelity for ``_sync_dgrid_boundary``."""

    def test_sync_only_owned_faces_match(self):
        """Direct ``_sync_dgrid_boundary`` MPI matches local on owned faces.

        Skips the full SW step (which has its own MPI fidelity
        concerns) and exercises ONLY the sync method.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("face-only mode only (1/2/3/6 ranks)")

        set_halo_backend("local")
        ref_model, dist_model, state = _build_sw_model_and_state()

        ref_synced = ref_model._sync_dgrid_boundary(state)
        jax.block_until_ready(ref_synced.u_d)

        initialize_distributed(global_n=state.h.shape[1])
        assert get_halo_backend() == "mpi"
        dist_synced = dist_model._sync_dgrid_boundary(state)
        jax.block_until_ready(dist_synced.u_d)

        _assert_mpi_matches_local(rank, size, ref_synced, dist_synced)

    def test_sync_per_edge_distinct_values(self):
        """iter-1053 (codex F-14 follow-up): every shared edge swap is
        exercised with DISTINCT values per (face, edge).

        The baseline test uses random init, but if any cube-edge
        sendrecv silently picks up the wrong neighbour strip (e.g.,
        ``is_reversed`` flag drop, swapped face indices), randomness
        could still pass.  This test fills each face's u_d, v_d with
        ``(face_index + 1) * 1000`` so the post-sync owned-face
        boundary cells from a non-owner edge MUST equal the owner
        face's marker value.  Any mis-pairing produces a wrong
        face-index marker on the boundary.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("face-only mode only (1/2/3/6 ranks)")

        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterConfig,
            CDGridShallowWaterModel,
            CDGridShallowWaterState,
        )

        n = 8
        grid = create_cubed_sphere(n)
        config = CDGridShallowWaterConfig()
        ref_model = CDGridShallowWaterModel(grid, config)
        dist_model = CDGridShallowWaterModel(grid, config)

        # Face-distinct marker fields.
        face_markers_u = jnp.arange(6).astype(jnp.float64)[:, None, None]
        face_markers_v = jnp.arange(6).astype(jnp.float64)[:, None, None] + 100.0
        # Add small spatial pattern so reversal mis-handling shows up
        # as more than just a uniform copy.
        u_d_init = (face_markers_u + 1) * 1000.0 + jnp.arange(
            (n + 1) * (n + 1), dtype=jnp.float64
        ).reshape(n + 1, n + 1)[None, :, :]
        v_d_init = (face_markers_v + 1) * 1000.0 - jnp.arange(
            (n + 1) * (n + 1), dtype=jnp.float64
        ).reshape(n + 1, n + 1)[None, :, :]
        state = CDGridShallowWaterState(
            h=jnp.full((6, n, n), 1000.0),
            u_d=u_d_init, v_d=v_d_init,
            h_s=jnp.zeros((6, n, n)),
        )

        set_halo_backend("local")
        ref_synced = ref_model._sync_dgrid_boundary(state)
        jax.block_until_ready(ref_synced.u_d)

        initialize_distributed(global_n=n)
        dist_synced = dist_model._sync_dgrid_boundary(state)
        jax.block_until_ready(dist_synced.u_d)

        _assert_mpi_matches_local(rank, size, ref_synced, dist_synced)

