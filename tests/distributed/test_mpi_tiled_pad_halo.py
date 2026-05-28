"""Tiled (sub-face) MPI halo exchange coverage (codex iter-1054 WARN #7).

The iter-1054 rank-sort deadlock fix touched 4 sites in
``halo_exchange.py``, including the 2 tiled-mode helpers
(``_pad_halo_mpi_tiled`` / ``_pad_halo_mpi_tiled_4d``).  The
existing ``test_mpi_interp_offsets`` skips ``size > 6``, so
the tiled-mode rank-sort fix was untested.

This file adds a minimal tiled-mode test at ``n_processes=24``
(tiles_per_face=4, k=2) — the smallest valid tiled config —
that exercises both scalar 2D and 4D MPI halo without
``interp_offsets`` (which iter-1040 refuses for tiled mode).

Run with::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 mpirun -np 24 --oversubscribe \\
        --timeout 300 python -m pytest \\
        tests/distributed/test_mpi_tiled_pad_halo.py -v

If 24 processes are not feasible on the test host, this file
is skipped.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.grids.halo import set_halo_backend, pad_halo, pad_halo_4d
from legoesm.parallel.comm import build_comm_topology


@pytest.fixture(autouse=True)
def reset_halo_backend():
    yield
    set_halo_backend("local")


def _require_tiled_topology():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if size <= 6 or size % 6 != 0:
        pytest.skip(f"tiled-mode requires n_processes = 6 * k^2 >= 24; got {size}")
    topology = build_comm_topology(rank, size)
    if topology.tiling == (1, 1):
        pytest.skip(f"topology not tiled at size={size}")
    return topology, rank


class TestTiledPadHaloMPI:
    """Tiled-mode pad_halo_mpi rank-sort fix coverage."""

    def test_scalar_2d_tiled_completes(self):
        """``pad_halo`` scalar 2D under tiled MPI completes without deadlock.

        Doesn't check bit-for-bit (would need a reference run on a
        single device that knows about the tile layout); checks that
        the call returns finite output and matches its own first-touch
        result on repeat — i.e., no hang and reproducible output.
        """
        topology, rank = _require_tiled_topology()
        # Each rank's local data is a tile.  Tile shape: (1, n_tile, n_tile).
        n_tile = topology.local_face_ids and 8 or 8  # n_tile per tile
        # In tiled mode each rank owns 1 face × 1 tile.
        key = jax.random.PRNGKey(20260528 + rank)
        data = jax.random.normal(key, (1, n_tile, n_tile), dtype=jnp.float64)

        set_halo_backend("mpi", topology)
        padded_1 = pad_halo(data)
        jax.block_until_ready(padded_1)
        padded_2 = pad_halo(data)
        jax.block_until_ready(padded_2)

        # Reproducibility: identical calls produce identical output.
        np.testing.assert_array_equal(
            np.asarray(padded_1), np.asarray(padded_2),
            err_msg=f"rank{rank} pad_halo (tiled) not reproducible",
        )
        # Sanity: no NaN/Inf in interior (halo cells may be zero from
        # nbr edges not yet exchanged — but pad_halo SHOULD have done
        # the exchange).
        assert np.all(np.isfinite(np.asarray(padded_1))), (
            f"rank{rank} pad_halo (tiled) produced non-finite values"
        )

    def test_4d_tiled_completes(self):
        """``pad_halo_4d`` 4D under tiled MPI completes without deadlock."""
        topology, rank = _require_tiled_topology()
        n_tile = 8
        nlev = 5
        key = jax.random.PRNGKey(20260528 + rank)
        data = jax.random.normal(
            key, (1, n_tile, n_tile, nlev), dtype=jnp.float64,
        )

        set_halo_backend("mpi", topology)
        padded_1 = pad_halo_4d(data)
        jax.block_until_ready(padded_1)
        padded_2 = pad_halo_4d(data)
        jax.block_until_ready(padded_2)

        np.testing.assert_array_equal(
            np.asarray(padded_1), np.asarray(padded_2),
            err_msg=f"rank{rank} pad_halo_4d (tiled) not reproducible",
        )
        assert np.all(np.isfinite(np.asarray(padded_1))), (
            f"rank{rank} pad_halo_4d (tiled) produced non-finite values"
        )
