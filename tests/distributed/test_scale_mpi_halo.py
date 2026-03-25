"""Category 7: MPI halo exchange tests.

Tests MPI halo exchange correctness. Skipped if mpi4jax is not available.
"""

from __future__ import annotations

import pytest
import numpy as np
import jax.numpy as jnp

# Skip entire module if mpi4jax is not available
mpi4jax = pytest.importorskip("mpi4jax", reason="mpi4jax not installed")

from legoesm.parallel.halo_exchange import (
    pad_halo_mpi,
    _place_strip,
    _place_strip_h2,
)
from legoesm.parallel.comm import build_comm_topology, CommTopology
from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH


N = 8


# ---------------------------------------------------------------------------
# _place_strip correctness
# ---------------------------------------------------------------------------

class TestPlaceStrip:
    """Test strip placement into padded arrays."""

    def test_place_west(self):
        padded = jnp.zeros((6, N + 2, N + 2), dtype=jnp.float64)
        strip = jnp.ones(N, dtype=jnp.float64) * 7.0
        padded = _place_strip(padded, 0, WEST, strip)
        np.testing.assert_allclose(padded[0, 0, 1:-1], 7.0)

    def test_place_east(self):
        padded = jnp.zeros((6, N + 2, N + 2), dtype=jnp.float64)
        strip = jnp.ones(N, dtype=jnp.float64) * 3.0
        padded = _place_strip(padded, 2, EAST, strip)
        np.testing.assert_allclose(padded[2, -1, 1:-1], 3.0)

    def test_place_south(self):
        padded = jnp.zeros((6, N + 2, N + 2), dtype=jnp.float64)
        strip = jnp.ones(N, dtype=jnp.float64) * 5.0
        padded = _place_strip(padded, 1, SOUTH, strip)
        np.testing.assert_allclose(padded[1, 1:-1, 0], 5.0)

    def test_place_north(self):
        padded = jnp.zeros((6, N + 2, N + 2), dtype=jnp.float64)
        strip = jnp.ones(N, dtype=jnp.float64) * 9.0
        padded = _place_strip(padded, 3, NORTH, strip)
        np.testing.assert_allclose(padded[3, 1:-1, -1], 9.0)

    def test_place_does_not_affect_other_faces(self):
        padded = jnp.zeros((6, N + 2, N + 2), dtype=jnp.float64)
        strip = jnp.ones(N, dtype=jnp.float64)
        padded = _place_strip(padded, 0, WEST, strip)
        # Faces 1-5 should still be all zeros
        for f in range(1, 6):
            np.testing.assert_allclose(padded[f], 0.0)


# ---------------------------------------------------------------------------
# CommTopology for MPI halo exchange
# ---------------------------------------------------------------------------

class TestCommTopologyForHalo:
    """Test that CommTopology provides correct info for halo exchange."""

    def test_neighbor_info_matches_connectivity(self):
        """neighbor_info should match the CONNECTIVITY table."""
        topo = build_comm_topology(0, 6)
        for (face, edge), (nbr_face, nbr_edge, rev) in topo.neighbor_info.items():
            expected = CONNECTIVITY[face][edge]
            assert (nbr_face, nbr_edge, rev) == expected

    def test_all_edges_have_neighbors(self):
        """Every face edge should have a neighbor rank."""
        topo = build_comm_topology(0, 6)
        face = topo.local_face_ids[0]
        for edge in [WEST, EAST, SOUTH, NORTH]:
            assert (face, edge) in topo.neighbor_ranks

    @pytest.mark.parametrize("n_procs", [1, 2, 3, 6])
    def test_topology_completeness(self, n_procs):
        """Every rank's local faces should have all 4 edges in neighbor_info."""
        for rank in range(n_procs):
            topo = build_comm_topology(rank, n_procs)
            for face in topo.local_face_ids:
                for edge in [WEST, EAST, SOUTH, NORTH]:
                    assert (face, edge) in topo.neighbor_info
                    assert (face, edge) in topo.neighbor_ranks
