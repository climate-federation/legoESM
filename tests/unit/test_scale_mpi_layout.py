"""Category 6: MPI distributed layout & communication.

Tests layout construction, topology computation, and scatter/gather
patterns WITHOUT requiring MPI.
"""

from __future__ import annotations

import pytest
import jax.numpy as jnp
import numpy as np

from legoesm.parallel.comm import (
    build_comm_topology,
    _face_to_rank,
    _rank_to_faces,
)
from legoesm.parallel.layout import (
    make_layout,
    scatter,
    DistributedLayout,
    SingleRankLayout,
)
from legoesm.parallel.runtime import validate_device_count


# =========================================================================
# 6a) CommTopology — face-only (≤6 ranks)
# =========================================================================

class TestCommTopologyFaceOnly:
    """Face-only topology for 1–6 ranks."""

    @pytest.mark.parametrize("world_size", [1, 2, 3, 6])
    def test_all_faces_assigned(self, world_size):
        """Every face is owned by exactly one rank."""
        all_faces = set()
        for rank in range(world_size):
            topo = build_comm_topology(rank, world_size)
            for f in topo.local_face_ids:
                assert f not in all_faces, f"Face {f} assigned to multiple ranks"
                all_faces.add(f)
        assert all_faces == {0, 1, 2, 3, 4, 5}

    @pytest.mark.parametrize("world_size", [1, 2, 3, 6])
    def test_face_to_rank_consistency(self, world_size):
        """_face_to_rank and _rank_to_faces should be consistent."""
        for rank in range(world_size):
            faces = _rank_to_faces(rank, world_size)
            for f in faces:
                assert _face_to_rank(f, world_size) == rank

    def test_neighbor_ranks_valid(self):
        """Neighbor ranks should be in [0, world_size)."""
        world_size = 6
        for rank in range(world_size):
            topo = build_comm_topology(rank, world_size)
            for key, nbr_rank in topo.neighbor_ranks.items():
                assert 0 <= nbr_rank < world_size, (
                    f"Rank {rank}: neighbor rank {nbr_rank} out of bounds"
                )


# =========================================================================
# 6b) CommTopology — sub-face tiling
# =========================================================================

class TestCommTopologyTiled:
    """Sub-face tiling for 6*k^2 ranks."""

    def test_24_ranks_4_tiles_per_face(self):
        """24 ranks = 6 faces × 4 tiles."""
        for rank in range(24):
            topo = build_comm_topology(rank, 24)
            assert topo.tiling == (2, 2)
            assert len(topo.local_face_ids) == 1

    def test_all_tiles_assigned(self):
        """Every tile is owned by exactly one rank."""
        world_size = 24
        tiles = set()
        for rank in range(world_size):
            topo = build_comm_topology(rank, world_size)
            face = topo.local_face_ids[0]
            ti, tj = topo.tile_index
            key = (face, ti, tj)
            assert key not in tiles, f"Tile {key} assigned to multiple ranks"
            tiles.add(key)
        assert len(tiles) == 24


# =========================================================================
# 6d) Layout — scatter correctness
# =========================================================================

class TestLayoutScatter:
    """scatter should extract correct subset of global array."""

    def test_single_rank_noop(self):
        """SingleRankLayout scatter is identity."""
        layout = make_layout(rank=0, n_ranks=1, global_n=8)
        assert isinstance(layout, SingleRankLayout)
        data = jnp.arange(6 * 8 * 8, dtype=jnp.float32).reshape(6, 8, 8)
        result = scatter(data, layout)
        np.testing.assert_array_equal(np.array(result), np.array(data))

    @pytest.mark.parametrize("n_ranks", [2, 3, 6])
    def test_face_layout_shapes(self, n_ranks):
        """Face-only layout scatter produces correct shapes."""
        n = 8
        faces_per_rank = 6 // n_ranks
        for rank in range(n_ranks):
            layout = make_layout(rank=rank, n_ranks=n_ranks, global_n=n)
            data = jnp.arange(6 * n * n, dtype=jnp.float32).reshape(6, n, n)
            result = scatter(data, layout)
            assert result.shape == (faces_per_rank, n, n)

    def test_tiled_layout_shape(self):
        """Tiled layout scatter produces tile-sized output."""
        n = 8
        layout = make_layout(rank=0, n_ranks=24, global_n=n)
        assert isinstance(layout, DistributedLayout)
        assert layout.is_tiled
        data = jnp.arange(6 * n * n, dtype=jnp.float32).reshape(6, n, n)
        result = scatter(data, layout)
        tile_n = n // 2  # k=2 for 24 ranks
        assert result.shape == (1, tile_n, tile_n)


# =========================================================================
# 6g) validate_device_count
# =========================================================================

class TestValidateDeviceCount:
    """validate_device_count should accept valid and reject invalid counts."""

    @pytest.mark.parametrize("n", [1, 2, 3, 6, 24, 54, 96, 150])
    def test_valid(self, n):
        validate_device_count(n)  # Should not raise

    @pytest.mark.parametrize("n", [0, 4, 5, 7, 8, 9, 10, 11, 12, 25])
    def test_invalid(self, n):
        with pytest.raises(ValueError):
            validate_device_count(n)

    def test_error_message_has_suggestions(self):
        """Error message should include nearby valid counts."""
        with pytest.raises(ValueError, match="next lower"):
            validate_device_count(10)
