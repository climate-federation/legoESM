"""Category 6: MPI layout & communication topology tests.

Tests CommTopology, DistributedLayout, SingleRankLayout,
scatter/gather (no actual MPI needed - pure logic tests).
"""

from __future__ import annotations

import pytest
import numpy as np
import jax.numpy as jnp

from legoesm.parallel.comm import (
    CommTopology,
    build_comm_topology,
    _face_to_rank,
    _rank_to_faces,
    _tile_rank,
    _rank_to_tile,
)
from legoesm.parallel.layout import (
    make_layout,
    DistributedLayout,
    SingleRankLayout,
    FaceOwnership,
    scatter,
    gather,
    scatter_pytree,
)


N = 8  # Global grid size per face edge


# ===========================================================================
# CommTopology: face-only mode
# ===========================================================================

class TestCommTopologyFaceOnly:
    """Face-only decomposition (1, 2, 3, or 6 processes)."""

    @pytest.mark.parametrize("n_procs", [1, 2, 3, 6])
    def test_all_faces_covered(self, n_procs):
        """Every face must be owned by exactly one rank."""
        all_faces = set()
        for rank in range(n_procs):
            topo = build_comm_topology(rank, n_procs)
            all_faces.update(topo.local_face_ids)
        assert all_faces == {0, 1, 2, 3, 4, 5}

    def test_single_rank_owns_all(self):
        topo = build_comm_topology(0, 1)
        assert topo.local_face_ids == (0, 1, 2, 3, 4, 5)
        assert topo.tiling == (1, 1)
        assert topo.tile_index == (0, 0)
        assert topo.tile_neighbors == {}

    def test_six_ranks_one_face_each(self):
        for rank in range(6):
            topo = build_comm_topology(rank, 6)
            assert len(topo.local_face_ids) == 1
            assert topo.local_face_ids[0] == rank

    def test_neighbor_ranks_valid(self):
        for rank in range(6):
            topo = build_comm_topology(rank, 6)
            for (face, edge), nbr_rank in topo.neighbor_ranks.items():
                assert 0 <= nbr_rank < 6
                assert face in topo.local_face_ids

    def test_invalid_n_processes_raises(self):
        with pytest.raises(ValueError):
            build_comm_topology(0, 4)  # 4 doesn't divide 6
        with pytest.raises(ValueError):
            build_comm_topology(0, 5)


# ===========================================================================
# CommTopology: sub-face tiling
# ===========================================================================

class TestCommTopologyTiled:
    """Sub-face tiling decomposition (24 = 6*2^2 processes)."""

    def test_24_processes_all_faces_covered(self):
        all_faces = set()
        for rank in range(24):
            topo = build_comm_topology(rank, 24)
            all_faces.update(topo.local_face_ids)
        assert all_faces == {0, 1, 2, 3, 4, 5}

    def test_24_processes_tiling(self):
        topo = build_comm_topology(0, 24)
        assert topo.tiling == (2, 2)

    def test_tile_neighbors_interior(self):
        """An interior tile should have all 4 same-face neighbors."""
        # With 54 = 6*3^2 processes, tile (1,1) is interior on each face
        # rank = face * 9 + 1*3 + 1 = face*9 + 4
        topo = build_comm_topology(4, 54)  # face=0, tile=(1,1)
        assert topo.tile_index == (1, 1)
        # All 4 directions should have same-face neighbors (not None)
        for direction in ["west", "east", "south", "north"]:
            assert topo.tile_neighbors[direction] is not None

    def test_tile_neighbors_corner(self):
        """A corner tile should have 2 None neighbors (inter-face)."""
        # rank=0 -> face=0, tile=(0,0) = SW corner
        topo = build_comm_topology(0, 24)
        assert topo.tile_index == (0, 0)
        assert topo.tile_neighbors["west"] is None  # face boundary
        assert topo.tile_neighbors["south"] is None  # face boundary


# ===========================================================================
# Face/rank mapping helpers
# ===========================================================================

class TestFaceRankMapping:
    def test_face_to_rank_6(self):
        for f in range(6):
            assert _face_to_rank(f, 6) == f

    def test_face_to_rank_3(self):
        assert _face_to_rank(0, 3) == 0
        assert _face_to_rank(1, 3) == 0
        assert _face_to_rank(2, 3) == 1
        assert _face_to_rank(3, 3) == 1
        assert _face_to_rank(4, 3) == 2
        assert _face_to_rank(5, 3) == 2

    def test_rank_to_faces_roundtrip(self):
        for n in [1, 2, 3, 6]:
            all_faces = []
            for r in range(n):
                all_faces.extend(_rank_to_faces(r, n))
            assert sorted(all_faces) == [0, 1, 2, 3, 4, 5]

    def test_tile_rank_roundtrip(self):
        tx, ty = 2, 2
        for face in range(6):
            for ti in range(tx):
                for tj in range(ty):
                    rank = _tile_rank(face, ti, tj, tx, ty)
                    f2, ti2, tj2 = _rank_to_tile(rank, tx, ty)
                    assert (f2, ti2, tj2) == (face, ti, tj)


# ===========================================================================
# DistributedLayout / SingleRankLayout
# ===========================================================================

class TestMakeLayout:
    def test_single_rank_layout(self):
        layout = make_layout(0, 1, N)
        assert isinstance(layout, SingleRankLayout)
        assert layout.rank == 0
        assert layout.n_ranks == 1
        assert layout.global_n == N
        assert layout.local_shape_2d == (6, N, N)
        assert layout.is_tiled is False

    def test_multi_rank_face_layout(self):
        layout = make_layout(0, 6, N)
        assert isinstance(layout, DistributedLayout)
        assert layout.rank == 0
        assert layout.n_ranks == 6
        assert len(layout.ownership.face_ids) == 1
        assert layout.is_tiled is False

    def test_tiled_layout(self):
        layout = make_layout(0, 24, N)
        assert isinstance(layout, DistributedLayout)
        assert layout.is_tiled is True
        assert layout.ownership.tiling == (2, 2)
        assert layout.ownership.tile_size == N // 2

    def test_all_ranks_cover_all_faces(self):
        for n_ranks in [2, 3, 6]:
            all_faces = set()
            for r in range(n_ranks):
                layout = make_layout(r, n_ranks, N)
                all_faces.update(layout.ownership.face_ids)
            assert all_faces == {0, 1, 2, 3, 4, 5}


# ===========================================================================
# scatter / gather (single-rank identity)
# ===========================================================================

class TestScatterGatherSingleRank:
    def test_scatter_identity(self):
        layout = make_layout(0, 1, N)
        arr = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        out = scatter(arr, layout)
        np.testing.assert_array_equal(out, arr)

    def test_scatter_pytree_identity(self):
        layout = make_layout(0, 1, N)
        tree = {"a": jnp.ones((6, N, N)), "b": jnp.zeros((3, 4))}
        out = scatter_pytree(tree, layout)
        np.testing.assert_array_equal(out["a"], tree["a"])
        np.testing.assert_array_equal(out["b"], tree["b"])


# ===========================================================================
# scatter multi-rank (face-only, mock with explicit layout)
# ===========================================================================

class TestScatterMultiRankFace:
    def test_face_scatter_extracts_correct_faces(self):
        """Rank 0 in a 6-rank layout should get only face 0."""
        layout = make_layout(0, 6, N)
        arr = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        local = scatter(arr, layout)
        assert local.shape == (1, N, N)
        np.testing.assert_array_equal(local[0], arr[0])

    def test_face_scatter_rank5_gets_face5(self):
        layout = make_layout(5, 6, N)
        arr = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        local = scatter(arr, layout)
        assert local.shape == (1, N, N)
        np.testing.assert_array_equal(local[0], arr[5])


# ===========================================================================
# scatter multi-rank (tiled)
# ===========================================================================

class TestScatterTiled:
    def test_tiled_scatter_shape(self):
        """Rank 0 in a 24-rank tiled layout should get tile (0,0) of face 0."""
        layout = make_layout(0, 24, N)
        arr = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        local = scatter(arr, layout)
        nt = N // 2
        assert local.shape == (1, nt, nt)

    def test_tiled_scatter_correct_data(self):
        layout = make_layout(0, 24, N)
        arr = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        local = scatter(arr, layout)
        nt = N // 2
        expected = arr[0, :nt, :nt]
        np.testing.assert_array_equal(local[0], expected)


# ===========================================================================
# Subprocess wrapper: run MPI bootstrap test via mpirun
# ===========================================================================

import shutil
import subprocess
import sys


class TestMPIBootstrapSubprocess:
    """Run MPI bootstrap integration test via subprocess.

    This allows the MPI test to be discovered by the regular pytest suite
    without requiring the user to invoke mpirun directly.
    """

    @pytest.fixture(autouse=True)
    def check_mpi(self):
        """Skip if mpirun or mpi4py is not available."""
        if shutil.which("mpirun") is None:
            pytest.skip("mpirun not available")
        try:
            import mpi4py  # noqa: F401
        except ImportError:
            pytest.skip("mpi4py not installed")

    def test_mpi_bootstrap_2_ranks(self):
        """MPI bootstrap with 2 ranks should succeed."""
        result = subprocess.run(
            [
                "mpirun", "-np", "2", "--oversubscribe",
                sys.executable, "-m", "pytest",
                "tests/distributed/test_mpi_bootstrap.py",
                "-v", "--tb=short", "-x",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        # Print output for debugging on failure.
        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        assert result.returncode == 0, (
            f"MPI bootstrap test failed with rc={result.returncode}"
        )
