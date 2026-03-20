"""Tests for the distributed layout system.

Validates:
- Layout construction for single-rank, face-only, and tiled modes.
- scatter/gather roundtrip preserves data.
- Local reductions give correct results.
- SingleRankLayout is identity (no-op scatter/gather).
- Backward-compatible shims in distributed.py.
- __init__.py exports are intact.

All tests run on a single process (no MPI required) by testing the
pure-JAX scatter path and mocking MPI for gather.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.parallel.layout import (
    FaceOwnership,
    DistributedLayout,
    SingleRankLayout,
    make_layout,
    scatter,
    scatter_pytree,
    gather,
    gather_pytree,
    local_sum,
    local_max,
    local_min,
    global_reduce,
)


# =========================================================================
# Helpers
# =========================================================================

def _global_field(n: int = 8, seed: int = 0) -> jnp.ndarray:
    """Create a deterministic (6, n, n) global field."""
    rng = np.random.default_rng(seed)
    return jnp.array(rng.standard_normal((6, n, n)), dtype=jnp.float32)


def _global_field_3d(n: int = 8, nlev: int = 5, seed: int = 0) -> jnp.ndarray:
    """Create a deterministic (6, n, n, nlev) global field."""
    rng = np.random.default_rng(seed)
    return jnp.array(rng.standard_normal((6, n, n, nlev)), dtype=jnp.float32)


# =========================================================================
# 1. Layout construction
# =========================================================================

class TestMakeLayout:
    """make_layout produces the correct layout type and shape."""

    def test_single_rank_layout(self):
        layout = make_layout(0, 1, 16)
        assert isinstance(layout, SingleRankLayout)
        assert layout.rank == 0
        assert layout.n_ranks == 1
        assert layout.global_n == 16
        assert layout.local_shape_2d == (6, 16, 16)
        assert layout.is_tiled is False

    @pytest.mark.parametrize("n_ranks", [1, 2, 3, 6])
    def test_face_only_valid_ranks(self, n_ranks):
        if n_ranks == 1:
            pytest.skip("Single rank handled separately")
        for rank in range(n_ranks):
            layout = make_layout(rank, n_ranks, 12)
            assert isinstance(layout, DistributedLayout)
            assert layout.is_tiled is False
            faces_per_rank = 6 // n_ranks
            assert len(layout.ownership.face_ids) == faces_per_rank
            assert layout.local_shape_2d == (faces_per_rank, 12, 12)

    def test_face_only_6_ranks_one_face_each(self):
        for rank in range(6):
            layout = make_layout(rank, 6, 16)
            assert layout.ownership.face_ids == (rank,)

    def test_face_only_2_ranks_three_faces_each(self):
        layout0 = make_layout(0, 2, 16)
        layout1 = make_layout(1, 2, 16)
        assert layout0.ownership.face_ids == (0, 1, 2)
        assert layout1.ownership.face_ids == (3, 4, 5)

    def test_face_only_invalid_ranks_raises(self):
        with pytest.raises(ValueError, match="does not evenly divide"):
            make_layout(0, 4, 16)

    def test_tiled_24_ranks(self):
        """24 = 6 × 4 → 2×2 tiles per face."""
        layout = make_layout(0, 24, 16)
        assert isinstance(layout, DistributedLayout)
        assert layout.is_tiled is True
        assert layout.ownership.tiling == (2, 2)
        assert layout.ownership.tile_size == 8
        assert layout.local_shape_2d == (1, 8, 8)

    def test_tiled_54_ranks(self):
        """54 = 6 × 9 → 3×3 tiles per face."""
        layout = make_layout(0, 54, 18)
        assert layout.is_tiled is True
        assert layout.ownership.tiling == (3, 3)
        assert layout.ownership.tile_size == 6

    def test_tiled_non_square_tiles_raises(self):
        """12 = 6 × 2, but 2 is not a perfect square."""
        with pytest.raises(ValueError, match="not a perfect square"):
            make_layout(0, 12, 16)

    def test_tiled_resolution_not_divisible_raises(self):
        """24 = 6×4 → k=2, but 15 not divisible by 2."""
        with pytest.raises(ValueError, match="not divisible"):
            make_layout(0, 24, 15)

    def test_tiled_not_multiple_of_6_raises(self):
        with pytest.raises(ValueError, match="not a multiple of 6"):
            make_layout(0, 10, 16)


# =========================================================================
# 2. FaceOwnership
# =========================================================================

class TestFaceOwnership:
    """FaceOwnership fields are consistent."""

    def test_face_only(self):
        layout = make_layout(1, 3, 12)
        own = layout.ownership
        assert own.tile == (0, 0)
        assert own.tiling == (1, 1)
        assert own.tile_size == 12

    def test_tiled(self):
        layout = make_layout(7, 24, 16)
        own = layout.ownership
        # rank 7: face = 7 // 4 = 1, tile_idx = 7 % 4 = 3
        # ti = 3 // 2 = 1, tj = 3 % 2 = 1
        assert own.face_ids == (1,)
        assert own.tile == (1, 1)
        assert own.tiling == (2, 2)
        assert own.tile_size == 8


# =========================================================================
# 3. Scatter (global → local)
# =========================================================================

class TestScatter:
    """scatter extracts the correct local portion."""

    def test_single_rank_identity(self):
        layout = make_layout(0, 1, 8)
        field = _global_field(8)
        local = scatter(field, layout)
        np.testing.assert_array_equal(local, field)

    def test_face_only_6_ranks(self):
        field = _global_field(8, seed=42)
        for rank in range(6):
            layout = make_layout(rank, 6, 8)
            local = scatter(field, layout)
            assert local.shape == (1, 8, 8)
            np.testing.assert_array_equal(local[0], field[rank])

    def test_face_only_2_ranks(self):
        field = _global_field(8, seed=42)
        layout0 = make_layout(0, 2, 8)
        local0 = scatter(field, layout0)
        assert local0.shape == (3, 8, 8)
        np.testing.assert_array_equal(local0, field[:3])

        layout1 = make_layout(1, 2, 8)
        local1 = scatter(field, layout1)
        np.testing.assert_array_equal(local1, field[3:])

    def test_tiled_24_ranks(self):
        """Each rank gets a (1, 4, 4) tile from a (6, 8, 8) field."""
        field = _global_field(8, seed=7)
        for rank in range(24):
            layout = make_layout(rank, 24, 8)
            local = scatter(field, layout)
            assert local.shape == (1, 4, 4)
            # Verify content matches the right tile
            own = layout.ownership
            face = own.face_ids[0]
            ti, tj = own.tile
            nt = own.tile_size
            expected = field[face, ti * nt:(ti + 1) * nt, tj * nt:(tj + 1) * nt]
            np.testing.assert_array_equal(local[0], expected)

    def test_scatter_with_trailing_dims(self):
        field = _global_field_3d(8, 5, seed=3)
        layout = make_layout(2, 6, 8)
        local = scatter(field, layout)
        assert local.shape == (1, 8, 8, 5)
        np.testing.assert_array_equal(local[0], field[2])

    def test_tiled_scatter_with_trailing_dims(self):
        field = _global_field_3d(8, 5, seed=3)
        layout = make_layout(0, 24, 8)
        local = scatter(field, layout)
        assert local.shape == (1, 4, 4, 5)


# =========================================================================
# 4. Scatter pytree
# =========================================================================

class TestScatterPytree:
    """scatter_pytree handles arbitrary pytrees."""

    def test_dict_pytree(self):
        field_a = _global_field(8, seed=0)
        field_b = _global_field(8, seed=1)
        pytree = {"a": field_a, "b": field_b}
        layout = make_layout(0, 6, 8)
        local_tree = scatter_pytree(pytree, layout)
        assert local_tree["a"].shape == (1, 8, 8)
        assert local_tree["b"].shape == (1, 8, 8)

    def test_non_face_arrays_pass_through(self):
        """Arrays without leading dim=6 are not scattered."""
        scalar = jnp.array(42.0)
        vec = jnp.ones(10)
        layout = make_layout(0, 6, 8)
        tree = {"scalar": scalar, "vec": vec}
        result = scatter_pytree(tree, layout)
        np.testing.assert_array_equal(result["scalar"], scalar)
        np.testing.assert_array_equal(result["vec"], vec)

    def test_single_rank_identity(self):
        pytree = {"x": _global_field(8)}
        layout = make_layout(0, 1, 8)
        result = scatter_pytree(pytree, layout)
        np.testing.assert_array_equal(result["x"], pytree["x"])


# =========================================================================
# 5. Scatter + gather roundtrip (face-only, simulated)
# =========================================================================

class TestScatterGatherRoundtrip:
    """Verify scatter then gather recovers the original field.

    Since MPI is not available in unit tests, we simulate multi-rank
    gather by concatenating local contributions from all ranks.
    """

    def test_face_only_6_ranks_roundtrip(self):
        """Scatter to 6 ranks, manually reassemble, verify exact match."""
        field = _global_field(8, seed=99)
        locals_ = []
        for rank in range(6):
            layout = make_layout(rank, 6, 8)
            locals_.append(scatter(field, layout))

        # Reassemble: each local is (1, 8, 8), concat along face dim
        reassembled = jnp.concatenate(locals_, axis=0)
        np.testing.assert_array_equal(reassembled, field)

    def test_face_only_2_ranks_roundtrip(self):
        field = _global_field(8, seed=42)
        local0 = scatter(field, make_layout(0, 2, 8))
        local1 = scatter(field, make_layout(1, 2, 8))
        reassembled = jnp.concatenate([local0, local1], axis=0)
        np.testing.assert_array_equal(reassembled, field)

    def test_face_only_3_ranks_roundtrip(self):
        field = _global_field(12, seed=11)
        locals_ = []
        for rank in range(3):
            layout = make_layout(rank, 3, 12)
            locals_.append(scatter(field, layout))
        reassembled = jnp.concatenate(locals_, axis=0)
        np.testing.assert_array_equal(reassembled, field)

    def test_tiled_24_ranks_roundtrip(self):
        """Scatter to 24 ranks (2×2 tiles), manually reassemble."""
        n = 8
        field = _global_field(n, seed=77)
        result = jnp.zeros_like(field)
        for rank in range(24):
            layout = make_layout(rank, 24, n)
            local = scatter(field, layout)
            own = layout.ownership
            face = own.face_ids[0]
            ti, tj = own.tile
            nt = own.tile_size
            result = result.at[face, ti * nt:(ti + 1) * nt, tj * nt:(tj + 1) * nt].set(local[0])
        np.testing.assert_array_equal(result, field)

    def test_tiled_roundtrip_with_trailing_dims(self):
        """Same but with (6, 8, 8, 5) field."""
        n, nlev = 8, 5
        field = _global_field_3d(n, nlev, seed=55)
        result = jnp.zeros_like(field)
        for rank in range(24):
            layout = make_layout(rank, 24, n)
            local = scatter(field, layout)
            own = layout.ownership
            face = own.face_ids[0]
            ti, tj = own.tile
            nt = own.tile_size
            result = result.at[face, ti * nt:(ti + 1) * nt, tj * nt:(tj + 1) * nt].set(local[0])
        np.testing.assert_array_equal(result, field)


# =========================================================================
# 6. SingleRankLayout gather is identity
# =========================================================================

class TestSingleRankGather:
    """SingleRankLayout scatter/gather are no-ops."""

    def test_gather_identity(self):
        layout = make_layout(0, 1, 8)
        field = _global_field(8)
        assert gather(field, layout) is field

    def test_gather_pytree_identity(self):
        layout = make_layout(0, 1, 8)
        tree = {"x": _global_field(8), "y": jnp.array(1.0)}
        result = gather_pytree(tree, layout)
        assert result is tree


# =========================================================================
# 7. Local reductions
# =========================================================================

class TestLocalReductions:
    """local_sum, local_max, local_min on rank-local data."""

    def test_local_sum_unweighted(self):
        data = jnp.ones((2, 8, 8))
        result = local_sum(data)
        np.testing.assert_allclose(float(result), 2 * 8 * 8)

    def test_local_sum_weighted(self):
        data = jnp.ones((1, 4, 4))
        weights = jnp.full((1, 4, 4), 2.0)
        result = local_sum(data, weights)
        np.testing.assert_allclose(float(result), 32.0)

    def test_local_max(self):
        data = jnp.array([[[1, 5], [3, 2]]])
        assert float(local_max(data)) == 5.0

    def test_local_min(self):
        data = jnp.array([[[1, 5], [3, -2]]])
        assert float(local_min(data)) == -2.0


# =========================================================================
# 8. global_reduce with SingleRankLayout
# =========================================================================

class TestGlobalReduce:
    """global_reduce is identity for SingleRankLayout."""

    def test_sum_single_rank(self):
        layout = make_layout(0, 1, 8)
        val = jnp.array(42.0)
        result = global_reduce(val, layout, op="sum")
        assert float(result) == 42.0

    def test_max_single_rank(self):
        layout = make_layout(0, 1, 8)
        val = jnp.array(7.0)
        result = global_reduce(val, layout, op="max")
        assert float(result) == 7.0

    def test_min_single_rank(self):
        layout = make_layout(0, 1, 8)
        val = jnp.array(-3.0)
        result = global_reduce(val, layout, op="min")
        assert float(result) == -3.0

    def test_invalid_op_raises(self):
        # Use a DistributedLayout (not SingleRankLayout) so that
        # global_reduce actually dispatches to the op lookup.
        layout = make_layout(0, 6, 8)
        with pytest.raises(ValueError, match="Unknown op"):
            global_reduce(jnp.array(1.0), layout, op="mean")


# =========================================================================
# 9. Multi-rank local sums equal single-rank global sum
# =========================================================================

class TestLocalSumConsistency:
    """Sum of local contributions equals global sum (no MPI needed)."""

    def test_face_only_local_sums_equal_global(self):
        field = _global_field(8, seed=123)
        global_total = float(jnp.sum(field))

        rank_sums = 0.0
        for rank in range(6):
            layout = make_layout(rank, 6, 8)
            local_field = scatter(field, layout)
            rank_sums += float(local_sum(local_field))

        np.testing.assert_allclose(rank_sums, global_total, rtol=1e-5)

    def test_tiled_local_sums_equal_global(self):
        field = _global_field(8, seed=456)
        global_total = float(jnp.sum(field))

        rank_sums = 0.0
        for rank in range(24):
            layout = make_layout(rank, 24, 8)
            local_field = scatter(field, layout)
            rank_sums += float(local_sum(local_field))

        np.testing.assert_allclose(rank_sums, global_total, rtol=1e-5)

    def test_weighted_local_sums_equal_global_weighted(self):
        """Area-weighted sums across ranks equal the global integral."""
        n = 8
        field = _global_field(n, seed=789)
        # Uniform area weights
        area = jnp.ones((6, n, n)) * 1e10

        global_integral = float(jnp.sum(field * area))

        rank_integrals = 0.0
        for rank in range(6):
            layout = make_layout(rank, 6, n)
            local_f = scatter(field, layout)
            local_a = scatter(area, layout)
            rank_integrals += float(local_sum(local_f, local_a))

        np.testing.assert_allclose(rank_integrals, global_integral, rtol=1e-5)


# =========================================================================
# 10. Legacy shims in distributed.py
# =========================================================================

class TestDistributedShims:
    """distributed.py exports new layout-based API."""

    def test_scatter_to_local_requires_layout(self):
        from legoesm.parallel.distributed import scatter_to_local, set_active_layout
        # Ensure no active layout
        set_active_layout(None)
        with pytest.raises(ValueError, match="No layout"):
            scatter_to_local({"x": jnp.ones((6, 8, 8))})

    def test_gather_to_global_requires_layout(self):
        from legoesm.parallel.distributed import gather_to_global, set_active_layout
        set_active_layout(None)
        with pytest.raises(ValueError, match="No layout"):
            gather_to_global({"x": jnp.ones((1, 8, 8))})

    def test_scatter_to_local_with_explicit_layout(self):
        from legoesm.parallel.distributed import scatter_to_local
        layout = make_layout(0, 6, 8)
        field = _global_field(8)
        result = scatter_to_local({"x": field}, layout=layout)
        assert result["x"].shape == (1, 8, 8)

    def test_get_set_active_layout(self):
        from legoesm.parallel.distributed import (
            get_active_layout, set_active_layout,
        )
        layout = make_layout(0, 1, 8)
        set_active_layout(layout)
        assert get_active_layout() is layout
        # Cleanup
        set_active_layout(None)


# =========================================================================
# 11. __init__.py exports
# =========================================================================

class TestParallelExports:
    """parallel.__init__.py exports layout types and functions."""

    def test_layout_types_exported(self):
        from legoesm.parallel import (
            DistributedLayout,
            SingleRankLayout,
            FaceOwnership,
        )
        assert DistributedLayout is not None
        assert SingleRankLayout is not None
        assert FaceOwnership is not None

    def test_layout_functions_exported(self):
        from legoesm.parallel import (
            make_layout,
            layout_scatter,
            layout_scatter_pytree,
            layout_gather,
            layout_gather_pytree,
            layout_local_sum,
            layout_local_max,
            layout_local_min,
            layout_global_reduce,
        )
        # Just verify they're callable
        assert callable(make_layout)
        assert callable(layout_scatter)
        assert callable(layout_scatter_pytree)
        assert callable(layout_gather)
        assert callable(layout_gather_pytree)
        assert callable(layout_local_sum)
        assert callable(layout_local_max)
        assert callable(layout_local_min)
        assert callable(layout_global_reduce)


# =========================================================================
# 12. Comparison: old zero-masked vs new rank-local
# =========================================================================

class TestZeroMaskedVsRankLocal:
    """New rank-local scatter gives same non-zero data as old masking."""

    def test_rank_local_matches_zero_masked(self):
        """For rank 0 of 6: local[0] == global[0] (both approaches)."""
        field = _global_field(8, seed=31)
        layout = make_layout(0, 6, 8)
        local = scatter(field, layout)

        # Simulate old zero-masked approach: keep face 0, zero rest
        mask = jnp.zeros(6, dtype=jnp.float32).at[0].set(1.0)
        masked = field * mask.reshape(6, 1, 1)

        # The non-zero slice of the masked array equals local
        np.testing.assert_array_equal(local[0], masked[0])

    def test_rank_local_uses_less_memory(self):
        """Rank-local shape is smaller than global (6, n, n)."""
        n = 64
        layout = make_layout(0, 6, n)
        # Rank-local: (1, 64, 64) = 4096 elements
        # Global masked: (6, 64, 64) = 24576 elements
        local_size = 1
        for d in layout.local_shape_2d:
            local_size *= d
        global_size = 6 * n * n
        assert local_size < global_size
        assert local_size == n * n  # exactly 1/6th
