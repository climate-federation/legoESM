"""Tests for async halo exchange utilities (single-process, local backend).

All tests use the local halo backend (no MPI).  They validate the
interior/boundary mask construction, the split-compute pattern, and
merging of results.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.parallel.async_halo import (
    InteriorBoundaryMasks,
    create_interior_boundary_masks,
    split_interior_boundary,
    merge_interior_boundary,
    boundary_slices,
    interior_slice,
    extract_interior_padded,
    build_boundary_stencil_padded,
    _pad_local_only,
    create_overlap_context,
    OverlapContext,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FACES = 6
N = 8  # grid points per face edge


# ===========================================================================
# TestInteriorBoundaryMasks
# ===========================================================================

class TestInteriorBoundaryMasks:
    """Tests for create_interior_boundary_masks."""

    def test_mask_shapes(self):
        """Masks have shape (n, n)."""
        masks = create_interior_boundary_masks(N, halo_width=1)
        assert masks.interior.shape == (N, N)
        assert masks.boundary.shape == (N, N)

    def test_mask_shapes_halo2(self):
        """Masks with halo_width=2 have correct shape."""
        masks = create_interior_boundary_masks(N, halo_width=2)
        assert masks.interior.shape == (N, N)
        assert masks.boundary.shape == (N, N)

    def test_interior_mask_true_in_center(self):
        """Interior mask is True for points away from edges."""
        masks = create_interior_boundary_masks(N, halo_width=1)
        interior = np.asarray(masks.interior)

        # Center point should be interior
        assert interior[N // 2, N // 2] == True
        # (1,1) should be interior for halo_width=1 with N=8
        assert interior[1, 1] == True
        # Interior should be True in the [1:7, 1:7] block for N=8, h=1
        assert np.all(interior[1:7, 1:7])

    def test_boundary_mask_at_edges(self):
        """Boundary mask is True at edges."""
        masks = create_interior_boundary_masks(N, halo_width=1)
        boundary = np.asarray(masks.boundary)

        # All edge cells should be boundary
        assert boundary[0, 0] == True
        assert boundary[0, N - 1] == True
        assert boundary[N - 1, 0] == True
        assert boundary[N - 1, N - 1] == True
        # Top row is all boundary
        assert np.all(boundary[0, :])
        # Bottom row
        assert np.all(boundary[N - 1, :])
        # Left column
        assert np.all(boundary[:, 0])
        # Right column
        assert np.all(boundary[:, N - 1])

    def test_boundary_is_complement_of_interior(self):
        """boundary = ~interior, they are complementary."""
        masks = create_interior_boundary_masks(N, halo_width=1)
        interior = np.asarray(masks.interior)
        boundary = np.asarray(masks.boundary)

        # They should be complements
        np.testing.assert_array_equal(boundary, ~interior)

    def test_interior_count(self):
        """Interior has (n-2h)^2 points, boundary has n^2 - (n-2h)^2."""
        h = 1
        masks = create_interior_boundary_masks(N, halo_width=h)
        n_interior = int(np.asarray(masks.interior).sum())
        n_boundary = int(np.asarray(masks.boundary).sum())

        expected_interior = (N - 2 * h) ** 2
        expected_boundary = N * N - expected_interior

        assert n_interior == expected_interior
        assert n_boundary == expected_boundary

    def test_interior_count_halo2(self):
        """Interior with halo_width=2 has (n-4)^2 points."""
        h = 2
        masks = create_interior_boundary_masks(N, halo_width=h)
        n_interior = int(np.asarray(masks.interior).sum())

        expected = (N - 2 * h) ** 2
        assert n_interior == expected

    def test_stored_attributes(self):
        """Masks store halo_width and n correctly."""
        masks = create_interior_boundary_masks(N, halo_width=2)
        assert masks.halo_width == 2
        assert masks.n == N

    def test_invalid_halo_width_zero(self):
        """halo_width < 1 raises ValueError."""
        with pytest.raises(ValueError, match="halo_width must be >= 1"):
            create_interior_boundary_masks(N, halo_width=0)

    def test_grid_too_small_for_halo(self):
        """Grid too small for halo_width raises ValueError."""
        with pytest.raises(ValueError, match="too small"):
            create_interior_boundary_masks(3, halo_width=2)


# ===========================================================================
# TestSplitCompute
# ===========================================================================

class TestSplitCompute:
    """Tests for split-compute pattern: split, compute, merge."""

    def test_split_preserves_total(self):
        """Split + add should recover original field everywhere."""
        masks = create_interior_boundary_masks(N, halo_width=1)
        field = jnp.ones((N_FACES, N, N)) * 7.0

        interior, boundary = split_interior_boundary(field, masks)

        # Interior + boundary should reconstruct the original
        reconstructed = interior + boundary
        np.testing.assert_allclose(
            np.asarray(reconstructed), np.asarray(field), atol=1e-14
        )

    def test_split_interior_zeros_boundary(self):
        """Interior part has zeros at boundary points."""
        masks = create_interior_boundary_masks(N, halo_width=1)
        field = jnp.ones((N_FACES, N, N)) * 3.0

        interior, _ = split_interior_boundary(field, masks)
        interior_np = np.asarray(interior)

        # Boundary points should be zero
        boundary_mask = np.asarray(masks.boundary)
        assert np.all(interior_np[:, boundary_mask] == 0.0)

    def test_split_boundary_zeros_interior(self):
        """Boundary part has zeros at interior points."""
        masks = create_interior_boundary_masks(N, halo_width=1)
        field = jnp.ones((N_FACES, N, N)) * 3.0

        _, boundary = split_interior_boundary(field, masks)
        boundary_np = np.asarray(boundary)

        interior_mask = np.asarray(masks.interior)
        assert np.all(boundary_np[:, interior_mask] == 0.0)

    def test_split_3d_field(self):
        """Split works for 3D fields (6, n, n, nlev)."""
        nlev = 5
        masks = create_interior_boundary_masks(N, halo_width=1)
        field = jnp.ones((N_FACES, N, N, nlev)) * 5.0

        interior, boundary = split_interior_boundary(field, masks)

        assert interior.shape == (N_FACES, N, N, nlev)
        assert boundary.shape == (N_FACES, N, N, nlev)

        # Sum should recover original
        reconstructed = interior + boundary
        np.testing.assert_allclose(
            np.asarray(reconstructed), np.asarray(field), atol=1e-14
        )

    def test_interior_only_computation(self):
        """Interior-only computation matches expected region."""
        masks = create_interior_boundary_masks(N, halo_width=1)
        field = jnp.arange(N_FACES * N * N, dtype=jnp.float64).reshape(
            N_FACES, N, N
        )

        interior, _ = split_interior_boundary(field, masks)

        # Interior should have original values in center, zeros at edges
        interior_np = np.asarray(interior)
        field_np = np.asarray(field)

        # Interior region [1:7, 1:7] should match original
        np.testing.assert_array_equal(
            interior_np[:, 1:7, 1:7], field_np[:, 1:7, 1:7]
        )

    def test_merge_selects_correctly(self):
        """merge_interior_boundary picks from the right source."""
        masks = create_interior_boundary_masks(N, halo_width=1)

        # Interior result: 10s everywhere
        interior_result = jnp.full((N_FACES, N, N), 10.0)
        # Boundary result: 20s everywhere
        boundary_result = jnp.full((N_FACES, N, N), 20.0)

        merged = merge_interior_boundary(
            interior_result, boundary_result, masks
        )
        merged_np = np.asarray(merged)

        # Interior points should be 10
        interior_mask = np.asarray(masks.interior)
        assert np.all(merged_np[:, interior_mask] == 10.0)

        # Boundary points should be 20
        boundary_mask = np.asarray(masks.boundary)
        assert np.all(merged_np[:, boundary_mask] == 20.0)

    def test_merge_3d_field(self):
        """merge works for 3D fields (6, n, n, nlev)."""
        nlev = 5
        masks = create_interior_boundary_masks(N, halo_width=1)

        interior_result = jnp.full((N_FACES, N, N, nlev), 10.0)
        boundary_result = jnp.full((N_FACES, N, N, nlev), 20.0)

        merged = merge_interior_boundary(
            interior_result, boundary_result, masks
        )

        assert merged.shape == (N_FACES, N, N, nlev)
        merged_np = np.asarray(merged)
        interior_mask = np.asarray(masks.interior)
        assert np.all(merged_np[:, interior_mask, :] == 10.0)


# ===========================================================================
# TestBoundarySlices
# ===========================================================================

class TestBoundarySlices:
    """Tests for boundary_slices and interior_slice helpers."""

    def test_boundary_slices_keys(self):
        """boundary_slices returns west/east/south/north."""
        slices = boundary_slices(N, halo_width=1)
        assert set(slices.keys()) == {"west", "east", "south", "north"}

    def test_boundary_slices_coverage(self):
        """All boundary cells are covered by the four strips."""
        h = 1
        slices = boundary_slices(N, halo_width=h)

        covered = np.zeros((N, N), dtype=bool)
        for _, (si, sj) in slices.items():
            covered[si, sj] = True

        # All edge cells should be covered
        assert covered[0, 0]
        assert covered[N - 1, N - 1]
        assert covered[0, N - 1]
        assert covered[N - 1, 0]

    def test_interior_slice_shape(self):
        """interior_slice returns correct interior region."""
        h = 1
        si, sj = interior_slice(N, halo_width=h)

        test = np.zeros((N, N))
        test[si, sj] = 1
        assert test.sum() == (N - 2 * h) ** 2

    def test_interior_slice_halo2(self):
        """interior_slice with halo_width=2."""
        h = 2
        si, sj = interior_slice(N, halo_width=h)

        test = np.zeros((N, N))
        test[si, sj] = 1
        assert test.sum() == (N - 2 * h) ** 2


# ===========================================================================
# TestPaddedDomainUtils
# ===========================================================================

class TestPaddedDomainUtils:
    """Tests for extract_interior_padded and build_boundary_stencil_padded."""

    def test_extract_interior_padded_shape(self):
        """extract_interior_padded strips halo to (6, n, n)."""
        h = 1
        padded = jnp.ones((N_FACES, N + 2 * h, N + 2 * h))
        result = extract_interior_padded(padded, halo_width=h)
        assert result.shape == (N_FACES, N, N)

    def test_extract_interior_padded_values(self):
        """Extracted interior values match the center of padded array."""
        h = 1
        data = jnp.arange(N_FACES * (N + 2) * (N + 2), dtype=jnp.float64)
        padded = data.reshape(N_FACES, N + 2, N + 2)
        result = extract_interior_padded(padded, halo_width=h)

        np.testing.assert_array_equal(
            np.asarray(result), np.asarray(padded[:, h:-h, h:-h])
        )

    def test_build_boundary_stencil_padded(self):
        """Interior region in padded domain is zeroed out."""
        h = 1
        masks = create_interior_boundary_masks(N, halo_width=h)
        padded = jnp.ones((N_FACES, N + 2 * h, N + 2 * h))

        result = build_boundary_stencil_padded(padded, masks, halo_width=h)

        result_np = np.asarray(result)
        # Interior block (within the non-halo region) should be zeroed
        # where masks.interior is True
        inner = result_np[:, h:-h, h:-h]
        interior_mask = np.asarray(masks.interior)
        assert np.all(inner[:, interior_mask] == 0.0)

        # Boundary points in the inner region should remain
        boundary_mask = np.asarray(masks.boundary)
        assert np.all(inner[:, boundary_mask] == 1.0)


# ===========================================================================
# TestPadLocalOnly
# ===========================================================================

class TestPadLocalOnly:
    """Tests for _pad_local_only edge extrapolation."""

    def test_pad_local_only_shape(self):
        """Padding increases shape by 2*h in each spatial dim."""
        h = 1
        field = jnp.ones((N_FACES, N, N))
        padded = _pad_local_only(field, halo_width=h)
        assert padded.shape == (N_FACES, N + 2 * h, N + 2 * h)

    def test_pad_local_only_shape_h2(self):
        """Padding with halo_width=2."""
        h = 2
        field = jnp.ones((N_FACES, N, N))
        padded = _pad_local_only(field, halo_width=h)
        assert padded.shape == (N_FACES, N + 2 * h, N + 2 * h)

    def test_pad_local_only_interior_preserved(self):
        """Interior values are preserved after padding."""
        h = 1
        field = jnp.arange(N_FACES * N * N, dtype=jnp.float64).reshape(
            N_FACES, N, N
        )
        padded = _pad_local_only(field, halo_width=h)

        np.testing.assert_array_equal(
            np.asarray(padded[:, h:-h, h:-h]), np.asarray(field)
        )

    def test_pad_local_only_edge_replication(self):
        """Halo values are edge-replicated (mode='edge')."""
        h = 1
        field = jnp.arange(N_FACES * N * N, dtype=jnp.float64).reshape(
            N_FACES, N, N
        )
        padded = _pad_local_only(field, halo_width=h)

        # Left halo should replicate leftmost column
        np.testing.assert_array_equal(
            np.asarray(padded[:, h:-h, 0]),
            np.asarray(field[:, :, 0]),
        )
        # Right halo should replicate rightmost column
        np.testing.assert_array_equal(
            np.asarray(padded[:, h:-h, -1]),
            np.asarray(field[:, :, -1]),
        )

    def test_pad_local_only_face_dim_unchanged(self):
        """Face dimension is not padded."""
        h = 1
        field = jnp.ones((N_FACES, N, N))
        padded = _pad_local_only(field, halo_width=h)
        assert padded.shape[0] == N_FACES


# ===========================================================================
# TestOverlapContext
# ===========================================================================

class TestOverlapContext:
    """Tests for the precomputed OverlapContext."""

    def test_create_overlap_context(self):
        """Context has correct attributes."""
        ctx = create_overlap_context(N, halo_width=1)
        assert isinstance(ctx, OverlapContext)
        assert ctx.halo_width == 1
        assert ctx.n == N
        assert ctx.interp_offsets is None
        assert isinstance(ctx.masks, InteriorBoundaryMasks)

    def test_context_masks_match_direct(self):
        """Context masks match directly created masks."""
        ctx = create_overlap_context(N, halo_width=2)
        direct = create_interior_boundary_masks(N, halo_width=2)

        np.testing.assert_array_equal(
            np.asarray(ctx.masks.interior),
            np.asarray(direct.interior),
        )
        np.testing.assert_array_equal(
            np.asarray(ctx.masks.boundary),
            np.asarray(direct.boundary),
        )


# ===========================================================================
# TestJITCompatibility
# ===========================================================================

class TestJITCompatibility:
    """Verify that key operations work under jax.jit."""

    def test_split_merge_jittable(self):
        """split and merge work under jax.jit."""
        masks = create_interior_boundary_masks(N, halo_width=1)
        field = jnp.ones((N_FACES, N, N)) * 5.0

        @jax.jit
        def split_and_merge(f):
            interior, boundary = split_interior_boundary(f, masks)
            return merge_interior_boundary(interior, boundary, masks)

        result = split_and_merge(field)
        np.testing.assert_allclose(
            np.asarray(result), np.asarray(field), atol=1e-14
        )

    def test_pad_local_only_jittable(self):
        """_pad_local_only works under jax.jit."""
        field = jnp.ones((N_FACES, N, N))

        @jax.jit
        def pad(f):
            return _pad_local_only(f, halo_width=1)

        result = pad(field)
        assert result.shape == (N_FACES, N + 2, N + 2)

    def test_extract_interior_jittable(self):
        """extract_interior_padded works under jax.jit."""
        padded = jnp.ones((N_FACES, N + 2, N + 2))

        @jax.jit
        def extract(p):
            return extract_interior_padded(p, halo_width=1)

        result = extract(padded)
        assert result.shape == (N_FACES, N, N)
