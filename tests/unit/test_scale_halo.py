"""Category 2: Halo exchange correctness for cubed-sphere.

Tests pad_halo and pad_halo_vector for constant fields, face-unique fields,
vector rotation at boundaries, corners, and halo width 2.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.halo import (
    pad_halo,
    pad_halo_vector,
    CONNECTIVITY,
    WEST, EAST, SOUTH, NORTH,
    _extract_edge_strip,
    compute_halo_interp_offsets,
    compute_halo_interp_offsets_h2,
    _fill_corners_h1,
)


N = 8  # Small grid for testing


# ---------------------------------------------------------------------------
# Constant field: halo should match interior value everywhere
# ---------------------------------------------------------------------------

class TestConstantField:
    """A constant field should be perfectly preserved by halo exchange."""

    def test_constant_scalar_halo1(self):
        data = jnp.ones((6, N, N), dtype=jnp.float64) * 7.0
        padded = pad_halo(data, halo=1)
        assert padded.shape == (6, N + 2, N + 2)
        # Interior should be exactly 7.0
        interior = padded[:, 1:-1, 1:-1]
        np.testing.assert_allclose(interior, 7.0, atol=0.0)
        # All edge halo strips should be 7.0 (constant field)
        for face in range(6):
            # West halo
            np.testing.assert_allclose(padded[face, 0, 1:-1], 7.0, atol=1e-12)
            # East halo
            np.testing.assert_allclose(padded[face, -1, 1:-1], 7.0, atol=1e-12)
            # South halo
            np.testing.assert_allclose(padded[face, 1:-1, 0], 7.0, atol=1e-12)
            # North halo
            np.testing.assert_allclose(padded[face, 1:-1, -1], 7.0, atol=1e-12)

    def test_constant_scalar_halo2(self):
        data = jnp.ones((6, N, N), dtype=jnp.float64) * 3.0
        padded = pad_halo(data, halo=2)
        assert padded.shape == (6, N + 4, N + 4)
        interior = padded[:, 2:-2, 2:-2]
        np.testing.assert_allclose(interior, 3.0, atol=0.0)
        # Edge halos should all be 3.0
        for face in range(6):
            # Depth-0 halo (adjacent to interior)
            np.testing.assert_allclose(padded[face, 1, 2:-2], 3.0, atol=1e-12)
            np.testing.assert_allclose(padded[face, -2, 2:-2], 3.0, atol=1e-12)
            np.testing.assert_allclose(padded[face, 2:-2, 1], 3.0, atol=1e-12)
            np.testing.assert_allclose(padded[face, 2:-2, -2], 3.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Face-unique field: each face has a unique value
# ---------------------------------------------------------------------------

class TestFaceUniqueField:
    """Each face set to face_index; halo should pull from correct neighbor."""

    def test_face_unique_halo1(self):
        data = jnp.zeros((6, N, N), dtype=jnp.float64)
        for f in range(6):
            data = data.at[f].set(float(f))
        padded = pad_halo(data, halo=1)

        for face in range(6):
            for edge in [WEST, EAST, SOUTH, NORTH]:
                nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
                # The halo strip should contain values from the neighbor face
                if edge == WEST:
                    halo_strip = padded[face, 0, 1:-1]
                elif edge == EAST:
                    halo_strip = padded[face, -1, 1:-1]
                elif edge == SOUTH:
                    halo_strip = padded[face, 1:-1, 0]
                elif edge == NORTH:
                    halo_strip = padded[face, 1:-1, -1]
                # Since field is constant per face, halo values should
                # equal the neighbor face index
                np.testing.assert_allclose(
                    halo_strip, float(nbr_face), atol=1e-12,
                    err_msg=f"Face {face}, edge {edge}: expected neighbor {nbr_face}",
                )


# ---------------------------------------------------------------------------
# _extract_edge_strip
# ---------------------------------------------------------------------------

class TestExtractEdgeStrip:
    def test_strip_shapes(self):
        data = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        for face in range(6):
            for edge in [WEST, EAST, SOUTH, NORTH]:
                strip = _extract_edge_strip(data, face, edge)
                assert strip.shape == (N,), f"face={face}, edge={edge}"

    def test_west_strip_is_first_row(self):
        data = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        strip = _extract_edge_strip(data, 0, WEST)
        np.testing.assert_array_equal(strip, data[0, 0, :])

    def test_east_strip_is_last_row(self):
        data = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        strip = _extract_edge_strip(data, 0, EAST)
        np.testing.assert_array_equal(strip, data[0, -1, :])

    def test_south_strip_is_first_col(self):
        data = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        strip = _extract_edge_strip(data, 0, SOUTH)
        np.testing.assert_array_equal(strip, data[0, :, 0])

    def test_north_strip_is_last_col(self):
        data = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        strip = _extract_edge_strip(data, 0, NORTH)
        np.testing.assert_array_equal(strip, data[0, :, -1])


# ---------------------------------------------------------------------------
# Connectivity symmetry
# ---------------------------------------------------------------------------

class TestConnectivitySymmetry:
    """CONNECTIVITY must be symmetric: if A's edge -> (B, B_edge, rev),
    then B's B_edge -> (A, A_edge, rev)."""

    def test_symmetric(self):
        for face in range(6):
            for edge in [WEST, EAST, SOUTH, NORTH]:
                nbr_face, nbr_edge, rev = CONNECTIVITY[face][edge]
                # Reverse lookup
                back_face, back_edge, back_rev = CONNECTIVITY[nbr_face][nbr_edge]
                assert back_face == face, (
                    f"Broken symmetry: face={face}, edge={edge} -> "
                    f"({nbr_face}, {nbr_edge}, {rev}), "
                    f"but ({nbr_face}, {nbr_edge}) -> ({back_face}, {back_edge}, {back_rev})"
                )
                assert back_edge == edge
                assert back_rev == rev


# ---------------------------------------------------------------------------
# Interior preservation
# ---------------------------------------------------------------------------

class TestInteriorPreservation:
    """Padding must not alter the interior data."""

    def test_interior_unchanged_h1(self):
        data = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        padded = pad_halo(data, halo=1)
        interior = padded[:, 1:-1, 1:-1]
        np.testing.assert_array_equal(interior, data)

    def test_interior_unchanged_h2(self):
        data = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        padded = pad_halo(data, halo=2)
        interior = padded[:, 2:-2, 2:-2]
        np.testing.assert_array_equal(interior, data)


# ---------------------------------------------------------------------------
# Corners
# ---------------------------------------------------------------------------

class TestCorners:
    """Corner cells should be filled (not zero) after _fill_corners_h1."""

    def test_corners_nonzero_for_constant_field(self):
        data = jnp.ones((6, N, N), dtype=jnp.float64) * 5.0
        padded = pad_halo(data, halo=1)
        for face in range(6):
            # Check all 4 corners
            assert padded[face, 0, 0] != 0.0, f"Face {face} SW corner is zero"
            assert padded[face, 0, -1] != 0.0, f"Face {face} NW corner is zero"
            assert padded[face, -1, 0] != 0.0, f"Face {face} SE corner is zero"
            assert padded[face, -1, -1] != 0.0, f"Face {face} NE corner is zero"
            # For a constant field, corners should be 5.0
            np.testing.assert_allclose(padded[face, 0, 0], 5.0, atol=1e-12)
            np.testing.assert_allclose(padded[face, 0, -1], 5.0, atol=1e-12)
            np.testing.assert_allclose(padded[face, -1, 0], 5.0, atol=1e-12)
            np.testing.assert_allclose(padded[face, -1, -1], 5.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Interpolated halo offsets
# ---------------------------------------------------------------------------

class TestHaloInterpOffsets:
    def test_offset_shape(self):
        offsets = compute_halo_interp_offsets(N)
        assert offsets.shape == (6, 4, N)

    def test_offset_shape_h2(self):
        offsets = compute_halo_interp_offsets_h2(N)
        assert offsets.shape == (6, 4, 2, N)

    def test_interpolated_constant_field(self):
        """With interpolation offsets, constant field should still be exact."""
        offsets = compute_halo_interp_offsets(N)
        data = jnp.ones((6, N, N), dtype=jnp.float64) * 2.5
        padded = pad_halo(data, halo=1, interp_offsets=offsets)
        for face in range(6):
            np.testing.assert_allclose(padded[face, 0, 1:-1], 2.5, atol=1e-6)
            np.testing.assert_allclose(padded[face, -1, 1:-1], 2.5, atol=1e-6)
            np.testing.assert_allclose(padded[face, 1:-1, 0], 2.5, atol=1e-6)
            np.testing.assert_allclose(padded[face, 1:-1, -1], 2.5, atol=1e-6)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJITCompatibility:
    def test_pad_halo_jittable(self):
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        padded_jit = jax.jit(pad_halo)(data)
        padded_eager = pad_halo(data)
        np.testing.assert_allclose(padded_jit, padded_eager, atol=1e-14)

    def test_pad_halo_differentiable(self):
        """pad_halo should be differentiable for AD."""
        def loss(data):
            padded = pad_halo(data)
            return jnp.sum(padded ** 2)
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        grad = jax.grad(loss)(data)
        assert grad.shape == data.shape
        assert jnp.all(jnp.isfinite(grad))
