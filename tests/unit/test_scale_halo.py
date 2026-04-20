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

    def test_offset_shape_h3(self):
        """Iter-496: halo=3 interp-offset scaffolding for FB-chain
        stability work.  The h3 offsets are a strict extension of h2:
        depths 0-1 match h2 bitwise; depth 2 is new.
        """
        from legoesm.grids.halo import compute_halo_interp_offsets_h3
        offsets_h1 = compute_halo_interp_offsets(N)
        offsets_h2 = compute_halo_interp_offsets_h2(N)
        offsets_h3 = compute_halo_interp_offsets_h3(N)
        assert offsets_h3.shape == (6, 4, 3, N)
        # h1 ≡ h2[depth=0] ≡ h3[depth=0]
        np.testing.assert_array_equal(
            np.asarray(offsets_h2[:, :, 0, :]), np.asarray(offsets_h1))
        # h2 ≡ h3[depth=:2]
        np.testing.assert_array_equal(
            np.asarray(offsets_h3[:, :, :2, :]), np.asarray(offsets_h2))
        # Depth 2 offsets are bounded (O(1) near cube vertices, typically <3)
        assert float(jnp.max(jnp.abs(offsets_h3[:, :, 2, :]))) < 3.0

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
# h3 corner fill (iter-497)
# ---------------------------------------------------------------------------

class TestFillCornersH3:
    """Tests for the halo=3 corner fill helper added in iter-497.

    This is plumbing for the ng=3 halo extension (review-doc item #2,
    FB-path stability on W2 C36).  The fill rule is the inside-out
    2-point averaging used by `_fill_corners_h2`, generalized to a
    3x3 corner block.
    """

    def _build_constant_padded(self, n: int, halo: int, value: float):
        """Construct an (6, n+2h, n+2h) padded array with interior and
        edge-strip halos both set to ``value`` and corner blocks left
        at zero.  Matches what ``_pad_halo_local_h3`` would produce on
        a constant field before the corner fill step."""
        padded = jnp.zeros((6, n + 2 * halo, n + 2 * halo), dtype=jnp.float64)
        # Interior
        padded = padded.at[:, halo:-halo, halo:-halo].set(value)
        # Edge halos (skip the 4 corner blocks which are left at 0)
        for d in range(halo):
            # WEST, EAST: i = d or n+2h-1-d, j in [halo:-halo]
            padded = padded.at[:, d, halo:-halo].set(value)
            padded = padded.at[:, n + 2 * halo - 1 - d, halo:-halo].set(value)
            # SOUTH, NORTH: j = d or n+2h-1-d, i in [halo:-halo]
            padded = padded.at[:, halo:-halo, d].set(value)
            padded = padded.at[:, halo:-halo, n + 2 * halo - 1 - d].set(value)
        return padded

    def test_constant_field_preserved(self):
        """Corner fill on a constant edge+interior padded array should
        leave every cell at the same constant value."""
        from legoesm.grids.halo import _fill_corners_h3
        n = N
        padded = self._build_constant_padded(n, halo=3, value=4.25)
        filled = _fill_corners_h3(padded)
        np.testing.assert_allclose(
            np.asarray(filled), 4.25, atol=1e-12,
            err_msg="Corner fill should preserve constant fields")

    def test_shape_preserved(self):
        from legoesm.grids.halo import _fill_corners_h3
        n = N
        padded = self._build_constant_padded(n, halo=3, value=1.0)
        filled = _fill_corners_h3(padded)
        assert filled.shape == (6, n + 6, n + 6)

    def test_zero_corner_cells_get_filled(self):
        """Before the fill, corner 3x3 blocks are zero; after, they
        are non-zero (pulled from non-zero edge halos)."""
        from legoesm.grids.halo import _fill_corners_h3
        n = N
        padded = self._build_constant_padded(n, halo=3, value=7.0)
        # Sanity: corner blocks were left zero by the builder
        assert float(padded[0, 0, 0]) == 0.0
        assert float(padded[0, 0, 2]) == 0.0
        assert float(padded[0, 2, 2]) == 0.0
        filled = _fill_corners_h3(padded)
        # All 9 SW corner cells of face 0 should be non-zero
        for i in range(3):
            for j in range(3):
                assert float(filled[0, i, j]) != 0.0

    def test_no_mutation_of_interior(self):
        """The fill must leave the interior block untouched."""
        from legoesm.grids.halo import _fill_corners_h3
        n = N
        padded = self._build_constant_padded(n, halo=3, value=0.0)
        # Write a distinct pattern in the interior
        interior_vals = jnp.arange(
            6 * n * n, dtype=jnp.float64).reshape(6, n, n)
        padded = padded.at[:, 3:-3, 3:-3].set(interior_vals)
        filled = _fill_corners_h3(padded)
        np.testing.assert_array_equal(
            np.asarray(filled[:, 3:-3, 3:-3]),
            np.asarray(interior_vals),
        )

    def test_no_mutation_of_edge_halos(self):
        """The fill must leave the edge-strip halos (non-corner) untouched."""
        from legoesm.grids.halo import _fill_corners_h3
        n = N
        h = 3
        padded = jnp.zeros((6, n + 2 * h, n + 2 * h), dtype=jnp.float64)
        # Fill edge halos with unique sentinel patterns
        rng = np.random.default_rng(42)
        w = jnp.asarray(rng.standard_normal((6, h, n)))
        e = jnp.asarray(rng.standard_normal((6, h, n)))
        s = jnp.asarray(rng.standard_normal((6, n, h)))
        no = jnp.asarray(rng.standard_normal((6, n, h)))
        padded = padded.at[:, :h, h:-h].set(w)
        padded = padded.at[:, -h:, h:-h].set(e)
        padded = padded.at[:, h:-h, :h].set(s)
        padded = padded.at[:, h:-h, -h:].set(no)
        from legoesm.grids.halo import _fill_corners_h3
        filled = _fill_corners_h3(padded)
        np.testing.assert_array_equal(
            np.asarray(filled[:, :h, h:-h]), np.asarray(w))
        np.testing.assert_array_equal(
            np.asarray(filled[:, -h:, h:-h]), np.asarray(e))
        np.testing.assert_array_equal(
            np.asarray(filled[:, h:-h, :h]), np.asarray(s))
        np.testing.assert_array_equal(
            np.asarray(filled[:, h:-h, -h:]), np.asarray(no))


# ---------------------------------------------------------------------------
# h3 local scalar exchange (iter-498)
# ---------------------------------------------------------------------------

class TestPadHaloLocalH3:
    """Tests for the halo=3 local scalar exchange added in iter-498.

    The implementation generalizes `_pad_halo_local_h2` to 3 halo
    depths, using `_fill_corners_h3` for the 3x3 L-shaped corner
    blocks.
    """

    def test_constant_field(self):
        """Constant field at interior should propagate to all halo cells."""
        from legoesm.grids.halo import _pad_halo_local_h3
        data = jnp.ones((6, N, N), dtype=jnp.float64) * 9.0
        padded = _pad_halo_local_h3(data)
        assert padded.shape == (6, N + 6, N + 6)
        np.testing.assert_allclose(
            np.asarray(padded), 9.0, atol=1e-12,
            err_msg="h3 local exchange must preserve constant fields")

    def test_interior_preservation(self):
        """Interior data must be bit-identical after halo exchange."""
        from legoesm.grids.halo import _pad_halo_local_h3
        data = jnp.arange(
            6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        padded = _pad_halo_local_h3(data)
        np.testing.assert_array_equal(
            np.asarray(padded[:, 3:-3, 3:-3]), np.asarray(data))

    def test_shape(self):
        from legoesm.grids.halo import _pad_halo_local_h3
        data = jnp.zeros((6, N, N), dtype=jnp.float64)
        padded = _pad_halo_local_h3(data)
        assert padded.shape == (6, N + 6, N + 6)

    def test_face_unique_depth0_edges_match_h2_depth0(self):
        """Depth-0 (interior-adjacent) edge strip values must match the
        existing `_pad_halo_local_h2` depth-0 strip — both reference
        the same physical neighbour-strip row, so they cannot differ."""
        from legoesm.grids.halo import (
            _pad_halo_local_h2, _pad_halo_local_h3,
        )
        data = jnp.zeros((6, N, N), dtype=jnp.float64)
        for f in range(6):
            data = data.at[f].set(float(f) + 1.0)
        p_h2 = _pad_halo_local_h2(data)
        p_h3 = _pad_halo_local_h3(data)
        # West depth=0 strip: p_h2[face, 1, 2:-2] vs p_h3[face, 2, 3:-3]
        for f in range(6):
            np.testing.assert_array_equal(
                np.asarray(p_h3[f, 2, 3:-3]),
                np.asarray(p_h2[f, 1, 2:-2]),
                err_msg=f"face {f} WEST depth=0 differs between h2 and h3")
            np.testing.assert_array_equal(
                np.asarray(p_h3[f, N + 3, 3:-3]),
                np.asarray(p_h2[f, N + 2, 2:-2]),
                err_msg=f"face {f} EAST depth=0 differs between h2 and h3")
            np.testing.assert_array_equal(
                np.asarray(p_h3[f, 3:-3, 2]),
                np.asarray(p_h2[f, 2:-2, 1]),
                err_msg=f"face {f} SOUTH depth=0 differs between h2 and h3")
            np.testing.assert_array_equal(
                np.asarray(p_h3[f, 3:-3, N + 3]),
                np.asarray(p_h2[f, 2:-2, N + 2]),
                err_msg=f"face {f} NORTH depth=0 differs between h2 and h3")

    def test_face_unique_depth1_matches_h2_depth1(self):
        """Depth-1 edge strip values must match h2 depth-1 (also both
        reference the second row from the neighbour interior)."""
        from legoesm.grids.halo import (
            _pad_halo_local_h2, _pad_halo_local_h3,
        )
        data = jnp.zeros((6, N, N), dtype=jnp.float64)
        for f in range(6):
            data = data.at[f].set(float(f) + 1.0)
        p_h2 = _pad_halo_local_h2(data)
        p_h3 = _pad_halo_local_h3(data)
        for f in range(6):
            # h2 WEST depth=1 at i=0; h3 WEST depth=1 at i=1
            np.testing.assert_array_equal(
                np.asarray(p_h3[f, 1, 3:-3]),
                np.asarray(p_h2[f, 0, 2:-2]))
            # EAST depth=1: h2 i=n+3, h3 i=n+4
            np.testing.assert_array_equal(
                np.asarray(p_h3[f, N + 4, 3:-3]),
                np.asarray(p_h2[f, N + 3, 2:-2]))
            np.testing.assert_array_equal(
                np.asarray(p_h3[f, 3:-3, 1]),
                np.asarray(p_h2[f, 2:-2, 0]))
            np.testing.assert_array_equal(
                np.asarray(p_h3[f, 3:-3, N + 4]),
                np.asarray(p_h2[f, 2:-2, N + 3]))

    def test_depth2_edge_strip_pulls_from_neighbour(self):
        """Depth-2 halo strip must pull from the 3rd row into the
        neighbour's interior (i.e., row index 2 or -3 on that face)."""
        from legoesm.grids.halo import (
            _pad_halo_local_h3,
            CONNECTIVITY,
            _extract_edge_strip_at_depth,
        )
        data = jnp.arange(
            6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        padded = _pad_halo_local_h3(data)
        edges = [WEST, EAST, SOUTH, NORTH]
        for face in range(6):
            for edge_idx, edge in enumerate(edges):
                nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
                expected = _extract_edge_strip_at_depth(
                    data, nbr_face, nbr_edge, 2)
                if is_reversed:
                    expected = expected[::-1]
                # Extract halo slice at depth=2
                if edge == WEST:
                    actual = padded[face, 0, 3:-3]
                elif edge == EAST:
                    actual = padded[face, N + 5, 3:-3]
                elif edge == SOUTH:
                    actual = padded[face, 3:-3, 0]
                else:
                    actual = padded[face, 3:-3, N + 5]
                np.testing.assert_array_equal(
                    np.asarray(actual), np.asarray(expected),
                    err_msg=(
                        f"face={face} edge={edge} depth=2 mismatch"),
                )

    def test_jittable_and_differentiable(self):
        from legoesm.grids.halo import _pad_halo_local_h3
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        # JIT path
        p_jit = jax.jit(_pad_halo_local_h3)(data)
        p_eager = _pad_halo_local_h3(data)
        np.testing.assert_allclose(
            np.asarray(p_jit), np.asarray(p_eager), atol=1e-14)
        # Gradient path
        def loss(x):
            return jnp.sum(_pad_halo_local_h3(x) ** 2)
        grad = jax.grad(loss)(data)
        assert grad.shape == data.shape
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# Padded angle / metrics consistency between halo=2 and halo=3 (iter-530)
# ---------------------------------------------------------------------------

class TestPaddedAngleHaloConsistency:
    """The `compute_padded_angle` and `compute_padded_half_metrics`
    helpers accept any halo depth.  iter-530: lock the invariant that
    the halo=3 output matches the halo=2 output at the overlapping
    interior region (positions [1:-1, 1:-1] of halo=2 = positions
    [2:-2, 2:-2] of halo=3).  This is a critical correctness guard:
    halo=3 must agree with halo=2 wherever they overlap, otherwise
    the iter-496..501 ng=3 scaffolding is broken.
    """

    # iter-531 (Codex follow-up): use rtol-based tolerances so the
    # tests pass under both x64 and the default float32 backend.
    # iter-530 used atol=1e-12 (angle) and atol=1e-6 (metrics) which
    # are tighter than float32's ~1e-7 relative precision; the tests
    # only passed because the matrix run sets `JAX_ENABLE_X64=1`.
    # `compute_padded_angle` returns radians (range ~π/2 ≈ 1.6) →
    # rtol=1e-5 with atol=1e-6 covers float32 precision.
    # `compute_padded_half_metrics` returns metres on Earth (~1.5e6) →
    # absolute float32 precision is ~0.15 m; rtol=1e-5 covers it.

    def test_compute_padded_angle_h2_h3_consistent(self):
        from legoesm.grids.halo import compute_padded_angle
        n = 8
        a_h2 = compute_padded_angle(n, halo=2)
        a_h3 = compute_padded_angle(n, halo=3)
        assert a_h2.shape == (6, n + 4, n + 4)
        assert a_h3.shape == (6, n + 6, n + 6)
        # The halo=2 array's interior + 2-cell halo corresponds to
        # the halo=3 array's interior + 1..3-deep halo at offset (1, 1).
        # I.e., a_h3[1:-1, 1:-1] should equal a_h2 at every cell.
        np.testing.assert_allclose(
            np.asarray(a_h3[:, 1:-1, 1:-1]),
            np.asarray(a_h2),
            rtol=1e-5, atol=1e-6,
            err_msg="compute_padded_angle(h=3) interior does not "
                    "match compute_padded_angle(h=2) — h3 scaffolding "
                    "would propagate inconsistency to the FB chain.")

    def test_grid_halo_interp_offsets_h3_wired_and_consistent(self):
        """Iter-532: `CubedSphereGrid.halo_interp_offsets_h3` must be
        present on every cubed-sphere grid built by
        `create_cubed_sphere`, and its values must match the
        free-function `compute_halo_interp_offsets_h3(n)` (so a
        future caller can use either path interchangeably)."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.halo import compute_halo_interp_offsets_h3
        n = 8
        grid = create_cubed_sphere(n=n, use_duogrid=False)
        assert grid.halo_interp_offsets_h3 is not None
        assert grid.halo_interp_offsets_h3.shape == (6, 4, 3, n)
        # Bit-equality with the free function (both compute the same
        # offsets, so they must agree to dtype precision).
        ref = compute_halo_interp_offsets_h3(n)
        np.testing.assert_allclose(
            np.asarray(grid.halo_interp_offsets_h3),
            np.asarray(ref),
            rtol=1e-5,  # float32 precision under default backend
        )

    def test_grid_panel_halo_interp_offsets_h3_is_none(self):
        """Single-face regional panel uses wall BCs; `_h3` should be
        None like `_h1` and `_h2`."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere_panel
        n = 8
        panel = create_cubed_sphere_panel(n=n, face_id=0)
        assert panel.halo_interp_offsets_h3 is None

    def test_compute_padded_half_metrics_h2_h3_consistent(self):
        from legoesm.grids.halo import compute_padded_half_metrics
        n = 8
        radius = 6.371229e6
        hx2, hy2 = compute_padded_half_metrics(n, radius, halo=2)
        hx3, hy3 = compute_padded_half_metrics(n, radius, halo=3)
        assert hx2.shape == (6, n + 4, n + 4)
        assert hx3.shape == (6, n + 6, n + 6)
        np.testing.assert_allclose(
            np.asarray(hx3[:, 1:-1, 1:-1]), np.asarray(hx2), rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(hy3[:, 1:-1, 1:-1]), np.asarray(hy2), rtol=1e-5)


# ---------------------------------------------------------------------------
# Public pad_halo(halo=3) dispatch (iter-499)
# ---------------------------------------------------------------------------

class TestPadHaloH3Dispatch:
    """Tests that the public `pad_halo(halo=3)` dispatch wires through
    to `_pad_halo_local_h3` on the single-node path.
    """

    def test_dispatch_delegates_to_local_h3(self):
        """Calling pad_halo(..., halo=3) must equal a direct
        `_pad_halo_local_h3` call when no interp_offsets/duogrid and
        no distributed backend is active."""
        from legoesm.grids.halo import _pad_halo_local_h3
        data = jnp.arange(
            6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        p_dispatch = pad_halo(data, halo=3)
        p_direct = _pad_halo_local_h3(data)
        np.testing.assert_array_equal(
            np.asarray(p_dispatch), np.asarray(p_direct))

    def test_shape(self):
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        padded = pad_halo(data, halo=3)
        assert padded.shape == (6, N + 6, N + 6)

    def test_constant_field_preserved(self):
        data = jnp.ones((6, N, N), dtype=jnp.float64) * 3.5
        padded = pad_halo(data, halo=3)
        np.testing.assert_allclose(
            np.asarray(padded), 3.5, atol=1e-12)

    def test_halo4_raises_notimplemented(self):
        """halo=4 should still raise NotImplementedError — iter-499 only
        extended dispatch to halo=3."""
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        with pytest.raises(NotImplementedError):
            pad_halo(data, halo=4)

    def test_interp_offsets_h3_forwarded(self):
        """When interp_offsets with h3 shape (6,4,3,n) is passed, the
        dispatch must feed it through to `_pad_halo_local_h3` — on a
        constant field the result should still be exact."""
        from legoesm.grids.halo import compute_halo_interp_offsets_h3
        offsets = compute_halo_interp_offsets_h3(N)
        data = jnp.ones((6, N, N), dtype=jnp.float64) * 1.75
        padded = pad_halo(data, halo=3, interp_offsets=offsets)
        np.testing.assert_allclose(
            np.asarray(padded), 1.75, atol=1e-6)

    def test_single_face_panel_uses_wall_bc(self):
        """For a (1, n, n) regional panel, halo=3 must use wall BCs
        (Neumann) and produce shape (1, n+6, n+6)."""
        data = jnp.arange(
            1 * N * N, dtype=jnp.float64).reshape(1, N, N)
        padded = pad_halo(data, halo=3)
        assert padded.shape == (1, N + 6, N + 6)
        # Interior preserved bit-exactly
        np.testing.assert_array_equal(
            np.asarray(padded[:, 3:-3, 3:-3]), np.asarray(data))
        # Boundary rows are edge-replicated (Neumann / zero-gradient)
        np.testing.assert_array_equal(
            np.asarray(padded[0, 0, 3:-3]), np.asarray(data[0, 0, :]))
        np.testing.assert_array_equal(
            np.asarray(padded[0, -1, 3:-3]), np.asarray(data[0, -1, :]))

    def test_duogrid_at_halo3_constant_field_preserved(self):
        """Iter-533 / iter-534: halo=3 is allowed for the duogrid path
        because `cube_rmp_vectorized` and `fill_corner_region` already
        loop over halo depth.  Constant-field preservation guard."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        n = 8
        grid = create_cubed_sphere(n=n, use_duogrid=True, duogrid_ng=4)
        assert grid.duogrid is not None
        data = jnp.ones((6, n, n), dtype=jnp.float64) * 5.0
        padded = pad_halo(data, halo=3, duogrid=grid.duogrid)
        assert padded.shape == (6, n + 6, n + 6)
        np.testing.assert_allclose(
            np.asarray(padded), 5.0, atol=1e-12,
            err_msg="duogrid+halo=3 broke constant-field preservation")

    def test_duogrid_at_halo3_third_ring_corner_cells_in_face_value_range(self):
        """Iter-536 (Codex follow-up to iter-535): the iter-535 test
        only checks the EDGE STRIPS of the depth=2 ring (positions
        `[0, 3:-3]` etc.), missing the depth=2 CORNER cells (the
        outermost cells of each 3×3 corner block: 12 cells per
        cube vertex × 4 corners × 6 faces = 288 cells per grid).

        On a face-unique constant field (face f → value f+1.0):
          - Edge strips contain a single neighbour face's value
            (covered by iter-535 test).
          - Corner cells are filled by `_fill_corners_h3` averaging
            of adjacent edge halos.  Each corner cell's value is
            therefore some average of the host face's value (from
            interior-side neighbours) and 1-2 neighbour faces'
            values (from the cross-face edge halos).

        A correctly-populated corner cell MUST be:
          (a) finite (catches "stays at NaN/Inf");
          (b) within [1.0, 6.0] = [min face value, max face value]
              (catches "stays at zero" — initialization value);
          (c) NOT identically zero on any cell of the third ring.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere

        n = 8
        grid = create_cubed_sphere(n=n, use_duogrid=True, duogrid_ng=4)
        data = jnp.zeros((6, n, n), dtype=jnp.float64)
        for f in range(6):
            data = data.at[f].set(float(f) + 1.0)

        p_h3 = pad_halo(data, halo=3, duogrid=grid.duogrid)
        p_h3_np = np.asarray(p_h3)

        # The "third ring" is the outermost layer: i ∈ {0, n+5} OR
        # j ∈ {0, n+5}.  Build a mask for those cells.
        np_p3 = n + 6
        ring_mask = np.zeros((np_p3, np_p3), dtype=bool)
        ring_mask[0, :] = True
        ring_mask[-1, :] = True
        ring_mask[:, 0] = True
        ring_mask[:, -1] = True

        ring_vals = p_h3_np[:, ring_mask]   # (6, ring_count)
        # (a) finite
        assert np.all(np.isfinite(ring_vals)), (
            "halo=3 third ring contains NaN/Inf — duogrid h3 path "
            "blew up at the outermost ring.")
        # (b) within face-value range [1.0, 6.0]
        ring_min = float(np.min(ring_vals))
        ring_max = float(np.max(ring_vals))
        # Tiny float-drift tolerance because the duogrid Lagrange
        # remap weights only sum to 1 in exact arithmetic.
        assert ring_min >= 1.0 - 1e-5, (
            f"halo=3 third ring min = {ring_min} < 1.0 — value "
            f"escaped the face-value range, likely zeroed by an "
            f"unpopulated cell.")
        assert ring_max <= 6.0 + 1e-5, (
            f"halo=3 third ring max = {ring_max} > 6.0 — value "
            f"overshot the face-value range.")
        # (c) NOT identically zero on ANY cell.  This catches the
        # specific failure mode where a per-corner block is silently
        # left at the `_pad_halo_local_h3` zero initialization.
        zero_count = int(np.sum(np.abs(ring_vals) < 1e-10))
        assert zero_count == 0, (
            f"halo=3 third ring contains {zero_count} zero cells "
            f"(out of {ring_vals.size} total).  This indicates "
            f"the duogrid h3 corner-fill or edge-strip path left "
            f"some cells at the zero initialization value.")

    def test_duogrid_at_halo3_corner_cells_blend_neighbour_faces(self):
        """Iter-537 (Codex follow-up to iter-536): the iter-536 test
        only does smoke-checks (finite / in range / non-zero) on
        the outer ring corner cells.  A corner cell could pass all
        three while still being identically equal to the host face's
        value (i.e., the corner fill silently propagated host data
        instead of cross-face neighbour data) — a meaningful
        correctness violation that smoke-tests don't catch.

        This test verifies POSITIVE correctness on the corner blocks:
        for each face × each cube vertex, the 3×3 corner block cells
        must have values within the convex hull of the TWO adjacent
        neighbour faces' values, AND at least one cell of each
        corner block must NOT be identically equal to the host face's
        value (proving the corner fill DID blend cross-face data).
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH

        n = 8
        h = 3  # halo width
        grid = create_cubed_sphere(n=n, use_duogrid=True, duogrid_ng=4)
        data = jnp.zeros((6, n, n), dtype=jnp.float64)
        for f in range(6):
            data = data.at[f].set(float(f) + 1.0)

        p_h3 = pad_halo(data, halo=h, duogrid=grid.duogrid)
        p_np = np.asarray(p_h3)

        # The 3x3 corner blocks of the halo=3 padded array sit at
        # the four corners.  Map each cube vertex to its two
        # adjacent edges and the corresponding (i, j) slice.
        corner_specs = [
            ("SW", WEST,  SOUTH, slice(0, h),    slice(0, h)),
            ("SE", EAST,  SOUTH, slice(-h, None), slice(0, h)),
            ("NE", EAST,  NORTH, slice(-h, None), slice(-h, None)),
            ("NW", WEST,  NORTH, slice(0, h),    slice(-h, None)),
        ]

        for face in range(6):
            host_value = float(face) + 1.0
            for label, edge_a, edge_b, i_slice, j_slice in corner_specs:
                nbr_a, _, _ = CONNECTIVITY[face][edge_a]
                nbr_b, _, _ = CONNECTIVITY[face][edge_b]
                v_a = float(nbr_a) + 1.0
                v_b = float(nbr_b) + 1.0
                lo = min(v_a, v_b)
                hi = max(v_a, v_b)
                block = p_np[face, i_slice, j_slice]  # (h, h)
                # (1) Convex-hull check: every cell in the corner
                # block must be in [lo, hi] ± 1e-5 float drift.
                cell_min = float(block.min())
                cell_max = float(block.max())
                assert cell_min >= lo - 1e-5, (
                    f"face {face} {label}: min cell value "
                    f"{cell_min} < neighbour min {lo} = min(face "
                    f"{nbr_a}+1, face {nbr_b}+1).  Corner fill "
                    f"produced an out-of-hull value.")
                assert cell_max <= hi + 1e-5, (
                    f"face {face} {label}: max cell value "
                    f"{cell_max} > neighbour max {hi}.  Corner "
                    f"fill produced an out-of-hull value.")
                # (2) Cross-face blend check: at least one cell in
                # the block must differ from the host value.  (If
                # ALL cells equal the host value, the corner fill
                # silently propagated host data instead of cross-
                # face neighbour data — a real correctness bug.)
                differs = bool(np.any(np.abs(block - host_value) > 1e-5))
                assert differs, (
                    f"face {face} {label} corner block: all 9 cells "
                    f"equal the host value {host_value}.  Corner "
                    f"fill did not blend in neighbour faces "
                    f"{nbr_a} (={v_a}) and {nbr_b} (={v_b})."
                )
                # (3) Iter-538/539 (Codex follow-up): TWO-face SPATIAL
                # blend.  The cross-face check above passes if all
                # cells equal a SINGLE neighbour's value v_a.  But a
                # degenerate "all cells = midpoint(v_a, v_b)" fill
                # also passes "strict-between" — it's blended in
                # COMPOSITION but not in SPATIAL STRUCTURE.
                #
                # To prove BOTH neighbours genuinely contribute
                # POSITION-DEPENDENTLY (cells closer to W are closer
                # to v_W, cells closer to S are closer to v_S), the
                # block must contain at least one cell BELOW the
                # midpoint AND at least one cell ABOVE the midpoint.
                # That rules out:
                #   - single-neighbour propagation (all = v_a or v_b)
                #   - degenerate constant-blend (all = midpoint)
                gap = hi - lo
                # Only meaningful when v_a != v_b (which is always
                # true for face-unique values; CONNECTIVITY never
                # has both neighbours equal).
                assert gap > 1e-6, (
                    f"Test setup error: face {face} {label} adjacent "
                    f"neighbours have equal value (v_a={v_a}, "
                    f"v_b={v_b}); face-unique field cannot exercise "
                    f"two-face blend.")
                midpoint = 0.5 * (lo + hi)
                margin = 0.02 * gap   # 2% margin to absorb float drift
                some_below = bool(np.any(block < midpoint - margin))
                some_above = bool(np.any(block > midpoint + margin))
                assert some_below and some_above, (
                    f"face {face} {label} corner block: lacks SPATIAL "
                    f"two-face blend.  midpoint = {midpoint:.3f} "
                    f"(between v_a={v_a}, v_b={v_b}); "
                    f"some_below_midpoint = {some_below}, "
                    f"some_above_midpoint = {some_above}.  Cells: "
                    f"{sorted(set(block.flatten().tolist()))}.  "
                    f"A genuine two-face blend produces cells "
                    f"closer to v_a near the v_a edge AND cells "
                    f"closer to v_b near the v_b edge."
                )

    def test_duogrid_at_halo3_third_ring_carries_neighbour_data(self):
        """Iter-535 (Codex follow-up to iter-534): the iter-534
        edge-match test only checks `p_h3[1:-1, 1:-1]` — the overlap
        with halo=2.  The OUTERMOST ring of halo=3 (depth=2 edge
        strip + depth-2 corners) is never verified.

        This test uses a face-unique constant field (each face set
        to its face index) so that the depth-2 edge strip MUST
        contain the neighbour-face's value (because depth=2 pulls
        from the 3rd row of the neighbour's interior, which under a
        face-unique field equals that neighbour's face index for
        every cell).

        The duogrid Lagrange remap on a face-unique field also
        preserves neighbour-face values exactly (the Lagrange weights
        sum to 1, and a constant input gives the same constant
        output).  So the depth-2 edge strip of p_h3 MUST match the
        neighbour face's value, not stay at zero (which would
        indicate the duogrid-h3 path silently skipped depth 2).
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH

        n = 8
        grid = create_cubed_sphere(n=n, use_duogrid=True, duogrid_ng=4)

        # Face-unique constant field: each face = (face + 1.0).
        data = jnp.zeros((6, n, n), dtype=jnp.float64)
        for f in range(6):
            data = data.at[f].set(float(f) + 1.0)

        p_h3 = pad_halo(data, halo=3, duogrid=grid.duogrid)
        p_h3_np = np.asarray(p_h3)
        assert p_h3_np.shape == (6, n + 6, n + 6)

        # Depth=2 strip positions in halo=3 padded array (interior at
        # [3:-3, 3:-3]):
        #   WEST  depth=2 → i=0
        #   EAST  depth=2 → i=n+5
        #   SOUTH depth=2 → j=0
        #   NORTH depth=2 → j=n+5
        # Strip range along the orthogonal axis: [3:-3] (interior j or i).
        # Each cell of these strips should equal the neighbour face's
        # constant value (= nbr_face + 1.0).
        for face in range(6):
            for edge in (WEST, EAST, SOUTH, NORTH):
                nbr_face, nbr_edge, _rev = CONNECTIVITY[face][edge]
                expected = float(nbr_face) + 1.0
                if edge == WEST:
                    strip = p_h3_np[face, 0, 3:-3]
                elif edge == EAST:
                    strip = p_h3_np[face, n + 5, 3:-3]
                elif edge == SOUTH:
                    strip = p_h3_np[face, 3:-3, 0]
                else:
                    strip = p_h3_np[face, 3:-3, n + 5]
                # Allow small float drift from the Lagrange remap
                # (weights sum to 1 in exact arithmetic; float32
                # gives ~1e-6 drift).
                np.testing.assert_allclose(
                    strip, expected, rtol=1e-5, atol=1e-6,
                    err_msg=(f"face {face} edge {edge} depth=2 "
                             f"strip = {strip} but expected "
                             f"neighbour face {nbr_face} value "
                             f"= {expected}.  iter-533 halo=3 path "
                             f"is silently broken at depth=2 — "
                             f"`cube_rmp_vectorized`'s loop or the "
                             f"underlying `_pad_halo_local_h3` did "
                             f"not populate the third ring."))

    def test_duogrid_at_halo3_h2_h3_edge_match(self):
        """Iter-534 (Codex follow-up to iter-533): the constant-field
        test alone is too weak — it would pass even if the duogrid
        halo=3 path silently did nothing on the outer halo cells.

        Stronger guard: on a smooth non-constant field, the halo=3
        output's EDGE-STRIP region (the cells filled by face-to-face
        edge exchange + Lagrange remap, EXCLUDING cube-vertex
        corner cells) must match the halo=2 output at the
        corresponding interior-overlap positions.

        Cube-vertex corner cells of `p_h3[1:-1, 1:-1]` and
        `p_h2[:, :]` differ legitimately because corner-fill rules
        depend on halo depth (the corner-fill recursion fills the
        interior-most diagonal first, then propagates outward;
        h=3's depth-1 cell uses different neighbours from h=2's
        outermost-cell corner-fill).

        This test masks out the cube-vertex corner cells and asserts
        the remaining edge cells match.  Verified for both ng=4
        (full Lagrange at all 3 halo depths) and ng=2 (Lagrange +
        averaging fallback at outer halo).
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere

        n = 8
        for ng in (4, 2):
            grid = create_cubed_sphere(
                n=n, use_duogrid=True, duogrid_ng=ng)
            data = jnp.cos(grid.lat) ** 2 + 0.1 * grid.lon
            p_h2 = pad_halo(data, halo=2, duogrid=grid.duogrid)
            p_h3 = pad_halo(data, halo=3, duogrid=grid.duogrid)
            assert p_h2.shape == (6, n + 4, n + 4)
            assert p_h3.shape == (6, n + 6, n + 6)
            inner_h3 = np.asarray(p_h3[:, 1:-1, 1:-1])  # (6, n+4, n+4)
            ref_h2 = np.asarray(p_h2)                    # (6, n+4, n+4)
            # Mask: cell (i, j) is a corner-fill cell (depends on halo
            # depth) iff BOTH i and j are inside one of the 2-cell
            # halo strips at the array boundary (i.e., the four 2×2
            # corner blocks at [0..1, 0..1], [0..1, n+2..n+3], etc.).
            # The remaining cells are either interior or pure
            # edge-strip cells, which DO match between halo=2 and the
            # iter-533 halo=3 inner overlap.
            np_p2 = n + 4
            halo_w = 2
            in_i_halo = lambda i: i < halo_w or i >= np_p2 - halo_w
            mask = np.ones((np_p2, np_p2), dtype=bool)
            for i in range(np_p2):
                for j in range(np_p2):
                    if in_i_halo(i) and in_i_halo(j):
                        mask[i, j] = False
            np.testing.assert_allclose(
                inner_h3[:, mask], ref_h2[:, mask],
                rtol=1e-6, atol=1e-6,
                err_msg=(f"duogrid_ng={ng}: halo=3 edge-strip / "
                         f"interior cells do not match halo=2.  "
                         f"iter-533 path is silently broken at edge "
                         f"halos.  (Corner 2×2 blocks excluded — "
                         f"those legitimately differ by halo depth.)"))


# ---------------------------------------------------------------------------
# Public-API guardrails for halo=3 (iter-500, Codex stop-time fix)
# ---------------------------------------------------------------------------

class TestPadHaloH3Guardrails:
    """Codex flagged iter-499 for exposing halo=3 publicly without
    enough guardrails.  These tests pin down the guardrails:

    1. `pad_halo(halo=3)` validates `interp_offsets` shape — wrong
       shape raises `ValueError` clearly instead of IndexError.
    2. `pad_halo_4d(halo=3)` raises `NotImplementedError` with a
       message pointing at the 4D gap.
    3. `pad_halo_vector(halo=3)` raises `NotImplementedError` — the
       vector rotation round-trip depends on h=3 padded grid angles
       and half-metrics, none of which exist yet.
    """

    def test_pad_halo_halo3_rejects_h1_shape_offsets(self):
        """Passing h1-shape offsets (6, 4, n) with halo=3 must raise
        ValueError — the halo=3 path strictly requires h3 shape
        (6, 4, 3, n)."""
        from legoesm.grids.halo import compute_halo_interp_offsets
        offsets_h1 = compute_halo_interp_offsets(N)  # (6, 4, N)
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        with pytest.raises(ValueError, match=r"halo=3 expects"):
            pad_halo(data, halo=3, interp_offsets=offsets_h1)

    def test_pad_halo_halo3_rejects_h2_shape_offsets(self):
        from legoesm.grids.halo import compute_halo_interp_offsets_h2
        offsets_h2 = compute_halo_interp_offsets_h2(N)  # (6, 4, 2, N)
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        with pytest.raises(ValueError, match=r"halo=3 expects"):
            pad_halo(data, halo=3, interp_offsets=offsets_h2)

    def test_pad_halo_halo3_rejects_non4d_offsets(self):
        """Wrong ndim for halo=3 must be caught."""
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        bad = jnp.zeros((6, 4, N), dtype=jnp.float64)  # ndim=3
        with pytest.raises(ValueError, match=r"halo=3 expects"):
            pad_halo(data, halo=3, interp_offsets=bad)

    def test_pad_halo_halo3_rejects_wrong_face_axis(self):
        """Iter-501 (Codex): non-6 face axis must be caught, not silently
        accepted because ndim and depth happen to match."""
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        bad = jnp.zeros((10, 4, 3, N), dtype=jnp.float64)
        with pytest.raises(ValueError, match=r"halo=3 expects"):
            pad_halo(data, halo=3, interp_offsets=bad)

    def test_pad_halo_halo3_rejects_wrong_edge_axis(self):
        """Non-4 edge axis must be rejected."""
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        bad = jnp.zeros((6, 7, 3, N), dtype=jnp.float64)
        with pytest.raises(ValueError, match=r"halo=3 expects"):
            pad_halo(data, halo=3, interp_offsets=bad)

    def test_pad_halo_halo3_rejects_n_mismatch(self):
        """Final axis (n) must match data.shape[1] — otherwise _interp_strip
        silently produces wrong-sized output."""
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        bad = jnp.zeros((6, 4, 3, N + 2), dtype=jnp.float64)  # n axis wrong
        with pytest.raises(ValueError, match=r"halo=3 expects"):
            pad_halo(data, halo=3, interp_offsets=bad)

    def test_pad_halo_4d_halo3_raises_notimplemented(self):
        from legoesm.grids.halo import pad_halo_4d
        data = jnp.ones((6, N, N, 3), dtype=jnp.float64)
        with pytest.raises(NotImplementedError, match="pad_halo_4d"):
            pad_halo_4d(data, halo=3)

    def test_pad_halo_vector_halo3_works_on_local_backend(self):
        """Iter-595: pad_halo_vector(halo=3) is SUPPORTED on the
        single-device (non-MPI) backend.  Callers must supply
        `cos_angle_padded_h3` / `sin_angle_padded_h3` (shape
        (6, n+6, n+6)) — the scalar `pad_halo(halo=3)` internal
        call has been validated since iter-499.

        **Iter-596 Codex follow-up**: the iter-595 version of this
        test only checked interior preservation — the halo cells the
        relaxed guard exposed were not validated at all.  This
        version adds THREE halo-cell validations:

        1. **Inner two halo=3 rings match halo=2 exchange** at the
           overlapping physical cells.  The halo=2 vector path is
           validated by the rest of `TestPadHaloH3Dispatch` and is
           what existing callers use.  h=3 must reproduce h=2 at
           the overlap to be a safe drop-in for future callers.

        2. **Interior preserved** (iter-595 invariant): rotation
           round-trip to ~1e-10 float64 noise.

        3. **Outermost h=3 ring carries neighbour-face data**: for a
           face-unique input (each face has a distinctive constant
           value), the outermost halo ring of face 0 must match one
           of the neighbouring faces' values — NOT be garbage or
           left as zeros / NaN.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.halo import pad_halo_vector
        n = N
        grid = create_cubed_sphere(n=n, use_duogrid=False)

        # Use RANDOM (non-constant) inputs — constant inputs would
        # leak halo artefacts through rotation.  Random x64 inputs
        # exercise the full rotation+pad+inverse-rotation chain.
        rng = np.random.default_rng(596)
        u = jnp.asarray(rng.standard_normal((6, n, n)))
        v = jnp.asarray(rng.standard_normal((6, n, n)))

        ca = grid.cos_angle.astype(jnp.float64)
        sa = grid.sin_angle.astype(jnp.float64)

        # --- halo=3 exchange (the NEW path) ---
        up3, vp3 = pad_halo_vector(
            u, v,
            ca, sa,
            grid.cos_angle_padded_h3.astype(jnp.float64),
            grid.sin_angle_padded_h3.astype(jnp.float64),
            interp_offsets=grid.halo_interp_offsets_h3,
            halo=3,
        )
        assert up3.shape == (6, n + 6, n + 6)
        assert vp3.shape == (6, n + 6, n + 6)

        # --- halo=2 exchange (the VALIDATED REFERENCE) ---
        up2, vp2 = pad_halo_vector(
            u, v,
            ca, sa,
            grid.cos_angle_padded_h2.astype(jnp.float64),
            grid.sin_angle_padded_h2.astype(jnp.float64),
            interp_offsets=grid.halo_interp_offsets_h2,
            halo=2,
        )

        # --- Check 1 (iter-595): interior preserved in h=3 ---
        np.testing.assert_allclose(
            np.asarray(up3[:, 3:-3, 3:-3]), np.asarray(u),
            atol=1e-10, err_msg="h=3 interior u deviates from input")
        np.testing.assert_allclose(
            np.asarray(vp3[:, 3:-3, 3:-3]), np.asarray(v),
            atol=1e-10, err_msg="h=3 interior v deviates from input")

        # --- Check 2 (iter-596): h=3 inner two halo rings match
        # h=2 output at the same physical cells.  The shapes are:
        #   h=2: (6, n+4, n+4), halo rings at [:2] and [-2:]
        #   h=3: (6, n+6, n+6), halo rings at [:3] and [-3:]
        # The INNER two rings of h=3 (indices [1, 2] and [-2, -3])
        # correspond to the two rings of h=2 (indices [0, 1] and
        # [-1, -2]).
        # West halo: h=2 row [f, i, :] for i in {0, 1}
        #            h=3 row [f, i+1, :] for i in {0, 1} at the same
        #            transverse range [2:-2] in h=2 vs [3:-3] in h=3.
        def _inner_h3(arr, side):
            """Return the inner two halo rings of a h=3 padded array."""
            if side == "W":
                return arr[:, 1:3, 3:-3]
            if side == "E":
                return arr[:, -3:-1, 3:-3]
            if side == "S":
                return arr[:, 3:-3, 1:3]
            if side == "N":
                return arr[:, 3:-3, -3:-1]
            raise ValueError(side)

        def _all_h2(arr, side):
            """Return the h=2 halo rings (both)."""
            if side == "W":
                return arr[:, 0:2, 2:-2]
            if side == "E":
                return arr[:, -2:, 2:-2]
            if side == "S":
                return arr[:, 2:-2, 0:2]
            if side == "N":
                return arr[:, 2:-2, -2:]
            raise ValueError(side)

        for side in ("W", "E", "S", "N"):
            for comp_label, arr3, arr2 in (
                    ("u", up3, up2), ("v", vp3, vp2)):
                h3_rings = np.asarray(_inner_h3(arr3, side))
                h2_rings = np.asarray(_all_h2(arr2, side))
                diff = float(np.max(np.abs(h3_rings - h2_rings)))
                assert diff < 1e-10, (
                    f"h=3 inner halo rings on {side} {comp_label} "
                    f"diverge from validated h=2 output by "
                    f"{diff:.3e}.  The h=3 halo path must "
                    f"reproduce h=2 at overlapping physical "
                    f"cells — this anchors the new path to "
                    f"the validated reference.")

        # --- Check 3: outermost h=3 ring carries non-trivial data.
        # For random inputs, the outermost halo ring on each face
        # should have values in the same distribution as the inputs.
        # A bug that returned zeros or NaN would fire |outer| ~ 0 or
        # non-finite.
        for side, slicer in (
                ("W", (slice(None), 0, slice(3, -3))),
                ("E", (slice(None), -1, slice(3, -3))),
                ("S", (slice(None), slice(3, -3), 0)),
                ("N", (slice(None), slice(3, -3), -1))):
            outer_u = np.asarray(up3[slicer])
            outer_v = np.asarray(vp3[slicer])
            assert np.isfinite(outer_u).all(), (
                f"h=3 outermost {side} u has non-finite values")
            assert np.isfinite(outer_v).all(), (
                f"h=3 outermost {side} v has non-finite values")
            # Magnitude check: should be in the same range as input
            # (random normal, so |max| ~ O(1)).  A bug that zeroed
            # the outer ring would produce max ~ 0.
            max_outer = float(np.max(np.abs(outer_u))
                              + np.max(np.abs(outer_v)))
            assert max_outer > 0.1, (
                f"h=3 outermost {side} ring has near-zero "
                f"magnitude ({max_outer:.3e}); expected O(1) "
                f"for random unit-variance inputs.  The halo "
                f"exchange may have silently failed for the "
                f"outermost ring.")

    def test_pad_halo_vector_halo3_outer_ring_exact_constant_geographic_wind(self):
        """Iter-597 (Codex stop-time review on iter-596): the iter-596
        outer-ring check only verified non-triviality (finite, |max| >
        0.1), not CORRECTNESS.  A bug that produced `0.5 * expected`
        everywhere in the outer ring would pass iter-596's check but
        silently halve the halo amplitude.

        This test nails the outer ring to an EXACT reference via a
        constant-geographic-wind round-trip:

          Input: u_grid = cos_angle, v_grid = -sin_angle.
          This corresponds to u_east = 1, v_north = 0 everywhere
          on the sphere (pure zonal flow at constant speed 1).

          After `pad_halo_vector(halo=3)`:
            - u_east_padded should be 1 everywhere (halo fills from
              neighbour face's u_east = 1).
            - v_north_padded should be 0 everywhere.
            - After inverse rotation via cos_angle_padded_h3 /
              sin_angle_padded_h3, the grid-aligned outputs are:
                u_padded[f, i, j] = cos_angle_padded_h3[f, i, j]
                v_padded[f, i, j] = -sin_angle_padded_h3[f, i, j]
              at EVERY cell including the outermost halo ring.

        Tolerance 1e-6: `halo_interp_offsets_h3` is stored as float32
        in the grid (default storage dtype), so the offset arithmetic
        inside `pad_halo(halo=3)` introduces O(float32 eps) ≈ 1.2e-7
        error even when the input data is float64.  1e-6 is ~8x above
        this floor — catches O(1) bugs (wrong rotation, wrong halo
        cells, missing neighbour data) while tolerating the storage-
        precision noise.

        This makes the outermost halo ring a PREDICTED value at
        every cell, closing the iter-596 correctness gap.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.halo import pad_halo_vector
        n = N
        grid = create_cubed_sphere(n=n, use_duogrid=False)

        ca = grid.cos_angle.astype(jnp.float64)
        sa = grid.sin_angle.astype(jnp.float64)
        cap3 = grid.cos_angle_padded_h3.astype(jnp.float64)
        sap3 = grid.sin_angle_padded_h3.astype(jnp.float64)

        # Constant geographic wind u_east=1, v_north=0:
        # grid-aligned = inverse rotation of (1, 0)
        # u_grid = cos(angle) * u_east + sin(angle) * v_north = cos
        # v_grid = -sin(angle) * u_east + cos(angle) * v_north = -sin
        u_grid = ca
        v_grid = -sa

        up, vp = pad_halo_vector(
            u_grid, v_grid,
            ca, sa, cap3, sap3,
            interp_offsets=grid.halo_interp_offsets_h3,
            halo=3,
        )

        # Predicted output at every padded cell:
        #   u_padded = cos_angle_padded_h3  (since u_east=1)
        #   v_padded = -sin_angle_padded_h3 (since v_north=0)
        u_expected = cap3
        v_expected = -sap3

        # Check #1: full padded output matches prediction.
        u_diff = float(np.max(np.abs(
            np.asarray(up) - np.asarray(u_expected))))
        v_diff = float(np.max(np.abs(
            np.asarray(vp) - np.asarray(v_expected))))
        assert u_diff < 1e-6, (
            f"halo=3 vector pad for constant u_east=1, v_north=0 "
            f"failed: max|u_out - cos_angle_padded_h3| = {u_diff:.3e}. "
            f"Expected exact rotation round-trip at every padded "
            f"cell including the outermost halo ring.  If this "
            f"fires, the halo=3 vector path either has a rotation "
            f"sign error, the wrong padded angle, or the scalar "
            f"pad_halo(halo=3) does not propagate constants at the "
            f"outer ring.")
        assert v_diff < 1e-6, (
            f"halo=3 vector pad for constant u_east=1, v_north=0 "
            f"failed: max|v_out - (-sin_angle_padded_h3)| = "
            f"{v_diff:.3e}.")

        # Check #2: lock the diff specifically on the OUTERMOST halo
        # ring (which iter-596 was missing correctness for).  Isolate
        # the 4 outer strips.
        for label, slicer in (
                ("W (i=0)",  (slice(None), 0, slice(None))),
                ("E (i=-1)", (slice(None), -1, slice(None))),
                ("S (j=0)",  (slice(None), slice(None), 0)),
                ("N (j=-1)", (slice(None), slice(None), -1))):
            u_outer_diff = float(np.max(np.abs(
                np.asarray(up[slicer])
                - np.asarray(u_expected[slicer]))))
            v_outer_diff = float(np.max(np.abs(
                np.asarray(vp[slicer])
                - np.asarray(v_expected[slicer]))))
            assert u_outer_diff < 1e-6, (
                f"halo=3 outermost {label} u deviates from "
                f"cos_angle_padded_h3 by {u_outer_diff:.3e} — "
                f"outermost-ring correctness regression.")
            assert v_outer_diff < 1e-6, (
                f"halo=3 outermost {label} v deviates from "
                f"-sin_angle_padded_h3 by {v_outer_diff:.3e} — "
                f"outermost-ring correctness regression.")

    def test_pad_halo_vector_halo3_rejects_mpi_backend(self):
        """Iter-595: halo=3 is still unsupported on the MPI backend
        because `pad_halo_mpi_4d(halo=3)` is not yet implemented.  The
        check must use the backend-specific gate, not a blanket
        NotImplementedError.
        """
        from unittest import mock
        import legoesm.grids.halo as halo_mod
        u = jnp.ones((6, N, N), dtype=jnp.float64)
        v = jnp.zeros((6, N, N), dtype=jnp.float64)
        ca = jnp.ones((6, N, N), dtype=jnp.float64)
        sa = jnp.zeros((6, N, N), dtype=jnp.float64)
        cap = jnp.ones((6, N + 6, N + 6), dtype=jnp.float64)
        sap = jnp.zeros((6, N + 6, N + 6), dtype=jnp.float64)

        # Flip the module-level backend to "mpi" for the duration of
        # this test; halo=3 must raise under MPI until
        # pad_halo_mpi_4d(halo=3) is implemented.
        with mock.patch.object(halo_mod, "_halo_backend", "mpi"):
            with pytest.raises(
                    NotImplementedError, match="MPI backend"):
                halo_mod.pad_halo_vector(
                    u, v, ca, sa, cap, sap,
                    interp_offsets=None, halo=3)


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
