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

    def test_duogrid_at_halo3_raises(self):
        """Duo-Grid remap at halo=3 not yet implemented — must error
        clearly rather than silently call the halo=2 remap path."""
        # Build a tiny DuoGridData-shaped object sufficient to trip the
        # guard — the duogrid-None check runs before any remap call.
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        class _FakeDuogrid:
            pass
        with pytest.raises(NotImplementedError, match="Duo-Grid"):
            pad_halo(data, halo=3, duogrid=_FakeDuogrid())


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
        with pytest.raises(ValueError, match=r"halo=3 expects.*shape \(6, 4, 3, n\)"):
            pad_halo(data, halo=3, interp_offsets=offsets_h1)

    def test_pad_halo_halo3_rejects_h2_shape_offsets(self):
        from legoesm.grids.halo import compute_halo_interp_offsets_h2
        offsets_h2 = compute_halo_interp_offsets_h2(N)  # (6, 4, 2, N)
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        with pytest.raises(ValueError, match=r"halo=3 expects.*shape \(6, 4, 3, n\)"):
            pad_halo(data, halo=3, interp_offsets=offsets_h2)

    def test_pad_halo_halo3_rejects_non4d_offsets(self):
        """Wrong ndim for halo=3 must be caught."""
        data = jnp.ones((6, N, N), dtype=jnp.float64)
        bad = jnp.zeros((6, 4, N), dtype=jnp.float64)  # ndim=3
        with pytest.raises(ValueError, match=r"halo=3 expects"):
            pad_halo(data, halo=3, interp_offsets=bad)

    def test_pad_halo_4d_halo3_raises_notimplemented(self):
        from legoesm.grids.halo import pad_halo_4d
        data = jnp.ones((6, N, N, 3), dtype=jnp.float64)
        with pytest.raises(NotImplementedError, match="pad_halo_4d"):
            pad_halo_4d(data, halo=3)

    def test_pad_halo_vector_halo3_raises_notimplemented(self):
        """The vector rotation round-trip needs h=3 grid angles and
        half-metrics; with neither in place, pad_halo_vector at halo=3
        must error rather than produce wrong numbers."""
        from legoesm.grids.halo import pad_halo_vector
        u = jnp.ones((6, N, N), dtype=jnp.float64)
        v = jnp.zeros((6, N, N), dtype=jnp.float64)
        ca = jnp.ones((6, N, N), dtype=jnp.float64)
        sa = jnp.zeros((6, N, N), dtype=jnp.float64)
        cap = jnp.ones((6, N + 6, N + 6), dtype=jnp.float64)
        sap = jnp.zeros((6, N + 6, N + 6), dtype=jnp.float64)
        with pytest.raises(NotImplementedError, match="pad_halo_vector"):
            pad_halo_vector(
                u, v, ca, sa, cap, sap, interp_offsets=None, halo=3)


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
