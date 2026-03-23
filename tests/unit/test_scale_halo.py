"""Category 2: Halo exchange correctness (single-node).

Verifies cubed-sphere halo exchange correctly fills ghost cells using the
local (JAX-only) backend.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.halo import (
    pad_halo,
    pad_halo_vector,
    compute_halo_interp_offsets,
    compute_halo_interp_offsets_h2,
    CONNECTIVITY,
    WEST, EAST, SOUTH, NORTH,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere


N = 8  # grid resolution per face


# =========================================================================
# 2a) Constant field — halo = interior
# =========================================================================

class TestConstantField:
    """Halo of a constant field should be the same constant."""

    @pytest.mark.parametrize("halo", [1, 2])
    def test_constant_halo(self, halo):
        data = jnp.full((6, N, N), 42.0)
        padded = pad_halo(data, halo=halo)
        assert padded.shape == (6, N + 2 * halo, N + 2 * halo)
        # All values should be 42
        np.testing.assert_allclose(np.array(padded), 42.0, atol=1e-12)

    def test_interior_preserved(self):
        """Interior of padded array should exactly equal input."""
        data = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        padded = pad_halo(data, halo=1)
        interior = padded[:, 1:-1, 1:-1]
        np.testing.assert_allclose(np.array(interior), np.array(data), atol=0)


# =========================================================================
# 2b) Face-unique field — correct neighbor
# =========================================================================

class TestFaceUniqueField:
    """Ghost cells should contain values from the correct neighbor face."""

    def test_face_unique_halo1(self):
        """Each face has unique value; ghost cells come from correct neighbor."""
        data = jnp.zeros((6, N, N))
        for f in range(6):
            data = data.at[f].set(float(f + 1))

        padded = pad_halo(data, halo=1)

        for face in range(6):
            for edge in [WEST, EAST, SOUTH, NORTH]:
                nbr_face, nbr_edge, _ = CONNECTIVITY[face][edge]
                expected_val = float(nbr_face + 1)
                # Extract halo strip
                if edge == WEST:
                    strip = padded[face, 0, 1:-1]
                elif edge == EAST:
                    strip = padded[face, -1, 1:-1]
                elif edge == SOUTH:
                    strip = padded[face, 1:-1, 0]
                elif edge == NORTH:
                    strip = padded[face, 1:-1, -1]
                np.testing.assert_allclose(
                    np.array(strip), expected_val, atol=1e-12,
                    err_msg=f"Face {face} edge {edge}: expected {expected_val} from face {nbr_face}",
                )


# =========================================================================
# 2c) Index reversal at swapped boundaries
# =========================================================================

class TestIndexReversal:
    """At reversed boundaries, halo strips should be flipped."""

    def test_reversed_edges(self):
        """Create strictly monotonic field and verify reversal at reversed boundaries."""
        # Field with strict gradient: data[f, i, j] = (f+1)*100 + i*10 + j
        ii = jnp.arange(N, dtype=jnp.float64)
        jj = jnp.arange(N, dtype=jnp.float64)
        data = jnp.zeros((6, N, N))
        for f in range(6):
            data = data.at[f].set((f + 1) * 100.0 + ii[:, None] * 10.0 + jj[None, :])

        padded = pad_halo(data, halo=1)

        # At least one reversed boundary should show a non-monotonic halo strip
        found_reversed = False
        for face in range(6):
            for edge in [WEST, EAST, SOUTH, NORTH]:
                _, _, reversed_ = CONNECTIVITY[face][edge]
                if reversed_:
                    if edge == WEST:
                        strip = padded[face, 0, 1:-1]
                    elif edge == EAST:
                        strip = padded[face, -1, 1:-1]
                    elif edge == SOUTH:
                        strip = padded[face, 1:-1, 0]
                    elif edge == NORTH:
                        strip = padded[face, 1:-1, -1]
                    arr = np.array(strip)
                    # The source strip was monotonic; if reversed, the halo strip
                    # should be flipped (decreasing where source was increasing)
                    diffs = np.diff(arr)
                    if np.any(diffs < 0):
                        found_reversed = True
        assert found_reversed, "Expected at least one reversed halo strip"


# =========================================================================
# 2d) Interpolation offsets
# =========================================================================

class TestInterpolationOffsets:
    """Interpolation offsets should be bounded and have correct shape."""

    def test_offsets_shape(self):
        offsets = compute_halo_interp_offsets(N)
        assert offsets.shape == (6, 4, N)

    def test_offsets_bounded(self):
        offsets = compute_halo_interp_offsets(N)
        assert float(jnp.max(jnp.abs(offsets))) <= 0.5 + 1e-6

    def test_offsets_h2_shape(self):
        offsets = compute_halo_interp_offsets_h2(N)
        assert offsets.shape == (6, 4, 2, N)

    def test_offsets_h2_bounded(self):
        offsets = compute_halo_interp_offsets_h2(N)
        # Depth-0 offsets (inner halo) should be bounded by 0.5
        assert float(jnp.max(jnp.abs(offsets[:, :, 0, :]))) <= 0.5 + 1e-6
        # Depth-1 offsets (outer halo) can be larger due to gnomonic projection
        # Just check they are finite and bounded
        assert jnp.all(jnp.isfinite(offsets))
        assert float(jnp.max(jnp.abs(offsets))) < 5.0


# =========================================================================
# 2e) Vector halo — rotation correctness
# =========================================================================

class TestVectorHalo:
    """Vector halo exchange should correctly rotate at face boundaries."""

    def test_pure_zonal_preserved(self):
        """Pure zonal wind (u_east=1, v_north=0) should be preserved in halos."""
        grid = create_cubed_sphere(N)
        cos_a = grid.cos_angle
        sin_a = grid.sin_angle
        cos_a_p = grid.cos_angle_padded
        sin_a_p = grid.sin_angle_padded

        # Pure zonal in geographic: u_east=1, v_north=0
        # In grid coordinates: u_grid = cos(angle), v_grid = -sin(angle)
        u_grid = cos_a
        v_grid = -sin_a

        u_pad, v_pad = pad_halo_vector(
            u_grid, v_grid, cos_a, sin_a, cos_a_p, sin_a_p,
            interp_offsets=grid.halo_interp_offsets, halo=1,
        )

        # All output values should be finite
        assert jnp.all(jnp.isfinite(u_pad))
        assert jnp.all(jnp.isfinite(v_pad))

        # Interior of padded should exactly match input (no modification)
        u_int = u_pad[:, 1:-1, 1:-1]
        v_int = v_pad[:, 1:-1, 1:-1]
        np.testing.assert_allclose(np.array(u_int), np.array(u_grid), atol=1e-12)
        np.testing.assert_allclose(np.array(v_int), np.array(v_grid), atol=1e-12)

        # Reconstruct geographic in interior using ORIGINAL angles (not padded)
        u_east_int = cos_a * u_int + sin_a * v_int
        v_north_int = -sin_a * u_int + cos_a * v_int
        np.testing.assert_allclose(np.array(u_east_int), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.array(v_north_int), 0.0, atol=1e-10)

        # Halo values should be finite; speed ~ 1.0 ± interpolation error
        speed_halo = jnp.sqrt(u_pad**2 + v_pad**2)
        assert jnp.all(jnp.isfinite(speed_halo))
        # Speed should be approximately 1 everywhere (solid-body is speed=1)
        np.testing.assert_allclose(np.array(speed_halo), 1.0, atol=0.15)


# =========================================================================
# 2f) Vector halo — solid-body rotation
# =========================================================================

class TestVectorSolidBody:
    """Solid-body rotation should be smooth across face boundaries."""

    def test_solid_body_continuity(self):
        grid = create_cubed_sphere(N)
        U0 = 20.0
        lat = grid.lat
        cos_a = grid.cos_angle
        sin_a = grid.sin_angle

        # Geographic: u = U0 * cos(lat), v = 0
        u_geo = U0 * jnp.cos(lat)
        v_geo = jnp.zeros_like(lat)

        # Grid-aligned
        u_grid = cos_a * u_geo - sin_a * v_geo
        v_grid = sin_a * u_geo + cos_a * v_geo

        u_pad, v_pad = pad_halo_vector(
            u_grid, v_grid,
            cos_a, sin_a,
            grid.cos_angle_padded, grid.sin_angle_padded,
            interp_offsets=grid.halo_interp_offsets, halo=1,
        )

        # Velocity magnitude should be smooth everywhere
        speed = jnp.sqrt(u_pad**2 + v_pad**2)
        assert jnp.all(jnp.isfinite(speed))
        # Max speed should be ≤ U0 * 1.1 (allowing small interpolation errors)
        assert float(jnp.max(speed)) < U0 * 1.1


# =========================================================================
# 2g) Symmetry — CONNECTIVITY is bidirectional
# =========================================================================

class TestConnectivitySymmetry:
    """CONNECTIVITY table should be symmetric."""

    def test_bidirectional(self):
        for face in range(6):
            for edge in [WEST, EAST, SOUTH, NORTH]:
                nbr_face, nbr_edge, rev = CONNECTIVITY[face][edge]
                # Reverse lookup
                back_face, back_edge, back_rev = CONNECTIVITY[nbr_face][nbr_edge]
                assert back_face == face, (
                    f"Asymmetry: ({face},{edge})->({nbr_face},{nbr_edge}) "
                    f"but ({nbr_face},{nbr_edge})->({back_face},{back_edge})"
                )


# =========================================================================
# 2h) Halo width 2
# =========================================================================

class TestHaloWidth2:
    """Halo=2 should produce correct output shape and fill both layers."""

    def test_shape(self):
        data = jnp.ones((6, N, N))
        padded = pad_halo(data, halo=2)
        assert padded.shape == (6, N + 4, N + 4)

    def test_interior_exact(self):
        data = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        padded = pad_halo(data, halo=2)
        interior = padded[:, 2:-2, 2:-2]
        np.testing.assert_allclose(np.array(interior), np.array(data), atol=0)

    def test_no_uninitialized_zeros(self):
        """Non-zero field should not have zero-filled halo (uninitialized)."""
        data = jnp.ones((6, N, N)) * 7.0
        padded = pad_halo(data, halo=2)
        # All values should be 7.0 for constant field
        np.testing.assert_allclose(np.array(padded), 7.0, atol=1e-12)


# =========================================================================
# 2i) Corner cells
# =========================================================================

class TestCornerCells:
    """Corner cells of padded array should be finite and reasonable."""

    def test_corners_finite(self):
        """All 4 corners of each face should be finite."""
        grid = create_cubed_sphere(N)
        data = jnp.ones((6, N, N)) * 3.14
        padded = pad_halo(data, halo=1)
        for f in range(6):
            assert jnp.isfinite(padded[f, 0, 0]), f"Face {f} SW corner not finite"
            assert jnp.isfinite(padded[f, 0, -1]), f"Face {f} NW corner not finite"
            assert jnp.isfinite(padded[f, -1, 0]), f"Face {f} SE corner not finite"
            assert jnp.isfinite(padded[f, -1, -1]), f"Face {f} NE corner not finite"

    def test_corners_not_zero_for_nonzero_field(self):
        """Corners should not be zero if the field is uniformly non-zero."""
        data = jnp.ones((6, N, N)) * 5.0
        padded = pad_halo(data, halo=1)
        for f in range(6):
            assert float(padded[f, 0, 0]) != 0.0
            assert float(padded[f, 0, -1]) != 0.0
            assert float(padded[f, -1, 0]) != 0.0
            assert float(padded[f, -1, -1]) != 0.0
