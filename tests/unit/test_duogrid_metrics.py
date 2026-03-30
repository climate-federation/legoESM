"""Unit tests for Duo-Grid sub-grid metrics (sin_sg, cos_sg).

The 9-point supergrid metrics encode the non-orthogonality angle at
9 positions within each cell.  The Duo-Grid approach (Mouallem, Harris
& Chen 2023) uses these direction-dependent metrics to eliminate
cubed-sphere edge artefacts.

These tests verify correctness of the metric computation and confirm
the expected discontinuity at face boundaries.
"""

import unittest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestDuoGridMetrics(unittest.TestCase):
    """Test sin_sg/cos_sg computation on CubedSphereCDGrid."""

    def setUp(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        self.n = 16
        grid = create_cubed_sphere(self.n)
        self.cdgrid = create_cubed_sphere_cdgrid(grid)

    def test_shapes(self):
        """sin_sg and cos_sg have shape (6, n, n, 9)."""
        self.assertEqual(self.cdgrid.sin_sg.shape, (6, self.n, self.n, 9))
        self.assertEqual(self.cdgrid.cos_sg.shape, (6, self.n, self.n, 9))

    def test_identity_sin2_plus_cos2(self):
        """sin²(α) + cos²(α) = 1 at all 9 positions."""
        check = self.cdgrid.sin_sg ** 2 + self.cdgrid.cos_sg ** 2
        max_err = float(jnp.max(jnp.abs(check - 1.0)))
        self.assertLess(max_err, 1e-6,
                        f"sin²+cos² max deviation from 1: {max_err}")

    def test_sin_sg_positive(self):
        """sin_sg > 0 everywhere (angles in (0, π))."""
        min_sin = float(jnp.min(self.cdgrid.sin_sg))
        self.assertGreater(min_sin, 0.0,
                           f"min sin_sg = {min_sin} should be > 0")

    def test_interior_edge_symmetry(self):
        """E-edge of cell (i,j) ≈ W-edge of cell (i+1,j) in the interior."""
        n = self.n
        # Interior cells only (i=1..n-3 to stay away from boundaries)
        sin_E = self.cdgrid.sin_sg[:, 1:n - 2, :, 2]  # E-edge
        sin_W = self.cdgrid.sin_sg[:, 2:n - 1, :, 0]  # W-edge of next cell
        max_diff = float(jnp.max(jnp.abs(sin_E - sin_W)))
        self.assertLess(max_diff, 1e-6,
                        f"Interior E/W edge mismatch: {max_diff}")

        cos_E = self.cdgrid.cos_sg[:, 1:n - 2, :, 2]
        cos_W = self.cdgrid.cos_sg[:, 2:n - 1, :, 0]
        max_diff_c = float(jnp.max(jnp.abs(cos_E - cos_W)))
        self.assertLess(max_diff_c, 1e-6,
                        f"Interior E/W cos_sg mismatch: {max_diff_c}")

    def test_cos_sg_discontinuity_at_face_boundary(self):
        """cos_sg flips sign at equatorial-to-polar face boundaries.

        Face 0 NORTH connects to Face 4 SOUTH.  The coordinate kink
        causes cos_sg at the shared edge to have opposite signs on the
        two faces, while sin_sg remains the same (|sin(α)| = |sin(π-α)|).
        """
        n = self.n
        cos_N_face0 = self.cdgrid.cos_sg[0, :, n - 1, 3]  # N edge, top row
        cos_S_face4 = self.cdgrid.cos_sg[4, :, 0, 1]       # S edge, bottom row
        # cos values should have opposite signs
        product = cos_N_face0 * cos_S_face4
        # At the face boundary, most products should be negative (opposite signs)
        n_negative = int(jnp.sum(product < 0))
        self.assertGreater(n_negative, n // 2,
                           "Expected cos_sg to flip sign at face boundary")

    def test_sin_sg_continuous_at_face_boundary(self):
        """sin_sg is the same on both sides of the face boundary.

        Since sin(α) = sin(π - α), the sine values match even though
        the cosine values differ.
        """
        n = self.n
        sin_N_face0 = self.cdgrid.sin_sg[0, :, n - 1, 3]
        sin_S_face4 = self.cdgrid.sin_sg[4, :, 0, 1]
        max_diff = float(jnp.max(jnp.abs(sin_N_face0 - sin_S_face4)))
        self.assertLess(max_diff, 1e-6,
                        f"sin_sg should be continuous at face boundary: "
                        f"max diff = {max_diff}")


if __name__ == "__main__":
    unittest.main()
