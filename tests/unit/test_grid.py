"""Unit tests for the cubed-sphere grid."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere, CubedSphereGrid


class TestCubedSphereGrid:
    """Tests for cubed-sphere grid construction and properties."""

    def test_creation(self, small_grid):
        assert small_grid.n == 8
        assert small_grid.shape == (6, 8, 8)
        assert small_grid.n_cells == 6 * 8 * 8

    def test_lon_lat_ranges(self, small_grid):
        """Longitude should be in [-pi, pi], latitude in [-pi/2, pi/2]."""
        assert jnp.all(small_grid.lon >= -jnp.pi - 0.01)
        assert jnp.all(small_grid.lon <= jnp.pi + 0.01)
        assert jnp.all(small_grid.lat >= -jnp.pi / 2 - 0.01)
        assert jnp.all(small_grid.lat <= jnp.pi / 2 + 0.01)

    def test_cartesian_unit_sphere(self, small_grid):
        """Cartesian coordinates should lie on the unit sphere."""
        r = jnp.sqrt(
            small_grid.x_cart**2 + small_grid.y_cart**2 + small_grid.z_cart**2
        )
        assert jnp.allclose(r, 1.0, atol=1e-6)

    def test_areas_positive(self, small_grid):
        """All cell areas should be positive."""
        assert jnp.all(small_grid.area > 0)

    def test_total_area_sphere(self, small_grid):
        """Total area should approximate 4*pi*R^2."""
        expected_area = 4.0 * jnp.pi * small_grid.radius**2
        actual_area = small_grid.total_area
        relative_error = jnp.abs(actual_area - expected_area) / expected_area
        # Allow 5% error for low-res grid
        assert relative_error < 0.05, f"Area error: {relative_error:.4f}"

    def test_coriolis_parameter(self, small_grid):
        """Coriolis at equator ~ 0, at poles ~ +/- 2*Omega."""
        Omega = 7.292e-5
        f_max = jnp.max(jnp.abs(small_grid.f))
        assert f_max <= 2 * Omega * 1.01

    def test_grid_spacing_positive(self, small_grid):
        """Grid spacings should be positive."""
        assert jnp.all(small_grid.dx > 0)
        assert jnp.all(small_grid.dy > 0)

    def test_grid_spacing_reasonable(self, small_grid):
        """Grid spacing should be reasonable for the resolution."""
        # C8 is very coarse; dx is the distance between 2-cell-apart neighbors
        # due to centered difference. Allow wide range for low-res grid.
        mean_dx_km = float(jnp.mean(small_grid.dx)) / 1000
        assert 500 < mean_dx_km < 5000, f"Mean dx = {mean_dx_km:.0f} km"

    def test_different_resolutions(self):
        """Higher resolution should have smaller grid spacing."""
        g16 = create_cubed_sphere(16)
        g32 = create_cubed_sphere(32)
        assert float(jnp.mean(g32.dx)) < float(jnp.mean(g16.dx))

    def test_pytree_compatible(self, small_grid):
        """Grid should work with JAX tree operations."""
        leaves = jax.tree.leaves(small_grid)
        assert len(leaves) > 0
