"""Unit tests for the lat-lon grid construction."""

import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import LatLonGrid, create_latlon_grid


@pytest.fixture
def grid():
    """Small lat-lon grid for testing."""
    return create_latlon_grid(16, 32)


class TestGridConstruction:
    """Tests for create_latlon_grid."""

    def test_shape(self, grid):
        """Grid arrays should have correct shapes."""
        assert grid.lat.shape == (16,)
        assert grid.lon.shape == (32,)
        assert grid.lat2d.shape == (16, 32)
        assert grid.lon2d.shape == (16, 32)
        assert grid.cos_lat.shape == (16,)
        assert grid.sin_lat.shape == (16,)
        assert grid.f.shape == (16, 32)
        assert grid.dx.shape == (16, 32)
        assert grid.area.shape == (16, 32)

    def test_n_lat_n_lon(self, grid):
        """n_lat and n_lon should match."""
        assert grid.n_lat == 16
        assert grid.n_lon == 32

    def test_default_n_lon(self):
        """n_lon should default to 2 * n_lat."""
        grid = create_latlon_grid(32)
        assert grid.n_lon == 64

    def test_lat_range(self, grid):
        """Latitudes should be within (-pi/2, pi/2) and avoid exact poles."""
        assert float(grid.lat[0]) > -jnp.pi / 2
        assert float(grid.lat[-1]) < jnp.pi / 2

    def test_lat_south_to_north(self, grid):
        """Latitudes should be monotonically increasing (S->N)."""
        assert jnp.all(jnp.diff(grid.lat) > 0)

    def test_lon_range(self, grid):
        """Longitudes should be in [0, 2*pi)."""
        assert float(grid.lon[0]) >= 0.0
        assert float(grid.lon[-1]) < 2.0 * jnp.pi

    def test_cos_lat_positive(self, grid):
        """cos_lat should be strictly positive (clamped away from zero)."""
        assert jnp.all(grid.cos_lat > 0)

    def test_coriolis_antisymmetric(self, grid):
        """Coriolis parameter should be antisymmetric about equator."""
        # Find equatorial latitude index (closest to 0)
        eq_idx = jnp.argmin(jnp.abs(grid.lat))
        n = grid.n_lat
        # f at symmetric latitudes should have opposite signs
        for i in range(n // 2):
            j = n - 1 - i
            assert jnp.allclose(
                grid.f[i, 0], -grid.f[j, 0], atol=1e-10
            ), f"f not antisymmetric at indices {i}, {j}"

    def test_dx_decreases_toward_poles(self, grid):
        """dx should decrease toward the poles (where cos_lat is smaller)."""
        # Compare equator vs high latitude
        eq_idx = grid.n_lat // 2
        pole_idx = 0  # southernmost
        assert float(grid.dx[eq_idx, 0]) > float(grid.dx[pole_idx, 0])

    def test_dy_is_1d_array(self, grid):
        """dy is a 1D jax.Array of shape (n_lat,) — see docs/mercator_grid_plan.md.

        For uniform-dlat grids (this one) all entries are equal.
        """
        assert grid.dy.shape == (grid.n_lat,)
        # Uniform-dlat: all entries equal to the equatorial value.
        assert jnp.allclose(grid.dy, grid.dy[0])

    def test_area_positive(self, grid):
        """All cell areas should be positive."""
        assert jnp.all(grid.area > 0)

    def test_total_area_sphere(self, grid):
        """Total area should approximate 4*pi*R^2."""
        expected = 4.0 * jnp.pi * grid.radius**2
        rel_err = abs(float(grid.total_area) - float(expected)) / float(expected)
        # For 16x32, the discrete sum won't be exact but should be close
        assert rel_err < 0.01, f"Total area relative error: {rel_err:.4e}"

    def test_dlat_dlon(self, grid):
        """dlat and dlon should be consistent with n_lat, n_lon."""
        assert abs(grid.dlat - jnp.pi / 16) < 1e-10
        assert abs(grid.dlon - 2 * jnp.pi / 32) < 1e-10


class TestGridResolutions:
    """Test that different resolutions work correctly."""

    @pytest.mark.parametrize("n_lat", [8, 16, 32, 64])
    def test_various_resolutions(self, n_lat):
        """Grid creation should work for various resolutions."""
        grid = create_latlon_grid(n_lat)
        assert grid.n_lat == n_lat
        assert grid.n_lon == 2 * n_lat
        assert jnp.all(jnp.isfinite(grid.area))
        assert jnp.all(grid.area > 0)


class TestGridOmega:
    """LatLonGrid stores its construction rotation rate (#521)."""

    def test_omega_stored_and_default(self):
        from legoesm import constants
        grid = create_latlon_grid(8)
        assert grid.omega == constants.Omega
        grid0 = create_latlon_grid(8, omega=0.0)
        assert grid0.omega == 0.0
        assert float(jnp.max(jnp.abs(grid0.f))) == 0.0

    def test_ensure_geometry_inherits_grid_omega(self):
        """ensure_geometry must not silently re-rotate an omega=0 grid
        (codex 2026-07-03 round-4 HIGH: the old constants.Omega default
        rebuilt f_T/f_u/f_v with Earth rotation)."""
        from legoesm import constants
        from legoesm.grids.latlon import ensure_geometry

        geom0 = ensure_geometry(create_latlon_grid(8, omega=0.0))
        assert float(jnp.max(jnp.abs(geom0.f_T))) == 0.0
        assert float(jnp.max(jnp.abs(geom0.f_u))) == 0.0
        assert float(jnp.max(jnp.abs(geom0.f_v))) == 0.0

        geom_e = ensure_geometry(create_latlon_grid(8))
        assert float(jnp.max(jnp.abs(geom_e.f_T))) > 0.0
        # Explicit override still wins over the stored scalar.
        geom_o = ensure_geometry(create_latlon_grid(8),
                                 omega=constants.Omega)
        assert float(jnp.max(jnp.abs(geom_o.f_T))) > 0.0

    def test_geometry_carries_omega_for_operators(self):
        """The operational path (ensure_geometry -> LatLonCGridGeometry)
        must expose the same omega scalar so stagger-rebuilding operators
        (QG-Leith vertex absolute vorticity) see the true rotation
        (codex 2026-07-03 round-6)."""
        from legoesm.grids.latlon import ensure_geometry

        geom0 = ensure_geometry(create_latlon_grid(8, omega=0.0))
        assert geom0.omega == 0.0
        geom_e = ensure_geometry(create_latlon_grid(8))
        from legoesm import constants
        assert geom_e.omega == constants.Omega
