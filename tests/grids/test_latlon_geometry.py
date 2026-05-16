"""Unit tests for ``LatLonCGridGeometry`` + ``ensure_geometry``.

Addresses PR #268 slopbuster WARN #8: ``ensure_geometry`` had no direct
test coverage, and the field-by-field parity between ``LatLonGrid`` and
``LatLonCGridGeometry`` was relied on but never asserted.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.grids.latlon import (
    LatLonCGridGeometry,
    create_latlon_geometry,
    create_latlon_grid,
    ensure_geometry,
)


@pytest.fixture
def regular_grid_pair():
    n_lat, n_lon = 16, 32
    return create_latlon_grid(n_lat, n_lon), create_latlon_geometry(n_lat, n_lon)


class TestEnsureGeometry:
    def test_passthrough_for_geometry(self):
        geom = create_latlon_geometry(8, 16)
        out = ensure_geometry(geom)
        assert out is geom

    def test_converts_latlongrid(self):
        grid = create_latlon_grid(8, 16)
        out = ensure_geometry(grid)
        assert isinstance(out, LatLonCGridGeometry)
        assert out.n_lat == grid.n_lat
        assert out.n_lon == grid.n_lon
        assert float(out.radius) == pytest.approx(grid.radius)

    def test_default_omega_matches_constants(self):
        grid = create_latlon_grid(8, 16)
        out = ensure_geometry(grid)
        out_explicit = ensure_geometry(grid, omega=constants.Omega)
        assert jnp.allclose(out.f_T, out_explicit.f_T)


class TestLatLonGridParity:
    """LatLonGrid ↔ LatLonCGridGeometry field-by-field agreement.

    Operators duck-type between both; if the analytical 2D metric arrays
    in ``LatLonCGridGeometry`` ever drifted from the legacy 1D
    ``LatLonGrid`` fields, the dispatch would silently pick the wrong
    branch.  These tests pin the parity.
    """

    def test_scalar_fields(self, regular_grid_pair):
        grid, geom = regular_grid_pair
        assert geom.n_lat == grid.n_lat
        assert geom.n_lon == grid.n_lon
        assert float(geom.dlon) == pytest.approx(float(grid.dlon))
        assert float(geom.dlat) == pytest.approx(float(grid.dlat))
        assert float(geom.radius) == pytest.approx(float(grid.radius))

    def test_lat_lon_arrays(self, regular_grid_pair):
        grid, geom = regular_grid_pair
        assert jnp.allclose(geom.lat, grid.lat, atol=1e-12)
        assert jnp.allclose(geom.lon, grid.lon, atol=1e-12)
        assert jnp.allclose(geom.cos_lat, grid.cos_lat, atol=1e-12)
        assert jnp.allclose(geom.sin_lat, grid.sin_lat, atol=1e-12)

    def test_coriolis(self, regular_grid_pair):
        grid, geom = regular_grid_pair
        assert jnp.allclose(geom.f, grid.f, atol=1e-12)
        assert jnp.allclose(geom.f_T, grid.f, atol=1e-12)

    def test_area_parity(self, regular_grid_pair):
        grid, geom = regular_grid_pair
        assert jnp.allclose(geom.area, grid.area, atol=1e-8)
        assert float(geom.total_area) == pytest.approx(
            float(grid.total_area), rel=1e-10,
        )

    def test_dx_dy_two_cell_parity(self, regular_grid_pair):
        """LatLonGrid.dx/dy are 2-cell spans; LatLonCGridGeometry exposes
        them via convenience properties that double the single-cell dx_T/dy_T."""
        grid, geom = regular_grid_pair
        assert jnp.allclose(geom.dx, grid.dx, atol=1e-8)
        assert jnp.allclose(geom.dy, grid.dy, atol=1e-8)

    def test_dy_v_constant_for_regular_grid(self, regular_grid_pair):
        _, geom = regular_grid_pair
        ref = float(geom.dy_v[0, 0])
        assert jnp.allclose(geom.dy_v, ref, atol=1e-6)


class TestFoldInactiveOnRegularGeometry:
    def test_fold_is_inactive(self):
        geom = create_latlon_geometry(8, 16)
        assert geom.fold.is_active is False

    def test_perm_T_length(self):
        geom = create_latlon_geometry(8, 16)
        assert geom.fold.perm_T.shape == (16,)
