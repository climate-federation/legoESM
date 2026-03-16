"""Tests for grid and vertical coordinate protocols."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.protocol import GridProtocol, VerticalCoordProtocol
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import (
    create_sigma_coordinate,
    make_hybrid_levels,
    pressure_from_sigma,
    pressure_from_hybrid,
    dp_from_hybrid,
)


class TestGridProtocol:
    """Verify all grid types satisfy GridProtocol."""

    def test_cubed_sphere_satisfies_protocol(self):
        grid = create_cubed_sphere(4)
        assert isinstance(grid, GridProtocol)

    def test_gaussian_satisfies_protocol(self):
        grid = create_gaussian_grid(10)
        assert isinstance(grid, GridProtocol)

    def test_latlon_satisfies_protocol(self):
        grid = create_latlon_grid(8)
        assert isinstance(grid, GridProtocol)

    def test_cubed_sphere_properties(self):
        grid = create_cubed_sphere(4)
        assert grid.grid_lat.shape == (6, 4, 4)
        assert grid.grid_lon.shape == (6, 4, 4)
        assert grid.grid_area.shape == (6, 4, 4)
        assert float(grid.grid_total_area) > 0
        assert grid.grid_coriolis.shape == (6, 4, 4)
        assert grid.grid_n_columns == 6 * 4 * 4

    def test_gaussian_properties(self):
        grid = create_gaussian_grid(10)
        assert grid.grid_lat.shape == (grid.n_lat, grid.n_lon)
        assert grid.grid_lon.shape == (grid.n_lat, grid.n_lon)
        assert grid.grid_coriolis.shape == (grid.n_lat, grid.n_lon)
        assert grid.grid_n_columns == grid.n_lat * grid.n_lon

    def test_latlon_properties(self):
        grid = create_latlon_grid(8)
        assert grid.grid_lat.shape == (8, 16)
        assert grid.grid_lon.shape == (8, 16)
        assert grid.grid_n_columns == 8 * 16

    def test_cubed_sphere_to_from_columns(self):
        grid = create_cubed_sphere(4)
        field = jnp.ones((6, 4, 4, 5))
        cols = grid.to_columns(field)
        assert cols.shape == (96, 5)
        back = grid.from_columns(cols)
        assert back.shape == (6, 4, 4, 5)
        assert jnp.allclose(field, back)

    def test_gaussian_to_from_columns(self):
        grid = create_gaussian_grid(10)
        field = jnp.ones((grid.n_lat, grid.n_lon, 5))
        cols = grid.to_columns(field)
        assert cols.shape == (grid.n_lat * grid.n_lon, 5)
        back = grid.from_columns(cols)
        assert back.shape == (grid.n_lat, grid.n_lon, 5)
        assert jnp.allclose(field, back)

    def test_latlon_to_from_columns(self):
        grid = create_latlon_grid(8)
        field = jnp.ones((8, 16, 5))
        cols = grid.to_columns(field)
        assert cols.shape == (128, 5)
        back = grid.from_columns(cols)
        assert back.shape == (8, 16, 5)
        assert jnp.allclose(field, back)

    def test_cubed_sphere_2d_roundtrip(self):
        grid = create_cubed_sphere(4)
        field = jnp.ones((6, 4, 4))
        cols = grid.to_columns(field)
        assert cols.shape == (96,)
        back = grid.from_columns(cols)
        assert back.shape == (6, 4, 4)


class TestVerticalCoordProtocol:
    """Verify vertical coordinates satisfy VerticalCoordProtocol."""

    def test_sigma_satisfies_protocol(self):
        coord = create_sigma_coordinate(20)
        assert isinstance(coord, VerticalCoordProtocol)

    def test_hybrid_satisfies_protocol(self):
        coord = make_hybrid_levels(20)
        assert isinstance(coord, VerticalCoordProtocol)

    def test_sigma_pressure_at_full(self):
        coord = create_sigma_coordinate(20)
        p_s = jnp.full((6, 4, 4), 1e5)
        p_full = coord.pressure_at_full(p_s)
        p_full_ref = pressure_from_sigma(coord.sigma_full, p_s)
        assert jnp.allclose(p_full, p_full_ref)

    def test_sigma_pressure_at_half(self):
        coord = create_sigma_coordinate(20)
        p_s = jnp.full((6, 4, 4), 1e5)
        p_half = coord.pressure_at_half(p_s)
        p_half_ref = pressure_from_sigma(coord.sigma_half, p_s)
        assert jnp.allclose(p_half, p_half_ref)

    def test_sigma_layer_thickness(self):
        coord = create_sigma_coordinate(20)
        p_s = jnp.full((6, 4, 4), 1e5)
        dp = coord.layer_thickness_dp(p_s)
        assert dp.shape == (6, 4, 4, 20)

    def test_hybrid_pressure_at_full(self):
        coord = make_hybrid_levels(20)
        p_s = jnp.full((6, 4, 4), 1e5)
        p_full = coord.pressure_at_full(p_s)
        p_full_ref = pressure_from_hybrid(coord, p_s, full=True)
        assert jnp.allclose(p_full, p_full_ref)

    def test_hybrid_pressure_at_half(self):
        coord = make_hybrid_levels(20)
        p_s = jnp.full((6, 4, 4), 1e5)
        p_half = coord.pressure_at_half(p_s)
        p_half_ref = pressure_from_hybrid(coord, p_s, full=False)
        assert jnp.allclose(p_half, p_half_ref)

    def test_hybrid_layer_thickness(self):
        coord = make_hybrid_levels(20)
        p_s = jnp.full((6, 4, 4), 1e5)
        dp = coord.layer_thickness_dp(p_s)
        dp_ref = dp_from_hybrid(coord, p_s)
        assert jnp.allclose(dp, dp_ref)
