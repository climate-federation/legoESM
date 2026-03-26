"""Category 4: Soil Grid -- Geometry & Discretization."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.land.soil_grid import make_soil_grid, make_soil_grid_custom, SoilGridConfig, SoilGrid


class Test4a_DefaultGrid:
    def test_n_layers(self):
        grid = make_soil_grid()
        assert grid.n_layers == 8
        assert grid.dz.shape == (8,)

    def test_all_dz_positive(self):
        grid = make_soil_grid()
        assert jnp.all(grid.dz > 0)

    def test_geometric_spacing(self):
        grid = make_soil_grid()
        ratios = grid.dz[1:] / grid.dz[:-1]
        assert jnp.allclose(ratios, 2.0, rtol=1e-10)

    def test_interface_surface(self):
        grid = make_soil_grid()
        assert jnp.allclose(grid.z_interface[0], 0.0)
        assert jnp.allclose(grid.z_interface[-1], jnp.sum(grid.dz))

    def test_node_midpoints(self):
        grid = make_soil_grid()
        expected = 0.5 * (grid.z_interface[:-1] + grid.z_interface[1:])
        assert jnp.allclose(grid.z_node, expected, atol=1e-12)


class Test4b_TotalDepth:
    def test_total_depth_3m(self):
        config = SoilGridConfig(total_depth=3.0, n_layers=8, growth_factor=2.0)
        grid = make_soil_grid(config)
        assert jnp.allclose(jnp.sum(grid.dz), 3.0, atol=1e-10)
        ratios = grid.dz[1:] / grid.dz[:-1]
        assert jnp.allclose(ratios, 2.0, rtol=1e-10)


class Test4c_CustomGrid:
    def test_custom_dz(self):
        dz_input = [0.05, 0.1, 0.2, 0.4, 0.8]
        grid = make_soil_grid_custom(dz_input)
        assert grid.n_layers == 5
        assert jnp.allclose(grid.dz, jnp.array(dz_input))
        expected_interfaces = jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.array(dz_input))])
        assert jnp.allclose(grid.z_interface, expected_interfaces, atol=1e-12)


class Test4d_InterfaceDistances:
    def test_dz_interface_positive(self):
        grid = make_soil_grid()
        assert jnp.all(grid.dz_interface > 0)
        assert grid.dz_interface.shape == (grid.n_layers - 1,)

    def test_dz_interface_formula(self):
        grid = make_soil_grid()
        expected = grid.z_node[1:] - grid.z_node[:-1]
        assert jnp.allclose(grid.dz_interface, expected, atol=1e-12)


class Test4e_UniformSpacing:
    def test_uniform(self):
        config = SoilGridConfig(n_layers=6, growth_factor=1.0, total_depth=3.0)
        grid = make_soil_grid(config)
        assert jnp.allclose(grid.dz, 0.5, atol=1e-10)
