"""Tests for the lat-lon FV ocean model.

Verifies that the lat-lon finite-volume ocean discretization produces
correct, stable, and differentiable results.  Mirrors the structure of
``test_ocean_fv.py`` (cubed-sphere C-D grid) for parity.
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon import rest_state_latlon_ocean
from legoesm.ocean.state import LatLonOceanConfig
from legoesm.ocean.dynamics.ocean_pe_latlon import latlon_ocean_baroclinic_tendencies
from legoesm.ocean.dynamics.ocean_model_latlon import LatLonOceanModel


@pytest.fixture
def grid():
    return create_latlon_grid(n_lat=36, n_lon=72)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=5, H_max=4000.0)


@pytest.fixture
def state(grid, z_coord):
    return rest_state_latlon_ocean(
        grid, z_coord,
        T_surface=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )


@pytest.fixture
def config():
    return LatLonOceanConfig()


class TestLatLonTendencies:
    """Lat-lon ocean baroclinic tendencies."""

    def test_tendencies_finite(self, state, grid, z_coord, config):
        tend = latlon_ocean_baroclinic_tendencies(state, grid, z_coord, config)
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
        assert jnp.all(jnp.isfinite(tend.deta_dt.data))

    def test_rest_state_small_tendencies(self, state, grid, z_coord, config):
        tend = latlon_ocean_baroclinic_tendencies(state, grid, z_coord, config)
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) < 1e-2
        assert float(jnp.max(jnp.abs(tend.dv_dt.data))) < 1e-2

    def test_static_fields_zero(self, state, grid, z_coord, config):
        tend = latlon_ocean_baroclinic_tendencies(state, grid, z_coord, config)
        assert jnp.all(tend.dH_bathy_dt.data == 0)
        assert jnp.all(tend.dland_mask_dt.data == 0)


class TestLatLonOceanFVModel:
    """Lat-lon FV ocean model integration."""

    def test_single_step(self, grid, z_coord, config, state):
        model = LatLonOceanModel(grid, z_coord, config)
        s1 = model.step(state, 3600.0)
        assert jnp.all(jnp.isfinite(s1.u.data))
        assert jnp.all(jnp.isfinite(s1.T.data))
        assert jnp.all(jnp.isfinite(s1.eta.data))

    def test_shapes_preserved(self, grid, z_coord, config, state):
        model = LatLonOceanModel(grid, z_coord, config)
        s1 = model.step(state, 3600.0)
        assert s1.u.data.shape == state.u.data.shape
        assert s1.T.data.shape == state.T.data.shape
        assert s1.eta.data.shape == state.eta.data.shape

    def test_static_fields_unchanged(self, grid, z_coord, config, state):
        model = LatLonOceanModel(grid, z_coord, config)
        s1 = model.step(state, 3600.0)
        assert jnp.allclose(s1.H_bathy.data, state.H_bathy.data)
        assert jnp.allclose(s1.land_mask.data, state.land_mask.data)

    def test_10_steps_stable(self, grid, z_coord, state):
        cfg = LatLonOceanConfig(A_h=1e4, K_h=1e3, A_v=1e-3, K_v=1e-4)
        model = LatLonOceanModel(grid, z_coord, cfg)
        s = state
        for _ in range(10):
            s = model.step(s, 3600.0)
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.T.data))
        assert jnp.all(jnp.isfinite(s.eta.data))

    def test_differentiable(self, grid, z_coord, config, state):
        """Lat-lon FV ocean should be differentiable through tendencies."""
        def loss_fn(eta_data):
            s = state._replace(eta=state.eta.replace(data=eta_data))
            tend = latlon_ocean_baroclinic_tendencies(s, grid, z_coord, config)
            return jnp.mean(tend.deta_dt.data ** 2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.eta.data)
        assert jnp.all(jnp.isfinite(g))
