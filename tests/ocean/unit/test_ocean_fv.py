"""Tests for the C-D grid ocean PE on the cubed-sphere.

Verifies that the cdgrid discretization produces correct, stable,
and differentiable results.
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init import rest_state_ocean
from legoesm.ocean.state import OceanConfig
from legoesm.ocean.dynamics.ocean_pe_cdgrid import ocean_baroclinic_tendencies_cdgrid
from legoesm.ocean.dynamics.ocean_model import OceanModel
from legoesm.core.operators_fv_cubed import default_div_damp_coeffs


@pytest.fixture
def grid():
    return create_cubed_sphere(8)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=10, H_max=4000.0)


@pytest.fixture
def state(grid, z_coord):
    return rest_state_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )


@pytest.fixture
def config(grid):
    nu2, nu4 = default_div_damp_coeffs(grid, dt=3600.0)
    return OceanConfig(
        A_h=1e3, K_h=1e2, A_v=1e-3, K_v=1e-4,
        n_barotropic_substeps=10,
        div_damp_2=nu2,
        div_damp_4=nu4,
    )


class TestOceanCDGridTendencies:
    """C-D grid ocean baroclinic tendencies."""

    def test_tendencies_finite(self, state, grid, z_coord, config):
        cdgrid = create_cubed_sphere_cdgrid(grid)
        tend = ocean_baroclinic_tendencies_cdgrid(state, grid, z_coord, cdgrid, config)
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend.dS_dt.data))
        assert jnp.all(jnp.isfinite(tend.deta_dt.data))

    def test_rest_state_small_tendencies(self, state, grid, z_coord, config):
        cdgrid = create_cubed_sphere_cdgrid(grid)
        tend = ocean_baroclinic_tendencies_cdgrid(state, grid, z_coord, cdgrid, config)
        # Rest state should have near-zero velocity tendencies
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) < 1e-2
        assert float(jnp.max(jnp.abs(tend.dv_dt.data))) < 1e-2

    def test_static_fields_zero(self, state, grid, z_coord, config):
        cdgrid = create_cubed_sphere_cdgrid(grid)
        tend = ocean_baroclinic_tendencies_cdgrid(state, grid, z_coord, cdgrid, config)
        assert jnp.all(tend.dH_bathy_dt.data == 0)
        assert jnp.all(tend.dland_mask_dt.data == 0)


class TestOceanFVModel:
    """FV ocean model integration via OceanModel."""

    def test_single_step(self, grid, z_coord, config, state):
        model = OceanModel(grid, z_coord, config, discretization="cdgrid")
        s1 = model.step(state, 3600.0)
        assert jnp.all(jnp.isfinite(s1.u.data))
        assert jnp.all(jnp.isfinite(s1.T.data))
        assert jnp.all(jnp.isfinite(s1.eta.data))

    def test_shapes_preserved(self, grid, z_coord, config, state):
        model = OceanModel(grid, z_coord, config, discretization="cdgrid")
        s1 = model.step(state, 3600.0)
        assert s1.u.data.shape == state.u.data.shape
        assert s1.T.data.shape == state.T.data.shape
        assert s1.eta.data.shape == state.eta.data.shape

    def test_static_fields_unchanged(self, grid, z_coord, config, state):
        model = OceanModel(grid, z_coord, config, discretization="cdgrid")
        s1 = model.step(state, 3600.0)
        assert jnp.allclose(s1.H_bathy.data, state.H_bathy.data)
        assert jnp.allclose(s1.land_mask.data, state.land_mask.data)

    def test_10_steps_stable(self, grid, z_coord, state):
        nu2, nu4 = default_div_damp_coeffs(grid, dt=3600.0)
        cfg = OceanConfig(
            A_h=1e4, K_h=1e3, A_v=1e-3, K_v=1e-4,
            n_barotropic_substeps=10,
            div_damp_2=nu2,
            div_damp_4=nu4,
        )
        model = OceanModel(grid, z_coord, cfg, discretization="cdgrid")
        s = state
        for _ in range(10):
            s = model.step(s, 3600.0)
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.T.data))
        assert jnp.all(jnp.isfinite(s.eta.data))

    def test_differentiable(self, grid, z_coord, config, state):
        """C-D grid ocean should be differentiable through tendencies."""
        cdgrid = create_cubed_sphere_cdgrid(grid)

        def loss_fn(eta_data):
            s = state._replace(eta=state.eta.replace(data=eta_data))
            tend = ocean_baroclinic_tendencies_cdgrid(s, grid, z_coord, cdgrid, config)
            return jnp.mean(tend.deta_dt.data ** 2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.eta.data)
        assert jnp.all(jnp.isfinite(g))
