"""Unit tests for the FV hydrostatic primitive equation model on the lat-lon grid."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.primitive_eq_fv_latlon import (
    FVLatLonPrimitiveEquationConfig,
    FVLatLonPrimitiveEquationModel,
    fv_latlon_hydrostatic_tendencies,
)


pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="Requires JAX_ENABLE_X64=1",
)


@pytest.fixture
def grid():
    return create_latlon_grid(8, 16)


@pytest.fixture
def sigma():
    return create_sigma_coordinate(5)


def _make_state(grid, sigma, T_val=300.0, u_val=0.0, v_val=0.0, p_s_val=1e5):
    shape_3d = (grid.n_lat, grid.n_lon, sigma.n_levels)
    shape_2d = (grid.n_lat, grid.n_lon)
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    return HydrostaticState(
        u=Field(data=jnp.ones(shape_3d) * u_val, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.ones(shape_3d) * v_val, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=jnp.ones(shape_3d) * T_val, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=jnp.ones(shape_2d) * p_s_val, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=dims_2d, units="m^2/s^2"),
    )


class TestFVLatLonHydrostaticTendencies:

    def test_isothermal_rest_state_small_tendencies(self, grid, sigma):
        """Isothermal atmosphere at rest should have near-zero tendencies."""
        state = _make_state(grid, sigma, T_val=250.0, u_val=0.0, v_val=0.0)
        config = FVLatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, use_polar_filter=False,
        )

        tend = fv_latlon_hydrostatic_tendencies(state, grid, sigma, config)

        assert jnp.allclose(tend.du_dt.data, 0.0, atol=1e-8), \
            f"du_dt max: {float(jnp.max(jnp.abs(tend.du_dt.data)))}"
        assert jnp.allclose(tend.dv_dt.data, 0.0, atol=1e-8), \
            f"dv_dt max: {float(jnp.max(jnp.abs(tend.dv_dt.data)))}"
        assert jnp.allclose(tend.dT_dt.data, 0.0, atol=1e-6), \
            f"dT_dt max: {float(jnp.max(jnp.abs(tend.dT_dt.data)))}"
        assert jnp.allclose(tend.dp_s_dt.data, 0.0, atol=1e-4), \
            f"dp_s_dt max: {float(jnp.max(jnp.abs(tend.dp_s_dt.data)))}"

    def test_tendencies_finite(self, grid, sigma):
        """Tendencies should be finite for any reasonable state."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        config = FVLatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, use_polar_filter=False,
        )

        tend = fv_latlon_hydrostatic_tendencies(state, grid, sigma, config)

        assert jnp.all(jnp.isfinite(tend.du_dt.data)), "du_dt not finite"
        assert jnp.all(jnp.isfinite(tend.dv_dt.data)), "dv_dt not finite"
        assert jnp.all(jnp.isfinite(tend.dT_dt.data)), "dT_dt not finite"
        assert jnp.all(jnp.isfinite(tend.dp_s_dt.data)), "dp_s_dt not finite"

    def test_tendencies_with_hyperdiffusion(self, grid, sigma):
        """Tendencies with hyperdiffusion should be finite."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        config = FVLatLonPrimitiveEquationConfig(
            hyperdiff_coeff=1e15, use_polar_filter=False,
        )

        tend = fv_latlon_hydrostatic_tendencies(state, grid, sigma, config)

        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))

    def test_tendencies_with_polar_filter(self, grid, sigma):
        """Tendencies with polar filter should be finite."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        config = FVLatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, use_polar_filter=True,
        )
        from legoesm.grids.polar_filter import compute_polar_filter_mask
        mask = compute_polar_filter_mask(grid, dt=300.0)

        tend = fv_latlon_hydrostatic_tendencies(
            state, grid, sigma, config, polar_mask=mask,
        )

        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dp_s_dt.data))


class TestFVLatLonPEModel:

    def test_single_step_finite(self, grid, sigma):
        """A single time step produces finite values."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=5.0, v_val=2.0)
        config = FVLatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, use_polar_filter=True,
        )
        model = FVLatLonPrimitiveEquationModel(grid, sigma, config, dt=300.0)
        state_new = model.step(state, 300.0)

        assert jnp.all(jnp.isfinite(state_new.u.data)), "u not finite after step"
        assert jnp.all(jnp.isfinite(state_new.v.data)), "v not finite after step"
        assert jnp.all(jnp.isfinite(state_new.T.data)), "T not finite after step"
        assert jnp.all(jnp.isfinite(state_new.p_s.data)), "p_s not finite after step"

    def test_multi_step_stability(self, grid, sigma):
        """Model should be stable for multiple time steps."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=5.0, v_val=2.0)
        config = FVLatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, use_polar_filter=True,
        )
        model = FVLatLonPrimitiveEquationModel(grid, sigma, config, dt=300.0)

        for _ in range(10):
            state = model.step(state, 300.0)

        assert jnp.all(jnp.isfinite(state.u.data)), "u not finite after 10 steps"
        assert jnp.all(jnp.isfinite(state.T.data)), "T not finite after 10 steps"

    def test_mass_conservation(self, grid, sigma):
        """Mass should be conserved with the fixer."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        config = FVLatLonPrimitiveEquationConfig(
            fix_mass=True, use_polar_filter=True,
        )
        model = FVLatLonPrimitiveEquationModel(grid, sigma, config, dt=300.0)

        mass_before = float(jnp.sum(state.p_s.data * grid.area))

        for _ in range(20):
            state = model.step(state, 300.0)

        mass_after = float(jnp.sum(state.p_s.data * grid.area))
        rel_err = abs(mass_after - mass_before) / abs(mass_before)
        assert rel_err < 1e-5, f"Mass conservation failed: rel_err={rel_err:.2e}"

    def test_grad_through_step(self, grid, sigma):
        """jax.grad should work through a single step."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=5.0, v_val=0.0)
        config = FVLatLonPrimitiveEquationConfig(
            use_conservation_fixer=False, use_polar_filter=True,
        )
        model = FVLatLonPrimitiveEquationModel(grid, sigma, config, dt=300.0)

        def loss(u_data):
            s = state._replace(u=state.u.replace(data=u_data))
            s_new = model.step(s, 300.0)
            return jnp.mean(s_new.u.data**2)

        grads = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(grads)), "Gradient contains NaN/Inf"

    def test_phis_static(self, grid, sigma):
        """Surface geopotential should not change."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=5.0, v_val=2.0)
        phis_init = state.phis.data
        config = FVLatLonPrimitiveEquationConfig(use_polar_filter=True)
        model = FVLatLonPrimitiveEquationModel(grid, sigma, config, dt=300.0)

        for _ in range(5):
            state = model.step(state, 300.0)

        assert jnp.allclose(state.phis.data, phis_init, atol=1e-12)
