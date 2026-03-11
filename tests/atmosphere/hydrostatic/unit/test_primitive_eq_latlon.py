"""Unit tests for the lat-lon hydrostatic primitive equation dynamical core."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.core.operators_latlon import global_integral
from legoesm.core.conservation import fix_mass_hydrostatic_latlon
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import (
    create_sigma_coordinate,
    compute_sigma_dot,
)
from legoesm.core.operators_latlon_3d import divergence_3d
from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
    LatLonPrimitiveEquationConfig,
    LatLonPrimitiveEquationModel,
    latlon_hydrostatic_tendencies,
)
from legoesm.grids.polar_filter import compute_polar_filter_mask


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def grid():
    """Small lat-lon grid for testing."""
    return create_latlon_grid(8, 16)


@pytest.fixture
def sigma():
    """5-level sigma coordinate for testing."""
    return create_sigma_coordinate(5)


@pytest.fixture
def sigma_20():
    """20-level sigma coordinate for testing."""
    return create_sigma_coordinate(20)


def _make_state(grid, sigma, T_val=300.0, u_val=0.0, v_val=0.0, p_s_val=1e5):
    """Helper: create a HydrostaticState with uniform fields on lat-lon."""
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


# ==============================================================================
# Hydrostatic Tendency Tests
# ==============================================================================

class TestLatLonHydrostaticTendencies:
    """Tests for the lat-lon hydrostatic PE tendency computation."""

    def test_isothermal_rest_state_small_tendencies(self, grid, sigma):
        """Isothermal atmosphere at rest should have near-zero tendencies."""
        state = _make_state(grid, sigma, T_val=250.0, u_val=0.0, v_val=0.0)
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, use_polar_filter=False,
        )

        tend = latlon_hydrostatic_tendencies(state, grid, sigma, config)

        assert jnp.allclose(tend.du_dt.data, 0.0, atol=1e-8), \
            f"du_dt max: {float(jnp.max(jnp.abs(tend.du_dt.data)))}"
        assert jnp.allclose(tend.dv_dt.data, 0.0, atol=1e-8), \
            f"dv_dt max: {float(jnp.max(jnp.abs(tend.dv_dt.data)))}"
        assert jnp.allclose(tend.dT_dt.data, 0.0, atol=1e-6), \
            f"dT_dt max: {float(jnp.max(jnp.abs(tend.dT_dt.data)))}"
        assert jnp.allclose(tend.dp_s_dt.data, 0.0, atol=1e-4), \
            f"dp_s_dt max: {float(jnp.max(jnp.abs(tend.dp_s_dt.data)))}"
        assert jnp.allclose(tend.dphis_dt.data, 0.0, atol=1e-15)

    def test_tendencies_finite(self, grid, sigma):
        """Tendencies should be finite for any reasonable state."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, use_polar_filter=False,
        )

        tend = latlon_hydrostatic_tendencies(state, grid, sigma, config)

        assert jnp.all(jnp.isfinite(tend.du_dt.data)), "du_dt not finite"
        assert jnp.all(jnp.isfinite(tend.dv_dt.data)), "dv_dt not finite"
        assert jnp.all(jnp.isfinite(tend.dT_dt.data)), "dT_dt not finite"
        assert jnp.all(jnp.isfinite(tend.dp_s_dt.data)), "dp_s_dt not finite"

    def test_tendencies_with_hyperdiffusion(self, grid, sigma):
        """Tendencies with hyperdiffusion should be finite."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=1e15, use_polar_filter=False,
        )

        tend = latlon_hydrostatic_tendencies(state, grid, sigma, config)

        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))

    def test_tendencies_with_polar_filter(self, grid, sigma):
        """Tendencies with polar filter should be finite."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, use_polar_filter=True,
        )
        polar_mask = compute_polar_filter_mask(grid, dt=600.0)

        tend = latlon_hydrostatic_tendencies(
            state, grid, sigma, config, polar_mask=polar_mask,
        )

        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend.dp_s_dt.data))

    def test_tendency_pytree_structure(self, grid, sigma):
        """Tendency should have same structure as state."""
        state = _make_state(grid, sigma)
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, use_polar_filter=False,
        )

        tend = latlon_hydrostatic_tendencies(state, grid, sigma, config)

        assert isinstance(tend, HydrostaticTendencies)
        assert tend.du_dt.data.shape == state.u.data.shape
        assert tend.dv_dt.data.shape == state.v.data.shape
        assert tend.dT_dt.data.shape == state.T.data.shape
        assert tend.dp_s_dt.data.shape == state.p_s.data.shape
        assert tend.dphis_dt.data.shape == state.phis.data.shape


# ==============================================================================
# Model Integration Tests
# ==============================================================================

class TestLatLonPrimitiveEquationModel:
    """Tests for the LatLonPrimitiveEquationModel class."""

    def test_single_step_finite(self, grid, sigma):
        """A single time step should produce finite results."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=5.0, v_val=2.0)
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            use_polar_filter=False,
        )
        model = LatLonPrimitiveEquationModel(grid, sigma, config)

        state_new = model.step(state, dt=300.0)

        assert jnp.all(jnp.isfinite(state_new.u.data)), "u not finite after step"
        assert jnp.all(jnp.isfinite(state_new.v.data)), "v not finite after step"
        assert jnp.all(jnp.isfinite(state_new.T.data)), "T not finite after step"
        assert jnp.all(jnp.isfinite(state_new.p_s.data)), "p_s not finite after step"

    def test_rest_state_stays_at_rest(self, grid, sigma):
        """A rest state should remain nearly at rest after one step."""
        state = _make_state(grid, sigma, T_val=250.0, u_val=0.0, v_val=0.0)
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            use_polar_filter=False,
        )
        model = LatLonPrimitiveEquationModel(grid, sigma, config)

        state_new = model.step(state, dt=300.0)

        assert jnp.allclose(state_new.u.data, 0.0, atol=1e-4), \
            f"u drifted: max={float(jnp.max(jnp.abs(state_new.u.data)))}"
        assert jnp.allclose(state_new.v.data, 0.0, atol=1e-4), \
            f"v drifted: max={float(jnp.max(jnp.abs(state_new.v.data)))}"

    def test_mass_conservation_with_fixer(self, grid, sigma):
        """Mass fixer should preserve global dry air mass."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=True,
            fix_mass=True,
            use_polar_filter=False,
        )
        model = LatLonPrimitiveEquationModel(grid, sigma, config)

        mass_before = float(global_integral(state.p_s, grid))
        state_new = model.step(state, dt=300.0)
        mass_after = float(global_integral(state_new.p_s, grid))

        assert abs(mass_after - mass_before) / abs(mass_before) < 1e-5, (
            f"Mass not conserved: before={mass_before:.6e}, after={mass_after:.6e}"
        )

    def test_step_with_physics(self, grid, sigma):
        """step_with_physics should work with Held-Suarez forcing."""
        from legoesm.atmosphere.physics.held_suarez_latlon import (
            held_suarez_forcing_latlon,
            held_suarez_init_latlon,
        )

        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            use_polar_filter=False,
        )
        model = LatLonPrimitiveEquationModel(grid, sigma, config)
        state = held_suarez_init_latlon(grid, sigma)

        state_new = model.step_with_physics(state, 300.0, held_suarez_forcing_latlon)

        assert jnp.all(jnp.isfinite(state_new.u.data))
        assert jnp.all(jnp.isfinite(state_new.v.data))
        assert jnp.all(jnp.isfinite(state_new.T.data))
        assert jnp.all(jnp.isfinite(state_new.p_s.data))


# ==============================================================================
# Conservation Tests
# ==============================================================================

class TestMassFixerLatLon:
    """Tests for the lat-lon hydrostatic mass fixer."""

    def test_mass_fixer_exact(self, grid, sigma):
        """Mass fixer should restore exact mass."""
        state_old = _make_state(grid, sigma, T_val=280.0, p_s_val=1e5)

        key = jax.random.PRNGKey(0)
        perturbation = jax.random.normal(key, state_old.p_s.data.shape) * 100.0
        state_new = state_old._replace(
            p_s=state_old.p_s.replace(data=state_old.p_s.data + perturbation)
        )

        state_fixed = fix_mass_hydrostatic_latlon(state_new, state_old, grid)

        mass_old = float(global_integral(state_old.p_s, grid))
        mass_fixed = float(global_integral(state_fixed.p_s, grid))
        assert abs(mass_fixed - mass_old) / abs(mass_old) < 1e-5


# ==============================================================================
# Continuity Closure Tests
# ==============================================================================

class TestContinuityClosureLatLon:
    """Test that the discrete continuity equation is exactly satisfied."""

    def test_continuity_closure(self, grid, sigma):
        """Continuity closure with uniform p_s."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=10.0, v_val=5.0)
        u, v = state.u.data, state.v.data
        p_s = state.p_s.data
        dsigma = sigma.dsigma
        sigma_top = float(sigma.sigma_half[0])
        sigma_range = 1.0 - sigma_top

        div_v = divergence_3d(u, v, grid)
        D_total = jnp.sum(div_v * dsigma[None, None, :], axis=-1)

        dp_s_dt = -p_s * D_total / sigma_range
        dlnps_dt = dp_s_dt / p_s
        sigma_dot = compute_sigma_dot(div_v, sigma)

        for k in range(sigma.n_levels):
            dsigma_dot = (sigma_dot[..., k + 1] - sigma_dot[..., k]) / dsigma[k]
            residual = dlnps_dt + div_v[..., k] + dsigma_dot
            max_res = float(jnp.max(jnp.abs(residual)))
            max_div = float(jnp.max(jnp.abs(div_v[..., k])))
            assert max_res < 1e-6 * max(max_div, 1e-15), (
                f"Level {k}: continuity residual {max_res:.2e} too large "
                f"relative to div_v {max_div:.2e}"
            )


# ==============================================================================
# Differentiability Tests
# ==============================================================================

class TestDifferentiabilityLatLon:
    """Tests that the lat-lon PE model is differentiable."""

    def test_grad_through_tendencies(self, grid, sigma):
        """jax.grad should work through tendency computation."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=5.0, v_val=2.0)
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, use_polar_filter=False,
        )

        def loss(u_data):
            state_new = state._replace(u=state.u.replace(data=u_data))
            tend = latlon_hydrostatic_tendencies(state_new, grid, sigma, config)
            return jnp.sum(tend.du_dt.data ** 2)

        grads = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(grads)), "Gradients through tendencies are not finite"

    def test_grad_through_single_step(self, grid, sigma):
        """jax.grad should work through a single model step."""
        state = _make_state(grid, sigma, T_val=280.0, u_val=5.0, v_val=2.0)
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            use_polar_filter=False,
        )
        model = LatLonPrimitiveEquationModel(grid, sigma, config)

        def loss(u_data):
            state_new = state._replace(u=state.u.replace(data=u_data))
            state_stepped = model.step(state_new, dt=60.0)
            return jnp.sum(state_stepped.u.data ** 2)

        grads = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(grads)), "Gradients through model step are not finite"


# ==============================================================================
# Held-Suarez Temperature Stability
# ==============================================================================

class TestTemperatureStabilityLatLon:
    """Regression tests for temperature staying within physical bounds."""

    def test_held_suarez_temperature_bounds(self, grid, sigma_20):
        """Held-Suarez forcing should keep T within bounds over 50 steps."""
        from legoesm.atmosphere.physics.held_suarez_latlon import (
            held_suarez_forcing_latlon,
            held_suarez_init_latlon,
        )

        model = LatLonPrimitiveEquationModel(
            grid, sigma_20,
            LatLonPrimitiveEquationConfig(
                hyperdiff_coeff=1e18,
                use_conservation_fixer=True,
                fix_mass=True,
                use_polar_filter=True,
            ),
        )
        state = held_suarez_init_latlon(grid, sigma_20)

        # Run 50 steps with dt=600s (~8 hours)
        for _ in range(50):
            state = model.step_with_physics(state, 600.0, held_suarez_forcing_latlon)

        T = state.T.data
        T_min = float(jnp.min(T))
        T_max = float(jnp.max(T))

        assert T_min > 100.0, (
            f"Temperature too cold: T_min = {T_min:.1f} K (expected > 100 K)"
        )
        assert T_max < 400.0, (
            f"Temperature too hot: T_max = {T_max:.1f} K (expected < 400 K)"
        )
        assert jnp.all(jnp.isfinite(T)), "Non-finite temperatures detected"
