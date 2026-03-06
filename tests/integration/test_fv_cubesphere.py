"""Regression tests for the consistent FV cubed-sphere dynamical cores.

Tests SW, hydrostatic PE, and non-hydrostatic CE with FV transport + divergence
damping on the cubed sphere. Verifies stability, conservation, and
differentiability.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
from legoesm.core.conservation import compute_conservation_diagnostics


# ============================================================================
# Shared fixtures
# ============================================================================

@pytest.fixture(scope="module")
def grid():
    return create_cubed_sphere(16)


# ============================================================================
# Shallow Water FV
# ============================================================================

class TestFVShallowWater:
    """FV shallow water on cubed sphere with divergence damping."""

    @pytest.fixture(scope="class")
    def grid_sw(self):
        return create_cubed_sphere(16)

    @pytest.fixture(scope="class")
    def model_and_state(self, grid_sw):
        from legoesm.atmosphere.dynamics.shallow_water_fv import (
            FVShallowWaterConfig, FVShallowWaterModel,
        )
        from tests.test_cases.williamson import williamson_test2

        dt = 600.0
        nu2, nu4 = default_div_damp_coeffs(grid_sw, dt=dt)
        config = FVShallowWaterConfig(
            div_damp_2=nu2,
            div_damp_4=nu4,
            hyperdiff_coeff=1e15,
            time_integrator="ssp_rk3",
            use_conservation_fixer=True,
            fix_mass=True,
            fix_energy=False,
        )
        model = FVShallowWaterModel(grid_sw, config)
        state = williamson_test2(grid_sw)
        return model, state, dt

    def test_tendencies_finite(self, grid_sw):
        from legoesm.atmosphere.dynamics.shallow_water_fv import (
            FVShallowWaterConfig, fv_shallow_water_tendencies,
        )
        from tests.test_cases.williamson import williamson_test2

        state = williamson_test2(grid_sw)
        nu2, nu4 = default_div_damp_coeffs(grid_sw, dt=600.0)
        config = FVShallowWaterConfig(div_damp_2=nu2, div_damp_4=nu4)
        tend = fv_shallow_water_tendencies(state, grid_sw, config)
        assert jnp.all(jnp.isfinite(tend.dh_dt.data))
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))

    def test_single_step(self, model_and_state):
        model, state, dt = model_and_state
        s1 = model.step(state, dt)
        assert jnp.all(jnp.isfinite(s1.h.data))
        assert jnp.all(jnp.isfinite(s1.u.data))
        assert jnp.all(jnp.isfinite(s1.v.data))

    def test_100_steps_stable(self, model_and_state):
        """100 steps of Williamson 2 should remain stable."""
        model, state, dt = model_and_state
        s = state
        for _ in range(100):
            s = model.step(s, dt)
        assert jnp.all(jnp.isfinite(s.h.data))
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.v.data))
        # Height should not drift far from initial mean
        h_mean_init = float(jnp.mean(state.h.data))
        h_mean_final = float(jnp.mean(s.h.data))
        assert abs(h_mean_final - h_mean_init) / h_mean_init < 1e-3

    def test_mass_conservation(self, model_and_state, grid_sw):
        """Mass should be conserved to near machine precision over 50 steps."""
        model, state, dt = model_and_state
        d0 = compute_conservation_diagnostics(state, grid_sw)
        s = state
        for _ in range(50):
            s = model.step(s, dt)
        d1 = compute_conservation_diagnostics(s, grid_sw)
        mass_drift = abs(float(d1["total_mass"] - d0["total_mass"])) / float(d0["total_mass"])
        assert mass_drift < 1e-10, f"Mass drift {mass_drift:.2e} exceeds threshold"

    def test_williamson5_stable(self, grid_sw):
        """Williamson 5 (mountain) should be stable for 50 steps."""
        from legoesm.atmosphere.dynamics.shallow_water_fv import (
            FVShallowWaterConfig, FVShallowWaterModel,
        )
        from tests.test_cases.williamson import williamson_test5

        dt = 450.0
        nu2, nu4 = default_div_damp_coeffs(grid_sw, dt=dt)
        config = FVShallowWaterConfig(
            div_damp_2=nu2,
            div_damp_4=nu4,
            hyperdiff_coeff=1e15,
            use_conservation_fixer=True,
            fix_mass=True,
            fix_energy=False,
        )
        model = FVShallowWaterModel(grid_sw, config)
        state = williamson_test5(grid_sw)
        s = state
        for _ in range(50):
            s = model.step(s, dt)
        assert jnp.all(jnp.isfinite(s.h.data))

    def test_differentiable_10_steps(self, grid_sw):
        """FV SW should be differentiable through 10 steps via scan."""
        from legoesm.atmosphere.dynamics.shallow_water_fv import (
            FVShallowWaterConfig, FVShallowWaterModel,
        )
        from tests.test_cases.williamson import williamson_test2

        dt = 600.0
        nu2, nu4 = default_div_damp_coeffs(grid_sw, dt=dt)
        config = FVShallowWaterConfig(
            div_damp_2=nu2,
            div_damp_4=nu4,
            hyperdiff_coeff=1e15,
            fix_energy=False,
        )
        model = FVShallowWaterModel(grid_sw, config)
        state = williamson_test2(grid_sw)
        # Ensure float64 for scan type consistency
        state = jax.tree.map(lambda x: x.astype(jnp.float64) if hasattr(x, 'dtype') else x, state)

        def loss_fn(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            final, _ = model.integrate_scan(s, 10, dt)
            return jnp.mean(final.h.data ** 2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.h.data)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 0


# ============================================================================
# Hydrostatic Primitive Equations FV
# ============================================================================

class TestFVPrimitiveEquations:
    """FV hydrostatic PE on cubed sphere with FV-consistent sigma_dot."""

    @pytest.fixture(scope="class")
    def grid_pe(self):
        return create_cubed_sphere(8)

    @pytest.fixture(scope="class")
    def model_state_dt(self, grid_pe):
        from legoesm.atmosphere.dynamics.primitive_eq_fv import (
            FVPrimitiveEquationConfig, FVPrimitiveEquationModel,
        )
        from legoesm.grids.vertical import create_sigma_coordinate

        nlev = 5
        sigma_coord = create_sigma_coordinate(nlev)
        dt = 300.0
        nu2, nu4 = default_div_damp_coeffs(grid_pe, dt=dt)
        config = FVPrimitiveEquationConfig(
            div_damp_2=nu2,
            div_damp_4=nu4,
            hyperdiff_coeff=1e14,
        )
        model = FVPrimitiveEquationModel(grid_pe, sigma_coord, config)

        # Isothermal rest state
        n = grid_pe.n
        T_data = jnp.full((6, n, n, nlev), 250.0)
        u_data = jnp.zeros((6, n, n, nlev))
        v_data = jnp.zeros((6, n, n, nlev))
        ps_data = jnp.full((6, n, n), 1e5)
        phis_data = jnp.zeros((6, n, n))

        from legoesm.core.field import Field
        from legoesm.core.state import HydrostaticState
        state = HydrostaticState(
            u=Field(data=u_data, name="u", dims=("face", "x", "y", "lev"), units="m/s"),
            v=Field(data=v_data, name="v", dims=("face", "x", "y", "lev"), units="m/s"),
            T=Field(data=T_data, name="T", dims=("face", "x", "y", "lev"), units="K"),
            p_s=Field(data=ps_data, name="p_s", dims=("face", "x", "y"), units="Pa"),
            phis=Field(data=phis_data, name="phis", dims=("face", "x", "y"), units="m^2/s^2"),
        )
        return model, state, dt

    def test_tendencies_finite(self, model_state_dt):
        model, state, dt = model_state_dt
        tend = model.tendencies(state)
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dp_s_dt.data))

    def test_single_step(self, model_state_dt):
        model, state, dt = model_state_dt
        s1 = model.step(state, dt)
        assert jnp.all(jnp.isfinite(s1.T.data))
        assert jnp.all(jnp.isfinite(s1.u.data))
        assert jnp.all(jnp.isfinite(s1.p_s.data))

    def test_50_steps_stable(self, model_state_dt):
        """Isothermal rest state should stay stable for 50 steps."""
        model, state, dt = model_state_dt
        s = state
        for _ in range(50):
            s = model.step(s, dt)
        assert jnp.all(jnp.isfinite(s.T.data))
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.p_s.data))
        # p_s should not drift from initial value
        ps_drift = float(jnp.max(jnp.abs(s.p_s.data - state.p_s.data))) / 1e5
        assert ps_drift < 1e-3, f"p_s drift {ps_drift:.2e} too large for rest state"

    def test_differentiable_10_steps(self, grid_pe, model_state_dt):
        """FV PE should be differentiable through 10 steps."""
        model, state, dt = model_state_dt

        def loss_fn(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            final, _ = model.integrate_scan(s, 10, dt)
            return jnp.mean(final.T.data ** 2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.T.data)
        assert jnp.all(jnp.isfinite(g))


# ============================================================================
# Non-Hydrostatic Compressible Euler FV
# ============================================================================

class TestFVCompressibleEuler:
    """FV non-hydrostatic CE on cubed sphere with divergence damping."""

    @pytest.fixture(scope="class")
    def grid_ce(self):
        return create_cubed_sphere(8)

    @pytest.fixture(scope="class")
    def model_state_dt(self, grid_ce):
        from legoesm.atmosphere.dynamics.compressible_euler_fv import (
            FVCompressibleEulerConfig, FVCompressibleEulerModel,
        )
        from legoesm.grids.vertical import create_height_coordinate, compute_terrain_metric

        nlev = 5
        z_top = 30000.0
        height_coord = create_height_coordinate(nlev, z_top)
        terrain = jnp.zeros((6, grid_ce.n, grid_ce.n))
        terrain_metric = compute_terrain_metric(terrain, height_coord)

        dt = 10.0
        nu2, nu4 = default_div_damp_coeffs(grid_ce, dt=dt)
        config = FVCompressibleEulerConfig(
            div_damp_2=nu2,
            div_damp_4=nu4,
            hyperdiff_coeff=1e14,
            n_acoustic_substeps=4,
        )
        model = FVCompressibleEulerModel(grid_ce, height_coord, terrain_metric, config)

        # Rest state (perturbations = 0)
        n = grid_ce.n
        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")

        from legoesm.core.field import Field
        from legoesm.core.state import NonHydrostaticState
        state = NonHydrostaticState(
            u=Field(data=jnp.zeros((6, n, n, nlev)), name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.zeros((6, n, n, nlev)), name="v", dims=dims_3d, units="m/s"),
            w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w", dims=dims_w, units="m/s"),
            theta_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="theta_prime", dims=dims_3d, units="K"),
            rho_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="rho_prime", dims=dims_3d, units="kg/m^3"),
            phis=Field(data=jnp.zeros((6, n, n)), name="phis", dims=dims_2d, units="m^2/s^2"),
            tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers", dims=("face", "x", "y", "level", "tracer"), units="kg/kg"),
        )
        return model, state, dt

    def test_tendencies_finite(self, model_state_dt):
        model, state, dt = model_state_dt
        tend = model.tendencies(state)
        assert jnp.all(jnp.isfinite(tend.du_dt.data))
        assert jnp.all(jnp.isfinite(tend.dv_dt.data))
        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
        assert jnp.all(jnp.isfinite(tend.drho_prime_dt.data))

    def test_single_step(self, model_state_dt):
        model, state, dt = model_state_dt
        s1 = model.step(state, dt)
        assert jnp.all(jnp.isfinite(s1.theta_prime.data))
        assert jnp.all(jnp.isfinite(s1.rho_prime.data))
        assert jnp.all(jnp.isfinite(s1.u.data))

    def test_30_steps_stable(self, model_state_dt):
        """Isothermal rest state should stay stable for 30 steps."""
        model, state, dt = model_state_dt
        s = state
        for _ in range(30):
            s = model.step(s, dt)
        assert jnp.all(jnp.isfinite(s.theta_prime.data))
        assert jnp.all(jnp.isfinite(s.rho_prime.data))
        assert jnp.all(jnp.isfinite(s.u.data))
        # rho_prime should stay near zero for rest state
        rho_drift = float(jnp.max(jnp.abs(s.rho_prime.data)))
        assert rho_drift < 1.0, f"rho_prime drift {rho_drift:.2e} too large for rest state"

    def test_differentiable_10_steps(self, model_state_dt):
        """FV CE should be differentiable through 10 steps."""
        model, state, dt = model_state_dt

        def loss_fn(theta_p_data):
            s = state._replace(theta_prime=state.theta_prime.replace(data=theta_p_data))
            final, _ = model.integrate_scan(s, 10, dt)
            return jnp.mean(final.theta_prime.data ** 2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.theta_prime.data)
        assert jnp.all(jnp.isfinite(g))
