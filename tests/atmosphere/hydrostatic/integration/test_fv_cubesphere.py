"""Regression tests for FV hydrostatic primitive equations on cubed sphere."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
from legoesm.grids.cubed_sphere import create_cubed_sphere


class TestFVPrimitiveEquations:
    """FV hydrostatic PE on cubed sphere with FV-consistent sigma_dot."""

    @pytest.fixture(scope="class")
    def grid_pe(self):
        return create_cubed_sphere(8)

    @pytest.fixture(scope="class")
    def model_state_dt(self, grid_pe):
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationConfig as FVPrimitiveEquationConfig,
            CDGridPrimitiveEquationModel as FVPrimitiveEquationModel,
        )
        from legoesm.grids.vertical import create_sigma_coordinate

        nlev = 5
        sigma_coord = create_sigma_coordinate(nlev)
        dt = 300.0
        nu2, _nu4 = default_div_damp_coeffs(grid_pe, dt=dt)
        config = FVPrimitiveEquationConfig(
            div_damp_coeff=nu2,
            hyperdiff_coeff=1e14,
        )
        model = FVPrimitiveEquationModel(grid_pe, sigma_coord, config)

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
            phis=Field(
                data=phis_data, name="phis", dims=("face", "x", "y"), units="m^2/s^2"
            ),
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
        ps_drift = float(jnp.max(jnp.abs(s.p_s.data - state.p_s.data))) / 1e5
        assert ps_drift < 5e-3, f"p_s drift {ps_drift:.2e} too large for rest state"

    def test_differentiable_10_steps(self, grid_pe, model_state_dt):
        """FV PE should be differentiable through 10 steps."""
        model, state, dt = model_state_dt

        def loss_fn(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            final, _ = model.integrate_scan(s, 10, dt)
            return jnp.mean(final.T.data**2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.T.data)
        assert jnp.all(jnp.isfinite(g))
