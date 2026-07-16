"""Regression tests for FV non-hydrostatic Euler on cubed sphere."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere


class TestFVCompressibleEuler:
    """FV non-hydrostatic CE on cubed sphere with divergence damping."""

    @pytest.fixture(scope="class")
    def grid_ce(self):
        return create_cubed_sphere(8)

    @pytest.fixture(scope="class")
    def model_state_dt(self, grid_ce):
        from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
            CDGridCompressibleEulerConfig as FVCompressibleEulerConfig,
            CDGridCompressibleEulerModel as FVCompressibleEulerModel,
        )
        from legoesm.grids.vertical import create_height_coordinate, compute_terrain_metric

        nlev = 5
        z_top = 30000.0
        height_coord = create_height_coordinate(nlev, z_top)
        terrain = jnp.zeros((6, grid_ce.n, grid_ce.n))
        terrain_metric = compute_terrain_metric(terrain, height_coord)

        dt = 10.0
        config = FVCompressibleEulerConfig(
            hyperdiff_coeff=1e14,
            n_acoustic_substeps=4,
        )
        model = FVCompressibleEulerModel(grid_ce, height_coord, terrain_metric, config)

        n = grid_ce.n
        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")

        from legoesm.core.field import Field
        from legoesm.core.state import NonHydrostaticState

        state = NonHydrostaticState(
            u=Field(data=jnp.zeros((6, n, n, nlev)), name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.zeros((6, n, n, nlev)), name="v", dims=dims_3d, units="m/s"),
            w=Field(
                data=jnp.zeros((6, n, n, nlev + 1)),
                name="w",
                dims=dims_w,
                units="m/s",
            ),
            theta_prime=Field(
                data=jnp.zeros((6, n, n, nlev)),
                name="theta_prime",
                dims=dims_3d,
                units="K",
            ),
            rho_prime=Field(
                data=jnp.zeros((6, n, n, nlev)),
                name="rho_prime",
                dims=dims_3d,
                units="kg/m^3",
            ),
            phis=Field(data=jnp.zeros((6, n, n)), name="phis", dims=dims_2d, units="m^2/s^2"),
            tracers=Field(
                data=jnp.zeros((6, n, n, nlev, 0)),
                name="tracers",
                dims=("face", "x", "y", "level", "tracer"),
                units="kg/kg",
            ),
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
        rho_drift = float(jnp.max(jnp.abs(s.rho_prime.data)))
        assert rho_drift < 1.0, f"rho_prime drift {rho_drift:.2e} too large for rest state"

    def test_differentiable_10_steps(self, model_state_dt):
        """FV CE should be differentiable through 10 steps."""
        model, state, dt = model_state_dt

        def loss_fn(theta_p_data):
            s = state._replace(theta_prime=state.theta_prime.replace(data=theta_p_data))
            final, _ = model.integrate_scan(s, 10, dt)
            return jnp.mean(final.theta_prime.data**2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.theta_prime.data)
        assert jnp.all(jnp.isfinite(g))
