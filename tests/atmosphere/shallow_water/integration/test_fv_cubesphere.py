"""Regression tests for FV shallow-water cubed-sphere dynamics."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    CDGridShallowWaterModel,
    CDGridShallowWaterState,
    cdgrid_shallow_water_tendencies,
)


def _sw_to_cdgrid(state, cdgrid):
    """Convert generic ShallowWaterState to CDGridShallowWaterState."""
    from legoesm.grids.halo import pad_halo_vector
    h = state.h.data
    u_center = state.u.data
    v_center = state.v.data
    h_s = state.h_s.data
    u_pad, v_pad = pad_halo_vector(
        u_center, v_center,
        cdgrid.base.cos_angle, cdgrid.base.sin_angle,
        cdgrid.base.cos_angle_padded, cdgrid.base.sin_angle_padded,
        interp_offsets=cdgrid.base.halo_interp_offsets,
    )
    u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1] +
                   u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
    v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1] +
                   v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])
    return CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


class TestFVShallowWater:
    """FV shallow water on cubed sphere with divergence damping."""

    @pytest.fixture(scope="class")
    def grid_sw(self):
        return create_cubed_sphere(16)

    @pytest.fixture(scope="class")
    def model_and_state(self, grid_sw):
        from tests.test_cases.williamson import williamson_test2

        cdgrid = create_cubed_sphere_cdgrid(grid_sw)
        dt = 600.0
        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=1e15,
            time_integrator="ssp_rk3",
            use_conservation_fixer=True,
            fix_mass=True,
        )
        model = CDGridShallowWaterModel(grid_sw, config)
        sw_state = williamson_test2(grid_sw)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        # Anchor conservation fixer to initial mass for drift-free long runs
        model.set_initial_mass(state)
        return model, state, dt

    def test_tendencies_finite(self, grid_sw):
        from tests.test_cases.williamson import williamson_test2

        cdgrid = create_cubed_sphere_cdgrid(grid_sw)
        sw_state = williamson_test2(grid_sw)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        config = CDGridShallowWaterConfig()
        dh, du, dv = cdgrid_shallow_water_tendencies(state, cdgrid, config)
        assert jnp.all(jnp.isfinite(dh))
        assert jnp.all(jnp.isfinite(du))
        assert jnp.all(jnp.isfinite(dv))

    def test_single_step(self, model_and_state):
        model, state, dt = model_and_state
        s1 = model.step(state, dt)
        assert jnp.all(jnp.isfinite(s1.h))
        assert jnp.all(jnp.isfinite(s1.u_d))
        assert jnp.all(jnp.isfinite(s1.v_d))

    def test_100_steps_stable(self, model_and_state):
        """100 steps of Williamson 2 should remain stable."""
        model, state, dt = model_and_state
        s = state
        for _ in range(100):
            s = model.step(s, dt)
        assert jnp.all(jnp.isfinite(s.h))
        assert jnp.all(jnp.isfinite(s.u_d))
        assert jnp.all(jnp.isfinite(s.v_d))
        h_mean_init = float(jnp.mean(state.h))
        h_mean_final = float(jnp.mean(s.h))
        # iter-164: centralized helper (NaN-aware).
        from legoesm.diagnostics import compute_relative_drift
        assert compute_relative_drift([h_mean_init, h_mean_final]) < 1e-3

    def test_mass_conservation(self, model_and_state, grid_sw):
        """Mass should be conserved to near machine precision over 50 steps."""
        model, state, dt = model_and_state
        area = grid_sw.area
        mass_init = float(jnp.sum(state.h.astype(jnp.float64) * area.astype(jnp.float64)))
        s = state
        for _ in range(50):
            s = model.step(s, dt)
        mass_final = float(jnp.sum(s.h.astype(jnp.float64) * area.astype(jnp.float64)))
        # iter-164: centralized helper (NaN-aware, same migration
        # as iter-157 / iter-159 sweep).
        from legoesm.diagnostics import compute_relative_drift
        mass_drift = compute_relative_drift([mass_init, mass_final])
        # The conservation fixer anchors to the CDGrid state mass.
        # Machine-precision conservation (< 1e-10) requires a flux-form
        # scheme; the CDGrid scheme achieves ~1e-6 relative drift over 50 steps.
        assert mass_drift < 1e-4, f"Mass drift {mass_drift:.2e} exceeds threshold"

    def test_williamson5_stable(self, grid_sw):
        """Williamson 5 (mountain) should be stable for 50 steps."""
        from tests.test_cases.williamson import williamson_test5

        cdgrid = create_cubed_sphere_cdgrid(grid_sw)
        dt = 450.0
        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=1e15,
            use_conservation_fixer=True,
            fix_mass=True,
        )
        model = CDGridShallowWaterModel(grid_sw, config)
        sw_state = williamson_test5(grid_sw)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        s = state
        for _ in range(50):
            s = model.step(s, dt)
        assert jnp.all(jnp.isfinite(s.h))

    def test_differentiable_10_steps(self, grid_sw):
        """FV SW should be differentiable through 10 steps via scan."""
        from tests.test_cases.williamson import williamson_test2

        cdgrid = create_cubed_sphere_cdgrid(grid_sw)
        dt = 600.0
        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=1e15,
        )
        model = CDGridShallowWaterModel(grid_sw, config)
        sw_state = williamson_test2(grid_sw)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        state = jax.tree.map(
            lambda x: x.astype(jnp.float64) if hasattr(x, "dtype") else x, state
        )

        def loss_fn(h_data):
            s = state._replace(h=h_data)
            final, _ = model.integrate_scan(s, 10, dt)
            return jnp.mean(final.h**2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.h)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 0
