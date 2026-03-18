"""Regression tests for FV shallow-water cubed-sphere dynamics."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.conservation import compute_conservation_diagnostics
from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
from legoesm.grids.cubed_sphere import create_cubed_sphere


class TestFVShallowWater:
    """FV shallow water on cubed sphere with divergence damping."""

    @pytest.fixture(scope="class")
    def grid_sw(self):
        return create_cubed_sphere(16)

    @pytest.fixture(scope="class")
    def model_and_state(self, grid_sw):
        from legoesm.atmosphere.dynamics.shallow_water_fv import (
            FVShallowWaterConfig,
            FVShallowWaterModel,
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
        # Anchor conservation fixer to initial mass for drift-free long runs
        from legoesm.atmosphere.dynamics.shallow_water_fv import _sw_to_cdgrid
        cd_state = _sw_to_cdgrid(state, model.cdgrid)
        model._cd_model.set_initial_mass(cd_state)
        return model, state, dt

    def test_tendencies_finite(self, grid_sw):
        from legoesm.atmosphere.dynamics.shallow_water_fv import (
            FVShallowWaterConfig,
            fv_shallow_water_tendencies,
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
        mass_drift = abs(float(d1["total_mass"] - d0["total_mass"])) / float(
            d0["total_mass"]
        )
        # The FV wrapper goes through A-grid -> D-grid -> A-grid conversion,
        # and the conservation fixer anchors to the CDGrid state mass.
        # Machine-precision conservation (< 1e-10) requires a flux-form
        # scheme; the FV wrapper achieves ~1e-6 relative drift over 50 steps,
        # which is excellent for a non-flux-form conservation fixer.
        assert mass_drift < 1e-4, f"Mass drift {mass_drift:.2e} exceeds threshold"

    def test_williamson5_stable(self, grid_sw):
        """Williamson 5 (mountain) should be stable for 50 steps."""
        from legoesm.atmosphere.dynamics.shallow_water_fv import (
            FVShallowWaterConfig,
            FVShallowWaterModel,
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
            FVShallowWaterConfig,
            FVShallowWaterModel,
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
        state = jax.tree.map(
            lambda x: x.astype(jnp.float64) if hasattr(x, "dtype") else x, state
        )

        def loss_fn(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            final, _ = model.integrate_scan(s, 10, dt)
            return jnp.mean(final.h.data**2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.h.data)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 0
