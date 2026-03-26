"""Differentiability tests for land model components.

Categories:
  4a) Slab land model
  4b) Multi-layer land model
  4c) Snow budget
  4d) Carbon cycle (GPP)
  4e) Stomatal conductance
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field


def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac*100:.1f}% non-zero (need {min_nonzero_frac*100:.0f}%)"
    )


def make_atm_forcing(shape):
    """Minimal AtmToSurface forcing with realistic values."""
    from legoesm.coupler.coupling_fields import AtmToSurface
    ones = jnp.ones(shape)
    return AtmToSurface(
        sw_down=200.0 * ones,
        lw_down=300.0 * ones,
        precip_total=1e-5 * ones,
        precip_snow=0.0 * ones,
        T_lowest=280.0 * ones,
        q_lowest=5e-3 * ones,
        u_lowest=5.0 * ones,
        v_lowest=2.0 * ones,
        p_lowest=1e5 * ones,
        p_surface=1.013e5 * ones,
        rho_lowest=1.2 * ones,
        cos_zenith=0.7 * ones,
        co2_ppmv=400.0 * ones,
        has_radiation=1.0 * ones,
        has_precipitation=1.0 * ones,
    )


# ============================================================================
# 4a  Slab land model
# ============================================================================

class TestSlabLandGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig
        from legoesm.land.state import LandState
        self.step_land = step_land
        self.config = LandConfig()

        ncol = 32
        self.state = LandState(
            T_soil=Field(280.0 * jnp.ones(ncol), name="T_soil"),
            W_bucket=Field(50.0 * jnp.ones(ncol), name="W_bucket"),
            snow_depth=Field(jnp.zeros(ncol), name="snow_depth"),
            snow_age=Field(jnp.zeros(ncol), name="snow_age"),
        )
        self.forcing = make_atm_forcing((ncol,))
        self.dt = 60.0

    def test_grad_wrt_T_soil(self):
        step_fn, state, forcing, config, dt = (
            self.step_land, self.state, self.forcing, self.config, self.dt
        )

        def loss(T_data):
            s = state._replace(T_soil=state.T_soil.replace(data=T_data))
            out, _, _ = step_fn(s, forcing, config, U_min=1.0, dt=dt)
            return jnp.sum(out.T_soil.data ** 2)

        grad = jax.grad(loss)(state.T_soil.data)
        assert_gradient_ok(grad, "Slab land w.r.t. T_soil")

    def test_grad_wrt_sw_down(self):
        step_fn, state, forcing, config, dt = (
            self.step_land, self.state, self.forcing, self.config, self.dt
        )

        def loss(sw):
            f = forcing._replace(sw_down=sw)
            out, _, _ = step_fn(state, f, config, U_min=1.0, dt=dt)
            return jnp.sum(out.T_soil.data ** 2)

        grad = jax.grad(loss)(forcing.sw_down)
        assert_gradient_ok(grad, "Slab land w.r.t. sw_down")


# ============================================================================
# 4b  Multi-layer land model
# ============================================================================

class TestMultiLayerLandGrad:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state
        from legoesm.land.config import MultiLayerLandConfig
        self.step_fn = step_multilayer_land

        ncol = 32
        self.config = MultiLayerLandConfig()
        self.state = init_multilayer_land_state(ncol, self.config)
        self.forcing = make_atm_forcing((ncol,))
        self.dt = 60.0

    def test_grad_wrt_T_soil(self):
        step_fn, state, forcing, config, dt = (
            self.step_fn, self.state, self.forcing, self.config, self.dt
        )

        def loss(T_data):
            s = state._replace(T_soil=T_data)
            out, _, _ = step_fn(s, forcing, config, U_min=1.0, dt=dt)
            return jnp.sum(out.T_soil ** 2)

        grad = jax.grad(loss)(state.T_soil)
        assert_gradient_ok(grad, "Multi-layer land w.r.t. T_soil")


# ============================================================================
# 4c  Snow budget
# ============================================================================

class TestSnowBudgetGrad:

    def test_grad_wrt_T_surface(self):
        from legoesm.land.snow_budget import update_snow

        ncol = 32
        snow = 10.0 * jnp.ones(ncol)  # 10 kg/m2 SWE
        snow_age = 1000.0 * jnp.ones(ncol)
        precip_snow = 1e-5 * jnp.ones(ncol)

        def loss(T_surface):
            snow_new, _, _ = update_snow(
                snow, snow_age, T_surface, precip_snow, dt=3600.0,
            )
            return jnp.sum(snow_new ** 2)

        T_sfc = 275.0 * jnp.ones(ncol)
        grad = jax.grad(loss)(T_sfc)
        assert_gradient_ok(grad, "Snow budget w.r.t. T_surface")


# ============================================================================
# 4d  Carbon cycle — GPP
# ============================================================================

class TestCarbonGPPGrad:

    def test_grad_wrt_T(self):
        from legoesm.land.carbon.carbon_cycle import compute_gpp
        from legoesm.land.carbon.config import CarbonConfig

        config = CarbonConfig(scheme="differland")
        ncol = 32

        def loss(T):
            gpp = compute_gpp(
                sw_down=200.0 * jnp.ones(ncol),
                T=T,
                LAI=3.0 * jnp.ones(ncol),
                co2_ppmv=400.0 * jnp.ones(ncol),
                beta=0.8 * jnp.ones(ncol),
                config=config,
            )
            return jnp.sum(gpp ** 2)

        T = 290.0 * jnp.ones(ncol)
        grad = jax.grad(loss)(T)
        assert_gradient_ok(grad, "GPP w.r.t. T")


# ============================================================================
# 4e  Stomatal conductance
# ============================================================================

class TestStomataGrad:

    def test_jarvis_grad_wrt_T(self):
        from legoesm.land.carbon.stomata import jarvis_gs, StomataConfig

        config = StomataConfig(enabled=True)
        ncol = 32

        def loss(T):
            gs = jarvis_gs(
                T=T,
                sw_down=300.0 * jnp.ones(ncol),
                q_air=5e-3 * jnp.ones(ncol),
                p_surface=1e5 * jnp.ones(ncol),
                beta_soil=0.7 * jnp.ones(ncol),
                config=config,
            )
            return jnp.sum(gs ** 2)

        T = 290.0 * jnp.ones(ncol)
        grad = jax.grad(loss)(T)
        assert_gradient_ok(grad, "Jarvis gs w.r.t. T")
