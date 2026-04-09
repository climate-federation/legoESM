"""Differentiability tests for coupled system chains.

Tests that gradients flow correctly through multi-component chains —
the most important tests for 4D-Var and parameter estimation.

Categories:
  8a) Atmosphere dynamics + physics
  8b) Atmosphere → coupler → land
  8c) Atmosphere → coupler → ocean (SST sensitivity)
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


# ============================================================================
# 8a  Atmosphere dynamics + Held-Suarez physics
# ============================================================================

class TestAtmDynPlusPhysics:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel,
        )
        from tests.test_cases.held_suarez import held_suarez_forcing
        from legoesm.core.state import FV3HydrostaticState

        n, nlev = 4, 5
        self.grid = create_cubed_sphere(n)
        self.sigma = create_sigma_coordinate(nlev)
        self.dt = 60.0
        self.model = CDGridPrimitiveEquationModel(self.grid, self.sigma)

        key = jax.random.PRNGKey(100)
        T_data = 250.0 * jnp.ones((6, n, n, nlev)) + 1.0 * jax.random.normal(key, (6, n, n, nlev))
        self.state = FV3HydrostaticState(
            u_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="u_d"),
            v_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="v_d"),
            T=Field(T_data, name="T"),
            p_s=Field(1e5 * jnp.ones((6, n, n)), name="p_s"),
            phis=Field(jnp.zeros((6, n, n)), name="phis"),
        )
        self.held_suarez = held_suarez_forcing

    def test_grad_dynamics_plus_hs(self):
        from legoesm.core.state import HydrostaticState
        model, state, grid, sigma, dt = (
            self.model, self.state, self.grid, self.sigma, self.dt
        )
        hs = self.held_suarez
        n = 4
        nlev = 5

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            # One dynamics step
            s = model.step(s, dt)
            # Build A-grid state for Held-Suarez (it only uses T, u, v, p_s, phis)
            hs_state = HydrostaticState(
                u=Field(jnp.zeros((6, n, n, nlev)), name="u"),
                v=Field(jnp.zeros((6, n, n, nlev)), name="v"),
                T=s.T,
                p_s=s.p_s,
                phis=s.phis,
            )
            tend = hs(hs_state, grid, sigma)
            T_final = s.T.data + dt * tend.dT_dt.data
            return jnp.sum(T_final ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "Atm dynamics + Held-Suarez")
        # Should have spatial structure
        assert not jnp.all(grad == grad.ravel()[0]), "Gradient has no spatial structure"


# ============================================================================
# 8b  Atmosphere → coupler → land (T_lowest → T_soil)
# ============================================================================

class TestAtmCouplerLand:

    def test_grad_T_lowest_to_T_soil(self):
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig
        from legoesm.land.state import LandState
        from legoesm.coupler.coupling_fields import AtmToSurface

        ncol = 32
        config = LandConfig()
        state = LandState(
            T_soil=Field(280.0 * jnp.ones(ncol), name="T_soil"),
            W_bucket=Field(50.0 * jnp.ones(ncol), name="W_bucket"),
            snow_depth=Field(jnp.zeros(ncol), name="snow_depth"),
            snow_age=Field(jnp.zeros(ncol), name="snow_age"),
        )
        ones = jnp.ones(ncol)

        def loss(T_lowest):
            forcing = AtmToSurface(
                sw_down=200.0 * ones, lw_down=300.0 * ones,
                precip_total=1e-5 * ones, precip_snow=0.0 * ones,
                T_lowest=T_lowest, q_lowest=5e-3 * ones,
                u_lowest=5.0 * ones, v_lowest=2.0 * ones,
                p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
                rho_lowest=1.2 * ones, cos_zenith=0.7 * ones,
                co2_ppmv=400.0 * ones,
                has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
            )
            out, _, _ = step_land(state, forcing, config, U_min=1.0, dt=300.0)
            return jnp.sum(out.T_soil.data ** 2)

        T_lowest = 280.0 * jnp.ones(ncol)
        grad = jax.grad(loss)(T_lowest)
        assert_gradient_ok(grad, "T_lowest → T_soil")
        # Warmer atmosphere → warmer soil (positive gradient expected)
        assert jnp.mean(grad) > 0, "Expected positive gradient: warmer air → warmer soil"


# ============================================================================
# 8c  SST → coupler → sensible heat flux
# ============================================================================

class TestSSTToFlux:

    def test_grad_sst_to_shflx(self):
        from legoesm.coupler.bulk_flux import compute_most_fluxes

        ncol = 32
        T_atm = 290.0 * jnp.ones(ncol)  # cooler air than SST
        u_rel = 5.0 * jnp.ones(ncol)
        v_rel = 2.0 * jnp.ones(ncol)
        q_atm = 5e-3 * jnp.ones(ncol)
        q_sfc = 8e-3 * jnp.ones(ncol)
        rho = 1.2 * jnp.ones(ncol)

        def loss(sst):
            _, _, shflx, _, _ = compute_most_fluxes(
                u_rel, v_rel, T_atm, q_atm, sst, q_sfc, rho,
                scheme="coare3", n_iter=5,
            )
            return jnp.sum(shflx ** 2)

        sst = 295.0 * jnp.ones(ncol)
        grad = jax.grad(loss)(sst)
        assert_gradient_ok(grad, "SST → shflx")
        # Warmer SST → more sensible heat upward when SST > T_atm
        # Sign depends on convention, but gradient should be non-zero


# ============================================================================
# 8d  Atmosphere → coupler → sea ice → albedo feedback
# ============================================================================

class TestIceAlbedoChain:

    def test_sw_down_ice_albedo_feedback(self):
        """Gradient of absorbed SW through the full ice step.

        sw_down -> ice_step -> T_ice change -> albedo change -> absorbed_SW.
        """
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState
        from legoesm.coupler.coupling_fields import AtmToSurface

        config = SeaIceConfig(dynamics="none", temp_dependent_albedo=True)
        shape = (6, 4, 4)
        ones = jnp.ones(shape)

        state = SeaIceState(
            h_ice=Field(1.0 * ones, name="h_ice"),
            T_ice=Field(268.0 * ones, name="T_ice"),
            concentration=Field(0.8 * ones, name="concentration"),
        )

        def loss(sw_down):
            forcing = AtmToSurface(
                sw_down=sw_down,
                lw_down=250.0 * ones,
                precip_total=0.0 * ones,
                precip_snow=0.0 * ones,
                T_lowest=260.0 * ones,
                q_lowest=1e-3 * ones,
                u_lowest=5.0 * ones,
                v_lowest=2.0 * ones,
                p_lowest=1e5 * ones,
                p_surface=1.013e5 * ones,
                rho_lowest=1.4 * ones,
                cos_zenith=0.5 * ones,
                co2_ppmv=400.0 * ones,
                has_radiation=1.0 * ones,
                has_precipitation=1.0 * ones,
            )
            _, response = step_sea_ice(
                state, forcing, 271.35 * ones,
                jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0,
            )
            # Absorbed SW = (1 - albedo) * sw_down
            return jnp.sum((1.0 - response.albedo) * sw_down)

        sw = 100.0 * ones
        grad = jax.grad(loss)(sw)
        assert jnp.all(jnp.isfinite(grad)), "Ice albedo chain gradient not finite"
        assert jnp.any(grad != 0), "Ice albedo chain gradient all zero"


# ============================================================================
# 8e  Full AMIP-like chain (small): dynamics + physics + coupler
# ============================================================================

class TestFullAMIPChain:

    def test_dynamics_plus_physics_plus_land(self):
        """Full chain: dynamics.step -> Held-Suarez -> land step -> scalar loss.

        Tests gradient through the entire atmosphere + surface chain.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel,
        )
        from tests.test_cases.held_suarez import held_suarez_forcing
        from legoesm.core.state import FV3HydrostaticState, HydrostaticState
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig
        from legoesm.land.state import LandState
        from legoesm.coupler.coupling_fields import AtmToSurface

        n, nlev = 4, 5
        grid = create_cubed_sphere(n)
        sigma = create_sigma_coordinate(nlev)
        dt = 60.0
        model = CDGridPrimitiveEquationModel(grid, sigma)

        key = jax.random.PRNGKey(200)
        T_data = 250.0 * jnp.ones((6, n, n, nlev)) + 1.0 * jax.random.normal(key, (6, n, n, nlev))
        state = FV3HydrostaticState(
            u_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="u_d"),
            v_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="v_d"),
            T=Field(T_data, name="T"),
            p_s=Field(1e5 * jnp.ones((6, n, n)), name="p_s"),
            phis=Field(jnp.zeros((6, n, n)), name="phis"),
        )

        land_config = LandConfig()
        shape2d = (6, n, n)
        ones = jnp.ones(shape2d)
        land_state = LandState(
            T_soil=Field(280.0 * ones, name="T_soil"),
            W_bucket=Field(50.0 * ones, name="W_bucket"),
            snow_depth=Field(jnp.zeros(shape2d), name="snow_depth"),
            snow_age=Field(jnp.zeros(shape2d), name="snow_age"),
        )

        def loss(T_init):
            s = state._replace(T=state.T.replace(data=T_init))
            # 1. dynamics step
            s = model.step(s, dt)
            # 2. physics (Held-Suarez tendency)
            hs_state = HydrostaticState(
                u=Field(jnp.zeros((6, n, n, nlev)), name="u"),
                v=Field(jnp.zeros((6, n, n, nlev)), name="v"),
                T=s.T,
                p_s=s.p_s,
                phis=s.phis,
            )
            tend = held_suarez_forcing(hs_state, grid, sigma)
            T_after_phys = s.T.data + dt * tend.dT_dt.data
            # 3. Extract lowest level temp as forcing for land
            T_lowest = T_after_phys[:, :, :, -1]
            forcing = AtmToSurface(
                sw_down=200.0 * ones, lw_down=300.0 * ones,
                precip_total=1e-5 * ones, precip_snow=0.0 * ones,
                T_lowest=T_lowest, q_lowest=5e-3 * ones,
                u_lowest=5.0 * ones, v_lowest=2.0 * ones,
                p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
                rho_lowest=1.2 * ones, cos_zenith=0.7 * ones,
                co2_ppmv=400.0 * ones,
                has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
            )
            land_out, _, _ = step_land(land_state, forcing, land_config, U_min=1.0, dt=dt)
            # Loss combines atmosphere and land
            return jnp.sum(T_after_phys ** 2) + jnp.sum(land_out.T_soil.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "Full AMIP chain: dynamics + physics + land")
        # Should have spatial structure
        assert not jnp.all(grad == grad.ravel()[0]), "Gradient has no spatial structure"


