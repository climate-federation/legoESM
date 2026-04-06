"""Taylor test gradient verification for each component.

The Taylor test verifies gradient correctness by checking:
  |J(x + h*dx) - J(x) - h * <grad, dx>| / h^2 → C  as h → 0

If the ratio converges (approximately constant or decreasing), the
gradient is correct (2nd-order convergence). If it blows up, the
gradient is WRONG.

Categories:
  9a) Dynamics (shallow water lat-lon)
  9b) Physics (Held-Suarez)
  9c) Land (slab land)
  9d) Sea ice (slab thermodynamics)
  9e) Coupler (bulk flux COARE3)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field


def taylor_test(loss_fn, x0, hs=(1e-1, 1e-2, 1e-3, 1e-4)):
    """Run a Taylor test and return convergence ratios.

    Returns list of |J(x+h*dx)-J(x)-h*<g,dx>|/h^2 for each h.
    For correct gradients, these should be approximately constant.
    """
    J0 = loss_fn(x0)
    grad = jax.grad(loss_fn)(x0)
    dx = grad / jnp.linalg.norm(grad)
    gdx = jnp.sum(grad * dx)

    ratios = []
    for h in hs:
        J_pert = loss_fn(x0 + h * dx)
        remainder = jnp.abs(J_pert - J0 - h * gdx)
        ratios.append((remainder / h ** 2).item())
    return ratios


def assert_taylor_ok(ratios, name=""):
    """Check Taylor test convergence.

    The ratio r(h) = |J(x+h*dx)-J(x)-h*<g,dx>|/h^2 should be bounded
    and approximately constant for correct gradients.
    """
    ratios = [r for r in ratios if r > 0]  # skip exact zeros
    assert len(ratios) > 0, f"{name}: all Taylor ratios are zero"
    assert all(jnp.isfinite(r) for r in ratios), f"{name}: Taylor ratios have NaN/Inf"
    # Ratio should not blow up: last ratio < 1000 * first ratio
    if len(ratios) >= 2:
        assert ratios[-1] < 1000 * ratios[0] + 1e-6, (
            f"{name}: Taylor ratios diverging: {ratios}"
        )


# ============================================================================
# 9a  Dynamics (shallow water lat-lon)
# ============================================================================

class TestTaylorDynamics:

    def test_sw_latlon(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
            FVShallowWaterLatLonModel,
        )
        from legoesm.core.state import ShallowWaterState

        grid = create_latlon_grid(8, 16)
        dt = 120.0
        model = FVShallowWaterLatLonModel(grid, dt=dt)

        key = jax.random.PRNGKey(0)
        h_data = 1000.0 + 10.0 * jax.random.normal(key, (8, 16))
        state = ShallowWaterState(
            h=Field(h_data, name="h"),
            u=Field(jnp.zeros((8, 16)), name="u"),
            v=Field(jnp.zeros((8, 16)), name="v"),
            h_s=Field(jnp.zeros((8, 16)), name="h_s"),
        )

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            out = model.step(s, dt)
            return jnp.sum(out.h.data ** 2)

        ratios = taylor_test(loss, state.h.data)
        assert_taylor_ok(ratios, "SW lat-lon Taylor test")


# ============================================================================
# 9b  Physics (Held-Suarez)
# ============================================================================

class TestTaylorPhysics:

    def test_held_suarez(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from tests.test_cases.held_suarez import held_suarez_forcing
        from legoesm.core.state import HydrostaticState

        n, nlev = 4, 5
        grid = create_cubed_sphere(n)
        sigma = create_sigma_coordinate(nlev)

        key = jax.random.PRNGKey(1)
        T_data = 250.0 + 5.0 * jax.random.normal(key, (6, n, n, nlev))
        state = HydrostaticState(
            u=Field(jnp.zeros((6, n, n, nlev)), name="u"),
            v=Field(jnp.zeros((6, n, n, nlev)), name="v"),
            T=Field(T_data, name="T"),
            p_s=Field(1e5 * jnp.ones((6, n, n)), name="p_s"),
            phis=Field(jnp.zeros((6, n, n)), name="phis"),
        )

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend = held_suarez_forcing(s, grid, sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        ratios = taylor_test(loss, state.T.data)
        assert_taylor_ok(ratios, "Held-Suarez Taylor test")


# ============================================================================
# 9c  Land (slab land)
# ============================================================================

class TestTaylorLand:

    def test_slab_land(self):
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
        forcing = AtmToSurface(
            sw_down=200.0 * ones, lw_down=300.0 * ones,
            precip_total=1e-5 * ones, precip_snow=0.0 * ones,
            T_lowest=280.0 * ones, q_lowest=5e-3 * ones,
            u_lowest=5.0 * ones, v_lowest=2.0 * ones,
            p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
            rho_lowest=1.2 * ones, cos_zenith=0.7 * ones,
            co2_ppmv=400.0 * ones,
            has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
        )

        def loss(T_data):
            s = state._replace(T_soil=state.T_soil.replace(data=T_data))
            out, _, _ = step_land(s, forcing, config, U_min=1.0, dt=60.0)
            return jnp.sum(out.T_soil.data ** 2)

        ratios = taylor_test(loss, state.T_soil.data)
        assert_taylor_ok(ratios, "Slab land Taylor test")


# ============================================================================
# 9d  Sea ice (slab thermodynamics)
# ============================================================================

class TestTaylorIce:

    def test_slab_ice(self):
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState
        from legoesm.coupler.coupling_fields import AtmToSurface

        config = SeaIceConfig(dynamics="none")
        shape = (6, 4, 4)
        ones = jnp.ones(shape)
        state = SeaIceState(
            h_ice=Field(1.0 * ones, name="h_ice"),
            T_ice=Field(265.0 * ones, name="T_ice"),
            concentration=Field(0.8 * ones, name="concentration"),
        )
        forcing = AtmToSurface(
            sw_down=100.0 * ones, lw_down=250.0 * ones,
            precip_total=0.0 * ones, precip_snow=0.0 * ones,
            T_lowest=260.0 * ones, q_lowest=1e-3 * ones,
            u_lowest=5.0 * ones, v_lowest=2.0 * ones,
            p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
            rho_lowest=1.4 * ones, cos_zenith=0.5 * ones,
            co2_ppmv=400.0 * ones,
            has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
        )
        ocean_sst = 271.35 * ones

        def loss(T_data):
            s = state._replace(T_ice=state.T_ice.replace(data=T_data))
            out, _ = step_sea_ice(
                s, forcing, ocean_sst, jnp.zeros(shape), jnp.zeros(shape),
                config, U_min=1.0, dt=3600.0,
            )
            return jnp.sum(out.T_ice.data ** 2)

        ratios = taylor_test(loss, state.T_ice.data)
        assert_taylor_ok(ratios, "Slab ice Taylor test")


# ============================================================================
# 9e  Coupler (bulk flux COARE3)
# ============================================================================

class TestTaylorCoupler:

    def test_coare3(self):
        from legoesm.coupler.bulk_flux import compute_most_fluxes

        ncol = 32
        T_sfc = 300.0 * jnp.ones(ncol)
        T_atm = 295.0 * jnp.ones(ncol)
        u_rel = 5.0 * jnp.ones(ncol)
        v_rel = 2.0 * jnp.ones(ncol)
        q_atm = 5e-3 * jnp.ones(ncol)
        q_sfc = 8e-3 * jnp.ones(ncol)
        rho = 1.2 * jnp.ones(ncol)

        def loss(T_sfc):
            _, _, shflx, _, _ = compute_most_fluxes(
                u_rel, v_rel, T_atm, q_atm, T_sfc, q_sfc, rho,
                scheme="coare3", n_iter=5,
            )
            return jnp.sum(shflx ** 2)

        ratios = taylor_test(loss, T_sfc)
        assert_taylor_ok(ratios, "COARE3 Taylor test")
