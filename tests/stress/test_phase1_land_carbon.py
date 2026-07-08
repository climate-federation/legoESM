"""Phase 1B: Land carbon cycle component stress tests."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from legoesm.land.carbon.config import CarbonConfig, CarbonState
from legoesm.land.carbon.carbon_cycle import (
    compute_gpp,
    step_carbon_differland,
    init_carbon_state,
)
from legoesm.land.stomata import StomataConfig, coupled_farquhar_stomata


class TestLandCarbon:
    """Land carbon cycle stress tests (1B)."""

    # -----------------------------------------------------------------
    # 1B.1 -- GPP CO2 response
    # -----------------------------------------------------------------
    def test_gpp_co2_response(self):
        """GPP should increase with rising CO2 (Michaelis-Menten fertilization)."""
        config = CarbonConfig(scheme="differland")

        sw_down = jnp.array([200.0])
        T = jnp.array([298.0])
        LAI = jnp.array([4.0])
        beta = jnp.array([0.8])

        gpp_280 = compute_gpp(sw_down, T, LAI, jnp.array([280.0]), beta, config)
        gpp_560 = compute_gpp(sw_down, T, LAI, jnp.array([560.0]), beta, config)

        assert jnp.all(jnp.isfinite(gpp_280)), "GPP(280) not finite"
        assert jnp.all(jnp.isfinite(gpp_560)), "GPP(560) not finite"
        assert jnp.all(gpp_280 > 0), "GPP(280) should be positive"
        assert jnp.all(gpp_560 > 0), "GPP(560) should be positive"
        assert jnp.all(gpp_560 > gpp_280), (
            f"GPP(560)={float(gpp_560)} should exceed GPP(280)={float(gpp_280)}"
        )

    # -----------------------------------------------------------------
    # 1B.2 -- Carbon pool conservation
    # -----------------------------------------------------------------
    def test_carbon_pool_conservation(self):
        """Total carbon (pools + cumulative NEE) should be approximately conserved."""
        config = CarbonConfig(scheme="differland")
        shape = (100,)
        state = init_carbon_state(shape, config)

        dt = 3600.0
        sw_down = jnp.full(shape, 200.0)
        T = jnp.full(shape, 290.0)
        co2 = jnp.full(shape, 400.0)
        beta = jnp.full(shape, 0.8)
        lat = jnp.full(shape, 0.7)  # ~40 deg N
        precip = jnp.full(shape, 5e-5)
        doy = 180.0

        def _total_pools(s):
            return s.C_lab + s.C_fol + s.C_root + s.C_wood + s.C_lit + s.C_som

        total_C_initial = jnp.sum(_total_pools(state))

        # gC/m2 -> kgCO2/m2/s is the co2_flux unit.
        # To convert cumulative kgCO2/m2/s * dt back to gC/m2:
        #   kgCO2/m2/s * dt_s * 1000 g/kg * (12/44) gC/gCO2
        kg_co2_to_gc = 1000.0 * 12.0 / 44.0

        cumulative_nee_gc = jnp.zeros(shape)
        for _ in range(100):
            state, co2_flux = step_carbon_differland(
                state, sw_down, T, co2, beta, lat, doy, precip, config, dt,
            )
            # co2_flux is kgCO2/m2/s, positive up = source.
            # NEE > 0 means carbon left the pools, so we ADD it to cumulative.
            cumulative_nee_gc = cumulative_nee_gc + co2_flux * dt * kg_co2_to_gc

        total_C_final = jnp.sum(_total_pools(state))

        # The initial carbon should equal (final pools) + (cumulative NEE out).
        # Note: softplus smoothing introduces a small bias, so use 1e-4 rtol.
        reconstructed = total_C_final + jnp.sum(cumulative_nee_gc)
        rel_err = jnp.abs(reconstructed - total_C_initial) / jnp.abs(total_C_initial)
        assert rel_err < 1e-4, (
            f"Pool conservation violated: relative error {float(rel_err):.6e}"
        )

    # -----------------------------------------------------------------
    # 1B.3 -- Respiration temperature sensitivity
    # -----------------------------------------------------------------
    def test_respiration_temperature_sensitivity(self):
        """NEE should differ meaningfully between cold and warm conditions."""
        config = CarbonConfig(scheme="differland")
        shape = (10,)
        state = init_carbon_state(shape, config)

        dt = 3600.0
        sw_down = jnp.full(shape, 200.0)
        co2 = jnp.full(shape, 400.0)
        beta = jnp.full(shape, 0.8)
        lat = jnp.full(shape, 0.7)
        precip = jnp.full(shape, 5e-5)
        doy = 180.0

        T_cold = jnp.full(shape, 280.0)
        T_warm = jnp.full(shape, 300.0)

        _, nee_cold = step_carbon_differland(
            state, sw_down, T_cold, co2, beta, lat, doy, precip, config, dt,
        )
        _, nee_warm = step_carbon_differland(
            state, sw_down, T_warm, co2, beta, lat, doy, precip, config, dt,
        )

        assert jnp.all(jnp.isfinite(nee_cold)), "NEE(cold) not finite"
        assert jnp.all(jnp.isfinite(nee_warm)), "NEE(warm) not finite"

        # NEE = Reco - GPP.  Higher T increases both GPP (via Gaussian)
        # and respiration (via Q10).  The key assertion is that they differ.
        diff = jnp.max(jnp.abs(nee_warm - nee_cold))
        assert diff > 1e-12, (
            f"NEE should differ between 280K and 300K, max|diff|={float(diff):.3e}"
        )

        # Additionally: warmer T should produce more positive (or less negative)
        # NEE due to enhanced heterotrophic respiration outpacing GPP increase
        # at 300K (above T_opt=25C).
        assert jnp.mean(nee_warm) > jnp.mean(nee_cold), (
            "Warmer conditions should yield higher (more positive) mean NEE"
        )

    # -----------------------------------------------------------------
    # 1B.4 -- Stomatal conductance
    # -----------------------------------------------------------------
    def test_stomatal_conductance(self):
        """Daytime stomatal conductance should exceed nighttime."""
        config = StomataConfig(enabled=True)
        shape = (5,)

        T_leaf = jnp.full(shape, 298.0)
        co2_ppmv = 400.0
        q_air = jnp.full(shape, 0.01)
        p_surface = jnp.full(shape, 101325.0)
        LAI = jnp.full(shape, 4.0)
        beta_soil = jnp.full(shape, 0.8)

        sw_night = jnp.full(shape, 0.0)
        sw_day = jnp.full(shape, 500.0)

        gs_night, _ = coupled_farquhar_stomata(
            T_leaf, sw_night, co2_ppmv, q_air, p_surface, LAI, beta_soil, config,
        )
        gs_day, _ = coupled_farquhar_stomata(
            T_leaf, sw_day, co2_ppmv, q_air, p_surface, LAI, beta_soil, config,
        )

        assert jnp.all(jnp.isfinite(gs_night)), "gs(night) not finite"
        assert jnp.all(jnp.isfinite(gs_day)), "gs(day) not finite"
        assert jnp.all(gs_night >= 0), "gs(night) should be non-negative"
        assert jnp.all(gs_day >= 0), "gs(day) should be non-negative"
        assert jnp.all(gs_day > gs_night), (
            f"gs_day={float(gs_day[0]):.6f} should exceed "
            f"gs_night={float(gs_night[0]):.6f}"
        )
