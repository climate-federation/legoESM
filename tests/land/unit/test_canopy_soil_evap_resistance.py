"""Canopy-path bare-soil evaporation resistance (Sellers 1992 / Lee & Pielke 1992).

The two-leaf canopy soil-evaporation efficiency is the product of the Kelvin pore
relative humidity ``h_r`` and the top-layer diffusion-crust throttle
``S_top ** soil_evap_resistance_exp``.  Raising the exponent must REDUCE bare-soil
evaporation for a sub-saturated surface (S_top < 1), and ``exp = 0`` recovers the
Kelvin-only behaviour.
"""
from __future__ import annotations

import unittest

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.land.multilayer_land import (init_multilayer_land_state,
                                          step_multilayer_land_with_diagnostics)


def _daytime_forcing(ncol=1, dtype=jnp.float64) -> AtmToSurface:
    return AtmToSurface(
        sw_down=jnp.full(ncol, 700.0, dtype),
        lw_down=jnp.full(ncol, 350.0, dtype),
        precip_total=jnp.zeros(ncol, dtype),
        precip_snow=jnp.zeros(ncol, dtype),
        T_lowest=jnp.full(ncol, 298.0, dtype),
        q_lowest=jnp.full(ncol, 0.010, dtype),      # ~45% RH -> evaporative demand
        u_lowest=jnp.full(ncol, 3.0, dtype),
        v_lowest=jnp.full(ncol, 2.0, dtype),
        p_lowest=jnp.full(ncol, 9.5e4, dtype),
        p_surface=jnp.full(ncol, 1e5, dtype),
        rho_lowest=jnp.full(ncol, 1.2, dtype),
        cos_zenith=jnp.full(ncol, 0.8, dtype),
        co2_ppmv=jnp.full(ncol, 412.0, dtype),
        has_radiation=jnp.ones(ncol, dtype),
        has_precipitation=jnp.ones(ncol, dtype),
    )


def _le_soil(exp: float) -> float:
    # ``soil_evap_series_resistance`` defaults to True, which is mutually
    # exclusive with the beta-efficiency S_top**exp path and would leave the
    # exponent inert (Kelvin-h_r only).  Opt into the legacy beta path so the
    # exponent this test varies actually reaches the soil energy balance.
    cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(max_iters=40),
                               soil_evap_resistance_exp=exp,
                               soil_evap_series_resistance=False)
    # Sub-saturated top layer (S_top ~ 0.5) so the crust throttle bites.
    s0 = init_multilayer_land_state(
        1, cfg, T_init=298.0,
        theta_init=0.6 * cfg.hydraulics.theta_sat, TgC_init=25.0)
    _ns, _r, _c, out = step_multilayer_land_with_diagnostics(
        s0, _daytime_forcing(), cfg, 1.0, 1800.0,
        lat=jnp.array([0.7]), carbon_state=None, doy=180.0, land_params=None)
    return float(out.LE_soil[0])


class TestCanopySoilEvapResistance(unittest.TestCase):
    def test_higher_exp_reduces_bare_soil_evaporation(self):
        le0 = _le_soil(0.0)     # Kelvin-h_r only (pre-fix behaviour)
        le3 = _le_soil(3.0)     # + S_top**3 diffusion-crust resistance
        self.assertTrue(jnp.isfinite(le0) and jnp.isfinite(le3))
        self.assertGreater(le0, 1.0)          # daytime bare soil IS evaporating
        self.assertLess(le3, le0)             # the crust resistance throttles it
        # The reduction is partial (not the full S_top**3 factor) because the
        # PROGNOSTIC skin T compensates: throttling evaporation warms the surface,
        # raising q_sat and partly restoring the flux — the same energy-balance
        # feedback that keeps soil-evap stubbornly high in the EC diagnosis.  Assert
        # a real, non-trivial reduction rather than the raw conductance ratio.
        self.assertGreater(le0 - le3, 20.0)

    def test_exp_zero_is_kelvin_only_default_field(self):
        # The config field exists and defaults to the #671 value (2.0).
        self.assertEqual(MultiLayerLandConfig().soil_evap_resistance_exp, 2.0)


if __name__ == "__main__":
    unittest.main()
