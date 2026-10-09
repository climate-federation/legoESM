"""Category 5: Snow Budget -- Accumulation, Melt, Age."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.land.snow_budget import update_snow
from legoesm.land.config import LandConfig
from legoesm.surface_albedo import (
    land_albedo as compute_land_albedo,
    snow_albedo, snow_cover_fraction,
    LandAlbedoConfig,
)
from legoesm import constants


CFG = LandAlbedoConfig()


class Test5a_SnowAccumulation:
    def test_accumulates(self):
        snow = jnp.array([0.0])
        snow_age = jnp.array([0.0])
        T_sfc = jnp.array([260.0])
        precip_snow = jnp.array([1e-4])  # kg/m2/s
        dt = 3600.0
        # Use Q_net=0 so no melt
        snow_new, age_new, melt = update_snow(
            snow, snow_age, T_sfc, precip_snow, dt, Q_net=jnp.array([0.0]), snow_age_activation_K=0.0
        )
        expected = precip_snow * dt  # 0.36 kg/m2
        assert jnp.allclose(snow_new, expected, rtol=0.01)
        assert float(age_new[0]) == 0.0  # fresh snow resets age


class Test5b_SnowMelt:
    def test_melts_above_freezing(self):
        snow = jnp.array([10.0])
        snow_age = jnp.array([1000.0])
        T_sfc = jnp.array([280.0])  # above freezing
        precip_snow = jnp.array([0.0])
        dt = 3600.0
        Q_net = jnp.array([50.0])  # positive Q_net => energy available for melt
        snow_new, _, melt = update_snow(
            snow, snow_age, T_sfc, precip_snow, dt, Q_net=Q_net, snow_age_activation_K=0.0
        )
        # melt = Q_net * dt / L_f = 50 * 3600 / 3.337e5 ~ 0.54 kg/m2
        expected_melt = 50.0 * 3600.0 / constants.L_f
        assert jnp.allclose(melt, expected_melt, rtol=0.01)
        assert float(snow_new[0]) < 10.0


class Test5c_CompleteMelt:
    def test_complete_melt(self):
        snow = jnp.array([0.01])
        snow_age = jnp.array([5000.0])
        T_sfc = jnp.array([300.0])
        precip_snow = jnp.array([0.0])
        dt = 3600.0
        Q_net = jnp.array([100.0])  # very strong energy
        snow_new, age_new, melt = update_snow(
            snow, snow_age, T_sfc, precip_snow, dt, Q_net=Q_net, snow_age_activation_K=0.0
        )
        assert float(snow_new[0]) == 0.0
        assert float(age_new[0]) == 0.0  # no snow => age reset


class Test5d_NoMeltBelowFreezing:
    def test_no_melt(self):
        snow = jnp.array([5.0])
        snow_age = jnp.array([1000.0])
        T_sfc = jnp.array([260.0])  # below freezing
        precip_snow = jnp.array([0.0])
        dt = 3600.0
        Q_net = jnp.array([50.0])  # positive but T < T_melt
        snow_new, _, melt = update_snow(
            snow, snow_age, T_sfc, precip_snow, dt, Q_net=Q_net, snow_age_activation_K=0.0
        )
        assert float(melt[0]) == 0.0
        assert float(snow_new[0]) == 5.0


class Test5e_SnowAgeEvolution:
    def test_age_increases_no_snowfall(self):
        snow = jnp.array([5.0])
        snow_age = jnp.array([1000.0])
        T_sfc = jnp.array([260.0])
        precip_snow = jnp.array([0.0])
        dt = 3600.0
        _, age_new, _ = update_snow(
            snow, snow_age, T_sfc, precip_snow, dt, Q_net=jnp.array([0.0]), snow_age_activation_K=0.0
        )
        assert float(age_new[0]) == 1000.0 + dt

    def test_age_partial_rejuvenation_with_snowfall(self):
        # Mass-weighted grain-age mixing: a trace flurry (0.36 kg/m2) onto a 5 kg/m2
        # aged pack only NUDGES the age down proportional to the fresh mass fraction --
        # it does NOT hard-reset to 0 (the old bug that over-brightened aged snow on any
        # dusting).  Expected: (age+dt)*swe_old/(swe_old+fresh)
        #   = (100000+3600)*5/5.36 ~ 96642 s, i.e. barely below the aged clock.
        snow = jnp.array([5.0])
        snow_age = jnp.array([100000.0])
        T_sfc = jnp.array([260.0])           # below freezing -> no melt
        precip_snow = jnp.array([1e-4])      # 0.36 kg/m2 fresh over the step
        dt = 3600.0
        _, age_new, _ = update_snow(
            snow, snow_age, T_sfc, precip_snow, dt, Q_net=jnp.array([0.0]), snow_age_activation_K=0.0
        )
        fresh = float(precip_snow[0]) * dt
        expected = (100000.0 + dt) * 5.0 / (5.0 + fresh)
        assert jnp.allclose(age_new, expected, rtol=1e-6)
        assert float(age_new[0]) < 100000.0            # rejuvenated (younger)
        assert float(age_new[0]) > 0.9 * 100000.0      # but only slightly (trace fresh)

    def test_age_resets_on_bare_ground_snowfall(self):
        # Fresh snow accumulating on BARE ground (swe_old -> 0) yields age ~0: the pack
        # is entirely fresh, so the mass-weighted mix collapses to the fresh (age-0) snow.
        snow = jnp.array([0.0])
        snow_age = jnp.array([100000.0])
        T_sfc = jnp.array([260.0])
        precip_snow = jnp.array([1e-4])
        dt = 3600.0
        _, age_new, _ = update_snow(
            snow, snow_age, T_sfc, precip_snow, dt, Q_net=jnp.array([0.0]), snow_age_activation_K=0.0
        )
        assert float(age_new[0]) == 0.0

    def test_age_zero_when_no_snow(self):
        snow = jnp.array([0.0])
        snow_age = jnp.array([0.0])
        T_sfc = jnp.array([280.0])
        precip_snow = jnp.array([0.0])
        dt = 3600.0
        _, age_new, _ = update_snow(
            snow, snow_age, T_sfc, precip_snow, dt, Q_net=jnp.array([0.0]), snow_age_activation_K=0.0
        )
        assert float(age_new[0]) == 0.0


class Test5f_SnowAlbedoFeedback:
    def test_snow_increases_albedo(self):
        lat = jnp.array([0.8])  # ~46 deg N
        alpha_no_snow = compute_land_albedo(lat, jnp.array([0.0]), jnp.array([0.0]), CFG)
        alpha_with_snow = compute_land_albedo(lat, jnp.array([100.0]), jnp.array([0.0]), CFG)
        assert float(alpha_with_snow[0]) > float(alpha_no_snow[0])

    def test_fresh_snow_high_albedo(self):
        alpha = snow_albedo(jnp.array([0.0]), CFG)
        assert jnp.allclose(alpha, CFG.alpha_snow_max, atol=1e-10)

    def test_old_snow_low_albedo(self):
        alpha = snow_albedo(jnp.array([1e8]), CFG)  # very old
        assert jnp.allclose(alpha, CFG.alpha_snow_min, atol=0.01)

    def test_albedo_monotone_decay(self):
        ages = jnp.linspace(0, 5e6, 100)
        alpha = snow_albedo(ages, CFG)
        dalpha = jnp.diff(alpha)
        assert jnp.all(dalpha <= 0.0)

    def test_albedo_bounded(self):
        ages = jnp.linspace(0, 1e8, 100)
        alpha = snow_albedo(ages, CFG)
        assert jnp.all(alpha >= CFG.alpha_snow_min - 1e-10)
        assert jnp.all(alpha <= CFG.alpha_snow_max + 1e-10)


class Test5g_SnowCoverFraction:
    # Niu & Yang (2007) saturating form f_snow = tanh(SWE / crit).
    def test_partial_cover(self):
        frac = snow_cover_fraction(jnp.array([CFG.snow_depth_crit]), CFG)  # SWE = crit
        assert jnp.allclose(frac, jnp.tanh(1.0), atol=1e-10)   # ~0.762, partial

    def test_saturates_high(self):
        # a thin snowpack already masks most of the surface -> ~1 by ~3x crit
        frac = snow_cover_fraction(jnp.array([3.0 * CFG.snow_depth_crit]), CFG)
        assert float(frac[0]) > 0.99

    def test_steep_near_zero(self):
        # rises steeply: a small SWE already gives substantial cover
        f_small = snow_cover_fraction(jnp.array([0.3 * CFG.snow_depth_crit]), CFG)
        assert float(f_small[0]) > 0.25

    def test_no_cover(self):
        frac = snow_cover_fraction(jnp.array([0.0]), CFG)
        assert jnp.allclose(frac, 0.0, atol=1e-10)

    def test_bounded(self):
        depths = jnp.linspace(0, 200, 100)
        frac = snow_cover_fraction(depths, CFG)
        assert jnp.all(frac >= 0.0)
        assert jnp.all(frac <= 1.0)
