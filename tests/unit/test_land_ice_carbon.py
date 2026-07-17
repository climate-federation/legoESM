"""Category 6: Carbon Cycle -- GPP, Allocation, Decomposition."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.land.carbon.carbon_cycle import (
    compute_gpp, compute_phenology, step_carbon_differland, step_carbon,
    seasonal_co2_flux, init_carbon_state,
)
from legoesm.land.carbon.config import CarbonConfig, CarbonState


SHAPE = (4,)
CONFIG = CarbonConfig(scheme="differland")


def _state(shape=SHAPE):
    return init_carbon_state(shape, CONFIG)


# ===================================================================
# 6a  GPP smoke test
# ===================================================================

class Test6a_GPPSmoke:
    def test_gpp_positive(self):
        gpp = compute_gpp(
            sw_down=jnp.full(SHAPE, 300.0),
            T=jnp.full(SHAPE, 293.15),
            LAI=jnp.full(SHAPE, 3.0),
            co2_ppmv=jnp.full(SHAPE, 415.0),
            beta=jnp.full(SHAPE, 0.8),
            config=CONFIG,
        )
        assert jnp.all(gpp > 0)
        assert jnp.all(gpp > 1e-7)
        assert jnp.all(gpp < 1e-4)


# ===================================================================
# 6b  GPP = 0 at night
# ===================================================================

class Test6b_GPPNight:
    def test_gpp_zero_no_light(self):
        gpp = compute_gpp(
            sw_down=jnp.zeros(SHAPE),
            T=jnp.full(SHAPE, 293.15),
            LAI=jnp.full(SHAPE, 3.0),
            co2_ppmv=jnp.full(SHAPE, 415.0),
            beta=jnp.full(SHAPE, 0.8),
            config=CONFIG,
        )
        assert jnp.allclose(gpp, 0.0)


# ===================================================================
# 6c  GPP = 0 with no leaves
# ===================================================================

class Test6c_GPPNoLeaves:
    def test_gpp_zero_no_LAI(self):
        gpp = compute_gpp(
            sw_down=jnp.full(SHAPE, 300.0),
            T=jnp.full(SHAPE, 293.15),
            LAI=jnp.zeros(SHAPE),
            co2_ppmv=jnp.full(SHAPE, 415.0),
            beta=jnp.full(SHAPE, 0.8),
            config=CONFIG,
        )
        assert jnp.allclose(gpp, 0.0, atol=1e-15)


# ===================================================================
# 6d  GPP scales with drivers
# ===================================================================

class Test6d_GPPScaling:
    def test_gpp_increases_with_light(self):
        gpp_low = compute_gpp(jnp.full(SHAPE, 100.0), jnp.full(SHAPE, 293.15),
                              jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 415.0),
                              jnp.full(SHAPE, 0.8), CONFIG)
        gpp_high = compute_gpp(jnp.full(SHAPE, 500.0), jnp.full(SHAPE, 293.15),
                               jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 415.0),
                               jnp.full(SHAPE, 0.8), CONFIG)
        assert jnp.all(gpp_high > gpp_low)

    def test_gpp_increases_with_co2(self):
        gpp_low = compute_gpp(jnp.full(SHAPE, 300.0), jnp.full(SHAPE, 293.15),
                              jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 280.0),
                              jnp.full(SHAPE, 0.8), CONFIG)
        gpp_high = compute_gpp(jnp.full(SHAPE, 300.0), jnp.full(SHAPE, 293.15),
                               jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 800.0),
                               jnp.full(SHAPE, 0.8), CONFIG)
        assert jnp.all(gpp_high > gpp_low)

    def test_gpp_increases_with_beta(self):
        gpp_dry = compute_gpp(jnp.full(SHAPE, 300.0), jnp.full(SHAPE, 293.15),
                              jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 415.0),
                              jnp.full(SHAPE, 0.1), CONFIG)
        gpp_wet = compute_gpp(jnp.full(SHAPE, 300.0), jnp.full(SHAPE, 293.15),
                              jnp.full(SHAPE, 3.0), jnp.full(SHAPE, 415.0),
                              jnp.full(SHAPE, 0.9), CONFIG)
        assert jnp.all(gpp_wet > gpp_dry)


# ===================================================================
# 6e  Carbon pool conservation
# ===================================================================

class Test6e_CarbonConservation:
    def test_total_carbon_budget(self):
        state = _state()
        dt = 3600.0
        sw = jnp.full(SHAPE, 300.0)
        T = jnp.full(SHAPE, 293.15)
        co2 = jnp.full(SHAPE, 415.0)
        beta = jnp.full(SHAPE, 0.8)
        lat = jnp.full(SHAPE, 0.8)
        precip = jnp.full(SHAPE, 3e-5)

        new_state, co2_flux = step_carbon_differland(
            state, sw, T, co2, beta, lat, 200.0, precip, CONFIG, dt,
        )

        C_total_old = sum(float(getattr(state, f).mean()) for f in CarbonState._fields)
        C_total_new = sum(float(getattr(new_state, f).mean()) for f in CarbonState._fields)

        # Co2_flux is NEE in kgCO2/m2/s; convert back to gC/m2
        # NEE_gC_per_step = co2_flux * dt / (44/12 * 1e-3)
        NEE_gC = float(co2_flux.mean()) * dt / ((44.0/12.0) * 1e-3)
        # NEE = R_auto + R_het - GPP => dC_total = -NEE (negative NEE = uptake => C increases)
        dC_expected = -NEE_gC

        dC_actual = C_total_new - C_total_old
        # Allow generous tolerance because pools are clamped to >= 1 gC/m2
        assert abs(dC_actual - dC_expected) < max(abs(dC_expected) * 0.1, 1.0), (
            f"Carbon budget: dC_actual={dC_actual:.4f}, dC_expected={dC_expected:.4f}"
        )


# ===================================================================
# 6f  NPP allocation
# ===================================================================

class Test6f_Allocation:
    def test_allocation_fractions(self):
        cfg = CONFIG
        f_fol = cfg.f_fol
        f_lab = (1.0 - f_fol) * cfg.f_lab
        f_root = (1.0 - f_fol - f_lab) * cfg.f_root
        f_wood = 1.0 - f_fol - f_lab - f_root
        total = f_fol + f_lab + f_root + f_wood
        assert abs(total - 1.0) < 1e-10
        assert 0 <= f_fol <= 1
        assert 0 <= f_lab <= 1
        assert 0 <= f_root <= 1
        assert 0 <= f_wood <= 1


# ===================================================================
# 6g  Pools stay positive
# ===================================================================

class Test6g_PoolsPositive:
    def test_100_steps_positive(self):
        state = _state()
        sw = jnp.full(SHAPE, 300.0)
        T = jnp.full(SHAPE, 293.15)
        co2 = jnp.full(SHAPE, 415.0)
        beta = jnp.full(SHAPE, 0.8)
        lat = jnp.full(SHAPE, 0.8)
        precip = jnp.full(SHAPE, 3e-5)
        dt = 3600.0

        for i in range(100):
            state, _ = step_carbon_differland(
                state, sw, T, co2, beta, lat, float(i % 365), precip, CONFIG, dt,
            )

        for name in CarbonState._fields:
            arr = getattr(state, name)
            # A2: all eight pools (incl. the live active/slow/passive SOM
            # cascade, seeded by the CENTURY init partition) stay above the
            # 1 gC/m2 floor and finite.
            assert jnp.all(arr >= 1.0), f"{name} dropped below 1 gC/m2"
            assert jnp.all(jnp.isfinite(arr)), f"{name} has NaN"


# ===================================================================
# 6h  Decomposition temperature sensitivity
# ===================================================================

class Test6h_DecompTemp:
    def test_warmer_more_decomp(self):
        state = _state()
        sw = jnp.zeros(SHAPE)  # night => no GPP, so NEE = R
        T_cold = jnp.full(SHAPE, 283.15)
        T_warm = jnp.full(SHAPE, 293.15)
        co2 = jnp.full(SHAPE, 415.0)
        beta = jnp.full(SHAPE, 0.8)
        lat = jnp.full(SHAPE, 0.8)
        precip = jnp.full(SHAPE, 3e-5)
        dt = 86400.0

        _, flux_cold = step_carbon_differland(
            state, sw, T_cold, co2, beta, lat, 200.0, precip, CONFIG, dt,
        )
        _, flux_warm = step_carbon_differland(
            state, sw, T_warm, co2, beta, lat, 200.0, precip, CONFIG, dt,
        )
        # Warmer => more respiration => larger positive co2_flux
        assert jnp.all(flux_warm > flux_cold)


# ===================================================================
# 6i  Phenology seasonality
# ===================================================================

class Test6i_Phenology:
    def test_lrf_nonneg(self):
        doys = jnp.linspace(0, 365, 366)
        lat = jnp.full(366, 0.8)
        lrf, lff = compute_phenology(doys, lat, CONFIG)
        assert jnp.all(lrf >= 0)
        assert jnp.all(lff >= 0)

    def test_nh_peak_timing(self):
        """Labile release should peak near Bday=100 in NH."""
        doys = jnp.arange(366, dtype=jnp.float64)
        lat = jnp.full(366, 0.8)  # NH
        lrf, lff = compute_phenology(doys, lat, CONFIG)
        peak_lrf = int(jnp.argmax(lrf))
        # Bday=100, allow +/- 50 days for Gaussian width
        assert 50 < peak_lrf < 150, f"LRF peaks at doy {peak_lrf}"


# ===================================================================
# 6j  NEE sign convention
# ===================================================================

class Test6j_NEESign:
    def test_nighttime_positive(self):
        """Nighttime (sw=0, GPP=0): NEE positive (respiration source)."""
        state = _state()
        _, flux = step_carbon_differland(
            state, jnp.zeros(SHAPE), jnp.full(SHAPE, 293.15),
            jnp.full(SHAPE, 415.0), jnp.full(SHAPE, 0.8),
            jnp.full(SHAPE, 0.8), 200.0, jnp.full(SHAPE, 3e-5),
            CONFIG, 3600.0,
        )
        assert jnp.all(flux > 0), f"Nighttime flux should be positive: {flux}"

    def test_daytime_negative(self):
        """Daytime (sw=300, high GPP): NEE could be negative (uptake)."""
        state = _state()
        _, flux = step_carbon_differland(
            state, jnp.full(SHAPE, 500.0), jnp.full(SHAPE, 298.15),
            jnp.full(SHAPE, 415.0), jnp.full(SHAPE, 1.0),
            jnp.full(SHAPE, 0.8), 200.0, jnp.full(SHAPE, 3e-5),
            CONFIG, 3600.0,
        )
        # With sufficient light, GPP > R => NEE < 0
        assert jnp.all(flux < 0), f"Daytime flux should be negative (uptake): {flux}"


# ===================================================================
# 6k  Seasonal CO2 flux scheme
# ===================================================================

class Test6k_SeasonalFlux:
    def test_nh_summer_uptake(self):
        lat = jnp.array([0.8])
        flux = seasonal_co2_flux(200.0, lat, CONFIG)  # peak uptake day
        assert float(flux[0]) < 0, "NH summer should be uptake (negative)"

    def test_nh_winter_release(self):
        lat = jnp.array([0.8])
        flux = seasonal_co2_flux(15.0, lat, CONFIG)  # winter
        assert float(flux[0]) > 0, "NH winter should be release (positive)"

    def test_sh_opposite_phase(self):
        lat_nh = jnp.array([0.8])
        lat_sh = jnp.array([-0.8])
        flux_nh = seasonal_co2_flux(200.0, lat_nh, CONFIG)
        flux_sh = seasonal_co2_flux(200.0, lat_sh, CONFIG)
        # Opposite signs
        assert float(flux_nh[0]) * float(flux_sh[0]) < 0

    def test_equator_minimal(self):
        lat = jnp.array([0.0])
        flux = seasonal_co2_flux(200.0, lat, CONFIG)
        # sin(2*0) = 0 => flux should be near zero
        assert abs(float(flux[0])) < 1e-15
