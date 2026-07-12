"""Tests for the land carbon cycle module.

Covers:
- CarbonConfig / CarbonState construction
- GPP light-use efficiency model
- Phenology (labile release, leaf fall)
- Decomposition modifier
- DifferLand prognostic step (single & multi-step)
- Seasonal cycle
- Initialization
- Integration with slab land
- Differentiability
"""

from __future__ import annotations

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from legoesm.land.carbon.config import (
    CarbonConfig,
    CarbonDiagnostics,
    CarbonState,
    som_total,
)
from legoesm.land.carbon.carbon_cycle import (
    compute_gpp,
    compute_phenology,
    init_carbon_state,
    seasonal_co2_flux,
    step_carbon,
    step_carbon_differland,
    _GC_TO_KG_CO2,
    _SPD,
    _DAYS_PER_YEAR,
    _temperate_modifier,
    _freeze_modifier,
    _som_decomp_modifier,
    _effective_rate,
    annual_frozen_fraction,
    perennial_frost_protection,
)
from legoesm import constants


# ===================================================================
# Helpers
# ===================================================================

def _default_config(**overrides):
    return CarbonConfig(**overrides)


def _make_carbon_state(shape=(4,), **overrides):
    # A2: the three SOM pools are live; seed them with a realistic CENTURY-like
    # partition of a 10000 gC/m2 bulk stock (active small, passive dominant) so
    # the cascade is exercised.  som_total == 10000 either way.
    defaults = dict(
        C_lab=jnp.full(shape, 100.0),
        C_fol=jnp.full(shape, 200.0),
        C_root=jnp.full(shape, 300.0),
        C_wood=jnp.full(shape, 10000.0),
        C_lit=jnp.full(shape, 100.0),
        C_som_active=jnp.full(shape, 300.0),
        C_som_slow=jnp.full(shape, 3200.0),
        C_som_passive=jnp.full(shape, 6500.0),
    )
    defaults.update(overrides)
    return CarbonState(**defaults)


# ===================================================================
# Config and State
# ===================================================================

class TestCarbonConfig(unittest.TestCase):

    def test_default_construction(self):
        cfg = CarbonConfig()
        self.assertEqual(cfg.scheme, "none")
        self.assertAlmostEqual(cfg.f_auto, 0.28)
        self.assertAlmostEqual(cfg.LCMA, 50.0)

    def test_differland_construction(self):
        cfg = CarbonConfig(scheme="differland", epsilon=1.5)
        self.assertEqual(cfg.scheme, "differland")
        self.assertAlmostEqual(cfg.epsilon, 1.5)

    def test_seasonal_construction(self):
        cfg = CarbonConfig(scheme="seasonal", nee_amplitude=1e-7)
        self.assertEqual(cfg.scheme, "seasonal")
        self.assertAlmostEqual(cfg.nee_amplitude, 1e-7)

    def test_carbon_state_fields(self):
        state = _make_carbon_state()
        # 8 pools since the SOM split (C_som -> active/slow/passive).
        self.assertEqual(len(state), 8)
        self.assertEqual(
            state._fields[-3:],
            ("C_som_active", "C_som_slow", "C_som_passive"))
        self.assertEqual(state.C_lab.shape, (4,))
        # A2: all three SOM pools are live (non-zero) and sum to the bulk stock.
        self.assertTrue(jnp.all(state.C_som_slow > 0.0))
        self.assertTrue(jnp.all(state.C_som_passive > 0.0))
        npt.assert_allclose(som_total(state), 10000.0, rtol=1e-9)


# ===================================================================
# GPP
# ===================================================================

class TestGPP(unittest.TestCase):

    def test_gpp_positive_with_light(self):
        """GPP should be positive when there is light, T, and moisture."""
        cfg = _default_config()
        sw = jnp.array([300.0, 500.0])
        T = jnp.array([290.0, 295.0])
        LAI = jnp.array([3.0, 5.0])
        co2 = jnp.array([400.0, 400.0])
        beta = jnp.array([0.8, 1.0])
        gpp = compute_gpp(sw, T, LAI, co2, beta, cfg)
        self.assertTrue(jnp.all(gpp > 0))
        self.assertTrue(jnp.all(jnp.isfinite(gpp)))

    def test_gpp_zero_at_night(self):
        """GPP should be zero when sw_down = 0."""
        cfg = _default_config()
        sw = jnp.zeros(3)
        T = jnp.full(3, 290.0)
        LAI = jnp.full(3, 3.0)
        gpp = compute_gpp(sw, T, LAI, 400.0, jnp.ones(3), cfg)
        npt.assert_allclose(gpp, 0.0, atol=1e-15)

    def test_gpp_zero_no_leaves(self):
        """GPP should be near zero when LAI=0 (no canopy)."""
        cfg = _default_config()
        sw = jnp.full(3, 400.0)
        T = jnp.full(3, 290.0)
        LAI = jnp.zeros(3)
        gpp = compute_gpp(sw, T, LAI, 400.0, jnp.ones(3), cfg)
        npt.assert_allclose(gpp, 0.0, atol=1e-15)

    def test_gpp_scales_with_radiation(self):
        """More radiation should give more GPP."""
        cfg = _default_config()
        T = jnp.full(2, 290.0)
        LAI = jnp.full(2, 3.0)
        beta = jnp.ones(2)
        gpp_lo = compute_gpp(jnp.array([100.0, 100.0]), T, LAI, 400.0, beta, cfg)
        gpp_hi = compute_gpp(jnp.array([500.0, 500.0]), T, LAI, 400.0, beta, cfg)
        self.assertTrue(jnp.all(gpp_hi > gpp_lo))

    def test_gpp_scales_with_co2(self):
        """Higher CO2 should give more GPP (Michaelis-Menten)."""
        cfg = _default_config()
        sw = jnp.full(2, 300.0)
        T = jnp.full(2, 290.0)
        LAI = jnp.full(2, 3.0)
        beta = jnp.ones(2)
        gpp_280 = compute_gpp(sw, T, LAI, 280.0, beta, cfg)
        gpp_800 = compute_gpp(sw, T, LAI, 800.0, beta, cfg)
        self.assertTrue(jnp.all(gpp_800 > gpp_280))

    def test_gpp_scales_with_moisture(self):
        """Drier conditions (lower beta) should reduce GPP."""
        cfg = _default_config()
        sw = jnp.full(2, 300.0)
        T = jnp.full(2, 290.0)
        LAI = jnp.full(2, 3.0)
        gpp_dry = compute_gpp(sw, T, LAI, 400.0, jnp.full(2, 0.2), cfg)
        gpp_wet = compute_gpp(sw, T, LAI, 400.0, jnp.ones(2), cfg)
        self.assertTrue(jnp.all(gpp_wet > gpp_dry))


# ===================================================================
# Phenology
# ===================================================================

class TestPhenology(unittest.TestCase):

    def test_phenology_shape(self):
        cfg = _default_config()
        lat = jnp.array([0.5, 0.8, -0.3])
        lrf, lff = compute_phenology(jnp.array(150.0), lat, cfg)
        self.assertEqual(lrf.shape, (3,))
        self.assertEqual(lff.shape, (3,))

    def test_phenology_non_negative(self):
        """Release/fall fractions should always be >= 0."""
        cfg = _default_config()
        lat = jnp.linspace(-1.2, 1.2, 20)
        for doy in [0.0, 100.0, 200.0, 300.0]:
            lrf, lff = compute_phenology(jnp.array(doy), lat, cfg)
            self.assertTrue(jnp.all(lrf >= 0), f"lrf < 0 at doy={doy}")
            self.assertTrue(jnp.all(lff >= 0), f"lff < 0 at doy={doy}")

    def test_labile_release_peaks_near_bday(self):
        """Labile release should peak near Bday (default 100)."""
        cfg = _default_config(Bday=100.0)
        lat = jnp.array([0.7])  # NH mid-latitude
        doys = jnp.arange(0, 365, 5.0)
        lrf_vals = jnp.array([compute_phenology(d, lat, cfg)[0][0] for d in doys])
        peak_doy = doys[jnp.argmax(lrf_vals)]
        self.assertTrue(abs(float(peak_doy) - 100.0) < 40,
                        f"lrf peak at doy={peak_doy}, expected ~100")

    def test_leaf_fall_peaks_near_fday(self):
        """Leaf fall should peak near Fday (default 280)."""
        cfg = _default_config(Fday=280.0)
        lat = jnp.array([0.7])
        doys = jnp.arange(0, 365, 5.0)
        lff_vals = jnp.array([compute_phenology(d, lat, cfg)[1][0] for d in doys])
        peak_doy = doys[jnp.argmax(lff_vals)]
        self.assertTrue(abs(float(peak_doy) - 280.0) < 40,
                        f"lff peak at doy={peak_doy}, expected ~280")

    def test_hemisphere_flip(self):
        """SH phenology should be shifted by ~6 months from NH."""
        cfg = _default_config()
        lat_nh = jnp.array([0.7])
        lat_sh = jnp.array([-0.7])
        lrf_nh, _ = compute_phenology(jnp.array(100.0), lat_nh, cfg)
        lrf_sh, _ = compute_phenology(jnp.array(100.0), lat_sh, cfg)
        # At doy=100 (NH spring), NH should have higher release than SH
        self.assertTrue(float(lrf_nh[0]) > float(lrf_sh[0]))

    def test_phenology_finite(self):
        cfg = _default_config()
        lat = jnp.linspace(-1.0, 1.0, 10)
        lrf, lff = compute_phenology(jnp.array(180.0), lat, cfg)
        self.assertTrue(jnp.all(jnp.isfinite(lrf)))
        self.assertTrue(jnp.all(jnp.isfinite(lff)))

    def test_evergreen_phenology_is_continuous(self):
        """Evergreen leaf-fall is a steady, near-constant per-day rate all year
        (no Gaussian dormant-season drop to ~0), unlike the deciduous pulse; its
        seasonal amplitude is far smaller than the deciduous branch's."""
        lat = jnp.array([0.7])  # NH mid-latitude
        doys = jnp.arange(0, 365, 5.0)
        ever = _default_config(evergreen=True)
        deci = _default_config(evergreen=False)
        lff_ever = jnp.array([compute_phenology(d, lat, ever)[1][0] for d in doys])
        lff_deci = jnp.array([compute_phenology(d, lat, deci)[1][0] for d in doys])
        # Evergreen sheds every day (canopy never fully stops) and is flat in doy.
        self.assertTrue(jnp.all(lff_ever > 0.0), "evergreen leaf-fall hit zero")
        amp_ever = float(lff_ever.max() - lff_ever.min())
        amp_deci = float(lff_deci.max() - lff_deci.min())
        self.assertLess(amp_ever, 1e-9, "evergreen leaf-fall not flat in doy")
        # Far smaller seasonal amplitude than the deciduous Gaussian pulse.
        self.assertGreater(amp_deci, 1e-3)          # deciduous pulse is real
        self.assertLess(amp_ever, 0.01 * amp_deci)  # evergreen << deciduous
        # The steady rate is 1 / (leaf_lifespan * year): continuous turnover.
        expected = 1.0 / (ever.leaf_lifespan * _DAYS_PER_YEAR)
        npt.assert_allclose(lff_ever, expected, rtol=1e-6)
        # Deciduous dormant season drops BELOW the evergreen steady rate.
        self.assertLess(float(lff_deci.min()), float(lff_ever.min()))

    def test_evergreen_labile_release_is_steady(self):
        """Evergreen labile release is also continuous (feeds the steady
        regrowth) at 1 / (lab_lifespan * year)."""
        lat = jnp.array([0.7])
        doys = jnp.arange(0, 365, 5.0)
        ever = _default_config(evergreen=True)
        lrf_ever = jnp.array([compute_phenology(d, lat, ever)[0][0] for d in doys])
        self.assertTrue(jnp.all(lrf_ever > 0.0))
        expected = 1.0 / (ever.lab_lifespan * _DAYS_PER_YEAR)
        npt.assert_allclose(lrf_ever, expected, rtol=1e-6)


# ===================================================================
# Decomposition
# ===================================================================

class TestDecomposition(unittest.TestCase):

    def test_temperate_modifier_reference(self):
        """At reference T and precip, modifier should be near 1."""
        cfg = _default_config()
        T = jnp.array([cfg.T_ref])
        precip = jnp.array([cfg.precip_ref])
        mod = _temperate_modifier(T, precip, cfg)
        npt.assert_allclose(mod, 1.0, atol=0.01)

    def test_warmer_increases_decomp(self):
        """Warmer temperatures should increase the modifier."""
        cfg = _default_config()
        precip = jnp.array([cfg.precip_ref])
        mod_cold = _temperate_modifier(jnp.array([270.0]), precip, cfg)
        mod_warm = _temperate_modifier(jnp.array([300.0]), precip, cfg)
        self.assertTrue(float(mod_warm[0]) > float(mod_cold[0]))

    def test_het_modifier_uses_q10_het_not_autotrophic_q10(self):
        """Decomposition modifier responds to Q10_het_exp, NOT Q10_exp."""
        precip = jnp.array([3e-5])
        T = jnp.array([300.0])   # warm, above T_ref
        base = _temperate_modifier(T, precip, _default_config(Q10_het_exp=0.09))
        stronger = _temperate_modifier(T, precip, _default_config(Q10_het_exp=0.12))
        weaker = _temperate_modifier(T, precip, _default_config(Q10_het_exp=0.04))
        self.assertTrue(float(stronger[0]) > float(base[0]) > float(weaker[0]))
        # Changing the AUTOTROPHIC Q10_exp must NOT change the het modifier.
        alt_auto = _temperate_modifier(T, precip, _default_config(Q10_exp=0.11))
        npt.assert_allclose(base, alt_auto, rtol=1e-9)

    def test_effective_rate_small_dt(self):
        """For small dt, effective rate ≈ raw rate."""
        rate = jnp.array(0.001)
        dt = jnp.array(0.01)  # 0.01 day = ~14 min
        eff = _effective_rate(rate, dt)
        npt.assert_allclose(eff, rate, rtol=0.01)


# ===================================================================
# Freeze suppression modifier (the high-latitude SOC control)
# ===================================================================

class TestFreezeModifier(unittest.TestCase):

    def test_freeze_bounds_and_limits(self):
        """f_freeze in [som_freeze_floor, 1]; ->1 warm, -> floor deep-frozen,
        == floor + (1-floor)/2 at T_freeze (the floored-sigmoid midpoint)."""
        cfg = _default_config()
        floor = cfg.som_freeze_floor
        self.assertGreater(floor, 0.0)  # default floor is nonzero (the fix)
        T = jnp.linspace(240.0, 320.0, 60)
        ff = _freeze_modifier(T, cfg)
        self.assertTrue(jnp.all(ff >= floor - 1e-9))
        self.assertTrue(jnp.all(ff <= 1.0))
        self.assertGreater(float(_freeze_modifier(jnp.array([300.0]), cfg)[0]), 0.99)
        # Deep-frozen soil floors at ``som_freeze_floor`` (nonzero), NOT 0.
        npt.assert_allclose(
            _freeze_modifier(jnp.array([245.0]), cfg), floor, atol=1e-3)
        npt.assert_allclose(
            _freeze_modifier(jnp.array([constants.T_freeze]), cfg),
            floor + (1.0 - floor) * 0.5, atol=1e-6)

    def test_freeze_floor_recovers_unfloored_sigmoid(self):
        """som_freeze_floor=0 reduces to the plain sigmoid (backward-compatible),
        and floor <= floored <= 1 while floored >= the un-floored value."""
        T = jnp.linspace(250.0, 300.0, 40)
        ff0 = _freeze_modifier(T, _default_config(som_freeze_floor=0.0))
        ff = _freeze_modifier(T, _default_config(som_freeze_floor=0.05))
        npt.assert_allclose(ff0, jax.nn.sigmoid(
            (T - constants.T_freeze) / _default_config().som_freeze_width_K),
            atol=1e-12)
        # The floor raises decomposition everywhere (f_freeze larger => faster
        # turnover => less runaway) but never above 1.
        self.assertTrue(jnp.all(ff >= ff0 - 1e-12))
        self.assertTrue(jnp.all(ff <= 1.0 + 1e-12))

    def test_freeze_floor_out_of_range_required(self):
        with self.assertRaises(ValueError):
            _freeze_modifier(jnp.array([275.0]),
                             _default_config(som_freeze_floor=1.0))
        with self.assertRaises(ValueError):
            _freeze_modifier(jnp.array([275.0]),
                             _default_config(som_freeze_floor=-0.1))

    def test_freeze_monotonic_increasing(self):
        cfg = _default_config()
        ff = _freeze_modifier(jnp.linspace(250.0, 300.0, 50), cfg)
        self.assertTrue(jnp.all(jnp.diff(ff) > 0.0))

    def test_freeze_width_positive_required(self):
        with self.assertRaises(ValueError):
            _freeze_modifier(jnp.array([275.0]),
                             _default_config(som_freeze_width_K=0.0))

    def test_som_modifier_suppressed_relative_to_temperate(self):
        """m_som == f_temp*f_moist*f_freeze <= the freeze-free _temperate_modifier,
        strictly smaller when cold, ~equal when warm."""
        cfg = _default_config()
        precip = jnp.array([cfg.precip_ref])
        for T in (jnp.array([263.0]), jnp.array([290.0]), jnp.array([305.0])):
            m_som = _som_decomp_modifier(T, precip, cfg)
            m_temp = _temperate_modifier(T, precip, cfg)
            self.assertTrue(jnp.all(m_som <= m_temp + 1e-12))
        Tc = jnp.array([263.0])
        self.assertLess(
            float(_som_decomp_modifier(Tc, precip, cfg)[0]),
            0.5 * float(_temperate_modifier(Tc, precip, cfg)[0]))
        Tw = jnp.array([300.0])
        npt.assert_allclose(_som_decomp_modifier(Tw, precip, cfg),
                            _temperate_modifier(Tw, precip, cfg), rtol=1e-3)

    def test_som_modifier_differentiable(self):
        cfg = _default_config()

        def loss(T):
            return jnp.sum(_som_decomp_modifier(T, jnp.array([3e-5]), cfg))

        g = jax.grad(loss)(jnp.array([274.0]))
        self.assertTrue(jnp.all(jnp.isfinite(g)))
        self.assertFalse(jnp.allclose(g, 0.0))


class TestPerennialFrostProtection(unittest.TestCase):
    """Permafrost / anaerobic SOM protection: f_perma(frozen_fraction) and the
    annual frozen-fraction index (Koven et al. 2013; Hugelius et al. 2014)."""

    def test_f_perma_bounds_and_direction(self):
        """f_perma in [permafrost_protection_min, 1]; ->1 for never-frozen
        (frozen_fraction=0), -> permafrost_protection_min for perennial frost
        (frozen_fraction=1); monotonically DECREASING in frozen_fraction (more
        frost -> less decomposition -> more retained SOC)."""
        cfg = _default_config()
        p_min = cfg.permafrost_protection_min
        phi = jnp.linspace(0.0, 1.0, 60)
        fp = perennial_frost_protection(phi, cfg)
        self.assertTrue(jnp.all(fp >= p_min - 1e-9))
        self.assertTrue(jnp.all(fp <= 1.0 + 1e-9))
        # Never-frozen -> ~1 (temperate/tropical UNCHANGED); the threshold sits
        # well above 0 so the warm tail is untouched.
        self.assertGreater(
            float(perennial_frost_protection(jnp.array([0.0]), cfg)[0]), 0.999)
        # Perennial frost -> approaches the protection floor.
        npt.assert_allclose(
            float(perennial_frost_protection(jnp.array([1.0]), cfg)[0]),
            p_min, atol=5e-3)
        # Monotonically decreasing (MORE frost -> SMALLER f_perma).
        self.assertTrue(jnp.all(jnp.diff(fp) < 0.0))

    def test_f_perma_validation_raises(self):
        with self.assertRaises(ValueError):
            perennial_frost_protection(
                jnp.array([0.5]),
                _default_config(permafrost_protection_min=0.0))  # must be >0
        with self.assertRaises(ValueError):
            perennial_frost_protection(
                jnp.array([0.5]),
                _default_config(permafrost_protection_min=1.5))  # <=1
        with self.assertRaises(ValueError):
            perennial_frost_protection(
                jnp.array([0.5]),
                _default_config(permafrost_frozen_fraction_threshold=1.0))
        with self.assertRaises(ValueError):
            perennial_frost_protection(
                jnp.array([0.5]),
                _default_config(permafrost_frozen_fraction_width=0.0))

    def test_f_perma_differentiable_in_perma_params(self):
        """grad of f_perma w.r.t. the two tunable perennial-frost params is
        finite and non-zero (calibration lever)."""
        phi = jnp.array([0.6])

        def loss(vec):
            cfg = _default_config(
                permafrost_protection_min=vec[0],
                permafrost_frozen_fraction_threshold=vec[1])
            return jnp.sum(perennial_frost_protection(phi, cfg))

        g = jax.grad(loss)(jnp.array([0.3, 0.6]))
        self.assertTrue(jnp.all(jnp.isfinite(g)))
        self.assertTrue(jnp.all(jnp.abs(g) > 0.0))

    def test_annual_frozen_fraction_warm_cold(self):
        """phi in [0,1]; ~0 for a warm never-frozen climate, high for a cold
        climate, and monotonically DECREASING in mean-annual temperature."""
        cfg = _default_config()
        warm = float(annual_frozen_fraction(
            jnp.array(298.0), jnp.array(5.0), cfg))
        cold = float(annual_frozen_fraction(
            jnp.array(260.0), jnp.array(15.0), cfg))
        self.assertLess(warm, 0.02)      # tropical/warm-temperate: ~never frozen
        self.assertGreater(cold, 0.7)    # boreal/tundra: frozen most of the year
        # Monotone decreasing in MAT at fixed amplitude.
        mats = jnp.linspace(255.0, 295.0, 40)
        phi = annual_frozen_fraction(mats, jnp.full_like(mats, 12.0), cfg)
        self.assertTrue(jnp.all(phi >= 0.0))
        self.assertTrue(jnp.all(phi <= 1.0))
        self.assertTrue(jnp.all(jnp.diff(phi) <= 1e-9))
        # A freezing-mean climate is roughly half frozen.
        half = float(annual_frozen_fraction(
            jnp.array(constants.T_freeze), jnp.array(10.0), cfg))
        npt.assert_allclose(half, 0.5, atol=0.05)

    def test_som_modifier_frozen_fraction_suppresses(self):
        """Passing a high frozen_fraction scales the SOM modifier down by
        f_perma; frozen_fraction=None is byte-identical to the unprotected
        modifier; frozen_fraction=0 is ~unchanged (warm-soil regression)."""
        cfg = _default_config()
        T = jnp.array([268.0])           # cold column
        precip = jnp.array([cfg.precip_ref])
        m_none = _som_decomp_modifier(T, precip, cfg)
        m_zero = _som_decomp_modifier(T, precip, cfg, frozen_fraction=jnp.array([0.0]))
        m_frozen = _som_decomp_modifier(T, precip, cfg, frozen_fraction=jnp.array([0.8]))
        # None == unprotected (exactly).
        npt.assert_allclose(np.asarray(m_none), np.asarray(m_zero), rtol=2e-3)
        # High frozen fraction suppresses decomposition (< unprotected).
        self.assertLess(float(m_frozen[0]), float(m_none[0]))
        expect = float(m_none[0]) * float(
            perennial_frost_protection(jnp.array([0.8]), cfg)[0])
        npt.assert_allclose(float(m_frozen[0]), expect, rtol=1e-6)

    def test_step_perennial_frost_retains_som_cold_and_warm_unchanged(self):
        """IDEALIZED column: a perennially-frozen cold column RETAINS more SOM
        (less decomposed, LOWER heterotrophic SOM respiration) WITH the
        protection than without -- carbon RETAINED, not created (the extra pool
        carbon exactly equals the un-respired flux; mass closes).  A warm column
        (frozen_fraction=0) is UNCHANGED."""
        cfg = _default_config(scheme="differland")
        st = _make_carbon_state(shape=(1,))
        common = dict(
            sw_down=jnp.array([200.0]), co2_ppmv=jnp.array([400.0]),
            beta=jnp.array([0.6]), lat=jnp.array([1.0]), doy=15.0,
            precip=jnp.array([2e-5]), config=cfg, dt=7200.0)

        # --- cold column: with vs without perennial-frost protection ----------
        T_cold = jnp.array([266.0])
        _s_prot, _f_prot, d_prot = step_carbon_differland(
            st, T=T_cold, return_diagnostics=True,
            soil_frozen_fraction=jnp.array([0.8]), **common)
        s_prot, f_prot = step_carbon_differland(
            st, T=T_cold, soil_frozen_fraction=jnp.array([0.8]), **common)
        _s_un, _f_un, d_un = step_carbon_differland(
            st, T=T_cold, return_diagnostics=True, **common)
        s_un, f_un = step_carbon_differland(st, T=T_cold, **common)
        som_prot = float((s_prot.C_som_active + s_prot.C_som_slow
                          + s_prot.C_som_passive)[0])
        som_un = float((s_un.C_som_active + s_un.C_som_slow
                        + s_un.C_som_passive)[0])
        # Protection RETAINS SOM (higher pool) and LOWERS SOM respiration.
        self.assertGreater(som_prot, som_un)
        self.assertLess(float(d_prot.r_het_som[0]), float(d_un.r_het_som[0]))
        # Carbon RETAINED, not created: the extra SOM carbon == the reduction in
        # SOM heterotrophic respiration over the step (mass closes to ~machine
        # precision; the SOM input is identical between the two runs).
        dt_days = 7200.0 / 86400.0
        retained = som_prot - som_un
        un_respired = (float(d_un.r_het_som[0])
                       - float(d_prot.r_het_som[0])) * dt_days
        npt.assert_allclose(retained, un_respired, rtol=1e-6, atol=1e-9)
        # ... and the reduced respiration shows up as a LOWER (less-positive) NEE.
        self.assertLess(float(f_prot[0]), float(f_un[0]))

        # --- warm column: frozen_fraction=0 leaves the step UNCHANGED ----------
        T_warm = jnp.array([300.0])
        sw0, fw0 = step_carbon_differland(st, T=T_warm, **common)
        sw1, fw1 = step_carbon_differland(
            st, T=T_warm, soil_frozen_fraction=jnp.array([0.0]), **common)
        # co2_flux is O(1e-8) kgCO2/m2/s and may straddle zero, so use atol.
        npt.assert_allclose(float(fw0[0]), float(fw1[0]), rtol=1e-3, atol=1e-11)
        npt.assert_allclose(
            float((sw0.C_som_active + sw0.C_som_slow + sw0.C_som_passive)[0]),
            float((sw1.C_som_active + sw1.C_som_slow + sw1.C_som_passive)[0]),
            rtol=1e-4)


# ===================================================================
# Multi-pool SOM cascade (active -> slow -> passive)
# ===================================================================

class TestSomCascade(unittest.TestCase):

    def _cascade_step(self, dt=86400.0, T=290.0, sw=0.0, **state_over):
        cfg = _default_config(scheme="differland")
        state = _make_carbon_state(shape=(1,), **state_over)
        new_state, co2_flux, diag = step_carbon_differland(
            state, jnp.full(1, sw), jnp.full(1, T), jnp.full(1, 400.0),
            jnp.full(1, 0.8), jnp.full(1, 0.7), 180.0, jnp.full(1, 3e-5),
            cfg, dt, return_diagnostics=True)
        return cfg, state, new_state, co2_flux, diag

    def test_losses_nonneg_and_bounded_by_stock(self):
        """Each pool's decomposition D_X is >= 0 and never exceeds its stock
        (the _effective_rate loss fraction is <= 1), so no pool goes negative."""
        cfg, state, new, _flux, diag = self._cascade_step()
        dt_days = 86400.0 / _SPD
        for pool, loss in (("C_som_active", diag.som_active_loss),
                           ("C_som_slow", diag.som_slow_loss),
                           ("C_som_passive", diag.som_passive_loss)):
            self.assertTrue(jnp.all(loss >= 0.0), f"{pool} loss < 0")
            self.assertTrue(
                jnp.all(loss * dt_days <= getattr(state, pool) + 1e-9),
                f"{pool} over-drained")
        for f in new._fields:
            self.assertTrue(jnp.all(getattr(new, f) >= 0.0), f"{f} negative")

    def test_transfers_flow_downward(self):
        """With a big active + tiny slow/passive, the humified transfer grows
        the slow AND passive pools (active->slow->passive direction)."""
        _c, st, new, _f, _d = self._cascade_step(
            C_som_active=jnp.full((1,), 10000.0),
            C_som_slow=jnp.full((1,), 1.0),
            C_som_passive=jnp.full((1,), 1.0))
        self.assertGreater(float(new.C_som_slow[0]), float(st.C_som_slow[0]))
        self.assertGreater(float(new.C_som_passive[0]), float(st.C_som_passive[0]))

    def test_respiration_accounting(self):
        """r_het_som == (1-f_as)*D_active + (1-f_sp)*D_slow + D_passive >= 0."""
        cfg, _st, _new, _flux, diag = self._cascade_step()
        expected = ((1.0 - cfg.f_active_to_slow) * diag.som_active_loss
                    + (1.0 - cfg.f_slow_to_passive) * diag.som_slow_loss
                    + diag.som_passive_loss)
        npt.assert_allclose(diag.r_het_som, expected, rtol=1e-9, atol=1e-12)
        self.assertTrue(jnp.all(diag.r_het_som >= 0.0))

    def test_som_subcolumn_conserves(self):
        """SOM sub-budget: humified input - sum(dC_som)/dt - R_het_som == 0
        (every gram of input stays in a SOM pool or respires)."""
        cfg, state, new, _flux, diag = self._cascade_step()
        dt_days = 86400.0 / _SPD
        dC_som = ((new.C_som_active - state.C_som_active)
                  + (new.C_som_slow - state.C_som_slow)
                  + (new.C_som_passive - state.C_som_passive))
        som_input = diag.lit_to_som + diag.wood_to_som
        residual = som_input - dC_som / dt_days - diag.r_het_som
        npt.assert_allclose(residual, 0.0, atol=1e-9)

    def test_full_column_conservation_all_pools_live(self):
        """Full 8-pool closure sum(dC) == -NEE*dt at machine precision with all
        three SOM pools live and evolving (day and night)."""
        for sw in (0.0, 400.0):
            cfg, state, new, flux, _d = self._cascade_step(sw=sw)
            dC = sum(getattr(new, f) - getattr(state, f) for f in state._fields)
            expected = -(flux / _GC_TO_KG_CO2) * 86400.0
            npt.assert_allclose(dC, expected, rtol=1e-9, atol=1e-9,
                                err_msg=f"multipool column not closed (sw={sw})")

    def test_evergreen_column_conserves(self):
        """The EVERGREEN phenology branch conserves carbon exactly: it only
        changes the (lrf, lff) turnover rates, and the pool update routes
        lab_release C_lab->C_fol and leaf_litter C_fol->C_lit either way, so the
        8-pool closure sum(dC) == -NEE*dt still holds to machine precision."""
        for sw in (0.0, 400.0):
            cfg = _default_config(scheme="differland", evergreen=True)
            state = _make_carbon_state(shape=(1,))
            new, flux, _d = step_carbon_differland(
                state, jnp.full(1, sw), jnp.full(1, 290.0), jnp.full(1, 400.0),
                jnp.full(1, 0.8), jnp.full(1, 0.7), 180.0, jnp.full(1, 3e-5),
                cfg, 86400.0, return_diagnostics=True)
            dC = sum(getattr(new, f) - getattr(state, f) for f in state._fields)
            expected = -(flux / _GC_TO_KG_CO2) * 86400.0
            npt.assert_allclose(dC, expected, rtol=1e-9, atol=1e-9,
                                err_msg=f"evergreen column not closed (sw={sw})")


# ===================================================================
# SOM transfer-fraction validation (fail-early on out-of-[0,1] config)
# ===================================================================

class TestSomTransferFractionValidation(unittest.TestCase):
    """FIX #3 (codex A2): step_carbon_differland must fail early (a plain
    Python ``if ... raise ValueError`` on the STATIC config value, matching
    the repo's dispatch-hardening pattern, e.g. ``_freeze_modifier``'s
    ``som_freeze_width_K > 0`` guard) on an out-of-[0, 1]
    f_active_to_slow/f_slow_to_passive.  The ``__param_spec__`` bounds
    (0.1-0.5, config.py) only constrain the TRAINING search range and are
    never enforced at runtime, so a direct
    ``CarbonConfig(f_active_to_slow=1.5)`` construction bypasses them and
    would otherwise silently drive active respiration negative (fraction > 1)
    or a downstream transfer input negative (fraction < 0)."""

    def _call(self, **cfg_over):
        cfg = _default_config(scheme="differland", **cfg_over)
        state = _make_carbon_state(shape=(2,))
        return step_carbon_differland(
            state, jnp.full(2, 300.0), jnp.full(2, 290.0), jnp.full(2, 400.0),
            jnp.full(2, 0.8), jnp.full(2, 0.7), 150.0, jnp.full(2, 3e-5),
            cfg, 600.0)

    def test_f_active_to_slow_above_one_raises(self):
        with self.assertRaises(ValueError):
            self._call(f_active_to_slow=1.5)

    def test_f_active_to_slow_negative_raises(self):
        with self.assertRaises(ValueError):
            self._call(f_active_to_slow=-0.1)

    def test_f_slow_to_passive_above_one_raises(self):
        with self.assertRaises(ValueError):
            self._call(f_slow_to_passive=1.5)

    def test_f_slow_to_passive_negative_raises(self):
        with self.assertRaises(ValueError):
            self._call(f_slow_to_passive=-0.1)

    def test_cwd_humification_eff_above_one_raises(self):
        """cwd_humification_eff > 1 makes R_het_cwd = wood_litter - eff*wood_litter
        negative (the CWD path would create carbon)."""
        with self.assertRaises(ValueError):
            self._call(cwd_humification_eff=1.5)

    def test_cwd_humification_eff_negative_raises(self):
        """cwd_humification_eff < 0 makes wood_to_som negative (destroys carbon
        in the wood->SOM transfer)."""
        with self.assertRaises(ValueError):
            self._call(cwd_humification_eff=-0.1)

    def test_boundary_fractions_do_not_raise(self):
        """0.0 and 1.0 are valid (inclusive) bounds for all three fractions."""
        self._call(f_active_to_slow=0.0, f_slow_to_passive=1.0,
                   cwd_humification_eff=0.0)
        self._call(f_active_to_slow=1.0, f_slow_to_passive=0.0,
                   cwd_humification_eff=1.0)


# ===================================================================
# Cold-vs-warm SOC realism (the scientific point of the change)
# ===================================================================

class TestColdWarmSocRealism(unittest.TestCase):
    """A COLD soil equilibrates to MORE total SOC than a WARM soil with the SAME
    litter input (freeze + temperature suppression -> carbon retained), and more
    than the former single bulk pool held for a cold case.  The ``som_freeze_floor``
    keeps the cold accumulation FINITE (no runaway) while preserving cold>warm."""

    @staticmethod
    def _cascade_soc_eq(cfg, T, precip, som_input_per_day):
        """Analytic total SOM equilibrium of the live cascade for a constant
        input and (T, precip), using the model's OWN modifier + config rates:
        C_active_eq = I/(m*k_a); C_slow_eq = f_as*I/(m*k_s);
        C_passive_eq = f_as*f_sp*I/(m*k_p) (derived in step_carbon_differland)."""
        m = float(_som_decomp_modifier(
            jnp.array([T]), jnp.array([precip]), cfg)[0])
        i = som_input_per_day
        f_as, f_sp = cfg.f_active_to_slow, cfg.f_slow_to_passive
        return (i / (m * cfg.tor_som_active)
                + f_as * i / (m * cfg.tor_som_slow)
                + f_as * f_sp * i / (m * cfg.tor_som_passive))

    def test_cold_holds_more_soc_than_warm(self):
        cfg = _default_config(scheme="differland")
        i, precip = 0.2, _default_config().precip_ref
        soc_cold = self._cascade_soc_eq(cfg, 270.0, precip, i)   # -3 C, frozen
        soc_warm = self._cascade_soc_eq(cfg, 298.0, precip, i)   # +25 C
        self.assertGreater(soc_cold, soc_warm)
        self.assertGreater(soc_cold / soc_warm, 10.0)

    def test_cold_multipool_exceeds_old_single_pool(self):
        """The cold multi-pool SOC exceeds what the former single bulk pool
        (tor_som=4e-5/day, NO freeze control) held at the same input — the fix
        direction for high-latitude / grassland SOC underestimation."""
        cfg = _default_config(scheme="differland")
        i, precip, T = 0.2, _default_config().precip_ref, 270.0
        soc_multi = self._cascade_soc_eq(cfg, T, precip, i)
        # Old single pool: freeze-free _temperate_modifier, bulk 4e-5/day turnover.
        m_old = float(_temperate_modifier(
            jnp.array([T]), jnp.array([precip]), cfg)[0])
        soc_old_single = i / (m_old * 4e-5)   # legacy tor_som default
        self.assertGreater(soc_multi, soc_old_single)

    def test_extreme_cold_soc_high_but_bounded_by_floor(self):
        """The ``som_freeze_floor`` caps runaway cold-soil SOC while KEEPING the
        cold-retains-more direction.

        Two facts, both from the model's OWN cascade equilibrium:

        (1) A genuinely cold column (-5 C) equilibrates to a HIGH but BOUNDED
            total SOC -- far above the warm case, yet below a sane physical cap
            (not the runaway the un-floored curve produced in a real ERA5 build,
            global max ~175 kgC/m2).
        (2) As the soil goes DEEP-frozen the un-floored modifier -> 0 so the
            millennial slow/passive equilibrium ``I/(m*k)`` runs away without
            bound; the floor keeps decomposition >= ``som_freeze_floor`` of the
            unfrozen rate, so the SAME column stays finite and dramatically
            smaller -- the floor MEANINGFULLY reduces the deep-cold equilibrium.
        """
        precip, i = _default_config().precip_ref, 0.2
        cfg = _default_config(scheme="differland")                 # floor 0.05
        cfg_nofloor = _default_config(scheme="differland", som_freeze_floor=0.0)

        # (1) extreme (but not permafrost) cold: high, bounded, well above warm.
        soc_cold = self._cascade_soc_eq(cfg, 268.0, precip, i)     # -5 C
        soc_warm = self._cascade_soc_eq(cfg, 298.0, precip, i)     # +25 C
        self.assertTrue(math.isfinite(soc_cold))
        self.assertGreater(soc_cold, soc_warm)          # cold retains more
        self.assertGreater(soc_cold / soc_warm, 10.0)   # well above warm
        # Bounded: below a sane physical cap (200 kgC/m2); the un-floored value
        # at this same column is NOT below the cap -> the floor is what bounds it.
        self.assertLess(soc_cold, 200_000.0)            # gC/m2 == 200 kgC/m2
        self.assertGreater(
            self._cascade_soc_eq(cfg_nofloor, 268.0, precip, i), soc_cold)

        # (2) DEEP freeze: floor turns an unbounded runaway into a finite value.
        soc_deep_floor = self._cascade_soc_eq(cfg, 255.0, precip, i)
        soc_deep_nofloor = self._cascade_soc_eq(cfg_nofloor, 255.0, precip, i)
        self.assertTrue(math.isfinite(soc_deep_floor))
        self.assertGreater(soc_deep_floor, soc_warm)
        # Meaningful reduction vs no floor (actual ~440x at -18 C).
        self.assertGreater(soc_deep_nofloor / soc_deep_floor, 50.0)


# ===================================================================
# Autotrophic / heterotrophic reference-temperature decoupling (CUE fix)
# ===================================================================

class TestReferenceTemperatureDecoupling(unittest.TestCase):
    """The autotrophic maintenance-respiration reference ``T_ref_ra`` and the
    heterotrophic (soil-decomposition) reference ``T_ref`` are ORTHOGONAL:
    ``T_ref_ra`` drives ONLY autotrophic ``R_maint``; ``T_ref`` drives ONLY
    heterotrophic decomposition.  Guards the decoupling introduced by the CUE fix
    (maintenance respiration was over-consuming GPP through the shared 10 degC
    reference, collapsing CUE = NPP/GPP to ~0.11 with negative NPP)."""

    def _diag(self, cfg, T_val=298.0):
        ncol = 3
        state = _make_carbon_state(shape=(ncol,))
        _s, _f, diag = step_carbon_differland(
            state, jnp.full(ncol, 350.0), jnp.full(ncol, T_val),
            jnp.full(ncol, 400.0), jnp.full(ncol, 0.8), jnp.full(ncol, 0.7),
            160.0, jnp.full(ncol, 3e-5), cfg, 1800.0, return_diagnostics=True)
        return diag

    def test_heterotrophic_modifier_invariant_to_T_ref_ra(self):
        """Changing T_ref_ra must NOT change the heterotrophic modifiers
        (``_temperate_modifier`` / ``_som_decomp_modifier`` use ``T_ref``, and
        ``analytic_som_soc`` is built from ``_som_decomp_modifier``)."""
        precip = jnp.array([3e-5, 3e-5])
        T = jnp.array([300.0, 270.0])
        lo = _default_config(scheme="differland", T_ref_ra=298.15)
        hi = _default_config(scheme="differland", T_ref_ra=350.0)
        npt.assert_allclose(_temperate_modifier(T, precip, lo),
                            _temperate_modifier(T, precip, hi), rtol=0, atol=0)
        npt.assert_allclose(_som_decomp_modifier(T, precip, lo),
                            _som_decomp_modifier(T, precip, hi), rtol=0, atol=0)

    def test_autotrophic_R_maint_invariant_to_T_ref(self):
        """Changing the heterotrophic ``T_ref`` must NOT change autotrophic
        ``R_maint`` (which uses ``T_ref_ra`` + ``Q10_exp`` only)."""
        base = self._diag(_default_config(scheme="differland", T_ref=283.15))
        alt = self._diag(_default_config(scheme="differland", T_ref=250.0))
        npt.assert_allclose(base.r_maint, alt.r_maint, rtol=1e-12, atol=0)

    def test_R_maint_exact_ratio_from_T_ref_ra(self):
        """+15 K in ``T_ref_ra`` scales ``R_maint`` by exactly
        ``exp(-Q10_exp*15) = 0.5488116`` at every temperature (a UNIFORM rescaling
        of the maintenance curve, not a warm-only correction)."""
        cfg_lo = _default_config(scheme="differland", T_ref_ra=283.15)
        cfg_hi = _default_config(scheme="differland", T_ref_ra=298.15)
        ratio = self._diag(cfg_hi).r_maint / self._diag(cfg_lo).r_maint
        expected = math.exp(-cfg_lo.Q10_exp * 15.0)  # 0.5488116...
        npt.assert_allclose(ratio, expected, rtol=1e-6, atol=0)

    def test_higher_T_ref_ra_lowers_rauto_raises_npp(self):
        """Raising ``T_ref_ra`` lowers ``R_maint`` and ``R_auto`` and RAISES NPP
        (the fix direction: less maintenance loss => more net production).
        ``R_auto = f_auto*GPP + (1-f_auto)*R_maint`` is strictly increasing in
        ``R_maint``, and ``NPP = GPP - R_auto``."""
        d_lo = self._diag(_default_config(scheme="differland", T_ref_ra=283.15))
        d_hi = self._diag(_default_config(scheme="differland", T_ref_ra=298.15))
        self.assertTrue(bool(jnp.all(d_hi.r_maint < d_lo.r_maint)))
        self.assertTrue(bool(jnp.all(d_hi.r_auto < d_lo.r_auto)))
        self.assertTrue(bool(jnp.all(d_hi.npp > d_lo.npp)))

    def test_jit_and_grad_smoke(self):
        """``step_carbon_differland`` stays JIT-able and differentiable through the
        new reference (``T_ref_ra`` is a static float leaf; no new control flow)."""
        cfg = _default_config(scheme="differland")
        ncol = 2
        state = _make_carbon_state(shape=(ncol,))
        sw = jnp.full(ncol, 320.0)
        co2 = jnp.full(ncol, 400.0)
        beta = jnp.full(ncol, 0.8)
        lat = jnp.full(ncol, 0.7)
        precip = jnp.full(ncol, 3e-5)
        jitted = jax.jit(lambda st, Tair: step_carbon_differland(
            st, sw, Tair, co2, beta, lat, 150.0, precip, cfg, 1800.0))
        s2, f2 = jitted(state, jnp.full(ncol, 295.0))
        self.assertTrue(bool(jnp.all(jnp.isfinite(f2))))
        for name in s2._fields:
            self.assertTrue(bool(jnp.all(jnp.isfinite(getattr(s2, name)))))

        def loss(Tair):
            _s, _f, diag = step_carbon_differland(
                state, sw, Tair, co2, beta, lat, 150.0, precip, cfg, 1800.0,
                return_diagnostics=True)
            return jnp.sum(diag.npp)
        g = jax.grad(loss)(jnp.full(ncol, 295.0))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))


# ===================================================================
# DifferLand step
# ===================================================================

class TestDifferLandStep(unittest.TestCase):

    def _step_once(self, config=None):
        cfg = config or _default_config(scheme="differland")
        ncol = 4
        state = _make_carbon_state(shape=(ncol,))
        sw = jnp.full(ncol, 300.0)
        T = jnp.full(ncol, 290.0)
        lat = jnp.full(ncol, 0.7)  # ~40°N
        precip = jnp.full(ncol, 3e-5)
        co2 = jnp.full(ncol, 400.0)
        beta = jnp.full(ncol, 0.8)
        dt = 600.0
        return step_carbon_differland(
            state, sw, T, co2, beta, lat, 150.0, precip, cfg, dt,
        )

    def test_step_finite(self):
        new_state, co2_flux = self._step_once()
        for name in new_state._fields:
            arr = getattr(new_state, name)
            self.assertTrue(jnp.all(jnp.isfinite(arr)), f"{name} not finite")
        self.assertTrue(jnp.all(jnp.isfinite(co2_flux)))

    def test_pools_stay_positive(self):
        new_state, _ = self._step_once()
        # A2: all eight pools (incl. the live slow/passive SOM pools) stay above
        # the 1 gC/m2 floor for a well-stocked column at a realistic dt.
        for name in new_state._fields:
            arr = getattr(new_state, name)
            self.assertTrue(jnp.all(arr >= 1.0), f"{name} below floor")

    def test_co2_flux_units(self):
        """CO2 flux should be in kgCO2/m2/s — typical magnitude ~1e-8 to 1e-6."""
        _, co2_flux = self._step_once()
        mag = jnp.abs(co2_flux)
        self.assertTrue(jnp.all(mag < 1e-3),
                        "CO2 flux too large (>1e-3 kgCO2/m2/s)")

    def test_daytime_uptake(self):
        """With strong light, ecosystem should be a net sink (co2_flux < 0)."""
        cfg = _default_config(scheme="differland", epsilon=2.0)
        ncol = 2
        state = _make_carbon_state(shape=(ncol,), C_fol=jnp.full(ncol, 500.0))
        sw = jnp.full(ncol, 500.0)  # strong sunlight
        T = jnp.full(ncol, 293.0)   # optimal-ish temperature
        lat = jnp.full(ncol, 0.7)
        precip = jnp.full(ncol, 3e-5)
        beta = jnp.ones(ncol)
        _, co2_flux = step_carbon_differland(
            state, sw, T, jnp.full(ncol, 400.0), beta, lat, 180.0,
            precip, cfg, 600.0,
        )
        # With high LAI and strong light, should be net sink
        self.assertTrue(jnp.all(co2_flux < 0),
                        f"Expected sink but got co2_flux={co2_flux}")

    def test_nighttime_source(self):
        """At night (sw=0), respiration dominates → net source."""
        cfg = _default_config(scheme="differland")
        ncol = 2
        state = _make_carbon_state(shape=(ncol,))
        sw = jnp.zeros(ncol)  # night
        T = jnp.full(ncol, 290.0)
        lat = jnp.full(ncol, 0.7)
        precip = jnp.full(ncol, 3e-5)
        beta = jnp.ones(ncol)
        _, co2_flux = step_carbon_differland(
            state, sw, T, jnp.full(ncol, 400.0), beta, lat, 180.0,
            precip, cfg, 600.0,
        )
        self.assertTrue(jnp.all(co2_flux > 0),
                        f"Expected source at night but got co2_flux={co2_flux}")

    def test_multi_step_stability(self):
        """100 steps at dt=600s should remain stable."""
        cfg = _default_config(scheme="differland")
        ncol = 4
        state = _make_carbon_state(shape=(ncol,))
        sw = jnp.full(ncol, 250.0)
        T = jnp.full(ncol, 288.0)
        lat = jnp.full(ncol, 0.7)
        precip = jnp.full(ncol, 3e-5)
        beta = jnp.full(ncol, 0.7)
        co2 = jnp.full(ncol, 400.0)
        for i in range(100):
            doy = float(i * 600 / 86400)  # fractional days
            state, flux = step_carbon_differland(
                state, sw, T, co2, beta, lat, doy, precip, cfg, 600.0,
            )
        for name in state._fields:
            arr = getattr(state, name)
            self.assertTrue(jnp.all(jnp.isfinite(arr)),
                            f"{name} not finite after 100 steps")

    def test_total_carbon_conservation_tendency(self):
        """Sum of all pool changes should equal NEE * dt (approx)."""
        cfg = _default_config(scheme="differland")
        ncol = 2
        state = _make_carbon_state(shape=(ncol,))
        sw = jnp.full(ncol, 300.0)
        T = jnp.full(ncol, 290.0)
        lat = jnp.full(ncol, 0.7)
        precip = jnp.full(ncol, 3e-5)
        beta = jnp.ones(ncol)
        co2 = jnp.full(ncol, 400.0)
        dt = 600.0

        new_state, co2_flux = step_carbon_differland(
            state, sw, T, co2, beta, lat, 150.0, precip, cfg, dt,
        )

        # Total carbon change [gC/m2]
        dC_total = sum(
            getattr(new_state, f) - getattr(state, f)
            for f in state._fields
        )
        # NEE in gC/m2/s (convert back from kgCO2/m2/s)
        nee_gC = co2_flux / ((44.0 / 12.0) * 1e-3)
        expected_dC = -nee_gC * dt  # negative because NEE>0 = loss from land

        npt.assert_allclose(dC_total, expected_dC, rtol=0.05,
                            err_msg="Carbon not approximately conserved")

    def test_total_carbon_conservation_under_carbon_starvation(self):
        """Conservation must hold even when GPP < R_auto (polar winter /
        drought) — the iter-62 fix draws the NPP deficit from C_lab so
        the atmosphere gain equals the biomass loss exactly.

        Before the fix, ``R_auto`` was emitted to NEE without debiting
        any pool when GPP < R_auto, creating ``(R_auto − GPP)·dt`` of
        ghost carbon every step.
        """
        cfg = _default_config(scheme="differland")
        ncol = 2
        state = _make_carbon_state(shape=(ncol,))
        # Carbon-starvation forcing: zero light + cold + dry.
        sw = jnp.zeros(ncol)
        T = jnp.full(ncol, 285.0)   # cold-ish (above freezing, still respiring)
        lat = jnp.full(ncol, 0.7)
        precip = jnp.full(ncol, 1e-6)
        beta = jnp.ones(ncol)
        co2 = jnp.full(ncol, 400.0)
        dt = 600.0

        new_state, co2_flux = step_carbon_differland(
            state, sw, T, co2, beta, lat, 15.0, precip, cfg, dt,
        )
        dC_total = sum(
            getattr(new_state, f) - getattr(state, f)
            for f in state._fields
        )
        nee_gC = co2_flux / ((44.0 / 12.0) * 1e-3)
        expected_dC = -nee_gC * dt
        # Tight tolerance — bounded only by the C_lab cap if respiration
        # exceeds the labile reserves over one step (not the case here).
        npt.assert_allclose(dC_total, expected_dC, rtol=1e-3,
                            err_msg="Carbon not conserved under starvation")

    def test_total_carbon_conservation_with_exhausted_labile_pool(self):
        """Carbon conservation holds even when C_lab is exhausted — the
        iter-63 cascade fix routes the remainder to C_fol → C_root →
        C_wood.

        Before iter-63 the deficit draw was capped at C_lab only, so a
        column with depleted labile reserves AND ongoing R_auto bleed
        leaked (R_auto·dt − C_lab) of ghost carbon per step.
        """
        cfg = _default_config(scheme="differland")
        ncol = 1
        # Build a state with C_lab ≈ 0 (exhausted) but biomass intact.
        from legoesm.land.carbon.config import CarbonState
        state = CarbonState(
            C_lab=jnp.full((ncol,), 0.5),     # near-zero labile
            C_fol=jnp.full((ncol,), 300.0),
            C_root=jnp.full((ncol,), 400.0),
            C_wood=jnp.full((ncol,), 10000.0),
            C_lit=jnp.full((ncol,), 600.0),
            C_som_active=jnp.full((ncol,), 12000.0),
            C_som_slow=jnp.zeros((ncol,)),
            C_som_passive=jnp.zeros((ncol,)),
        )
        sw = jnp.zeros(ncol)            # no GPP
        T = jnp.full(ncol, 290.0)
        lat = jnp.full(ncol, 0.7)
        precip = jnp.full(ncol, 1e-6)
        beta = jnp.ones(ncol)
        co2 = jnp.full(ncol, 400.0)
        dt = 600.0

        new_state, co2_flux = step_carbon_differland(
            state, sw, T, co2, beta, lat, 15.0, precip, cfg, dt,
        )
        dC_total = sum(
            getattr(new_state, f) - getattr(state, f)
            for f in state._fields
        )
        nee_gC = co2_flux / ((44.0 / 12.0) * 1e-3)
        expected_dC = -nee_gC * dt
        npt.assert_allclose(
            dC_total, expected_dC, rtol=1e-3,
            err_msg="Carbon not conserved when C_lab exhausted",
        )

    def test_total_carbon_conservation_machine_precision(self):
        """Iter-64: with the natural-turnover-aware cascade cap and the
        ``jnp.maximum`` (vs softplus) clamp, conservation should hold
        at near-machine precision under any forcing — not just the
        loose 5% / 0.1% bounds of the earlier tests.

        Tests carbon-starvation forcing AND a state with a depleted
        labile pool, then asserts rtol=1e-9 conservation.
        """
        cfg = _default_config(scheme="differland")
        ncol = 1
        from legoesm.land.carbon.config import CarbonState
        state = CarbonState(
            C_lab=jnp.full((ncol,), 0.5),
            C_fol=jnp.full((ncol,), 300.0),
            C_root=jnp.full((ncol,), 400.0),
            C_wood=jnp.full((ncol,), 10000.0),
            C_lit=jnp.full((ncol,), 600.0),
            C_som_active=jnp.full((ncol,), 12000.0),
            C_som_slow=jnp.zeros((ncol,)),
            C_som_passive=jnp.zeros((ncol,)),
        )
        sw = jnp.zeros(ncol)            # no GPP
        T = jnp.full(ncol, 290.0)
        lat = jnp.full(ncol, 0.7)
        precip = jnp.full(ncol, 1e-6)
        beta = jnp.ones(ncol)
        co2 = jnp.full(ncol, 400.0)
        dt = 600.0

        new_state, co2_flux = step_carbon_differland(
            state, sw, T, co2, beta, lat, 15.0, precip, cfg, dt,
        )
        dC_total = sum(
            getattr(new_state, f) - getattr(state, f)
            for f in state._fields
        )
        nee_gC = co2_flux / ((44.0 / 12.0) * 1e-3)
        expected_dC = -nee_gC * dt
        npt.assert_allclose(
            dC_total, expected_dC, rtol=1e-9, atol=1e-9,
            err_msg="Carbon conservation must hold at near-machine precision",
        )


# ===================================================================
# DifferLand diagnostics (return_diagnostics=True)
# ===================================================================

class TestCarbonDiagnostics(unittest.TestCase):

    def _step_diag(self, **forcing_overrides):
        cfg = _default_config(scheme="differland")
        ncol = 3
        state = _make_carbon_state(shape=(ncol,))
        args = dict(
            sw_down=jnp.full(ncol, 350.0),
            T=jnp.full(ncol, 293.0),
            co2_ppmv=jnp.full(ncol, 400.0),
            beta=jnp.full(ncol, 0.8),
            lat=jnp.full(ncol, 0.7),
            precip=jnp.full(ncol, 3e-5),
        )
        args.update(forcing_overrides)
        new_state, co2_flux, diag = step_carbon_differland(
            state, args["sw_down"], args["T"], args["co2_ppmv"], args["beta"],
            args["lat"], 160.0, args["precip"], cfg, 1800.0,
            return_diagnostics=True,
        )
        return state, new_state, co2_flux, diag

    def test_returns_three_tuple_and_type(self):
        _, new_state, co2_flux, diag = self._step_diag()
        self.assertIsInstance(diag, CarbonDiagnostics)
        self.assertIsInstance(new_state, CarbonState)
        for name in diag._fields:
            self.assertTrue(jnp.all(jnp.isfinite(getattr(diag, name))),
                            f"diag.{name} not finite")

    def test_state_update_unchanged_by_diag_flag(self):
        """return_diagnostics must not alter the state update or flux."""
        cfg = _default_config(scheme="differland")
        ncol = 3
        state = _make_carbon_state(shape=(ncol,))
        common = (state, jnp.full(ncol, 350.0), jnp.full(ncol, 293.0),
                  jnp.full(ncol, 400.0), jnp.full(ncol, 0.8),
                  jnp.full(ncol, 0.7), 160.0, jnp.full(ncol, 3e-5), cfg, 1800.0)
        s2, f2 = step_carbon_differland(*common)
        s3, f3, _ = step_carbon_differland(*common, return_diagnostics=True)
        npt.assert_allclose(f2, f3, rtol=0, atol=0)
        for name in s2._fields:
            npt.assert_allclose(getattr(s2, name), getattr(s3, name),
                                rtol=0, atol=0)

    def test_allocation_closes_to_npp(self):
        """a_fol + a_lab + a_root + a_wood == max(npp, 0), exactly."""
        _, _, _, diag = self._step_diag()
        alloc_sum = diag.a_fol + diag.a_lab + diag.a_root + diag.a_wood
        npp_pos = jnp.maximum(diag.npp, 0.0)
        npt.assert_allclose(alloc_sum, npp_pos, rtol=1e-6, atol=1e-9)
        # Every allocation flux is non-negative (no pool "steals" carbon).
        for name in ("a_fol", "a_lab", "a_root", "a_wood"):
            self.assertTrue(jnp.all(getattr(diag, name) >= 0.0),
                            f"{name} negative")

    def test_npp_and_rauto_identities(self):
        """npp == gpp - r_auto and r_auto == r_maint + r_growth."""
        _, _, _, diag = self._step_diag()
        npt.assert_allclose(diag.npp, diag.gpp - diag.r_auto, rtol=1e-6, atol=1e-9)
        npt.assert_allclose(diag.r_auto, diag.r_maint + diag.r_growth,
                            rtol=1e-6, atol=1e-9)
        npt.assert_allclose(
            diag.r_het, diag.r_het_lit + diag.r_het_som + diag.r_het_cwd,
            rtol=1e-6, atol=1e-9)

    def test_cwd_humification_split(self):
        """Wood turnover splits into humified SOM input + CWD respiration."""
        cfg = _default_config(scheme="differland", cwd_humification_eff=0.3)
        ncol = 2
        state = _make_carbon_state(shape=(ncol,))
        _, _, diag = step_carbon_differland(
            state, jnp.full(ncol, 300.0), jnp.full(ncol, 293.0),
            jnp.full(ncol, 400.0), jnp.full(ncol, 0.8), jnp.full(ncol, 0.7),
            160.0, jnp.full(ncol, 3e-5), cfg, 1800.0, return_diagnostics=True)
        # wood_to_som + r_het_cwd == wood_litter (conserved split)
        npt.assert_allclose(diag.wood_to_som + diag.r_het_cwd, diag.wood_litter,
                            rtol=1e-9, atol=1e-12)
        npt.assert_allclose(diag.wood_to_som, 0.3 * diag.wood_litter,
                            rtol=1e-6, atol=1e-12)

    def test_nee_matches_co2_flux(self):
        """diag.nee [gC/m2/day] converts to the returned co2_flux exactly."""
        _, _, co2_flux, diag = self._step_diag()
        expected_flux = diag.nee / _SPD * _GC_TO_KG_CO2
        npt.assert_allclose(co2_flux, expected_flux, rtol=1e-9, atol=1e-30)

    def test_nee_budget_identity(self):
        """nee == r_auto - unmet_npp_deficit + r_het - gpp (definition)."""
        _, _, _, diag = self._step_diag()
        expected = (diag.r_auto - diag.unmet_npp_deficit
                    + diag.r_het - diag.gpp)
        npt.assert_allclose(diag.nee, expected, rtol=1e-6, atol=1e-9)

    def test_diag_conservation(self):
        """Pool change equals -nee*dt_days using the diagnostics only."""
        state, new_state, _, diag = self._step_diag()
        dt_days = 1800.0 / _SPD
        dC = sum(getattr(new_state, f) - getattr(state, f)
                 for f in state._fields)
        npt.assert_allclose(dC, -diag.nee * dt_days, rtol=1e-6, atol=1e-9)

    def test_lai_definition(self):
        cfg = _default_config(scheme="differland")
        _, _, _, diag = self._step_diag()
        state = _make_carbon_state(shape=(3,))
        npt.assert_allclose(diag.lai, state.C_fol / cfg.LCMA, rtol=1e-6)

    def test_dispatcher_diag_differland(self):
        cfg = _default_config(scheme="differland")
        state = _make_carbon_state()
        out = step_carbon(
            state, sw_down=jnp.full(4, 300.0), T=jnp.full(4, 290.0),
            co2_ppmv=jnp.full(4, 400.0), beta=jnp.full(4, 0.8),
            lat=jnp.full(4, 0.7), doy=150.0, precip=jnp.full(4, 3e-5),
            config=cfg, dt=600.0, return_diagnostics=True,
        )
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[2], CarbonDiagnostics)

    def test_dispatcher_diag_raises_for_non_differland(self):
        for scheme in ("none", "seasonal"):
            cfg = _default_config(scheme=scheme)
            with self.assertRaises(ValueError):
                step_carbon(
                    None, sw_down=jnp.full(2, 300.0), T=jnp.full(2, 290.0),
                    co2_ppmv=jnp.full(2, 400.0), beta=jnp.full(2, 0.8),
                    lat=jnp.full(2, 0.7), doy=150.0, precip=jnp.full(2, 3e-5),
                    config=cfg, dt=600.0, return_diagnostics=True,
                )


# ===================================================================
# Woody vs herbaceous allocation
# ===================================================================

class TestWoodyAllocation(unittest.TestCase):

    def _diag(self, woody):
        cfg = _default_config(scheme="differland", woody=woody)
        ncol = 2
        state = _make_carbon_state(shape=(ncol,))
        _, _, diag = step_carbon_differland(
            state, jnp.full(ncol, 400.0), jnp.full(ncol, 295.0),
            jnp.full(ncol, 400.0), jnp.full(ncol, 1.0), jnp.full(ncol, 0.5),
            180.0, jnp.full(ncol, 3e-5), cfg, 1800.0, return_diagnostics=True)
        return diag

    def test_woody_allocates_to_wood(self):
        diag = self._diag(woody=True)
        self.assertTrue(jnp.all(diag.a_wood > 0.0))

    def test_herbaceous_no_wood_allocation(self):
        diag = self._diag(woody=False)
        npt.assert_allclose(diag.a_wood, 0.0, atol=1e-15)

    def test_herbaceous_redirects_wood_to_root(self):
        """The structural (wood) fraction is invested in roots instead."""
        dw = self._diag(woody=True)
        dh = self._diag(woody=False)
        self.assertTrue(jnp.all(dh.a_root > dw.a_root))
        # The extra root allocation equals the woody-case wood allocation.
        npt.assert_allclose(dh.a_root - dw.a_root, dw.a_wood, rtol=1e-6, atol=1e-9)

    def test_allocation_closes_both_woodiness(self):
        for woody in (True, False):
            diag = self._diag(woody=woody)
            alloc = diag.a_fol + diag.a_lab + diag.a_root + diag.a_wood
            npt.assert_allclose(alloc, jnp.maximum(diag.npp, 0.0),
                                rtol=1e-6, atol=1e-9,
                                err_msg=f"allocation not closed (woody={woody})")


class TestSequentialAllocation(unittest.TestCase):
    """Direct test of the shared DALEC allocation helper (also used by the closed-form
    live-pool forward) -- the exact telescoping partition step_carbon_differland uses."""

    def test_partition_matches_hand_value_and_closes(self):
        from legoesm.land.carbon.carbon_cycle import sequential_allocation
        npp = jnp.asarray([100.0, 0.0])
        A_fol, A_lab, A_root_base, A_wood_raw = sequential_allocation(
            npp, 0.15, 0.10, 0.25)
        # a_fol=.15; a_lab=(1-.15)*.10=.085; a_root_base=(1-.15)(1-.10)*.25=.19125;
        # a_wood_raw = 1 - .15 - .085 - .19125 = .57375 (of NPP).
        npt.assert_allclose(A_fol, [15.0, 0.0], rtol=1e-12)
        npt.assert_allclose(A_lab, [8.5, 0.0], rtol=1e-12)
        npt.assert_allclose(A_root_base, [19.125, 0.0], rtol=1e-12)
        npt.assert_allclose(A_wood_raw, [57.375, 0.0], rtol=1e-12)
        # The four fluxes sum to NPP EXACTLY (telescoping remainder).
        npt.assert_allclose(A_fol + A_lab + A_root_base + A_wood_raw, npp,
                            rtol=0, atol=0)

    def test_matches_step_carbon_allocation(self):
        """The helper reproduces the step's absolute allocation fluxes (shared numerics)."""
        from legoesm.land.carbon.carbon_cycle import sequential_allocation
        cfg = _default_config(scheme="differland", woody=True)
        diag = TestWoodyAllocation()._diag(woody=True)
        npp_pos = jnp.maximum(diag.npp, 0.0)
        A_fol, A_lab, A_root_base, A_wood_raw = sequential_allocation(
            npp_pos, cfg.f_fol, cfg.f_lab, cfg.f_root)
        npt.assert_allclose(A_fol, diag.a_fol, rtol=1e-12)
        npt.assert_allclose(A_lab, diag.a_lab, rtol=1e-12)
        # Woody: a_root == A_root_base, a_wood == max(A_wood_raw, 0).
        npt.assert_allclose(A_root_base, diag.a_root, rtol=1e-12)
        npt.assert_allclose(jnp.maximum(A_wood_raw, 0.0), diag.a_wood, rtol=1e-12)


# ===================================================================
# Seasonal Cycle
# ===================================================================

class TestSeasonalCycle(unittest.TestCase):

    def test_seasonal_shape(self):
        cfg = _default_config(scheme="seasonal")
        lat = jnp.linspace(-1.0, 1.0, 10)
        flux = seasonal_co2_flux(180.0, lat, cfg)
        self.assertEqual(flux.shape, (10,))

    def test_seasonal_nh_summer_sink(self):
        """NH mid-latitudes should have negative flux (sink) at peak_day."""
        cfg = _default_config(scheme="seasonal", nee_peak_day=200.0)
        lat = jnp.array([0.7])  # ~40°N
        flux = seasonal_co2_flux(200.0, lat, cfg)
        self.assertTrue(float(flux[0]) < 0, "NH should be sink at peak_day")

    def test_seasonal_nh_winter_source(self):
        """NH mid-latitudes should have positive flux (source) in winter."""
        cfg = _default_config(scheme="seasonal", nee_peak_day=200.0)
        lat = jnp.array([0.7])
        flux = seasonal_co2_flux(17.0, lat, cfg)
        self.assertTrue(float(flux[0]) > 0, "NH should be source in winter")

    def test_seasonal_sh_opposite(self):
        """SH should have opposite sign from NH at same doy."""
        cfg = _default_config(scheme="seasonal")
        flux_nh = seasonal_co2_flux(200.0, jnp.array([0.7]), cfg)
        flux_sh = seasonal_co2_flux(200.0, jnp.array([-0.7]), cfg)
        self.assertTrue(float(flux_nh[0]) * float(flux_sh[0]) < 0,
                        "NH and SH should have opposite sign")

    def test_seasonal_equator_zero(self):
        """Equator (lat=0) should have near-zero flux."""
        cfg = _default_config(scheme="seasonal")
        flux = seasonal_co2_flux(200.0, jnp.array([0.0]), cfg)
        npt.assert_allclose(flux, 0.0, atol=1e-15)

    def test_seasonal_midlat_peak(self):
        """Amplitude should be largest at mid-latitudes (~45°)."""
        cfg = _default_config(scheme="seasonal")
        lat_25 = jnp.array([25 * jnp.pi / 180])
        lat_45 = jnp.array([45 * jnp.pi / 180])
        lat_80 = jnp.array([80 * jnp.pi / 180])
        amp_25 = jnp.abs(seasonal_co2_flux(200.0, lat_25, cfg))
        amp_45 = jnp.abs(seasonal_co2_flux(200.0, lat_45, cfg))
        amp_80 = jnp.abs(seasonal_co2_flux(200.0, lat_80, cfg))
        self.assertTrue(float(amp_45[0]) > float(amp_25[0]))
        self.assertTrue(float(amp_45[0]) > float(amp_80[0]))


# ===================================================================
# Dispatch (step_carbon)
# ===================================================================

class TestStepCarbon(unittest.TestCase):

    def _args(self, ncol=4):
        return dict(
            sw_down=jnp.full(ncol, 300.0),
            T=jnp.full(ncol, 290.0),
            co2_ppmv=jnp.full(ncol, 400.0),
            beta=jnp.full(ncol, 0.8),
            lat=jnp.full(ncol, 0.7),
            doy=150.0,
            precip=jnp.full(ncol, 3e-5),
            dt=600.0,
        )

    def test_none_returns_zeros(self):
        cfg = _default_config(scheme="none")
        state_out, flux = step_carbon(None, **self._args(), config=cfg)
        self.assertIsNone(state_out)
        npt.assert_allclose(flux, 0.0, atol=1e-15)

    def test_seasonal_no_state_needed(self):
        cfg = _default_config(scheme="seasonal")
        state_out, flux = step_carbon(None, **self._args(), config=cfg)
        self.assertIsNone(state_out)
        self.assertTrue(jnp.all(jnp.isfinite(flux)))

    def test_differland_requires_state(self):
        cfg = _default_config(scheme="differland")
        with self.assertRaises(ValueError):
            step_carbon(None, **self._args(), config=cfg)

    def test_differland_dispatch(self):
        cfg = _default_config(scheme="differland")
        state = _make_carbon_state()
        new_state, flux = step_carbon(state, **self._args(), config=cfg)
        self.assertIsNotNone(new_state)
        self.assertTrue(jnp.all(jnp.isfinite(flux)))


# ===================================================================
# Initialization
# ===================================================================

class TestInitCarbonState(unittest.TestCase):

    def test_init_shape(self):
        cfg = _default_config()
        state = init_carbon_state((6, 4, 4), cfg)
        for name in state._fields:
            self.assertEqual(getattr(state, name).shape, (6, 4, 4))

    def test_init_values(self):
        cfg = _default_config(C_lab_init=50.0, C_som_init=5000.0)
        state = init_carbon_state((3,), cfg)
        npt.assert_allclose(state.C_lab, 50.0)
        # A2: C_som_init is partitioned across the three pools by CENTURY
        # fractions (0.03 / 0.32 / 0.65); som_total is preserved exactly.
        npt.assert_allclose(som_total(state), 5000.0, rtol=1e-9)
        npt.assert_allclose(state.C_som_active, 0.03 * 5000.0, rtol=1e-9)
        npt.assert_allclose(state.C_som_slow, 0.32 * 5000.0, rtol=1e-9)
        npt.assert_allclose(state.C_som_passive, 0.65 * 5000.0, rtol=1e-9)

    def test_woody_init_keeps_wood_pool(self):
        cfg = _default_config(woody=True, C_wood_init=8000.0)
        state = init_carbon_state((3,), cfg)
        npt.assert_allclose(state.C_wood, 8000.0)

    def test_herbaceous_init_zeroes_wood_pool(self):
        """woody=False must start with NO wood, not the C_wood_init default."""
        cfg = _default_config(woody=False, C_wood_init=10000.0)
        state = init_carbon_state((3,), cfg)
        npt.assert_allclose(state.C_wood, 0.0)


# ===================================================================
# Integration with slab land
# ===================================================================

class TestSlabLandIntegration(unittest.TestCase):

    def test_slab_land_with_differland(self):
        """step_land with differland carbon returns non-zero co2_flux."""
        from legoesm.core.field import Field
        from legoesm.land.config import LandConfig
        from legoesm.land.state import LandState
        from legoesm.land.slab_land import step_land
        from legoesm.core.coupling_fields import AtmToSurface

        shape = (6, 4, 4)
        dims_2d = ("face", "x", "y")
        carbon_cfg = CarbonConfig(scheme="differland", epsilon=2.0)
        land_cfg = LandConfig(carbon=carbon_cfg)

        state = LandState(
            T_soil=Field(data=jnp.full(shape, 280.0), name="T", dims=dims_2d, units="K"),
            W_bucket=Field(data=jnp.full(shape, 75.0), name="W", dims=dims_2d, units="kg/m2"),
            snow_depth=Field(data=jnp.zeros(shape), name="snow", dims=dims_2d, units="kg/m2"),
            snow_age=Field(data=jnp.zeros(shape), name="age", dims=dims_2d, units="s"),
        )
        forcing = AtmToSurface(
            sw_down=jnp.full(shape, 300.0),
            lw_down=jnp.full(shape, 300.0),
            precip_total=jnp.full(shape, 3e-5),
            precip_snow=jnp.zeros(shape),
            T_lowest=jnp.full(shape, 280.0),
            q_lowest=jnp.full(shape, 0.005),
            u_lowest=jnp.full(shape, 3.0),
            v_lowest=jnp.zeros(shape),
            p_lowest=jnp.full(shape, 95000.0),
            p_surface=jnp.full(shape, 101325.0),
            rho_lowest=jnp.full(shape, 1.2),
            cos_zenith=jnp.full(shape, 0.5),
            co2_ppmv=jnp.full(shape, 400.0),
            has_radiation=jnp.ones(shape),
            has_precipitation=jnp.ones(shape),
        )
        carbon_state = init_carbon_state(shape, carbon_cfg)
        lat = jnp.full(shape, 0.7)

        new_state, resp, carbon_new = step_land(
            state, forcing, land_cfg, U_min=1.0, dt=600.0,
            lat=lat, carbon_state=carbon_state, doy=150.0,
        )

        self.assertIsNotNone(carbon_new)
        self.assertFalse(jnp.all(resp.co2_flux == 0),
                         "co2_flux should be non-zero with differland")
        self.assertTrue(jnp.all(jnp.isfinite(resp.co2_flux)))

    def test_slab_land_without_carbon(self):
        """Default config (carbon=none) returns zero co2_flux and None carbon."""
        from legoesm.core.field import Field
        from legoesm.land.config import LandConfig
        from legoesm.land.state import LandState
        from legoesm.land.slab_land import step_land
        from legoesm.core.coupling_fields import AtmToSurface

        shape = (6, 4, 4)
        dims_2d = ("face", "x", "y")
        land_cfg = LandConfig()

        state = LandState(
            T_soil=Field(data=jnp.full(shape, 280.0), name="T", dims=dims_2d, units="K"),
            W_bucket=Field(data=jnp.full(shape, 75.0), name="W", dims=dims_2d, units="kg/m2"),
            snow_depth=Field(data=jnp.zeros(shape), name="snow", dims=dims_2d, units="kg/m2"),
            snow_age=Field(data=jnp.zeros(shape), name="age", dims=dims_2d, units="s"),
        )
        forcing = AtmToSurface(
            sw_down=jnp.full(shape, 300.0),
            lw_down=jnp.full(shape, 300.0),
            precip_total=jnp.full(shape, 3e-5),
            precip_snow=jnp.zeros(shape),
            T_lowest=jnp.full(shape, 280.0),
            q_lowest=jnp.full(shape, 0.005),
            u_lowest=jnp.full(shape, 3.0),
            v_lowest=jnp.zeros(shape),
            p_lowest=jnp.full(shape, 95000.0),
            p_surface=jnp.full(shape, 101325.0),
            rho_lowest=jnp.full(shape, 1.2),
            cos_zenith=jnp.full(shape, 0.5),
            co2_ppmv=jnp.full(shape, 400.0),
            has_radiation=jnp.ones(shape),
            has_precipitation=jnp.ones(shape),
        )

        new_state, resp, carbon_new = step_land(
            state, forcing, land_cfg, U_min=1.0, dt=600.0,
        )
        self.assertIsNone(carbon_new)
        npt.assert_allclose(resp.co2_flux, 0.0, atol=1e-15)


# ===================================================================
# Differentiability
# ===================================================================

class TestDifferentiability(unittest.TestCase):

    def test_gpp_grad(self):
        """GPP should be differentiable w.r.t. T."""
        cfg = _default_config()

        def loss(T):
            return jnp.sum(compute_gpp(
                jnp.array([300.0]), T, jnp.array([3.0]),
                400.0, jnp.array([1.0]), cfg,
            ))

        grad = jax.grad(loss)(jnp.array([290.0]))
        self.assertTrue(jnp.all(jnp.isfinite(grad)))
        self.assertFalse(jnp.allclose(grad, 0.0))

    def test_step_grad(self):
        """Full carbon step should be differentiable w.r.t. radiation."""
        cfg = _default_config(scheme="differland")
        ncol = 2

        def loss(sw):
            state = _make_carbon_state(shape=(ncol,))
            new_state, flux = step_carbon_differland(
                state, sw, jnp.full(ncol, 290.0),
                jnp.full(ncol, 400.0), jnp.ones(ncol),
                jnp.full(ncol, 0.7), 150.0,
                jnp.full(ncol, 3e-5), cfg, 600.0,
            )
            return jnp.sum(flux)

        grad = jax.grad(loss)(jnp.full(ncol, 300.0))
        self.assertTrue(jnp.all(jnp.isfinite(grad)))
        # More SW -> more GPP -> more negative NEE -> negative gradient
        self.assertTrue(jnp.all(grad < 0),
                        f"Expected negative grad w.r.t. sw, got {grad}")

    def test_seasonal_grad(self):
        """Seasonal cycle should be differentiable w.r.t. latitude."""
        cfg = _default_config(scheme="seasonal")

        def loss(lat):
            return jnp.sum(seasonal_co2_flux(200.0, lat, cfg))

        grad = jax.grad(loss)(jnp.array([0.7]))
        self.assertTrue(jnp.all(jnp.isfinite(grad)))


if __name__ == "__main__":
    unittest.main()
