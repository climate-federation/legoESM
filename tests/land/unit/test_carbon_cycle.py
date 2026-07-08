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

import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from legoesm.land.carbon.config import (
    CarbonConfig,
    CarbonDiagnostics,
    CarbonState,
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
    _temperate_modifier,
    _effective_rate,
)


# ===================================================================
# Helpers
# ===================================================================

def _default_config(**overrides):
    return CarbonConfig(**overrides)


def _make_carbon_state(shape=(4,), **overrides):
    defaults = dict(
        C_lab=jnp.full(shape, 100.0),
        C_fol=jnp.full(shape, 200.0),
        C_root=jnp.full(shape, 300.0),
        C_wood=jnp.full(shape, 10000.0),
        C_lit=jnp.full(shape, 100.0),
        C_som=jnp.full(shape, 10000.0),
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
        self.assertEqual(len(state), 6)
        self.assertEqual(state.C_lab.shape, (4,))


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
            C_som=jnp.full((ncol,), 12000.0),
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
            C_som=jnp.full((ncol,), 12000.0),
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
        npt.assert_allclose(state.C_som, 5000.0)

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
