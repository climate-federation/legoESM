"""Tests for plant physiology: Farquhar photosynthesis and stomatal conductance."""

import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

from legoesm.land.stomata import (
    StomataConfig,
    ball_berry_gs,
    medlyn_gs,
    jarvis_gs,
    coupled_farquhar_stomata,
    solve_coupled_farquhar_ci,
    compute_stomatal_beta,
)
from legoesm.land.carbon.config import CarbonConfig, CarbonState
from legoesm.land.carbon.carbon_cycle import (
    step_carbon,
    step_carbon_differland,
    init_carbon_state,
)


class TestBallBerry(unittest.TestCase):
    """Ball-Berry stomatal conductance."""

    def setUp(self):
        self.cfg = StomataConfig()

    def test_minimum_conductance(self):
        """gs >= g0 always."""
        gs = ball_berry_gs(
            jnp.array(-5.0), jnp.array(0.8), jnp.array(400.0), self.cfg.g1_bb, self.cfg.g0)
        self.assertAlmostEqual(float(gs), self.cfg.g0, places=5)

    def test_increases_with_A(self):
        """gs increases with assimilation."""
        gs_low = ball_berry_gs(
            jnp.array(5.0), jnp.array(0.8), jnp.array(400.0), self.cfg.g1_bb, self.cfg.g0)
        gs_high = ball_berry_gs(
            jnp.array(20.0), jnp.array(0.8), jnp.array(400.0), self.cfg.g1_bb, self.cfg.g0)
        self.assertGreater(float(gs_high), float(gs_low))

    def test_increases_with_RH(self):
        """gs increases with relative humidity."""
        gs_dry = ball_berry_gs(
            jnp.array(10.0), jnp.array(0.3), jnp.array(400.0), self.cfg.g1_bb, self.cfg.g0)
        gs_wet = ball_berry_gs(
            jnp.array(10.0), jnp.array(0.9), jnp.array(400.0), self.cfg.g1_bb, self.cfg.g0)
        self.assertGreater(float(gs_wet), float(gs_dry))

    def test_decreases_with_Cs(self):
        """gs decreases with surface CO2."""
        gs_low_co2 = ball_berry_gs(
            jnp.array(10.0), jnp.array(0.8), jnp.array(200.0), self.cfg.g1_bb, self.cfg.g0)
        gs_high_co2 = ball_berry_gs(
            jnp.array(10.0), jnp.array(0.8), jnp.array(800.0), self.cfg.g1_bb, self.cfg.g0)
        self.assertGreater(float(gs_low_co2), float(gs_high_co2))


class TestMedlyn(unittest.TestCase):
    """Medlyn optimal stomatal conductance."""

    def setUp(self):
        self.cfg = StomataConfig()

    def test_minimum_conductance(self):
        """gs >= g0."""
        gs = medlyn_gs(
            jnp.array(-5.0), jnp.array(1.0), jnp.array(400.0), self.cfg.g1_med, self.cfg.g0)
        self.assertAlmostEqual(float(gs), self.cfg.g0, places=5)

    def test_increases_with_A(self):
        """gs increases with assimilation."""
        gs_low = medlyn_gs(
            jnp.array(5.0), jnp.array(1.0), jnp.array(400.0), self.cfg.g1_med, self.cfg.g0)
        gs_high = medlyn_gs(
            jnp.array(20.0), jnp.array(1.0), jnp.array(400.0), self.cfg.g1_med, self.cfg.g0)
        self.assertGreater(float(gs_high), float(gs_low))

    def test_decreases_with_VPD(self):
        """gs decreases with VPD."""
        gs_wet = medlyn_gs(
            jnp.array(10.0), jnp.array(0.5), jnp.array(400.0), self.cfg.g1_med, self.cfg.g0)
        gs_dry = medlyn_gs(
            jnp.array(10.0), jnp.array(3.0), jnp.array(400.0), self.cfg.g1_med, self.cfg.g0)
        self.assertGreater(float(gs_wet), float(gs_dry))

    def test_reasonable_magnitude(self):
        """gs in a reasonable range (g0 to ~0.5 mol/m2/s)."""
        gs = medlyn_gs(
            jnp.array(15.0), jnp.array(1.0), jnp.array(400.0), self.cfg.g1_med, self.cfg.g0)
        self.assertGreater(float(gs), self.cfg.g0)
        self.assertLess(float(gs), 1.0)


class TestJarvis(unittest.TestCase):
    """Jarvis multiplicative stomatal conductance."""

    def setUp(self):
        self.cfg = StomataConfig()
        self.T = jnp.array(298.15)
        self.sw = jnp.array(500.0)
        self.q = jnp.array(0.008)
        self.p = jnp.array(101325.0)

    def test_positive_under_good_conditions(self):
        """gs > 0 under favourable conditions."""
        gs = jarvis_gs(self.T, self.sw, self.q, self.p,
                       jnp.array(0.8), self.cfg)
        self.assertGreater(float(gs), 0.0)

    def test_decreases_with_dry_soil(self):
        """gs decreases when soil is dry."""
        gs_wet = jarvis_gs(self.T, self.sw, self.q, self.p,
                           jnp.array(1.0), self.cfg)
        gs_dry = jarvis_gs(self.T, self.sw, self.q, self.p,
                           jnp.array(0.1), self.cfg)
        self.assertGreater(float(gs_wet), float(gs_dry))

    def test_zero_in_dark(self):
        """gs ~ 0 in darkness (PAR response)."""
        gs = jarvis_gs(self.T, jnp.array(0.0), self.q, self.p,
                       jnp.array(0.8), self.cfg)
        self.assertLess(float(gs), 0.01)

    def test_temperature_sensitivity(self):
        """gs lower at extreme temperatures."""
        gs_opt = jarvis_gs(self.T, self.sw, self.q, self.p,
                           jnp.array(0.8), self.cfg)
        gs_cold = jarvis_gs(jnp.array(268.15), self.sw, self.q, self.p,
                            jnp.array(0.8), self.cfg)
        self.assertGreater(float(gs_opt), float(gs_cold))

    def test_batch_shape(self):
        """Works with batched inputs."""
        T = jnp.array([288.15, 298.15, 308.15])
        sw = jnp.array([200.0, 500.0, 800.0])
        q = jnp.full(3, 0.008)
        p = jnp.full(3, 101325.0)
        beta = jnp.full(3, 0.8)
        gs = jarvis_gs(T, sw, q, p, beta, self.cfg)
        self.assertEqual(gs.shape, (3,))

    def test_VPD_uses_specific_humidity_form(self):
        """``e_air = q · p / (ε + (1−ε)·q)`` is the correct form when
        q is specific humidity (iter-22 fix).  The earlier
        mixing-ratio form ``q · p / (ε + q)`` biases e_air upward by
        ~1.5 % at q = 0.02 — biasing VPD downward and falsely
        *opening* stomata.

        Regression strategy: pin the live ``jarvis_gs`` output at
        precise inputs against the expected value derived from the
        corrected formula.  Then assert the same inputs would have
        produced a *different* (larger) ``gs`` under the buggy
        mixing-ratio formula — proving the test would catch a
        regression.
        """
        from legoesm import constants
        from legoesm.thermo import saturation_vapor_pressure
        # Pick inputs that put the VPD response solidly in its linear
        # regime (not clipped at f_VPD_min=0.01 nor saturated at 1):
        #   T = 305 K  →  e_sat ≈ 4720 Pa
        #   q = 0.012  →  e_air ≈ 1907 Pa (corrected) / 1928 Pa (buggy)
        #   VPD ≈ 28-29 hPa, with cfg.a_vpd = 0.05/hPa → f_VPD ≈ -0.4
        #   clipped at f_VPD_min so the diff is observable in the
        #   linear region just above the clip threshold.
        # Use cfg with a_vpd small enough to stay above the clip.
        cfg = StomataConfig(a_vpd=0.01, f_VPD_min=0.001)
        q = 0.012
        p = 101325.0
        T = 305.0
        eps = constants.epsilon

        # CORRECTED formula (specific-humidity)
        e_corrected = q * p / (eps + (1.0 - eps) * q)
        # BUGGY formula (mixing-ratio applied to specific humidity)
        e_buggy = q * p / (eps + q)
        # The corrected denom is smaller (because (1−ε)·q < q for q>0
        # and ε<1), so e_corrected > e_buggy.  Therefore the buggy
        # version under-reports e_air, OVER-reports VPD, and
        # UNDER-reports f_VPD → smaller gs.
        assert e_corrected > e_buggy

        # Compute expected gs analytically from the corrected formula
        e_sat = float(saturation_vapor_pressure(jnp.array(T)))
        VPD_corrected_hPa = max(e_sat - e_corrected, 0.0) / 100.0
        VPD_buggy_hPa = max(e_sat - e_buggy, 0.0) / 100.0
        f_VPD_corrected = max(min(1.0 - cfg.a_vpd * VPD_corrected_hPa, 1.0), cfg.f_VPD_min)
        f_VPD_buggy = max(min(1.0 - cfg.a_vpd * VPD_buggy_hPa, 1.0), cfg.f_VPD_min)
        # f_PAR and f_T at saturating PAR and near-optimal T should
        # be ≈ 1 each, but compute them exactly via the same formulas
        # used in jarvis_gs to keep the comparison bit-faithful.
        T_C = T - constants.T_freeze
        dT_norm = (T_C - cfg.T_opt_jarvis_C) / cfg.T_range_jarvis_C
        f_T = max(1.0 - dT_norm ** 2, 0.0)
        PAR_FRAC = 0.48  # _PAR_FRAC in stomata.py
        PAR = PAR_FRAC * 2000.0
        f_PAR = PAR / (PAR + cfg.K_PAR + 1e-10)
        f_soil = 1.0  # beta = 1
        gs_expected_corrected = cfg.gs_max * f_PAR * f_T * f_VPD_corrected * f_soil
        gs_expected_buggy = cfg.gs_max * f_PAR * f_T * f_VPD_buggy * f_soil

        # Live code under the corrected formula
        gs_live = float(jarvis_gs(
            jnp.array(T), jnp.array(2000.0), jnp.array(q),
            jnp.array(p), jnp.array(1.0), cfg,
        ))

        # Live code MUST agree with the corrected expected value
        # (within float tolerance), and MUST disagree with the buggy
        # expected value by more than that tolerance — otherwise this
        # test is vacuous.
        self.assertAlmostEqual(gs_live, gs_expected_corrected, places=8)
        self.assertNotAlmostEqual(
            gs_live, gs_expected_buggy, places=6,
            msg="Live gs matched the buggy mixing-ratio formula — "
            "this test would not catch the iter-22 regression.",
        )


class TestCoupledFarquharStomata(unittest.TestCase):
    """Coupled Farquhar-stomata solver."""

    def _make_inputs(self, n=1):
        T = jnp.full(n, 298.15)
        sw = jnp.full(n, 500.0)
        co2 = 400.0
        q = jnp.full(n, 0.008)
        p = jnp.full(n, 101325.0)
        LAI = jnp.full(n, 3.0)
        beta = jnp.full(n, 0.8)
        return T, sw, co2, q, p, LAI, beta

    def _s(self, x):
        """Squeeze to scalar for float() conversion."""
        return float(jnp.squeeze(x))

    def test_ball_berry_positive(self):
        """Coupled solver returns positive gs and gpp (Ball-Berry)."""
        cfg = StomataConfig(enabled=True, stomata_model="ball_berry")
        T, sw, co2, q, p, LAI, beta = self._make_inputs()
        gs, gpp = coupled_farquhar_stomata(T, sw, co2, q, p, LAI, beta, cfg)
        self.assertGreater(self._s(gs), cfg.g0)
        self.assertGreater(self._s(gpp), 0.0)

    def test_medlyn_positive(self):
        """Coupled solver returns positive gs and gpp (Medlyn)."""
        cfg = StomataConfig(enabled=True, stomata_model="medlyn")
        T, sw, co2, q, p, LAI, beta = self._make_inputs()
        gs, gpp = coupled_farquhar_stomata(T, sw, co2, q, p, LAI, beta, cfg)
        self.assertGreater(self._s(gs), cfg.g0)
        self.assertGreater(self._s(gpp), 0.0)

    def test_gpp_in_reasonable_range(self):
        """GPP in a reasonable range (0-30 gC/m2/day)."""
        cfg = StomataConfig(enabled=True)
        T, sw, co2, q, p, LAI, beta = self._make_inputs()
        _, gpp = coupled_farquhar_stomata(T, sw, co2, q, p, LAI, beta, cfg)
        gpp_day = self._s(gpp) * 86400.0
        self.assertGreater(gpp_day, 0.0)
        self.assertLess(gpp_day, 30.0)

    def test_drought_reduces_gpp(self):
        """Soil moisture stress reduces GPP."""
        cfg = StomataConfig(enabled=True)
        T, sw, co2, q, p, LAI, _ = self._make_inputs()
        _, gpp_wet = coupled_farquhar_stomata(
            T, sw, co2, q, p, LAI, jnp.array([1.0]), cfg)
        _, gpp_dry = coupled_farquhar_stomata(
            T, sw, co2, q, p, LAI, jnp.array([0.1]), cfg)
        self.assertGreater(self._s(gpp_wet), self._s(gpp_dry))

    def test_co2_fertilization(self):
        """Higher CO2 increases GPP."""
        cfg = StomataConfig(enabled=True)
        T, sw, _, q, p, LAI, beta = self._make_inputs()
        _, gpp_low = coupled_farquhar_stomata(
            T, sw, 280.0, q, p, LAI, beta, cfg)
        _, gpp_high = coupled_farquhar_stomata(
            T, sw, 600.0, q, p, LAI, beta, cfg)
        self.assertGreater(self._s(gpp_high), self._s(gpp_low))

    def test_batch(self):
        """Works with batched inputs."""
        cfg = StomataConfig(enabled=True)
        T, sw, co2, q, p, LAI, beta = self._make_inputs(n=5)
        gs, gpp = coupled_farquhar_stomata(T, sw, co2, q, p, LAI, beta, cfg)
        self.assertEqual(gs.shape, (5,))
        self.assertEqual(gpp.shape, (5,))

    def test_dark_gives_minimum_gs(self):
        """In darkness, gs ~ g0 (minimum conductance)."""
        cfg = StomataConfig(enabled=True)
        T, _, co2, q, p, LAI, beta = self._make_inputs()
        gs, _ = coupled_farquhar_stomata(
            T, jnp.array([0.0]), co2, q, p, LAI, beta, cfg)
        self.assertAlmostEqual(self._s(gs), cfg.g0, places=2)


class TestStomatalBeta(unittest.TestCase):
    """Beta computation from stomatal conductance."""

    def setUp(self):
        self.cfg = StomataConfig(gs_ref=0.3)

    def test_full_opening_gives_high_beta(self):
        """gs = gs_ref gives beta ~ 1 for full canopy."""
        beta = compute_stomatal_beta(
            jnp.array(0.3), jnp.array(5.0), jnp.array(0.8), self.cfg)
        self.assertGreater(float(beta), 0.9)

    def test_closed_stomata_gives_low_beta(self):
        """Very low gs gives low beta."""
        beta = compute_stomatal_beta(
            jnp.array(0.01), jnp.array(5.0), jnp.array(0.8), self.cfg)
        self.assertLess(float(beta), 0.3)

    def test_no_LAI_uses_minimum(self):
        """Without LAI, beta = min(beta_soil, beta_canopy)."""
        # Stomata more limiting
        beta = compute_stomatal_beta(
            jnp.array(0.03), None, jnp.array(0.8), self.cfg)
        self.assertAlmostEqual(float(beta), 0.1, places=1)

    def test_bounded_0_1(self):
        """Beta is always in [0, 1]."""
        beta = compute_stomatal_beta(
            jnp.array(0.5), jnp.array(3.0), jnp.array(0.5), self.cfg)
        self.assertGreaterEqual(float(beta), 0.0)
        self.assertLessEqual(float(beta), 1.0)

    def test_bare_soil_dominates_low_LAI(self):
        """At low LAI, beta is close to beta_soil."""
        beta = compute_stomatal_beta(
            jnp.array(0.1), jnp.array(0.1), jnp.array(0.8), self.cfg)
        # f_canopy ~ 1-exp(-0.5*0.1) ~ 0.05, so mostly soil
        self.assertAlmostEqual(float(beta), 0.8, delta=0.1)


class TestGPPOverride(unittest.TestCase):
    """Farquhar GPP replaces LUE GPP in carbon cycle."""

    def _make_args(self):
        shape = (4,)
        return dict(
            sw_down=jnp.full(shape, 300.0),
            T=jnp.full(shape, 290.0),
            co2_ppmv=jnp.full(shape, 400.0),
            beta=jnp.full(shape, 0.5),
            lat=jnp.full(shape, 0.7),
            doy=180.0,
            precip=jnp.full(shape, 3e-5),
            dt=3600.0,
        )

    def test_gpp_override_changes_result(self):
        """Passing gpp_override changes the carbon cycle output."""
        cfg = CarbonConfig(scheme="differland")
        state = init_carbon_state((4,), cfg)
        args = self._make_args()

        # Without override
        state1, flux1 = step_carbon_differland(
            state, config=cfg, **args)

        # With override (double the GPP)
        gpp_override = jnp.full((4,), 1e-4)  # large GPP
        state2, flux2 = step_carbon_differland(
            state, config=cfg, gpp_override=gpp_override, **args)

        # Higher GPP -> more carbon uptake -> more negative NEE (co2_flux)
        self.assertLess(float(jnp.mean(flux2)), float(jnp.mean(flux1)))

    def test_step_carbon_passes_override(self):
        """step_carbon correctly passes gpp_override through."""
        cfg = CarbonConfig(scheme="differland")
        state = init_carbon_state((4,), cfg)
        args = self._make_args()

        gpp_override = jnp.full((4,), 5e-5)
        state_out, flux_out = step_carbon(
            state, config=cfg, gpp_override=gpp_override, **args)
        self.assertIsNotNone(state_out)
        self.assertTrue(jnp.all(jnp.isfinite(flux_out)))


class TestSlabLandIntegration(unittest.TestCase):
    """Integration of stomatal conductance with slab land model."""

    def _make_forcing(self, shape):
        from legoesm.core.coupling_fields import AtmToSurface
        return AtmToSurface(
            sw_down=jnp.full(shape, 300.0),
            lw_down=jnp.full(shape, 350.0),
            precip_total=jnp.full(shape, 3e-5),
            precip_snow=jnp.zeros(shape),
            T_lowest=jnp.full(shape, 290.0),
            q_lowest=jnp.full(shape, 0.008),
            u_lowest=jnp.full(shape, 3.0),
            v_lowest=jnp.full(shape, 2.0),
            p_surface=jnp.full(shape, 101325.0),
            rho_lowest=jnp.full(shape, 1.2),
            cos_zenith=jnp.full(shape, 0.6),
            p_lowest=jnp.full(shape, 100000.0),
            co2_ppmv=jnp.full(shape, 400.0),
            has_radiation=True,
            has_precipitation=True,
        )

    def _make_state(self, shape):
        from legoesm.core.field import Field
        from legoesm.land.state import LandState
        dims = ("x",) if len(shape) == 1 else ("face", "x", "y")
        return LandState(
            T_soil=Field(data=jnp.full(shape, 290.0),
                         name="T_soil", dims=dims, units="K"),
            W_bucket=Field(data=jnp.full(shape, 100.0),
                           name="W_bucket", dims=dims, units="kg/m2"),
            snow_depth=Field(data=jnp.zeros(shape),
                             name="snow_depth", dims=dims, units="kg/m2"),
            snow_age=Field(data=jnp.zeros(shape),
                           name="snow_age", dims=dims, units="s"),
        )

    def test_backward_compat_disabled(self):
        """When stomata disabled, result identical to original."""
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig

        shape = (4,)
        cfg = LandConfig()  # stomata disabled by default
        state = self._make_state(shape)
        forcing = self._make_forcing(shape)

        new_state, resp, _ = step_land(state, forcing, cfg, 1.0, 3600.0)
        self.assertTrue(jnp.all(jnp.isfinite(resp.lhflx)))
        self.assertTrue(jnp.all(jnp.isfinite(resp.shflx)))

    def test_jarvis_changes_ET(self):
        """Jarvis model (no carbon) changes latent heat flux."""
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig
        from legoesm.land.surface_scheme import SimpleSEBConfig   # stomata is the bulk-flux option
        from legoesm.land.stomata import StomataConfig

        shape = (4,)
        cfg_off = LandConfig(surface_scheme=SimpleSEBConfig())
        cfg_jarvis = LandConfig(surface_scheme=SimpleSEBConfig(), stomata=StomataConfig(enabled=True))
        state = self._make_state(shape)
        forcing = self._make_forcing(shape)

        _, resp_off, _ = step_land(state, forcing, cfg_off, 1.0, 3600.0)
        _, resp_jarvis, _ = step_land(state, forcing, cfg_jarvis, 1.0, 3600.0)

        # Jarvis should give different (typically lower) LH
        self.assertFalse(
            jnp.allclose(resp_off.lhflx, resp_jarvis.lhflx, atol=1e-3))

    def test_farquhar_with_carbon(self):
        """Farquhar+BB stomata work with active carbon cycle."""
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig
        from legoesm.land.stomata import StomataConfig
        from legoesm.land.carbon.config import CarbonConfig
        from legoesm.land.carbon.carbon_cycle import init_carbon_state

        shape = (4,)
        cfg = LandConfig(
            carbon=CarbonConfig(scheme="differland"),
            stomata=StomataConfig(enabled=True, stomata_model="ball_berry"),
        )
        state = self._make_state(shape)
        forcing = self._make_forcing(shape)
        carbon = init_carbon_state(shape, cfg.carbon)
        lat = jnp.full(shape, 0.7)

        new_state, resp, carbon_new = step_land(
            state, forcing, cfg, 1.0, 3600.0,
            lat=lat, carbon_state=carbon, doy=180.0)

        self.assertTrue(jnp.all(jnp.isfinite(resp.lhflx)))
        self.assertTrue(jnp.all(jnp.isfinite(resp.co2_flux)))
        self.assertIsNotNone(carbon_new)

    def test_farquhar_medlyn(self):
        """Farquhar+Medlyn stomata give different results from Ball-Berry."""
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig
        from legoesm.land.surface_scheme import SimpleSEBConfig   # stomata is the bulk-flux option
        from legoesm.land.stomata import StomataConfig
        from legoesm.land.carbon.config import CarbonConfig
        from legoesm.land.carbon.carbon_cycle import init_carbon_state

        shape = (4,)
        cfg_bb = LandConfig(
            surface_scheme=SimpleSEBConfig(),
            carbon=CarbonConfig(scheme="differland"),
            stomata=StomataConfig(enabled=True, stomata_model="ball_berry"),
        )
        cfg_med = LandConfig(
            surface_scheme=SimpleSEBConfig(),
            carbon=CarbonConfig(scheme="differland"),
            stomata=StomataConfig(enabled=True, stomata_model="medlyn"),
        )
        state = self._make_state(shape)
        forcing = self._make_forcing(shape)
        carbon = init_carbon_state(shape, cfg_bb.carbon)
        lat = jnp.full(shape, 0.7)

        _, resp_bb, _ = step_land(
            state, forcing, cfg_bb, 1.0, 3600.0,
            lat=lat, carbon_state=carbon, doy=180.0)
        _, resp_med, _ = step_land(
            state, forcing, cfg_med, 1.0, 3600.0,
            lat=lat, carbon_state=carbon, doy=180.0)

        # They should produce different fluxes
        self.assertFalse(
            jnp.allclose(resp_bb.lhflx, resp_med.lhflx, atol=1e-6))


class TestDifferentiability(unittest.TestCase):
    """JAX differentiability of stomatal models."""

    def test_ball_berry_differentiable(self):
        """Ball-Berry is differentiable w.r.t. A."""
        cfg = StomataConfig()

        def f(A):
            return jnp.sum(ball_berry_gs(
                A, jnp.array(0.8), jnp.array(400.0), cfg.g1_bb, cfg.g0))

        grad = jax.grad(f)(jnp.array(10.0))
        self.assertTrue(jnp.isfinite(grad))
        self.assertGreater(float(grad), 0.0)

    def test_coupled_solver_differentiable(self):
        """Coupled Farquhar-stomata solver is differentiable w.r.t. T."""
        cfg = StomataConfig(enabled=True, n_iter_ags=3)

        def f(T):
            gs, gpp = coupled_farquhar_stomata(
                T, jnp.array([500.0]), 400.0,
                jnp.array([0.008]), jnp.array([101325.0]),
                jnp.array([3.0]), jnp.array([0.8]), cfg)
            return jnp.sum(gpp)

        grad = jax.grad(f)(jnp.array([298.15]))
        self.assertTrue(jnp.all(jnp.isfinite(grad)))

    def test_jarvis_differentiable(self):
        """Jarvis model is differentiable w.r.t. T."""
        cfg = StomataConfig()

        def f(T):
            return jnp.sum(jarvis_gs(
                T, jnp.array(500.0), jnp.array(0.008),
                jnp.array(101325.0), jnp.array(0.8), cfg))

        grad = jax.grad(f)(jnp.array(298.15))
        self.assertTrue(jnp.isfinite(grad))


class TestCanopyLAIScaling(unittest.TestCase):
    """Big-leaf GPP scales with the canopy integral L_c = (1-exp(-k*LAI))/k,
    now folded into the effective Vcmax25 inside solve_coupled_farquhar_ci."""

    def _gpp(self, LAI):
        cfg = StomataConfig(enabled=True, stomata_model="ball_berry")
        st = solve_coupled_farquhar_ci(
            jnp.array([298.15]), jnp.array([600.0]), 400.0,
            jnp.array([0.008]), jnp.array([101325.0]),
            jnp.array([LAI]), jnp.array([1.0]), cfg)
        return float(jnp.squeeze(st.gpp))

    def test_gpp_increases_with_LAI(self):
        self.assertGreater(self._gpp(3.0), self._gpp(0.5))

    def test_gpp_saturates_at_high_LAI(self):
        # L_c -> 1/k as LAI -> inf, so the 5->8 GPP increment is much smaller
        # than the 0.5->3 one.
        low = self._gpp(3.0) - self._gpp(0.5)
        high = self._gpp(8.0) - self._gpp(5.0)
        self.assertLess(high, low)

    def test_leafless_column_assimilates_little(self):
        # L_c -> LAI -> 0 as LAI -> 0, so GPP -> ~0.
        self.assertLess(self._gpp(0.01), self._gpp(2.0))


if __name__ == "__main__":
    unittest.main()
