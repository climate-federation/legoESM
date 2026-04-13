"""Tests for stomatal conductance models and coupled Leuning A-gs solver.

Covers ``legoesm.land.canopy.stomatal`` and the ``StomataConfig`` NamedTuple.
The Leuning Farquhar model itself is tested in ``test_canopy_photosynthesis.py``;
this file focuses on the stomatal models (Ball-Berry, Medlyn, Jarvis), the
coupled solver for the SimpleSEB path, and integration with the slab / multi-
layer land models via ``compute_effective_beta``.
"""

import unittest

import jax
import jax.numpy as jnp

from legoesm.land.canopy.stomatal import (
    ball_berry_gs,
    medlyn_gs,
    jarvis_gs,
    coupled_farquhar_stomata,
    compute_stomatal_beta,
)
from legoesm.land.carbon.config import CarbonConfig, CarbonState, StomataConfig
from legoesm.land.carbon.carbon_cycle import (
    step_carbon,
    step_carbon_differland,
    init_carbon_state,
)


class TestBallBerry(unittest.TestCase):
    """Ball-Berry stomatal conductance — explicit m / b0 signature."""

    def setUp(self):
        self.cfg = StomataConfig()
        # Convenience aliases — both passed explicitly, not via config.
        self.m = self.cfg.g1_bb
        self.b0 = self.cfg.g0

    def test_minimum_conductance(self):
        """gs >= b0 always."""
        gs = ball_berry_gs(
            jnp.array(-5.0), jnp.array(0.8), jnp.array(400.0),
            self.m, self.b0)
        self.assertAlmostEqual(float(gs), self.b0, places=5)

    def test_increases_with_A(self):
        gs_low = ball_berry_gs(
            jnp.array(5.0), jnp.array(0.8), jnp.array(400.0),
            self.m, self.b0)
        gs_high = ball_berry_gs(
            jnp.array(20.0), jnp.array(0.8), jnp.array(400.0),
            self.m, self.b0)
        self.assertGreater(float(gs_high), float(gs_low))

    def test_increases_with_RH(self):
        gs_dry = ball_berry_gs(
            jnp.array(10.0), jnp.array(0.3), jnp.array(400.0),
            self.m, self.b0)
        gs_wet = ball_berry_gs(
            jnp.array(10.0), jnp.array(0.9), jnp.array(400.0),
            self.m, self.b0)
        self.assertGreater(float(gs_wet), float(gs_dry))

    def test_decreases_with_Cs(self):
        gs_low_co2 = ball_berry_gs(
            jnp.array(10.0), jnp.array(0.8), jnp.array(200.0),
            self.m, self.b0)
        gs_high_co2 = ball_berry_gs(
            jnp.array(10.0), jnp.array(0.8), jnp.array(800.0),
            self.m, self.b0)
        self.assertGreater(float(gs_low_co2), float(gs_high_co2))

    def test_per_column_m_b0(self):
        """Ball-Berry accepts per-column m / b0 arrays (needed by canopy)."""
        An = jnp.array([5.0, 10.0, 15.0])
        RH = jnp.full(3, 0.8)
        Cs = jnp.full(3, 400.0)
        m = jnp.array([9.0, 7.0, 4.0])
        b0 = jnp.array([0.01, 0.02, 0.04])
        gs = ball_berry_gs(An, RH, Cs, m, b0)
        self.assertEqual(gs.shape, (3,))
        self.assertTrue(jnp.all(gs >= b0 - 1e-12))


class TestMedlyn(unittest.TestCase):
    """Medlyn optimal stomatal conductance — explicit g1 / g0 signature."""

    def setUp(self):
        self.cfg = StomataConfig()
        self.g1 = self.cfg.g1_med
        self.g0 = self.cfg.g0

    def test_minimum_conductance(self):
        gs = medlyn_gs(
            jnp.array(-5.0), jnp.array(1.0), jnp.array(400.0),
            self.g1, self.g0)
        self.assertAlmostEqual(float(gs), self.g0, places=5)

    def test_increases_with_A(self):
        gs_low = medlyn_gs(
            jnp.array(5.0), jnp.array(1.0), jnp.array(400.0),
            self.g1, self.g0)
        gs_high = medlyn_gs(
            jnp.array(20.0), jnp.array(1.0), jnp.array(400.0),
            self.g1, self.g0)
        self.assertGreater(float(gs_high), float(gs_low))

    def test_decreases_with_VPD(self):
        gs_wet = medlyn_gs(
            jnp.array(10.0), jnp.array(0.5), jnp.array(400.0),
            self.g1, self.g0)
        gs_dry = medlyn_gs(
            jnp.array(10.0), jnp.array(3.0), jnp.array(400.0),
            self.g1, self.g0)
        self.assertGreater(float(gs_wet), float(gs_dry))

    def test_reasonable_magnitude(self):
        gs = medlyn_gs(
            jnp.array(15.0), jnp.array(1.0), jnp.array(400.0),
            self.g1, self.g0)
        self.assertGreater(float(gs), self.g0)
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
        gs = jarvis_gs(self.T, self.sw, self.q, self.p,
                       jnp.array(0.8), self.cfg)
        self.assertGreater(float(gs), 0.0)

    def test_decreases_with_dry_soil(self):
        gs_wet = jarvis_gs(self.T, self.sw, self.q, self.p,
                           jnp.array(1.0), self.cfg)
        gs_dry = jarvis_gs(self.T, self.sw, self.q, self.p,
                           jnp.array(0.1), self.cfg)
        self.assertGreater(float(gs_wet), float(gs_dry))

    def test_zero_in_dark(self):
        gs = jarvis_gs(self.T, jnp.array(0.0), self.q, self.p,
                       jnp.array(0.8), self.cfg)
        self.assertLess(float(gs), 0.01)

    def test_temperature_sensitivity(self):
        gs_opt = jarvis_gs(self.T, self.sw, self.q, self.p,
                           jnp.array(0.8), self.cfg)
        gs_cold = jarvis_gs(jnp.array(268.15), self.sw, self.q, self.p,
                            jnp.array(0.8), self.cfg)
        self.assertGreater(float(gs_opt), float(gs_cold))

    def test_batch_shape(self):
        T = jnp.array([288.15, 298.15, 308.15])
        sw = jnp.array([200.0, 500.0, 800.0])
        q = jnp.full(3, 0.008)
        p = jnp.full(3, 101325.0)
        beta = jnp.full(3, 0.8)
        gs = jarvis_gs(T, sw, q, p, beta, self.cfg)
        self.assertEqual(gs.shape, (3,))


class TestCoupledLeuningStomata(unittest.TestCase):
    """Coupled Leuning Farquhar + Ball-Berry/Medlyn solver."""

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
        return float(jnp.squeeze(x))

    def test_ball_berry_positive(self):
        cfg = StomataConfig(enabled=True, stomata_model="ball_berry")
        T, sw, co2, q, p, LAI, beta = self._make_inputs()
        gs, gpp = coupled_farquhar_stomata(T, sw, co2, q, p, LAI, beta, cfg)
        self.assertGreater(self._s(gs), cfg.g0)
        self.assertGreater(self._s(gpp), 0.0)

    def test_medlyn_positive(self):
        cfg = StomataConfig(enabled=True, stomata_model="medlyn")
        T, sw, co2, q, p, LAI, beta = self._make_inputs()
        gs, gpp = coupled_farquhar_stomata(T, sw, co2, q, p, LAI, beta, cfg)
        self.assertGreater(self._s(gs), cfg.g0)
        self.assertGreater(self._s(gpp), 0.0)

    def test_gpp_in_reasonable_range(self):
        cfg = StomataConfig(enabled=True)
        T, sw, co2, q, p, LAI, beta = self._make_inputs()
        _, gpp = coupled_farquhar_stomata(T, sw, co2, q, p, LAI, beta, cfg)
        gpp_day = self._s(gpp) * 86400.0
        self.assertGreater(gpp_day, 0.0)
        self.assertLess(gpp_day, 30.0)

    def test_drought_reduces_gpp(self):
        cfg = StomataConfig(enabled=True)
        T, sw, co2, q, p, LAI, _ = self._make_inputs()
        _, gpp_wet = coupled_farquhar_stomata(
            T, sw, co2, q, p, LAI, jnp.array([1.0]), cfg)
        _, gpp_dry = coupled_farquhar_stomata(
            T, sw, co2, q, p, LAI, jnp.array([0.1]), cfg)
        self.assertGreater(self._s(gpp_wet), self._s(gpp_dry))

    def test_co2_fertilization(self):
        cfg = StomataConfig(enabled=True)
        T, sw, _, q, p, LAI, beta = self._make_inputs()
        _, gpp_low = coupled_farquhar_stomata(
            T, sw, 280.0, q, p, LAI, beta, cfg)
        _, gpp_high = coupled_farquhar_stomata(
            T, sw, 600.0, q, p, LAI, beta, cfg)
        self.assertGreater(self._s(gpp_high), self._s(gpp_low))

    def test_batch(self):
        cfg = StomataConfig(enabled=True)
        T, sw, co2, q, p, LAI, beta = self._make_inputs(n=5)
        gs, gpp = coupled_farquhar_stomata(T, sw, co2, q, p, LAI, beta, cfg)
        self.assertEqual(gs.shape, (5,))
        self.assertEqual(gpp.shape, (5,))

    def test_dark_gives_minimum_gs(self):
        cfg = StomataConfig(enabled=True)
        T, _, co2, q, p, LAI, beta = self._make_inputs()
        gs, _ = coupled_farquhar_stomata(
            T, jnp.array([0.0]), co2, q, p, LAI, beta, cfg)
        self.assertAlmostEqual(self._s(gs), cfg.g0, places=2)

    def test_c4_fraction_changes_gpp(self):
        """Mixed-PFT C4 fraction is a continuous weighted average."""
        T, sw, co2, q, p, LAI, beta = self._make_inputs()
        cfg_c3 = StomataConfig(enabled=True, fC4=0.0)
        cfg_c4 = StomataConfig(enabled=True, fC4=1.0)
        _, gpp_c3 = coupled_farquhar_stomata(T, sw, co2, q, p, LAI, beta, cfg_c3)
        _, gpp_c4 = coupled_farquhar_stomata(T, sw, co2, q, p, LAI, beta, cfg_c4)
        # C3 and C4 GPPs differ at 25 C, 400 ppm, 500 W/m2.
        self.assertFalse(jnp.allclose(gpp_c3, gpp_c4, atol=1e-8))


class TestStomatalBeta(unittest.TestCase):
    """Beta computation from stomatal conductance."""

    def setUp(self):
        self.cfg = StomataConfig(gs_ref=0.3)

    def test_full_opening_gives_high_beta(self):
        beta = compute_stomatal_beta(
            jnp.array(0.3), jnp.array(5.0), jnp.array(0.8), self.cfg)
        self.assertGreater(float(beta), 0.9)

    def test_closed_stomata_gives_low_beta(self):
        beta = compute_stomatal_beta(
            jnp.array(0.01), jnp.array(5.0), jnp.array(0.8), self.cfg)
        self.assertLess(float(beta), 0.3)

    def test_no_LAI_uses_minimum(self):
        beta = compute_stomatal_beta(
            jnp.array(0.03), None, jnp.array(0.8), self.cfg)
        self.assertAlmostEqual(float(beta), 0.1, places=1)

    def test_bounded_0_1(self):
        beta = compute_stomatal_beta(
            jnp.array(0.5), jnp.array(3.0), jnp.array(0.5), self.cfg)
        self.assertGreaterEqual(float(beta), 0.0)
        self.assertLessEqual(float(beta), 1.0)

    def test_bare_soil_dominates_low_LAI(self):
        beta = compute_stomatal_beta(
            jnp.array(0.1), jnp.array(0.1), jnp.array(0.8), self.cfg)
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
        cfg = CarbonConfig(scheme="differland")
        state = init_carbon_state((4,), cfg)
        args = self._make_args()

        state1, flux1 = step_carbon_differland(
            state, config=cfg, **args)

        gpp_override = jnp.full((4,), 1e-4)
        state2, flux2 = step_carbon_differland(
            state, config=cfg, gpp_override=gpp_override, **args)

        # Higher GPP -> more carbon uptake -> more negative NEE (co2_flux).
        self.assertLess(float(jnp.mean(flux2)), float(jnp.mean(flux1)))

    def test_step_carbon_passes_override(self):
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
        from legoesm.coupler.coupling_fields import AtmToSurface
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
        """When stomata disabled, step_land still runs."""
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

        shape = (4,)
        cfg_off = LandConfig()
        cfg_jarvis = LandConfig(stomata=StomataConfig(enabled=True))
        state = self._make_state(shape)
        forcing = self._make_forcing(shape)

        _, resp_off, _ = step_land(state, forcing, cfg_off, 1.0, 3600.0)
        _, resp_jarvis, _ = step_land(state, forcing, cfg_jarvis, 1.0, 3600.0)

        self.assertFalse(
            jnp.allclose(resp_off.lhflx, resp_jarvis.lhflx, atol=1e-3))

    def test_leuning_farquhar_with_carbon(self):
        """Leuning Farquhar + BB stomata work with an active carbon cycle."""
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig

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

    def test_leuning_medlyn_differs_from_ball_berry(self):
        """Leuning + Medlyn stomata give different results from Ball-Berry."""
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig

        shape = (4,)
        cfg_bb = LandConfig(
            carbon=CarbonConfig(scheme="differland"),
            stomata=StomataConfig(enabled=True, stomata_model="ball_berry"),
        )
        cfg_med = LandConfig(
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

        self.assertFalse(
            jnp.allclose(resp_bb.lhflx, resp_med.lhflx, atol=1e-6))


class TestDifferentiability(unittest.TestCase):
    """JAX differentiability of stomatal models."""

    def test_ball_berry_differentiable(self):
        cfg = StomataConfig()

        def f(A):
            return jnp.sum(ball_berry_gs(
                A, jnp.array(0.8), jnp.array(400.0),
                cfg.g1_bb, cfg.g0))

        grad = jax.grad(f)(jnp.array(10.0))
        self.assertTrue(jnp.isfinite(grad))
        self.assertGreater(float(grad), 0.0)

    def test_medlyn_differentiable(self):
        cfg = StomataConfig()

        def f(A):
            return jnp.sum(medlyn_gs(
                A, jnp.array(1.0), jnp.array(400.0),
                cfg.g1_med, cfg.g0))

        grad = jax.grad(f)(jnp.array(10.0))
        self.assertTrue(jnp.isfinite(grad))
        self.assertGreater(float(grad), 0.0)

    def test_coupled_solver_differentiable(self):
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
        cfg = StomataConfig()

        def f(T):
            return jnp.sum(jarvis_gs(
                T, jnp.array(500.0), jnp.array(0.008),
                jnp.array(101325.0), jnp.array(0.8), cfg))

        grad = jax.grad(f)(jnp.array(298.15))
        self.assertTrue(jnp.isfinite(grad))


if __name__ == "__main__":
    unittest.main()
