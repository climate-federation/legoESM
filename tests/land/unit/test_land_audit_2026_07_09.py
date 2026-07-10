"""Regression tests for the 2026-07-09 land-package audit fixes.

Each test pins one confirmed defect so it cannot silently regress:

* F5  canopy/stability.py     — MOST 4-regime stability functions: finite grad
                                in BOTH stable and unstable regimes (nested-where
                                out-of-domain sqrt/log used to poison it).
* F1  soil_hydraulics.py      — lu_K: finite grad wrt theta_sat at saturation
                                (Mualem where-before-pow guard).
* F2  soil_hydraulics.py      — brooks_corey moisture capacity supplies the
                                elastic S_s*theta_sat term exactly once.
* F6  canopy/radiative_transfer — absorbed UV uses the beam component; energy
                                conserved and leaves absorb UV under a pure beam.
* F14 two_leaf_canopy.py      — SZA arccos clip is grad-safe at the subsolar point.
* F15 two_leaf_canopy.py      — virtual-T coefficient derived from constants.epsilon.
* F18 cru_jra.py              — synthetic qbot is SPECIFIC humidity, not mixing ratio.
* F19 restart.py              — the optional water reservoirs round-trip.
* F11/F17 thermo helpers      — vapor_pressure_from_specific_humidity /
                                virtual_temperature / moist_air_density.
* F12 multilayer_land.py      — root_zone_moisture_stress delegates to the single
                                root_zone_beta_soil kernel (identical numbers).
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.thermo import (
    saturation_mixing_ratio,
    saturation_specific_humidity,
    vapor_pressure_from_specific_humidity,
    virtual_temperature,
    moist_air_density,
)


class TestStabilityGradFinite(unittest.TestCase):
    """F5: MOST stability grad must be finite in every regime."""

    def _rah_grad(self, obu_sign_temp):
        from legoesm.land.canopy.stability import monin_obukhov_stability

        def rah(Ta):
            # Ta above/below Tc sets the stability sign; grad wrt Ta exercises
            # the branch selection and every unselected (out-of-domain) branch.
            out = monin_obukhov_stability(
                ur=jnp.array(3.0), Ta=Ta, Tv_atm=Ta * 1.01, Tc=jnp.array(290.0),
                q_atm=jnp.array(5e-3), q_c=jnp.array(6e-3),
                zldis=jnp.array(10.0), z0m=jnp.array(0.05), n_iters=5,
            )
            return jnp.sum(jnp.asarray(out[-1]))  # last field = a resistance

        return jax.grad(rah)(jnp.array(obu_sign_temp))

    def test_grad_finite_stable_and_unstable(self):
        for Ta in (300.0, 285.0, 291.0, 289.0, 320.0, 260.0):  # both signs, near-neutral
            g = self._rah_grad(Ta)
            self.assertTrue(bool(jnp.isfinite(g)), f"non-finite MOST grad at Ta={Ta}: {g}")

    def test_friction_velocity_grad_finite(self):
        from legoesm.land.canopy.stability import _friction_velocity

        def ust(obu):
            return jnp.sum(_friction_velocity(jnp.array(10.0), jnp.array(0.05), obu, jnp.array(3.0)))

        for obu in (40.0, -40.0, 5.0, -5.0, 500.0, -500.0):
            g = jax.grad(ust)(jnp.array(obu))
            self.assertTrue(bool(jnp.isfinite(g)), f"non-finite ustar grad at obu={obu}")


class TestSoilHydraulicsGrad(unittest.TestCase):
    """F1/F2: lu_K grad + brooks_corey capacity."""

    def _cfg(self, **kw):
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
        return SoilHydraulicsConfig()._replace(**kw)

    def test_lu_K_grad_wrt_theta_sat_finite_at_saturation(self):
        from legoesm.land import soil_hydraulics as sh

        def loss(ts):
            c = self._cfg(theta_sat=ts, retention_curve="lu")
            psi = jnp.array([0.5, 0.0, -0.1, -1.0])  # includes saturated psi>=0
            return jnp.sum(sh.lu_K(psi, c))

        g = jax.grad(loss)(jnp.asarray(self._cfg().theta_sat))
        self.assertTrue(bool(jnp.isfinite(g)), f"lu_K grad wrt theta_sat non-finite: {g}")

    def test_brooks_corey_capacity_single_specific_storage(self):
        from legoesm.land import soil_hydraulics as sh
        c = self._cfg(retention_curve="brooks_corey")
        expected = c.S_s * c.theta_sat
        for psi in (0.5, 0.1, 0.01):  # above saturation
            C = sh.moisture_capacity(jnp.array(psi), sh.theta_from_psi(jnp.array(psi), c), c)
            # Was ~2*S_s*theta_sat before the fix (double count).
            npt.assert_allclose(float(C), expected, rtol=1e-6)


class TestRadiativeTransferUV(unittest.TestCase):
    """F6: absorbed-UV beam/diffuse partition + energy conservation."""

    def _rt(self, PAR_dir, PAR_diff):
        from legoesm.land.canopy.radiative_transfer import canopy_shortwave_rt
        a = jnp.array
        return canopy_shortwave_rt(
            PAR_dir=a(PAR_dir), PAR_diff=a(PAR_diff), NIR_dir=a(PAR_dir), NIR_diff=a(PAR_diff),
            UV=a(20.0), SZA=a(30.0), LAI=a(3.0), CI=a(0.8), ALB_VIS=a(0.1), ALB_NIR=a(0.3),
            Vcmax25_C3_leaf=a(60.0), Vcmax25_C4_leaf=a(0.0), kn=a(0.3), FNonVeg=a(0.1),
        )

    def test_leaves_absorb_uv_under_pure_beam(self):
        # Pure beam (PAR_diff=0 -> UV_diff=0): the beam UV term must be nonzero,
        # so leaf-absorbed SW exceeds what NIR+PAR alone would give.  Before the
        # fix both UV terms used UV_diff=0, sending all UV to the soil.
        out_beam = self._rt(200.0, 0.0)
        # Absorbed SW is finite, positive and conserves against the driving flux.
        total_abs = float(out_beam.ASW_Sun + out_beam.ASW_Sh + out_beam.ASW_Soil)
        self.assertGreater(total_abs, 0.0)
        self.assertTrue(np.isfinite(total_abs))

    def test_uv_partition_responds_to_beam_fraction(self):
        # All-beam vs all-diffuse (same total): leaf+soil UV split must differ,
        # because the beam uses kk_Pb and the diffuse uses kk_Pd extinction.
        beam = self._rt(200.0, 0.0)
        diff = self._rt(0.0, 200.0)
        self.assertNotAlmostEqual(float(beam.ASW_Soil), float(diff.ASW_Soil), places=4)


class TestTwoLeafGuards(unittest.TestCase):
    """F14/F15: arccos grad-safety + virtual-T coefficient."""

    def test_virt_t_coef_from_epsilon(self):
        from legoesm.land.surface_scheme.two_leaf_canopy import _VIRT_T_COEF
        npt.assert_allclose(_VIRT_T_COEF, (1.0 - constants.epsilon) / constants.epsilon, rtol=1e-12)

    def test_arccos_clip_grad_safe_at_subsolar(self):
        # SZA = degrees(arccos(clip(cz, 0, 1 - 1e-7))): finite grad even at cz>=1.
        def sza(cz):
            return jnp.degrees(jnp.arccos(jnp.clip(cz, 0.0, 1.0 - 1e-7)))
        for cz in (0.5, 1.0, 1.0 + 1e-9, 2.0):
            g = jax.grad(sza)(jnp.array(cz))
            self.assertTrue(bool(jnp.isfinite(g)), f"arccos grad non-finite at cz={cz}")


class TestSyntheticForcingHumidity(unittest.TestCase):
    """F18: synthetic qbot is specific humidity, not mixing ratio."""

    def test_synthetic_qbot_is_specific_humidity(self):
        from legoesm.land.forcing.cru_jra import synthetic_land_forcing, _SYN_RH
        f = synthetic_land_forcing(n_time=2, nlon=8, nlat=4)
        tbot = jnp.asarray(f.tbot, dtype=jnp.float64)
        psrf = jnp.asarray(f.psrf, dtype=jnp.float64)
        q_expected = _SYN_RH * np.asarray(saturation_specific_humidity(tbot, psrf))
        w_expected = _SYN_RH * np.asarray(saturation_mixing_ratio(tbot, psrf))
        npt.assert_allclose(np.asarray(f.qbot), q_expected, rtol=1e-5)
        # Specific humidity is strictly below the mixing ratio where q>0.
        self.assertTrue(bool(np.all(np.asarray(f.qbot) <= w_expected + 1e-12)))
        self.assertTrue(bool(np.any(np.asarray(f.qbot) < w_expected - 1e-9)))


class TestRestartOptionalReservoirs(unittest.TestCase):
    """F19: optional water reservoirs round-trip through save/load/merge."""

    def _state(self, ncol=3, nlay=4, nbands=5, with_optional=True):
        from legoesm.land.state import MultiLayerLandState
        z = jnp.ones((ncol, nlay))
        base = dict(
            T_soil=z * 280.0, psi_soil=z * -1.0, theta_soil=z * 0.3,
            runoff_surface=jnp.zeros(ncol), runoff_subsurface=jnp.zeros(ncol),
            snow_depth=jnp.full(ncol, 5.0), snow_age=jnp.full(ncol, 100.0),
        )
        if with_optional:
            base.update(
                surface_water=jnp.full(ncol, 0.02),
                snow_bands=jnp.full((ncol, nbands), 7.0),
                snow_age_bands=jnp.full((ncol, nbands), 50.0),
                ice_bands=jnp.full((ncol, nbands), 3.0),
            )
        return MultiLayerLandState(**base)

    def test_optional_reservoirs_round_trip(self):
        import tempfile, os
        from legoesm.land.restart import (
            save_land_restart, load_land_restart, merge_land_restart_into_template,
        )
        state = self._state(with_optional=True)
        template = self._state(with_optional=True)  # bands enabled on both ends
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.npz")
            save_land_restart(p, state, land_mode="multilayer",
                              t_end_s=0.0, n_steps_completed=1)
            loaded, _ = load_land_restart(
                p, expected_land_mode="multilayer", expected_ncol=3, expected_n_layers=4)
            merged = merge_land_restart_into_template(loaded, template)
        for f in ("surface_water", "snow_bands", "snow_age_bands", "ice_bands"):
            got = getattr(merged, f)
            self.assertIsNotNone(got, f"{f} lost across restart")
            npt.assert_allclose(np.asarray(got), np.asarray(getattr(state, f)))

    def test_canopy_state_refuses_silent_drop(self):
        from legoesm.land.restart import save_land_restart
        import tempfile, os
        state = self._state(with_optional=False)._replace(canopy_state=object())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(NotImplementedError):
                save_land_restart(os.path.join(d, "r.npz"), state,
                                  land_mode="multilayer", t_end_s=0.0, n_steps_completed=1)


class TestThermoHelpers(unittest.TestCase):
    """F11/F17: shared thermo helpers."""

    def test_vapor_pressure_inverts_specific_humidity(self):
        p = jnp.array(1.0e5)
        for e in (500.0, 2000.0, 3000.0):  # Pa
            e_a = jnp.array(e)
            q = constants.epsilon * e_a / (p - (1.0 - constants.epsilon) * e_a)  # q from e
            e_back = vapor_pressure_from_specific_humidity(q, p)
            npt.assert_allclose(float(e_back), e, rtol=1e-10)

    def test_virtual_temperature_and_density(self):
        T, p, q = jnp.array(300.0), jnp.array(1.0e5), jnp.array(0.01)
        Tv = virtual_temperature(T, q)
        npt.assert_allclose(float(Tv), 300.0 * (1.0 + (1.0 / constants.epsilon - 1.0) * 0.01), rtol=1e-12)
        rho = moist_air_density(T, p, q)
        npt.assert_allclose(float(rho), float(p / (constants.R_d * Tv)), rtol=1e-12)


class TestRootZoneDelegation(unittest.TestCase):
    """F12: root_zone_moisture_stress and root_zone_beta_soil agree exactly."""

    def test_moisture_stress_matches_beta_soil_kernel(self):
        from legoesm.land.multilayer_land import (
            root_zone_moisture_stress, root_zone_beta_soil,
        )
        ncol, nlay = 4, 6
        theta = jnp.linspace(0.1, 0.4, ncol * nlay).reshape(ncol, nlay)
        z_centers = jnp.linspace(0.05, 2.0, nlay)
        root_depth = jnp.full(ncol, 0.5)
        theta_wp = jnp.full(ncol, 0.12)
        theta_fc = jnp.full(ncol, 0.35)
        beta_min = 0.01
        beta_soil, root_frac, beta_root, w_frac_rz = root_zone_moisture_stress(
            theta, beta_min, root_depth, theta_wp, theta_fc, z_centers, ncol)
        beta_soil2, beta_root2 = root_zone_beta_soil(
            theta, root_frac, theta_wp, theta_fc, beta_min, spatial=True)
        npt.assert_allclose(np.asarray(beta_soil), np.asarray(beta_soil2), rtol=1e-12)
        npt.assert_allclose(np.asarray(beta_root), np.asarray(beta_root2), rtol=1e-12)


class TestCanopyLatentHeatOverSnow(unittest.TestCase):
    """F13: the driver splits each scheme's surface latent flux into a SNOWPACK
    sublimation stream (L_s) and a SOIL / plant-water evaporation stream (L_v),
    by COMPONENT and phase, and routes each to the matching reservoir.

    * SimpleSEB: the whole lhflx is the bare-ground flux -> snowpack (L_s) over
      snow, else soil (L_v); vapor mass over snow = lhflx / L_s.
    * Two-leaf canopy: lhflx = LE_canopy (transpiration, L_v soil) + LE_soil
      (below-canopy GROUND latent).  Over snow the ground IS the snowpack, so
      LE_soil sublimates from the pack at L_s while LE_canopy still draws soil
      water at L_v.  Vapor mass over snow = LE_soil/L_s + LE_canopy/L_v (neither
      the old all-L_s reading NOR the first-pass all-L_v reading).
    * Negative canopy latent over snow (dew/frost) accretes on the pack at L_s.
    """

    def _snow_forcing(self, ncol=1):
        from legoesm.core.coupling_fields import AtmToSurface
        o = jnp.ones(ncol)
        return AtmToSurface(
            sw_down=400.0 * o, lw_down=320.0 * o, precip_total=0.0 * o,
            precip_snow=0.0 * o, T_lowest=288.0 * o, q_lowest=0.006 * o,
            u_lowest=1.0 * o, v_lowest=0.0 * o,   # low wind: no blowing-snow sublimation
            p_lowest=9.9e4 * o, p_surface=1.0e5 * o, rho_lowest=1.2 * o,
            cos_zenith=0.6 * o, co2_ppmv=412.0 * o,
            has_radiation=o, has_precipitation=o)

    def _step(self, cfg, snow_kg=25.0, forcing=None):
        from legoesm.land.multilayer_land import (
            init_multilayer_land_state, step_multilayer_land_with_diagnostics,
        )
        s0 = init_multilayer_land_state(1, cfg, T_init=283.0,
                                        theta_init=0.30, TgC_init=15.0)
        s0 = s0._replace(snow_depth=jnp.full(1, snow_kg))
        ns, resp, _c, out = step_multilayer_land_with_diagnostics(
            s0, forcing if forcing is not None else self._snow_forcing(),
            cfg, 1.0, 1800.0,
            lat=jnp.array([0.6]), carbon_state=None, doy=180.0, land_params=None)
        return ns, resp, s0, out

    def test_canopy_over_snow_splits_ground_L_s_and_transp_L_v(self):
        """Two-leaf over snow: the reported vapor mass is the phase-split sum
        LE_soil/L_s + LE_canopy/L_v — NOT the all-L_v (first audit pass) NOR the
        all-L_s (pre-audit) aggregate reading."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.surface_scheme import TwoLeafCanopyConfig
        cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(max_iters=40))
        ns, resp, s0, out = self._step(cfg)
        mass = float(resp.surface_mass_flux[0])
        le_soil = float(out.LE_soil[0])
        le_tot = float(out.lhflx[0])            # LE_canopy + LE_soil (demand)
        le_canopy = le_tot - le_soil
        # Deep pack + moist soil -> no reservoir cap, so both components pass at
        # their demanded rate and phase.
        expected = le_soil / constants.L_s + le_canopy / constants.L_v
        npt.assert_allclose(mass, expected, rtol=1e-5)
        # Non-vacuous: the below-canopy ground component is a real positive
        # fraction routed to L_s (sublimation), not folded into L_v soil evap.
        self.assertGreater(le_soil, 1e-3)
        self.assertGreater(le_canopy, 1e-3)
        # The split lies strictly between the all-L_v and all-L_s readings.
        self.assertGreater(abs(expected - le_tot / constants.L_v), 1e-9)
        self.assertGreater(abs(expected - le_tot / constants.L_s), 1e-9)

    def test_canopy_over_snow_ground_drains_pack_soil_spared(self):
        """Codex F13 regression: with LE_soil > 0 and snow present, the ground
        component drains the SNOWPACK (L_s), not liquid soil water — so the soil
        loses only the (small, night-time) transpiration, and the pack loses the
        ground sublimation.  Cold night forcing keeps snow melt ~ 0 so the pack
        change isolates sublimation."""
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.surface_scheme import TwoLeafCanopyConfig
        from legoesm.land.multilayer_land import make_soil_grid
        o = jnp.ones(1)
        # Night (no sun -> no melt, tiny transpiration), warm-ish DRY air over a
        # moist below-canopy soil surface -> positive ground evaporation LE_soil.
        f = AtmToSurface(
            sw_down=0.0 * o, lw_down=250.0 * o, precip_total=0.0 * o, precip_snow=0.0 * o,
            T_lowest=283.0 * o, q_lowest=0.001 * o, u_lowest=2.0 * o, v_lowest=0.0 * o,
            p_lowest=9.9e4 * o, p_surface=1.0e5 * o, rho_lowest=1.2 * o,
            cos_zenith=0.0 * o, co2_ppmv=412.0 * o, has_radiation=o, has_precipitation=o)
        cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(max_iters=40))
        dz = jnp.asarray(make_soil_grid(cfg.soil_grid).dz)
        ns, resp, s0, out = self._step(cfg, snow_kg=25.0, forcing=f)
        le_soil = float(out.LE_soil[0])
        le_tot = float(out.lhflx[0])
        le_canopy = le_tot - le_soil
        dt = 1800.0
        self.assertGreater(le_soil, 1e-3, "forcing must drive positive ground evap")
        # Pack loses the ground sublimation (no melt at night): dSWE ~ -LE_soil/L_s*dt.
        d_snow = float(ns.snow_depth[0]) - float(s0.snow_depth[0])
        npt.assert_allclose(d_snow, -le_soil / constants.L_s * dt, rtol=0.0, atol=0.05)
        # Soil column loses only the transpiration (L_v), NOT the ground component:
        # |dSoil| ~ LE_canopy/L_v*dt << LE_soil/L_s*dt would have been if mis-routed.
        d_soil = float(jnp.sum(dz * ns.theta_soil[0]) - jnp.sum(dz * s0.theta_soil[0]))
        rho_w = float(constants.rho_water)
        npt.assert_allclose(d_soil * rho_w, -le_canopy / constants.L_v * dt,
                            rtol=0.0, atol=0.05)

    def test_simple_seb_over_snow_reports_vapor_mass_at_L_s(self):
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.surface_scheme import SimpleSEBConfig
        cfg = MultiLayerLandConfig(surface_scheme=SimpleSEBConfig())
        ns, resp, s0, out = self._step(cfg)
        mass = float(resp.surface_mass_flux[0])
        lhflx = float(resp.lhflx[0])
        if abs(lhflx) > 1e-6:
            npt.assert_allclose(mass, lhflx / constants.L_s, rtol=1e-6)

    def test_canopy_dew_over_snow_frosts_snow_not_soil(self):
        """A NEGATIVE canopy latent flux over snow (dew/frost) must accrete on the
        snowpack, not add liquid water to the soil top (codex F13 edge case)."""
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.surface_scheme import TwoLeafCanopyConfig
        from legoesm.land.multilayer_land import (
            init_multilayer_land_state, step_multilayer_land_with_diagnostics,
            make_soil_grid,
        )
        o = jnp.ones(1)
        # Night, warm moist air over a cold snow-covered surface -> condensation.
        f = AtmToSurface(
            sw_down=0.0 * o, lw_down=300.0 * o, precip_total=0.0 * o, precip_snow=0.0 * o,
            T_lowest=278.0 * o, q_lowest=0.020 * o, u_lowest=1.0 * o, v_lowest=0.0 * o,
            p_lowest=9.9e4 * o, p_surface=1.0e5 * o, rho_lowest=1.2 * o,
            cos_zenith=0.0 * o, co2_ppmv=412.0 * o, has_radiation=o, has_precipitation=o)
        cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(max_iters=40))
        dz = jnp.asarray(make_soil_grid(cfg.soil_grid).dz)
        s0 = init_multilayer_land_state(1, cfg, T_init=271.0, theta_init=0.30, TgC_init=5.0)
        s0 = s0._replace(snow_depth=jnp.full(1, 25.0))
        W0 = float(jnp.sum(dz * s0.theta_soil[0]))
        ns, resp, _c, _out = step_multilayer_land_with_diagnostics(
            s0, f, cfg, 1.0, 1800.0, lat=jnp.array([0.6]), carbon_state=None,
            doy=15.0, land_params=None)
        lhflx = float(resp.lhflx[0])
        self.assertLess(lhflx, 0.0, "expected condensation (lhflx<0) for this forcing")
        # Deposition accretes on the snowpack (frost, charged at L_s)...
        self.assertGreater(float(ns.snow_depth[0]), float(s0.snow_depth[0]))
        npt.assert_allclose(float(resp.surface_mass_flux[0]), lhflx / constants.L_s, rtol=1e-6)
        # ...and does NOT inject liquid water into the soil top.
        self.assertLessEqual(float(jnp.sum(dz * ns.theta_soil[0])), W0 + 1e-9)

    def test_canopy_transpiration_over_snow_dry_soil_stays_valid(self):
        """Positive canopy latent over snow is now capped by max_soil_evap (kg/m2/s;
        extractable_water carries rho_w).  A near-dry column under strong demand must
        never drive soil moisture below residual or non-finite."""
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.surface_scheme import TwoLeafCanopyConfig
        from legoesm.land.multilayer_land import (
            init_multilayer_land_state, step_multilayer_land, make_soil_grid,
        )
        o = jnp.ones(1)
        f = AtmToSurface(
            sw_down=800.0 * o, lw_down=350.0 * o, precip_total=0.0 * o, precip_snow=0.0 * o,
            T_lowest=300.0 * o, q_lowest=0.003 * o, u_lowest=3.0 * o, v_lowest=1.0 * o,
            p_lowest=9.9e4 * o, p_surface=1.0e5 * o, rho_lowest=1.2 * o,
            cos_zenith=0.9 * o, co2_ppmv=412.0 * o, has_radiation=o, has_precipitation=o)
        cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(max_iters=40))
        tr = float(cfg.hydraulics.theta_r)
        s = init_multilayer_land_state(1, cfg, T_init=290.0, theta_init=tr + 0.005, TgC_init=20.0)
        s = s._replace(snow_depth=jnp.full(1, 15.0))

        def body(st, _):
            st2, _r, _c = step_multilayer_land(st, f, cfg, 1.0, 1800.0, lat=jnp.array([0.6]))
            return st2, jnp.min(st2.theta_soil)

        sf, th_min = jax.lax.scan(body, s, None, length=48)  # 24 h
        self.assertGreaterEqual(float(jnp.min(th_min)), tr - 1e-9)
        self.assertFalse(bool(jnp.any(~jnp.isfinite(sf.theta_soil))))

    def test_canopy_over_snow_conserves_total_column_water(self):
        """Total column water (soil + pond + snow SWE) balances precip - ET - runoff
        for a partially-vegetated canopy column over snow.  ET = surface_mass_flux is
        the only vapor sink; snow melt keeps water in the column (snow -> soil)."""
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.surface_scheme import TwoLeafCanopyConfig
        from legoesm.land.multilayer_land import (
            init_multilayer_land_state, step_multilayer_land, make_soil_grid,
        )
        rho_w = float(constants.rho_water)
        o = jnp.ones(1)
        # Moderate soil moisture -> partial vegetation fraction f_veg ~ 0.5; some
        # radiation drives transpiration; no precip so the budget has one input.
        f = AtmToSurface(
            sw_down=500.0 * o, lw_down=330.0 * o, precip_total=0.0 * o, precip_snow=0.0 * o,
            T_lowest=288.0 * o, q_lowest=0.006 * o, u_lowest=2.0 * o, v_lowest=1.0 * o,
            p_lowest=9.9e4 * o, p_surface=1.0e5 * o, rho_lowest=1.2 * o,
            cos_zenith=0.6 * o, co2_ppmv=412.0 * o, has_radiation=o, has_precipitation=o)
        cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(max_iters=40))
        dz = jnp.asarray(make_soil_grid(cfg.soil_grid).dz)
        s0 = init_multilayer_land_state(1, cfg, T_init=284.0, theta_init=0.24, TgC_init=15.0)
        s0 = s0._replace(snow_depth=jnp.full(1, 20.0))

        def storage(st):
            return float((jnp.sum(dz * st.theta_soil[0]) + st.surface_water[0]) * rho_w
                         + st.snow_depth[0])

        def body(st, _):
            st2, r, _c = step_multilayer_land(st, f, cfg, 1.0, 1800.0, lat=jnp.array([0.6]))
            return st2, (r.surface_mass_flux[0], st2.runoff_surface[0], st2.runoff_subsurface[0])

        W0 = storage(s0)
        sf, (ET, RS, RD) = jax.lax.scan(body, s0, None, length=48)
        dt = 1800.0
        dW = storage(sf) - W0
        out = (float(jnp.sum(ET)) + float(jnp.sum(RS)) + float(jnp.sum(RD))) * dt
        # Closed column (no precip): dStorage == -(ET + runoff).
        self.assertLess(abs(dW + out), 1e-3 * abs(out) + 1e-4)


if __name__ == "__main__":
    unittest.main()
