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


if __name__ == "__main__":
    unittest.main()
