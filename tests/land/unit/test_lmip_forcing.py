"""Tests for the synthetic single-column LMIP forcing generator."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy.testing as npt

jax.config.update("jax_enable_x64", True)

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.lmip_forcing import make_synthetic_lmip_forcing

_MIDLAT = 45.5 * jnp.pi / 180.0
_TROPICS = 0.0
_POLE = 80.0 * jnp.pi / 180.0


class TestSyntheticForcing(unittest.TestCase):

    def test_returns_atmtosurface_shape1(self):
        f = make_synthetic_lmip_forcing(_MIDLAT, 0.0, 180.0, 12.0)
        self.assertIsInstance(f, AtmToSurface)
        for name in ("sw_down", "T_lowest", "q_lowest", "precip_total"):
            self.assertEqual(getattr(f, name).shape, (1,), name)

    def test_night_no_shortwave(self):
        """Local midnight -> cos(zenith) clipped to 0 -> sw_down == 0."""
        f = make_synthetic_lmip_forcing(_MIDLAT, 0.0, 180.0, 0.0)
        npt.assert_allclose(f.sw_down, 0.0, atol=1e-12)
        self.assertGreaterEqual(float(f.cos_zenith[0]), 0.0)

    def test_noon_summer_has_shortwave(self):
        f = make_synthetic_lmip_forcing(_MIDLAT, 0.0, 180.0, 12.0)
        self.assertGreater(float(f.sw_down[0]), 100.0)

    def test_pole_colder_than_tropics(self):
        f_trop = make_synthetic_lmip_forcing(_TROPICS, 0.0, 200.0, 12.0)
        f_pole = make_synthetic_lmip_forcing(_POLE, 0.0, 200.0, 12.0)
        self.assertGreater(float(f_trop.T_lowest[0]), float(f_pole.T_lowest[0]))

    def test_seasonal_nh_summer_warmer(self):
        """NH mid-latitude July (doy 200) warmer than January (doy 15)."""
        f_jul = make_synthetic_lmip_forcing(_MIDLAT, 0.0, 200.0, 6.0)
        f_jan = make_synthetic_lmip_forcing(_MIDLAT, 0.0, 15.0, 6.0)
        self.assertGreater(float(f_jul.T_lowest[0]), float(f_jan.T_lowest[0]))

    def test_precip_snow_when_cold(self):
        """Deep-winter pole -> air below the snow threshold -> all snow."""
        f = make_synthetic_lmip_forcing(_POLE, 0.0, 15.0, 0.0)
        self.assertLess(float(f.T_lowest[0]), 275.0)
        npt.assert_allclose(f.precip_snow, f.precip_total, rtol=1e-9)

    def test_precip_rain_when_warm(self):
        f = make_synthetic_lmip_forcing(_TROPICS, 0.0, 200.0, 12.0)
        self.assertGreater(float(f.T_lowest[0]), 275.0)
        npt.assert_allclose(f.precip_snow, 0.0, atol=1e-15)

    def test_precip_rate_passthrough(self):
        f = make_synthetic_lmip_forcing(_TROPICS, 0.0, 200.0, 12.0, precip_rate=5e-5)
        npt.assert_allclose(f.precip_total, 5e-5, rtol=1e-9)

    def test_jit_traceable_day_hour(self):
        """day/hour must be traceable (scan-compatible time loop)."""
        @jax.jit
        def sw(day, hour):
            return make_synthetic_lmip_forcing(_MIDLAT, 0.3, day, hour).sw_down

        out = sw(jnp.asarray(180.0), jnp.asarray(12.0))
        self.assertTrue(jnp.all(jnp.isfinite(out)))

    def test_relative_humidity_positive_subsaturated(self):
        f = make_synthetic_lmip_forcing(_TROPICS, 0.0, 200.0, 12.0)
        self.assertGreater(float(f.q_lowest[0]), 0.0)


class TestLatitudeFeatureHelpers(unittest.TestCase):
    """The latitude->feature pieces factored out for reuse by the zonal
    carbon-IC climatology builder (must keep the same latitude dependence)."""

    def test_mat_decreases_poleward(self):
        from legoesm.land.lmip_forcing import latitude_mean_annual_temp_k
        eq = float(latitude_mean_annual_temp_k(_TROPICS))
        pole = float(latitude_mean_annual_temp_k(_POLE))
        self.assertGreater(eq, pole)
        self.assertTrue(jnp.isfinite(latitude_mean_annual_temp_k(_MIDLAT)))

    def test_seasonal_amp_increases_poleward(self):
        from legoesm.land.lmip_forcing import latitude_seasonal_amp_k
        npt.assert_allclose(float(latitude_seasonal_amp_k(_TROPICS)), 0.0, atol=1e-9)
        self.assertGreater(float(latitude_seasonal_amp_k(_POLE)),
                           float(latitude_seasonal_amp_k(_MIDLAT)))

    def test_daily_mean_sw_positive_and_vmappable(self):
        from legoesm.land.lmip_forcing import latitude_daily_mean_sw_w
        self.assertGreater(float(latitude_daily_mean_sw_w(_TROPICS, 200.0)), 0.0)
        # batched (vmap over lat, day) matches the scalar calls (the carbon-IC
        # driver evaluates it over all cell-months in one vmap).
        lats = jnp.array([_TROPICS, _MIDLAT])
        days = jnp.array([200.0, 200.0])
        batched = jax.vmap(latitude_daily_mean_sw_w)(lats, days)
        npt.assert_allclose(
            float(batched[0]), float(latitude_daily_mean_sw_w(_TROPICS, 200.0)),
            rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
