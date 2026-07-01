"""Realistic (Berger 1978) orbital insolation for AMIP-II.

Validates the eccentricity / Earth-Sun-distance corrections added to
``legoesm.atmosphere.physics.radiation.solar``:

* ``earth_sun_distance_factor`` ``(a/r)^2`` range + perihelion/aphelion
  asymmetry + annual mean ~ 1 (orbital forcing redistributes insolation
  seasonally without changing the global-annual total),
* ``solar_declination`` with an orbit stays within +/- obliquity and has the
  correct solstice signs,
* ``orbit=None`` is the unchanged circular-orbit baseline,
* global-annual-mean insolation is conserved to the analytic ``1/sqrt(1-e^2)``,
* the orbital path is jit-able and differentiable.

Run with JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.radiation.solar import (
    OrbitalParameters,
    daily_mean_insolation,
    earth_orbit,
    earth_sun_distance_factor,
    solar_declination,
)


def _obliquity_rad(orbit: OrbitalParameters) -> float:
    return orbit.obliquity_deg * float(constants.DEG_TO_RAD)


class TestDistanceFactor:
    def test_range_matches_analytic_bounds(self):
        orbit = earth_orbit()
        e = orbit.eccentricity
        days = np.arange(1.0, 366.0)
        fd = np.array([float(earth_sun_distance_factor(float(d), orbit))
                       for d in days])
        # (a/r)^2 in [1/(1+e)^2, 1/(1-e)^2].
        lo = 1.0 / (1.0 + e) ** 2
        hi = 1.0 / (1.0 - e) ** 2
        assert fd.min() >= lo - 1e-6
        assert fd.max() <= hi + 1e-6
        # The bounds are actually attained (within the daily sampling).
        assert fd.max() > 1.03
        assert fd.min() < 0.97

    def test_perihelion_early_january_exceeds_aphelion_july(self):
        orbit = earth_orbit()
        fd_jan = float(earth_sun_distance_factor(3.0, orbit))    # ~ perihelion
        fd_jul = float(earth_sun_distance_factor(185.0, orbit))  # ~ aphelion
        assert fd_jan > 1.0 > fd_jul
        assert fd_jan > fd_jul + 0.05  # ~6.8% peak-to-peak for e~0.0167

    def test_annual_mean_is_unity_to_analytic(self):
        """Time-mean of (a/r)^2 over the orbit = 1/sqrt(1-e^2) (~1.0001)."""
        orbit = earth_orbit()
        days = np.arange(0.5, 365.0)  # mid-day samples, uniform in time
        fd = np.array([float(earth_sun_distance_factor(float(d), orbit))
                       for d in days])
        expected = 1.0 / np.sqrt(1.0 - orbit.eccentricity ** 2)
        assert abs(fd.mean() - expected) < 2e-4
        assert abs(fd.mean() - 1.0) < 1e-3

    def test_zero_eccentricity_is_circular(self):
        circ = OrbitalParameters(eccentricity=0.0)
        for d in (1.0, 80.0, 172.0, 266.0, 355.0):
            assert float(earth_sun_distance_factor(d, circ)) == pytest.approx(
                1.0, abs=1e-12)


class TestOrbitalDeclination:
    def test_within_obliquity_envelope(self):
        orbit = earth_orbit()
        eps = _obliquity_rad(orbit)
        days = np.arange(1.0, 366.0)
        dec = np.array([float(solar_declination(float(d), orbit=orbit))
                        for d in days])
        assert np.max(np.abs(dec)) <= eps + 1e-9

    def test_solstice_signs_and_magnitude(self):
        orbit = earth_orbit()
        eps = _obliquity_rad(orbit)
        # NH summer solstice ~ Jun 21 (day 172): declination near +obliquity.
        dec_jun = float(solar_declination(172.0, orbit=orbit))
        assert dec_jun > 0.40
        assert dec_jun == pytest.approx(eps, abs=0.02)
        # NH winter solstice ~ Dec 21 (day 355): near -obliquity.
        dec_dec = float(solar_declination(355.0, orbit=orbit))
        assert dec_dec < -0.40
        # Equinox (day 80): near zero.
        assert abs(float(solar_declination(80.0, orbit=orbit))) < 0.02

    def test_orbit_none_is_legacy_circular_formula(self):
        # orbit=None must reproduce the simple obliquity*sin baseline exactly.
        for d in (1.0, 80.0, 172.0, 266.0, 355.0):
            got = float(solar_declination(d))  # default orbit=None
            obl = 23.45 * float(constants.DEG_TO_RAD)
            expect = obl * np.sin(2.0 * np.pi * (d - 80.0) / 365.0)
            assert got == pytest.approx(expect, rel=1e-12, abs=1e-12)


class TestDailyMeanInsolationOrbital:
    def test_orbit_none_unchanged(self):
        lat = jnp.deg2rad(jnp.linspace(-89.0, 89.0, 37))
        for d in (15.0, 172.0, 266.0):
            base = daily_mean_insolation(lat, d)
            explicit_none = daily_mean_insolation(lat, d, orbit=None)
            assert np.allclose(np.asarray(base), np.asarray(explicit_none),
                               rtol=0, atol=0)

    def test_perihelion_boosts_summer_hemisphere(self):
        """NH-winter / SH-summer (perihelion, ~Jan) total insolation gets the
        +3% distance boost vs the circular orbit."""
        orbit = earth_orbit()
        lat = jnp.deg2rad(jnp.linspace(-89.0, 89.0, 91))
        w = np.cos(np.asarray(lat))
        day = 3.0  # near perihelion
        circ = np.asarray(daily_mean_insolation(lat, day))
        orb = np.asarray(daily_mean_insolation(lat, day, orbit=orbit))
        gm_circ = (circ * w).sum() / w.sum()
        gm_orb = (orb * w).sum() / w.sum()
        assert gm_orb > gm_circ * 1.02  # perihelion ~ +3.4%

    def test_global_annual_energy_conserved(self):
        """Orbital insolation redistributes seasonally but conserves the
        global-annual mean to 1/sqrt(1-e^2)."""
        orbit = earth_orbit()
        lat = jnp.deg2rad(jnp.linspace(-89.5, 89.5, 180))
        w = np.cos(np.asarray(lat))
        days = np.arange(0.5, 365.0, 5.0)

        def gmean(use_orbit):
            tot = 0.0
            for d in days:
                q = np.asarray(daily_mean_insolation(
                    lat, float(d), orbit=orbit if use_orbit else None))
                tot += (q * w).sum() / w.sum()
            return tot / len(days)

        ratio = gmean(True) / gmean(False)
        expected = 1.0 / np.sqrt(1.0 - orbit.eccentricity ** 2)
        assert ratio == pytest.approx(expected, abs=5e-4)
        assert ratio == pytest.approx(1.0, abs=1e-3)


class TestJitAndGrad:
    def test_jit_distance_factor(self):
        orbit = earth_orbit()
        f = jax.jit(lambda d: earth_sun_distance_factor(d, orbit))
        assert float(f(3.0)) == pytest.approx(
            float(earth_sun_distance_factor(3.0, orbit)), rel=1e-10)

    def test_grad_daily_mean_is_finite(self):
        orbit = earth_orbit()
        lat = jnp.deg2rad(jnp.linspace(-80.0, 80.0, 33))

        def loss(latitudes):
            return jnp.sum(daily_mean_insolation(latitudes, 172.0, orbit=orbit))

        g = jax.grad(loss)(lat)
        assert np.all(np.isfinite(np.asarray(g)))
