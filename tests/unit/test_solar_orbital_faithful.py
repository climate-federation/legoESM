"""Oracle-faithfulness pins for the Berger (1978) orbital insolation.

Oracle: Berger, A. (1978), *J. Atmos. Sci.* 35, 2362-2367 (orbital elements and
the eccentricity series for the true solar longitude), the Cooper (1969)
single-harmonic declination, the classical spherical-astronomy zenith/hour-angle
identity, and the Hartmann/Frierson exact diurnal-mean-insolation integral.  The
Berger series COEFFICIENTS are those reproduced in CESM ``shr_orb_mod`` and
climlab ``solar_longitude`` (the calendar day_VE=80 follows climlab; see
departure #4).  Each closed form is pinned to round-off (rel 1e-12) against
an INDEPENDENT numpy reimplementation whose coefficients are typed from the
reference (NOT read back from ``legoesm...radiation.solar`` — non-circular):

  true solar longitude   λ = λ_m + C(M),  M = λ_m − ϖ  (equation of centre)
    λ_m  = λ_m0 + (day − day_VE)·2π/365
    λ_m0 = −2[ (e/2 + e³/8)(1+β) sin(−ϖ) − (e²/4)(½+β) sin(−2ϖ)
                                          + (e³/8)(⅓+β) sin(−3ϖ) ],  β=√(1−e²)
    C(M) = (2e − e³/4) sin M + (5/4)e² sin 2M + (13/12)e³ sin 3M   (classical)
  Earth-Sun distance      (a/r)² = ((1 + e cos ν)/(1 − e²))²,  ν = λ − ϖ
  declination (circular)  δ = ε · sin(2π(day − day_VE)/365)          (Cooper 1969)
  declination (orbital)   δ = arcsin(sin ε · sin λ)                   (Berger)
  cosine zenith           cos θ_z = sinφ sinδ + cosφ cosδ cos h
                          h = 2π·hour/24 + lon − π   (hour angle)
  daily-mean insolation   Q = (S₀/π)(h_s sinφ sinδ + cosφ cosδ sin h_s),
                          cos h_s = −tanφ tanδ  (× (a/r)² when orbital)
  daylight fraction       h_s/π
  perpetual-equinox       Q = (S₀/π) max(cosφ, 0)                     (Frierson 2006)

The equation-of-centre coefficients (2e − e³/4, 5e²/4, 13e³/12) are the standard
Fourier expansion of the equation of centre in powers of eccentricity (textbook
celestial mechanics); the λ_m0 β-series is Berger's calendar→orbit conversion.

DEPARTURES from the closed form (all deliberate, all tested):
  1. **AD-safe polar branch** in daily_mean_insolation / daylight_fraction: the
     sunset hour angle clips cos h_s to ±(1 − 1e-7) (``_AD_SAFE_BOUND``) via a
     where-before-divide, so the adjoint stays finite at the polar-day/night
     singularity (issue #249).  The pins here run at NON-polar latitudes where
     |cos h_s| < 1 − 1e-7, so the clip is inert and the formula is exact; the
     polar branch is tested by pinning the module to the exact CLIPPED-h_s
     formula (round-off) and bounding the shortfall from the ideal 24 h value.
     That shortfall is ~(S₀/π)·√(2·1e-7)·(a/r)²·sinφ sinδ ≈ 0.07 W/m² deep in
     polar day (the DOMINANT term is the h_s angular error √(2ε), not the
     cos-clip magnitude ε); AT the exact polar-day boundary (sinφ sinδ = cosφ
     cosδ) it collapses to O(ε^{3/2}) ~ 2e-9 W/m².  Physically negligible either
     way; the module comment is corrected to state these bounds.
  2. **Circular declination is the Cooper (1969) approximation** δ = ε sin(...),
     a first-harmonic surrogate for the Berger δ = arcsin(sin ε sin λ); the two
     branches are pinned to their OWN closed form and shown to differ.
  3. cos_zenith clips to [−1,1] (night = sun below horizon); the interior pins
     use configurations with |cos θ_z| < 1 so the clip is inert.
  4. **Vernal-equinox calendar day day_VE = 80** (``_SOLSTICE_OFFSET_DAYS``)
     follows climlab's convention; CESM/CLM5 document ``d_ve = 80.5`` (a ~½-day
     offset of the whole insolation calendar).  The oracle uses 80 to match the
     module (a convention choice, not a coefficient error).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.radiation import solar
from legoesm.atmosphere.physics.radiation.solar import (
    OrbitalParameters,
    _orbital_solar_longitude,  # white-box: the Berger series core (tests/ is
    #                            excluded from the private-import ratchet)
    cos_zenith_angle,
    daily_mean_insolation,
    daylight_fraction,
    earth_sun_distance_factor,
    perpetual_equinox_insolation,
    solar_declination,
)

from legoesm import constants

jax.config.update("jax_enable_x64", True)

# --- Berger (1978) / classical orbital series coefficients (typed from the
#     reference, canaried against the module's calendar constants below). ---
_O_DAY_VE = 80.0            # solar._SOLSTICE_OFFSET_DAYS (vernal-equinox day)
_O_DAYS_PER_YEAR = 365.0    # solar._DAYS_PER_YEAR
_O_EOC1_E = 2.0             # equation of centre: coeff of e   in sin(M)
_O_EOC1_E3 = -1.0 / 4.0     #                     coeff of e³  in sin(M)
_O_EOC2 = 5.0 / 4.0         #                     coeff of e²  in sin(2M)
_O_EOC3 = 13.0 / 12.0       #                     coeff of e³  in sin(3M)
_O_OBLIQUITY_DEFAULT_DEG = 23.45   # solar._EARTH_OBLIQUITY_DEG (circular default)

# Present-day (~year 2000) orbit used by earth_orbit()/constants.
_E0 = constants.orbital_eccentricity          # 0.016704
_OBL0 = constants.orbital_obliquity_deg        # 23.439
_W0 = constants.orbital_long_perihelion_deg    # 282.895

# Test orbits: present-day + off-nominal (larger e / different ϖ) so the e², e³
# series terms are exercised well above round-off.
_ORBITS = [
    (_E0, _OBL0, _W0),
    (0.05, 22.0, 100.0),
    (0.20, 24.5, 0.0),
    (0.30, 18.0, 250.0),
]
_DAYS = [1.0, 15.0, 80.0, 105.0, 172.0, 266.0, 300.0, 355.0]


# ------------------------- independent numpy oracle -------------------------

def _lambda_oracle(day, e, w_deg):
    """Berger-1978 true solar ecliptic longitude λ [rad] (independent)."""
    w = math.radians(w_deg)
    beta = math.sqrt(1.0 - e * e)
    lm0 = -2.0 * (
        (e / 2.0 + e ** 3 / 8.0) * (1.0 + beta) * math.sin(-w)
        - (e ** 2 / 4.0) * (0.5 + beta) * math.sin(-2.0 * w)
        + (e ** 3 / 8.0) * (1.0 / 3.0 + beta) * math.sin(-3.0 * w)
    )
    lm = lm0 + (day - _O_DAY_VE) * (2.0 * math.pi / _O_DAYS_PER_YEAR)
    m = lm - w
    return (
        lm
        + (_O_EOC1_E * e + _O_EOC1_E3 * e ** 3) * math.sin(m)
        + _O_EOC2 * e ** 2 * math.sin(2.0 * m)
        + _O_EOC3 * e ** 3 * math.sin(3.0 * m)
    )


def _dist_factor_oracle(day, e, w_deg):
    nu = _lambda_oracle(day, e, w_deg) - math.radians(w_deg)
    return ((1.0 + e * math.cos(nu)) / (1.0 - e * e)) ** 2


def _decl_circular_oracle(day, obliquity_deg):
    return math.radians(obliquity_deg) * math.sin(
        2.0 * math.pi * (day - _O_DAY_VE) / _O_DAYS_PER_YEAR)


def _decl_orbital_oracle(day, orbit):
    e, obl, w = orbit
    return math.asin(math.sin(math.radians(obl)) * math.sin(_lambda_oracle(day, e, w)))


def _decl_oracle(day, obliquity_deg, orbit):
    return _decl_orbital_oracle(day, orbit) if orbit else _decl_circular_oracle(day, obliquity_deg)


def _cosz_oracle(lat, lon, day, hour, obliquity_deg, orbit):
    delta = _decl_oracle(day, obliquity_deg, orbit)
    h = 2.0 * math.pi * (hour / 24.0) + lon - math.pi
    cz = math.sin(lat) * math.sin(delta) + math.cos(lat) * math.cos(delta) * math.cos(h)
    return max(-1.0, min(1.0, cz))


def _daily_oracle(lat, day, S_0, obliquity_deg, orbit):
    delta = _decl_oracle(day, obliquity_deg, orbit)
    cos_hs = -math.tan(lat) * math.tan(delta)   # caller keeps |cos_hs| < 1
    h_s = math.acos(cos_hs)
    q = (S_0 / math.pi) * (
        h_s * math.sin(lat) * math.sin(delta)
        + math.cos(lat) * math.cos(delta) * math.sin(h_s)
    )
    if orbit:
        q *= _dist_factor_oracle(day, orbit[0], orbit[2])
    return max(q, 0.0)


def _orbit(o):
    return OrbitalParameters(eccentricity=o[0], obliquity_deg=o[1], long_perihelion_deg=o[2])


# ===================== calendar / constant canaries (non-circular) =====================

def test_module_calendar_constants_match_oracle():
    assert solar._SOLSTICE_OFFSET_DAYS == _O_DAY_VE
    assert solar._DAYS_PER_YEAR == _O_DAYS_PER_YEAR
    assert solar._EARTH_OBLIQUITY_DEG == _O_OBLIQUITY_DEFAULT_DEG
    assert solar._POLE_THRESHOLD == pytest.approx(1.0e-7, rel=1e-12)


def test_present_day_orbital_constants():
    assert constants.S_0 == 1361.0
    assert float(constants.DEG_TO_RAD) == pytest.approx(math.pi / 180.0, rel=1e-15)
    assert constants.orbital_eccentricity == pytest.approx(0.016704, rel=1e-12)
    assert constants.orbital_obliquity_deg == pytest.approx(23.439, rel=1e-12)
    assert constants.orbital_long_perihelion_deg == pytest.approx(282.895, rel=1e-12)


# ===================== Berger true solar longitude λ (the core) =====================

@pytest.mark.parametrize("orbit", _ORBITS)
@pytest.mark.parametrize("day", _DAYS)
def test_orbital_solar_longitude_matches_berger_series(orbit, day):
    got = float(_orbital_solar_longitude(day, _orbit(orbit)))
    assert got == pytest.approx(_lambda_oracle(day, orbit[0], orbit[2]), rel=1e-12, abs=1e-12)


def test_lambda_equation_of_centre_coefficients_are_non_vacuous():
    # High-e orbit so the e², e³ equation-of-centre terms are well above round-off;
    # each wrong coefficient must move λ detectably (proves the pin canaries them).
    e, obl, w = 0.30, 20.0, 60.0
    day = 200.0
    ref = _lambda_oracle(day, e, w)
    got = float(_orbital_solar_longitude(day, _orbit((e, obl, w))))
    assert got == pytest.approx(ref, rel=1e-12)

    def wrong(c1e=_O_EOC1_E, c1e3=_O_EOC1_E3, c2=_O_EOC2, c3=_O_EOC3):
        wv = math.radians(w)
        beta = math.sqrt(1.0 - e * e)
        lm0 = -2.0 * (
            (e / 2.0 + e ** 3 / 8.0) * (1.0 + beta) * math.sin(-wv)
            - (e ** 2 / 4.0) * (0.5 + beta) * math.sin(-2.0 * wv)
            + (e ** 3 / 8.0) * (1.0 / 3.0 + beta) * math.sin(-3.0 * wv)
        )
        lm = lm0 + (day - _O_DAY_VE) * (2.0 * math.pi / _O_DAYS_PER_YEAR)
        m = lm - wv
        return (lm + (c1e * e + c1e3 * e ** 3) * math.sin(m)
                + c2 * e ** 2 * math.sin(2.0 * m) + c3 * e ** 3 * math.sin(3.0 * m))

    assert abs(got - wrong(c1e3=0.0)) > 1e-6      # drop the −e³/4 term
    assert abs(got - wrong(c2=1.0)) > 1e-4        # 5/4 → 1
    assert abs(got - wrong(c3=1.0)) > 1e-6        # 13/12 → 1


def test_lambda_reduces_to_uniform_motion_at_zero_eccentricity():
    # e=0 ⇒ λ_m0=0 and C(M)=0 ⇒ λ = (day − day_VE)·2π/365 exactly.
    for day in _DAYS:
        got = float(_orbital_solar_longitude(day, _orbit((0.0, 23.439, 100.0))))
        assert got == pytest.approx(
            (day - _O_DAY_VE) * (2.0 * math.pi / _O_DAYS_PER_YEAR), rel=1e-12, abs=1e-12)


# ===================== Earth-Sun distance factor (a/r)² =====================

@pytest.mark.parametrize("orbit", _ORBITS)
@pytest.mark.parametrize("day", _DAYS)
def test_earth_sun_distance_factor_matches_orbit_equation(orbit, day):
    got = float(earth_sun_distance_factor(day, _orbit(orbit)))
    assert got == pytest.approx(_dist_factor_oracle(day, orbit[0], orbit[2]), rel=1e-12)


# ===================== declination (both branches) =====================

@pytest.mark.parametrize("day", _DAYS)
def test_declination_circular_matches_cooper(day):
    # Default obliquity = solar._EARTH_OBLIQUITY_DEG (23.45), NOT the orbit's.
    got = float(solar_declination(day))
    assert got == pytest.approx(
        _decl_circular_oracle(day, _O_OBLIQUITY_DEFAULT_DEG), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("orbit", _ORBITS)
@pytest.mark.parametrize("day", _DAYS)
def test_declination_orbital_matches_berger(orbit, day):
    got = float(solar_declination(day, orbit=_orbit(orbit)))
    assert got == pytest.approx(_decl_orbital_oracle(day, orbit), rel=1e-12, abs=1e-12)


def test_circular_declination_is_a_cooper_surrogate_not_berger():
    # DEPARTURE #2: the two declination branches are genuinely different closed
    # forms (Cooper first-harmonic vs Berger arcsin), not equal.
    orbit = (_E0, _OBL0, _W0)
    diffs = [abs(float(solar_declination(d, orbit=_orbit(orbit)))
                 - _decl_circular_oracle(d, _OBL0)) for d in _DAYS]
    assert max(diffs) > 1e-3      # clearly distinct schemes


# ===================== cosine of the zenith angle =====================

@pytest.mark.parametrize("orbit", [None] + _ORBITS)
@pytest.mark.parametrize("lat_deg,lon_deg,day,hour", [
    (0.0, 0.0, 172.0, 12.0),
    (30.0, 45.0, 80.0, 9.0),
    (-45.0, -120.0, 355.0, 15.0),
    (60.0, 200.0, 266.0, 6.0),
])
def test_cos_zenith_matches_spherical_astronomy(orbit, lat_deg, lon_deg, day, hour):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    got = float(cos_zenith_angle(
        jnp.asarray(lat), jnp.asarray(lon), day, hour,
        orbit=_orbit(orbit) if orbit else None))
    ref = _cosz_oracle(lat, lon, day, hour, _O_OBLIQUITY_DEFAULT_DEG, orbit)
    assert got == pytest.approx(ref, rel=1e-12, abs=1e-12)


# ===================== daily-mean insolation (Hartmann integral) =====================

# Non-polar latitudes: |cos h_s| = |tanφ tanδ| < 1 − 1e-7 so the AD-safe clip is
# inert and the closed form is exact.
_INTERIOR_LATS = [-60.0, -30.0, 0.0, 30.0, 45.0, 60.0]


@pytest.mark.parametrize("orbit", [None] + _ORBITS)
@pytest.mark.parametrize("day", [15.0, 80.0, 172.0, 266.0, 355.0])
def test_daily_mean_insolation_matches_hartmann_integral(orbit, day):
    lat = jnp.deg2rad(jnp.asarray(_INTERIOR_LATS))
    got = np.asarray(daily_mean_insolation(lat, day, orbit=_orbit(orbit) if orbit else None))
    for i, lat_deg in enumerate(_INTERIOR_LATS):
        ref = _daily_oracle(math.radians(lat_deg), day, constants.S_0,
                            _O_OBLIQUITY_DEFAULT_DEG, orbit)
        assert got[i] == pytest.approx(ref, rel=1e-12, abs=1e-9)


def test_daily_mean_custom_S0_scales_linearly():
    lat = jnp.deg2rad(jnp.asarray([-30.0, 0.0, 45.0]))
    q1 = np.asarray(daily_mean_insolation(lat, 172.0, S_0=1000.0))
    q2 = np.asarray(daily_mean_insolation(lat, 172.0, S_0=2000.0))
    assert np.allclose(q2, 2.0 * q1, rtol=1e-12, atol=0.0)


# ===================== daylight fraction =====================

@pytest.mark.parametrize("orbit", [None] + _ORBITS)
@pytest.mark.parametrize("day", [15.0, 172.0, 355.0])
def test_daylight_fraction_matches_sunset_hour_angle(orbit, day):
    lat = jnp.deg2rad(jnp.asarray(_INTERIOR_LATS))
    got = np.asarray(daylight_fraction(lat, day, orbit=_orbit(orbit) if orbit else None))
    for i, lat_deg in enumerate(_INTERIOR_LATS):
        delta = _decl_oracle(day, _O_OBLIQUITY_DEFAULT_DEG, orbit)
        h_s = math.acos(-math.tan(math.radians(lat_deg)) * math.tan(delta))
        assert got[i] == pytest.approx(h_s / math.pi, rel=1e-12, abs=1e-12)


def test_daylight_fraction_half_at_equinox():
    lat = jnp.deg2rad(jnp.asarray(_INTERIOR_LATS))
    got = np.asarray(daylight_fraction(lat, _O_DAY_VE))    # equinox ⇒ δ≈0 ⇒ 0.5
    assert np.allclose(got, 0.5, rtol=0, atol=1e-6)


@pytest.mark.parametrize("lat", [math.pi / 2.0, -math.pi / 2.0])
def test_daylight_fraction_half_at_pole_on_equinox(lat):
    # Regression (codex): at the EXACT pole on the equinox, δ=0 ⇒ numerator=0 ⇒
    # near_pole; the pole fill must give cos_hs=0 (h_s=π/2, daylight 0.5), the
    # grazing-sun limit — NOT the polar-day/night saturation the old 2-way tie
    # picked (which returned ~0.99986). The daily-mean insolation there is 0.
    assert float(daylight_fraction(jnp.asarray(lat), _O_DAY_VE)) == pytest.approx(0.5, abs=1e-6)
    assert float(daily_mean_insolation(jnp.asarray(lat), _O_DAY_VE)) == pytest.approx(0.0, abs=1e-9)


def test_pole_polar_day_night_unaffected_at_solstice():
    # The 3-way fill must NOT change the solstice poles: δ≠0 ⇒ numerator≠0 ⇒
    # still polar day (NH summer, daylight→1) / polar night (SH, →0).
    assert float(daylight_fraction(jnp.asarray(math.pi / 2.0), 172.0)) > 0.999
    assert float(daylight_fraction(jnp.asarray(-math.pi / 2.0), 172.0)) < 1e-3


# ===================== perpetual-equinox insolation =====================

@pytest.mark.parametrize("lat_deg", [-89.0, -60.0, -30.0, 0.0, 30.0, 60.0, 89.0])
def test_perpetual_equinox_matches_frierson(lat_deg):
    lat = math.radians(lat_deg)
    got = float(perpetual_equinox_insolation(jnp.asarray(lat)))
    assert got == pytest.approx(
        (constants.S_0 / math.pi) * max(math.cos(lat), 0.0), rel=1e-12, abs=1e-12)


def test_perpetual_equinox_is_equinox_limit_of_daily_mean():
    # At δ=0 the daily-mean integral collapses to (S₀/π)cosφ (Frierson limit).
    lat = jnp.deg2rad(jnp.asarray(_INTERIOR_LATS))
    q_daily = np.asarray(daily_mean_insolation(lat, _O_DAY_VE))   # circular, equinox
    q_perp = np.asarray(perpetual_equinox_insolation(lat))
    assert np.allclose(q_daily, q_perp, rtol=1e-6, atol=1e-4)


# ===================== polar branches (DEPARTURE #1) =====================

def test_polar_day_and_night_insolation():
    orbit = _orbit((_E0, _OBL0, _W0))
    # NH summer solstice (day 172): North pole = polar day (24 h sun) ⇒ Q>0;
    # South pole = polar night ⇒ Q=0.
    q_np = float(daily_mean_insolation(jnp.asarray(math.radians(89.9)), 172.0, orbit=orbit))
    q_sp = float(daily_mean_insolation(jnp.asarray(math.radians(-89.9)), 172.0, orbit=orbit))
    assert q_np > 0.0
    assert q_sp == 0.0
    # daylight fraction saturates to 1 (polar day) / 0 (polar night).
    assert float(daylight_fraction(jnp.asarray(math.radians(89.9)), 172.0, orbit=orbit)) > 0.999
    assert float(daylight_fraction(jnp.asarray(math.radians(-89.9)), 172.0, orbit=orbit)) < 1e-3


def test_polar_day_clipped_hour_angle_pinned_and_departure_bounded():
    # DEPARTURE #1: deep in polar day (well inside the pole but NOT within the
    # _POLE_THRESHOLD shell, so the CLIP — not the pole_fill — acts) cos h_s
    # saturates to -(1-1e-7), and the module evaluates the daily-mean formula at
    # the CLIPPED sunset angle h_s = arccos(-(1-1e-7)) = π - √(2·1e-7).  Pin that
    # exact clipped computation to round-off, then bound the shortfall from the
    # ideal 24 h insolation S₀ sinφ sinδ (a/r)².  The shortfall is
    # ~(S₀/π)·√(2·1e-7)·sinφ sinδ ≈ 0.074 W/m² here — the DOMINANT term is the
    # h_s ANGULAR error √(2ε), NOT the cos-clip magnitude ε (so it is ~0.07, not
    # the (S₀/π)·ε ≈ 4e-5 a naive reading would give; still physically tiny).
    orbit = (_E0, _OBL0, _W0)
    lat = math.radians(89.9)
    day = 172.0
    assert math.cos(lat) * math.cos(_decl_orbital_oracle(day, orbit)) > solar._POLE_THRESHOLD
    delta = _decl_orbital_oracle(day, orbit)
    dist = _dist_factor_oracle(day, orbit[0], orbit[2])
    h_s_clip = math.acos(-(1.0 - 1.0e-7))    # cos h_s saturated to -(1-1e-7)
    q_clip = (constants.S_0 / math.pi) * (
        h_s_clip * math.sin(lat) * math.sin(delta)
        + math.cos(lat) * math.cos(delta) * math.sin(h_s_clip)) * dist
    got = float(daily_mean_insolation(jnp.asarray(lat), day, orbit=_orbit(orbit)))
    assert got == pytest.approx(q_clip, rel=1e-12)          # faithful to the clipped formula
    q_24 = constants.S_0 * math.sin(lat) * math.sin(delta) * dist
    shortfall = q_24 - got
    assert 1.0e-3 < shortfall < (constants.S_0 / math.pi) * math.sqrt(2.0e-7)


# ===================== differentiability =====================

def test_grad_daily_mean_finite_and_correct_interior():
    orbit = _orbit((_E0, _OBL0, _W0))

    def q_of_lat(lat):
        return jnp.sum(daily_mean_insolation(lat, 172.0, orbit=orbit))

    lat = jnp.deg2rad(jnp.asarray(_INTERIOR_LATS))
    g = jax.grad(q_of_lat)(lat)
    assert np.all(np.isfinite(np.asarray(g)))
    # central FD check on one interior latitude
    eps = 1e-6
    lat0 = math.radians(30.0)
    fp = float(daily_mean_insolation(jnp.asarray(lat0 + eps), 172.0, orbit=orbit))
    fm = float(daily_mean_insolation(jnp.asarray(lat0 - eps), 172.0, orbit=orbit))
    g0 = float(jax.grad(lambda x: daily_mean_insolation(x, 172.0, orbit=orbit))(jnp.asarray(lat0)))
    assert g0 == pytest.approx((fp - fm) / (2.0 * eps), rel=1e-5)


def test_grad_finite_through_polar_branch():
    orbit = _orbit((_E0, _OBL0, _W0))
    # Includes the poles; the where-before-divide must keep the gradient finite.
    lat = jnp.deg2rad(jnp.linspace(-90.0, 90.0, 37))
    g = jax.grad(lambda x: jnp.sum(daily_mean_insolation(x, 172.0, orbit=orbit)))(lat)
    assert np.all(np.isfinite(np.asarray(g)))
