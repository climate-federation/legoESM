"""Solar geometry for atmospheric radiation.

Provides functions to compute:
- Solar declination angle from day of year
- Cosine of the solar zenith angle
- Daily-mean insolation
- Perpetual-equinox (annual+daily mean) insolation

All functions are pure JAX and compatible with jit/grad.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)
# Pole-detection threshold on ``cos_lat * cos_delta``.  Must exceed the
# fp32 roundoff of ``cos(π/2) ≈ -4.37e-8`` so that any latitude
# numerically reaching the pole is treated as the singular branch and
# routed through the AD-safe fill (issue #249 codex round 5).  The
# threshold is on the PRODUCT ``cos_lat cos_delta``: at the minimum
# tropical ``cos_delta ≈ 0.91`` it fires for ``cos_lat < 1e-7/0.91 ≈
# 1.1e-7``, i.e. a latitude shell of width ``≈ 1.1e-7 rad ≈ 6e-6°``
# around each pole — far below any physical or grid resolution.
_POLE_THRESHOLD = 1.0e-7

from legoesm import constants


# Earth orbital / calendar constants (fixed).
_EARTH_OBLIQUITY_DEG = 23.45
# Day of the vernal equinox in a 365-day no-leap calendar (≈ March 21).
# The simplified declination crosses zero here; the orbital longitude is
# measured from this point.
_SOLSTICE_OFFSET_DAYS = 80.0
_DAYS_PER_YEAR = 365.0

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "Solar geometry / insolation: solar declination, cosine of the zenith "
        "angle, exact daily-mean and perpetual-equinox TOA insolation, and the "
        "Berger (1978) eccentric-orbit Earth-Sun distance factor."
    ),
    "inputs": {
        "lat": "rad", "lon": "rad", "day_of_year": "day (1-365)",
        "hour": "h (UTC, 0-24)", "S_0": "W/m^2", "obliquity": "degree",
    },
    "outputs": {
        "insolation": "W/m^2 (TOA daily-mean / perpetual-equinox)",
        "cos_zenith": "1 (cosine of solar zenith angle)",
        "declination": "rad", "distance_factor": "1 ((a/r)^2)",
        "daylight_fraction": "1",
    },
    "sign_convention": (
        "insolation>=0 (polar night floored to 0); cos_zenith in [-1,1] "
        "(negative = sun below the horizon); declination in "
        "[-obliquity,+obliquity]; the Earth-Sun distance factor (a/r)^2>0 with "
        "annual mean ~1. Diagnostic geometry only: nothing is added to or "
        "removed from any budget."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Berger (1978), J. Atmos. Sci. 35, 2362-2367 (orbital elements); "
        "Cooper (1969) declination; Frierson et al. (2006) equinox insolation."
    ),
    "idealized_test": (
        "tests/unit/test_solar_orbital.py + "
        "tests/unit/test_orbital_insolation_pipeline.py: equinox gives "
        "Q=(S_0/pi)cos(lat); global-annual-mean (a/r)^2 = 1; AD-finite "
        "gradients through the polar-day/night branch."
    ),
}


class OrbitalParameters(NamedTuple):
    """Earth orbital elements for realistic (non-circular) insolation.

    Berger (1978) convention (as in CESM ``shr_orb_mod`` / climlab): the
    longitude of perihelion is measured from the moving vernal equinox.
    Defaults are the present-day (≈ year 2000) values from
    ``legoesm.constants``.  ``eccentricity = 0`` recovers a circular orbit
    (the AMIP-II realistic insolation uses the present-day ellipse).
    """

    eccentricity: float = constants.orbital_eccentricity
    obliquity_deg: float = constants.orbital_obliquity_deg
    long_perihelion_deg: float = constants.orbital_long_perihelion_deg


def earth_orbit() -> "OrbitalParameters":
    """Present-day Earth orbit (convenience constructor)."""
    return OrbitalParameters()


def _orbital_solar_longitude(
    day_of_year: float, orbit: OrbitalParameters
) -> jnp.ndarray:
    """True solar ecliptic longitude lambda [rad], measured from the
    vernal equinox.

    Berger (1978) eccentricity series: the series COEFFICIENTS match CESM
    ``shr_orb_mod`` / climlab ``solar_longitude``, while the vernal-equinox
    calendar day (``_SOLSTICE_OFFSET_DAYS = 80``) follows climlab's convention
    (CESM/CLM5 document ``d_ve = 80.5`` — a ~½-day calendar offset).  Captures
    the non-uniform apparent motion of the Sun caused by the elliptical orbit,
    which the simple ``obliquity·sin`` declination ignores.
    """
    e = orbit.eccentricity
    w = orbit.long_perihelion_deg * constants.DEG_TO_RAD
    delta_lambda = (day_of_year - _SOLSTICE_OFFSET_DAYS) * (
        2.0 * jnp.pi / _DAYS_PER_YEAR
    )
    beta = jnp.sqrt(1.0 - e * e)
    # Mean longitude at the vernal equinox (series in e); Berger (1978).
    lambda_m0 = -2.0 * (
        (e / 2.0 + e ** 3 / 8.0) * (1.0 + beta) * jnp.sin(-w)  # coeff-ok: Berger 1978 series
        - e ** 2 / 4.0 * (0.5 + beta) * jnp.sin(-2.0 * w)
        + e ** 3 / 8.0 * (1.0 / 3.0 + beta) * jnp.sin(-3.0 * w)  # coeff-ok: Berger 1978 series
    )
    lambda_m = lambda_m0 + delta_lambda
    lam = (
        lambda_m
        + (2.0 * e - e ** 3 / 4.0) * jnp.sin(lambda_m - w)
        + (5.0 / 4.0) * e ** 2 * jnp.sin(2.0 * (lambda_m - w))  # coeff-ok: Berger 1978 series
        + (13.0 / 12.0) * e ** 3 * jnp.sin(3.0 * (lambda_m - w))  # coeff-ok: Berger 1978 series
    )
    return lam


def earth_sun_distance_factor(
    day_of_year: float, orbit: OrbitalParameters
) -> jnp.ndarray:
    """``(a/r)^2`` Earth-Sun distance factor that scales TOA insolation.

    From the orbit equation ``r = a(1−e²)/(1 + e cos nu)`` with true anomaly
    ``nu = lambda − varpi``:  ``(a/r)² = ((1 + e cos nu)/(1 − e²))²``.  The
    annual mean is ≈ 1 (orbital forcing redistributes insolation seasonally
    without changing the global-annual total); it ranges from ≈ 1/(1−e)² at
    perihelion (early January) to ≈ 1/(1+e)² at aphelion (early July).
    """
    e = orbit.eccentricity
    w = orbit.long_perihelion_deg * constants.DEG_TO_RAD
    lam = _orbital_solar_longitude(day_of_year, orbit)
    nu = lam - w
    return ((1.0 + e * jnp.cos(nu)) / (1.0 - e * e)) ** 2


def solar_declination(
    day_of_year: float,
    obliquity: float = _EARTH_OBLIQUITY_DEG,
    orbit: OrbitalParameters | None = None,
) -> float:
    """Compute solar declination angle.

    With ``orbit=None`` (default) uses the simplified "Spencer" formula
    ``delta = obliquity * sin(2*pi * (day_of_year - 80) / 365)``.

    The circular-orbit form above is the Cooper (1969) single-harmonic
    declination approximation.  With an ``OrbitalParameters`` the realistic
    Berger (1978) declination ``sin(delta) = sin(obliquity) * sin(lambda)`` is
    used instead, where ``lambda`` is the true solar longitude (the obliquity
    is then taken from the orbit, not the ``obliquity`` argument).

    Parameters
    ----------
    day_of_year : float
        Day of year (1-365).
    obliquity : float
        Obliquity of the ecliptic [degrees] (ignored when ``orbit`` is set).
    orbit : OrbitalParameters or None
        Realistic Earth orbit; ``None`` ⇒ circular-orbit approximation.

    Returns
    -------
    float
        Solar declination [radians].
    """
    if orbit is not None:
        eps = orbit.obliquity_deg * constants.DEG_TO_RAD
        lam = _orbital_solar_longitude(day_of_year, orbit)
        # |sin(eps) sin(lam)| <= sin(23.4 deg) ≈ 0.40, so arcsin is well
        # inside (-1, 1): no AD-singular clip needed.
        return jnp.arcsin(jnp.sin(eps) * jnp.sin(lam))
    obliquity_rad = obliquity * constants.DEG_TO_RAD
    return obliquity_rad * jnp.sin(2.0 * jnp.pi * (day_of_year - _SOLSTICE_OFFSET_DAYS) / _DAYS_PER_YEAR)


def cos_zenith_angle(
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    day_of_year: float,
    hour: float,
    obliquity: float = _EARTH_OBLIQUITY_DEG,
    orbit: OrbitalParameters | None = None,
) -> jnp.ndarray:
    """Compute cosine of the solar zenith angle.

    cos(theta_z) = sin(lat)*sin(delta) + cos(lat)*cos(delta)*cos(h)

    where delta is solar declination and h is the hour angle.

    Parameters
    ----------
    lat : jnp.ndarray
        Latitude [radians].
    lon : jnp.ndarray
        Longitude [radians].
    day_of_year : float
        Day of year (1-365).
    hour : float
        UTC hour (0-24).
    obliquity : float
        Obliquity [degrees].
    orbit : OrbitalParameters or None
        Realistic Earth orbit for the declination; ``None`` ⇒ circular.
        The Earth-Sun distance (1/r²) factor is NOT applied here — it scales
        the incoming flux, so callers multiply ``S_0`` by
        :func:`earth_sun_distance_factor`.

    Returns
    -------
    jnp.ndarray
        Cosine of zenith angle, clipped to [-1, 1].
    """
    delta = solar_declination(day_of_year, obliquity, orbit=orbit)
    # Hour angle: h = 2*pi * (hour/24 + lon/(2*pi)) - pi
    h = 2.0 * jnp.pi * (hour / 24.0) + lon - jnp.pi  # coeff-ok: hours/day diurnal phase

    cos_z = (
        jnp.sin(lat) * jnp.sin(delta)
        + jnp.cos(lat) * jnp.cos(delta) * jnp.cos(h)
    )
    return jnp.clip(cos_z, -1.0, 1.0)


def daily_mean_insolation(
    lat: jnp.ndarray,
    day_of_year: float,
    S_0: float = constants.S_0,
    obliquity: float = _EARTH_OBLIQUITY_DEG,
    orbit: OrbitalParameters | None = None,
) -> jnp.ndarray:
    """Compute daily-mean insolation at the top of atmosphere.

    Uses the exact integral over the diurnal cycle:

        Q = (S_0 / pi) * (h_s * sin(lat)*sin(delta) + cos(lat)*cos(delta)*sin(h_s))

    where h_s is the sunset hour angle:

        cos(h_s) = -tan(lat) * tan(delta)

    Polar day (24h sun) and polar night (0h sun) are handled.

    Parameters
    ----------
    lat : jnp.ndarray
        Latitude [radians].
    day_of_year : float
        Day of year (1-365).
    S_0 : float
        Total solar irradiance [W/m^2].
    obliquity : float
        Obliquity [degrees].
    orbit : OrbitalParameters or None
        Realistic Earth orbit; ``None`` ⇒ circular orbit (no eccentricity).
        When set, the declination is the Berger (1978) orbital declination
        AND the result is scaled by the Earth-Sun distance factor
        ``(a/r)^2`` (:func:`earth_sun_distance_factor`), so the seasonal
        perihelion/aphelion asymmetry is captured.

    Returns
    -------
    jnp.ndarray
        Daily-mean TOA insolation [W/m^2].
    """
    delta = solar_declination(day_of_year, obliquity, orbit=orbit)
    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)
    sin_delta = jnp.sin(delta)
    cos_delta = jnp.cos(delta)

    # Sunset hour angle.
    # cos(h_s) = -tan(lat) * tan(delta), clipped for polar day / night.
    # AD-safe arccos: clip *strictly inside* (-1, 1) so the derivative
    # d/dx arccos(x) = -1/sqrt(1 - x²) stays finite at the boundary.
    # Clipping exactly to ±1 makes the next ``arccos`` blow up to ±∞ on
    # the backward pass; combined with the zero subgradient through the
    # ``clip`` itself, JAX evaluates 0 · ∞ = NaN.  The ε = 1e-7 cap caps the
    # sunset angle at ``√(2ε) ≈ 4.5e-4`` from 0 (or ``π − √(2ε)`` from π deep in
    # polar day).  DEEP in polar day the shrink of the daily-mean insolation is
    # ``≈ (S_0/π)·√(2ε)·(a/r)²·sinφ sinδ`` (dominated by the h_s ANGULAR error
    # √(2ε), NOT the cos-clip magnitude ε) — for present-day S_0≈1361 that is
    # ≲ 0.2 W/m² (S_0-/orbit-dependent).  AT the exact polar-day boundary
    # (sinφ sinδ = cosφ cosδ) it collapses to the far smaller
    # ``(S_0/π)(a/r)² sinφ sinδ·(α−sin α), α=√(2ε)``, i.e. O(ε^{3/2}) ~ 2e-9 W/m².
    # Either way far below any physical or observational threshold, while keeping
    # ``|d arccos / dx| ≤ 1/sqrt(2ε) ≈ 2236``, safely in fp64/fp32 dynamic range.
    _AD_SAFE_BOUND = 1.0 - 1.0e-7
    # AD-safe polar singularity (issue #249 codex round 5).  The legacy
    # ``-sin_lat sin_delta / clip(cos_lat cos_delta, _TINY, None)`` form
    # was AD-safe at *equinox* (where ``sin_delta = 0`` makes the
    # numerator vanish too) but produced NaN gradients at solstice in
    # fp32 — the divide's ``-num / denom**2`` cotangent at
    # ``denom = _TINY ≈ 1e-38`` reaches ``∼1e76``, then chains through
    # the outer ``arccos`` saturation and ``maximum(Q, 0)`` reductions
    # before the saturation clip can kill it.  Replace with a
    # where-before-divide that masks the pole branch out of the divide
    # entirely and routes the polar-day / polar-night branch through a
    # numerator-sign-driven fill that the outer clip then saturates to
    # ``±_AD_SAFE_BOUND``.
    numerator = -sin_lat * sin_delta
    denom = cos_lat * cos_delta
    near_pole = denom < _POLE_THRESHOLD
    safe_denom = jnp.where(near_pole, jnp.asarray(1.0, denom.dtype), denom)
    ratio_div = numerator / safe_denom
    # In the pole branch, drive ``cos_hs`` from the *correct* side using the
    # numerator's sign:
    #   numerator > 0 → polar night (``cos_hs → +bound``)
    #   numerator < 0 → polar day  (``cos_hs → -bound``)
    #   numerator = 0 → equinox at the pole (sin_delta = 0): ``cos_hs = 0``
    #                   (h_s = π/2, daylight 0.5) — the grazing-sun limit.
    pole_fill = jnp.where(
        numerator > 0.0,
        jnp.asarray(1.0e30, denom.dtype),
        jnp.where(
            numerator < 0.0,
            jnp.asarray(-1.0e30, denom.dtype),
            # numerator == 0 (sin_delta = 0, i.e. equinox AT the pole): the sun
            # grazes the horizon all day ⇒ cos_hs = 0, h_s = π/2 (daylight 0.5),
            # NOT the polar-day/night saturation the 2-way tie would pick.
            jnp.asarray(0.0, denom.dtype),
        ),
    )
    ratio = jnp.where(near_pole, pole_fill, ratio_div)
    cos_hs = jnp.clip(ratio, -_AD_SAFE_BOUND, _AD_SAFE_BOUND)
    h_s = jnp.arccos(cos_hs)

    # Daily-mean insolation
    Q = (S_0 / jnp.pi) * (
        h_s * sin_lat * sin_delta
        + cos_lat * cos_delta * jnp.sin(h_s)
    )
    # Eccentricity: scale by the Earth-Sun distance factor (a/r)^2.  The
    # factor is positive, so it commutes with the polar-night floor below.
    if orbit is not None:
        Q = Q * earth_sun_distance_factor(day_of_year, orbit)
    return jnp.maximum(Q, 0.0)


def daylight_fraction(
    lat: jnp.ndarray,
    day_of_year: float,
    obliquity: float = _EARTH_OBLIQUITY_DEG,
    orbit: OrbitalParameters | None = None,
) -> jnp.ndarray:
    """Fraction of the day with sunlight (h_s / pi).

    Returns a value in [0, 1] where 0 = polar night and 1 = polar day.
    At equinox, returns 0.5 everywhere.

    Parameters
    ----------
    lat : jnp.ndarray
        Latitude [radians].
    day_of_year : float
        Day of year (1-365).
    obliquity : float
        Obliquity [degrees].
    orbit : OrbitalParameters or None
        Realistic Earth orbit for the declination; ``None`` ⇒ circular.

    Returns
    -------
    jnp.ndarray
        Daylight fraction, same shape as lat.
    """
    delta = solar_declination(day_of_year, obliquity, orbit=orbit)
    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)
    sin_delta = jnp.sin(delta)
    cos_delta = jnp.cos(delta)

    # Same AD-safe arccos pattern as in ``daily_mean_insolation`` — keep
    # the bound strictly inside ±1 so the polar-day / polar-night
    # gradient does not NaN.
    _AD_SAFE_BOUND = 1.0 - 1.0e-7
    # Same where-before-divide pattern as ``daily_mean_insolation`` —
    # required for AD-safe gradients at fp32 poles on solstice (issue
    # #249 codex round 5).  See that function for the derivation.
    numerator = -sin_lat * sin_delta
    denom = cos_lat * cos_delta
    near_pole = denom < _POLE_THRESHOLD
    safe_denom = jnp.where(near_pole, jnp.asarray(1.0, denom.dtype), denom)
    ratio_div = numerator / safe_denom
    pole_fill = jnp.where(
        numerator > 0.0,
        jnp.asarray(1.0e30, denom.dtype),
        jnp.where(
            numerator < 0.0,
            jnp.asarray(-1.0e30, denom.dtype),
            # numerator == 0 (sin_delta = 0, i.e. equinox AT the pole): the sun
            # grazes the horizon all day ⇒ cos_hs = 0, h_s = π/2 (daylight 0.5),
            # NOT the polar-day/night saturation the 2-way tie would pick.
            jnp.asarray(0.0, denom.dtype),
        ),
    )
    ratio = jnp.where(near_pole, pole_fill, ratio_div)
    cos_hs = jnp.clip(ratio, -_AD_SAFE_BOUND, _AD_SAFE_BOUND)
    h_s = jnp.arccos(cos_hs)
    return h_s / jnp.pi


def perpetual_equinox_insolation(
    lat: jnp.ndarray,
    S_0: float = constants.S_0,
) -> jnp.ndarray:
    """Compute annual+daily mean insolation at perpetual equinox.

    At equinox (delta=0), the daily-mean insolation simplifies to:

        Q(lat) = (S_0 / pi) * cos(lat)   for |lat| < pi/2

    This is the standard forcing for idealized aquaplanet experiments
    (Frierson et al. 2006).

    Parameters
    ----------
    lat : jnp.ndarray
        Latitude [radians].
    S_0 : float
        Total solar irradiance [W/m^2].

    Returns
    -------
    jnp.ndarray
        Perpetual-equinox insolation [W/m^2].
    """
    return (S_0 / jnp.pi) * jnp.maximum(jnp.cos(lat), 0.0)
