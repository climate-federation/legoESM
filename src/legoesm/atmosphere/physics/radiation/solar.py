"""Solar geometry for atmospheric radiation.

Provides functions to compute:
- Solar declination angle from day of year
- Cosine of the solar zenith angle
- Daily-mean insolation
- Perpetual-equinox (annual+daily mean) insolation

All functions are pure JAX and compatible with jit/grad.
"""

from __future__ import annotations

import jax.numpy as jnp

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)
# Pole-detection threshold on ``cos_lat * cos_delta``.  Must exceed the
# fp32 roundoff of ``cos(π/2) ≈ -4.37e-8`` so that any latitude
# numerically reaching the pole is treated as the singular branch and
# routed through the AD-safe fill (issue #249 codex round 5).  At the
# minimum tropical ``cos_delta ≈ 0.91``, this corresponds to a
# latitude shell of width ``arccos(1 − 1e-7) ≈ 4.5e-4 rad ≈ 0.025°``
# around each pole — far below any physical or grid resolution.
_POLE_THRESHOLD = 1.0e-7

from legoesm import constants


def solar_declination(day_of_year: float, obliquity: float = 23.45) -> float:
    """Compute solar declination angle.

    delta = obliquity * sin(2*pi * (day_of_year - 80) / 365)

    This is the simplified "Spencer" formula approximation.

    Parameters
    ----------
    day_of_year : float
        Day of year (1-365).
    obliquity : float
        Obliquity of the ecliptic [degrees].

    Returns
    -------
    float
        Solar declination [radians].
    """
    obliquity_rad = obliquity * constants.DEG_TO_RAD
    return obliquity_rad * jnp.sin(2.0 * jnp.pi * (day_of_year - 80.0) / 365.0)


def cos_zenith_angle(
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    day_of_year: float,
    hour: float,
    obliquity: float = 23.45,
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

    Returns
    -------
    jnp.ndarray
        Cosine of zenith angle, clipped to [-1, 1].
    """
    delta = solar_declination(day_of_year, obliquity)
    # Hour angle: h = 2*pi * (hour/24 + lon/(2*pi)) - pi
    h = 2.0 * jnp.pi * (hour / 24.0) + lon - jnp.pi

    cos_z = (
        jnp.sin(lat) * jnp.sin(delta)
        + jnp.cos(lat) * jnp.cos(delta) * jnp.cos(h)
    )
    return jnp.clip(cos_z, -1.0, 1.0)


def daily_mean_insolation(
    lat: jnp.ndarray,
    day_of_year: float,
    S_0: float = constants.S_0,
    obliquity: float = 23.45,
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

    Returns
    -------
    jnp.ndarray
        Daily-mean TOA insolation [W/m^2].
    """
    delta = solar_declination(day_of_year, obliquity)
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
    # ``clip`` itself, JAX evaluates 0 · ∞ = NaN.  An ε = 1e-7 cap shrinks
    # the daily-mean insolation by at most ``(S_0/π) · ε ≈ 4e-5 W/m²`` at
    # the polar-day boundary — far below any physical or observational
    # threshold — while keeping ``|d arccos / dx| ≤ 1/sqrt(2ε) ≈ 2236``,
    # safely within fp64 and fp32 dynamic range.
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
    # In the pole branch, drive ``cos_hs`` to the saturation bound from
    # the *correct* side using the numerator's sign:
    #   numerator > 0 → polar night (``cos_hs → +bound``)
    #   numerator < 0 → polar day  (``cos_hs → -bound``)
    pole_fill = jnp.where(
        numerator > 0.0,
        jnp.asarray(1.0e30, denom.dtype),
        jnp.asarray(-1.0e30, denom.dtype),
    )
    ratio = jnp.where(near_pole, pole_fill, ratio_div)
    cos_hs = jnp.clip(ratio, -_AD_SAFE_BOUND, _AD_SAFE_BOUND)
    h_s = jnp.arccos(cos_hs)

    # Daily-mean insolation
    Q = (S_0 / jnp.pi) * (
        h_s * sin_lat * sin_delta
        + cos_lat * cos_delta * jnp.sin(h_s)
    )
    return jnp.maximum(Q, 0.0)


def daylight_fraction(
    lat: jnp.ndarray,
    day_of_year: float,
    obliquity: float = 23.45,
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

    Returns
    -------
    jnp.ndarray
        Daylight fraction, same shape as lat.
    """
    delta = solar_declination(day_of_year, obliquity)
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
        jnp.asarray(-1.0e30, denom.dtype),
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
