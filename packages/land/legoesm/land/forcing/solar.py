"""Shared solar-zenith geometry for the land forcing path.

Canonical ``cos(solar zenith angle)`` used by BOTH the CRU-JRA variable map
(:func:`legoesm.land.forcing.cru_jra.forcing_to_atm_surface`) and the M2
shortwave 6h->model-dt disaggregation.  Factored here so the
declination / hour-angle formula is written exactly once (CLAUDE.md: no
re-derived solar geometry).  This mirrors the inline formula that previously
lived only in ``scripts/run/run_lmip_smoke.py:make_global_forcing``; that
copy should be unified onto this helper in a later cleanup.

Convention: ``doy`` is day-of-year on a 365-day (noleap) calendar to match
the CRU-JRA / CLM datm forcing; ``hour`` is UTC hour-of-day in ``[0, 24)``
and longitude supplies the local-solar-time offset.
"""

from __future__ import annotations

import jax.numpy as jnp

# --- solar geometry (astronomical constants) ---
_AXIAL_TILT_DEG = 23.45      # Earth obliquity used in the declination approximation
_EQUINOX_DOY = 80.0          # ~vernal-equinox day-of-year offset in the declination sine
_DAYS_PER_YEAR = 365.0       # noleap calendar (CRU-JRA / CLM datm)
_DEG_PER_HOUR = 15.0         # 360 deg / 24 h Earth rotation
_SOLAR_NOON_HOUR = 12.0      # local solar noon
_RAD_PER_DEG = jnp.pi / 180.0


def cos_solar_zenith(lat_rad, lon_rad, doy, hour):
    """Cosine of the solar zenith angle, clamped at 0 (night -> 0).

    Parameters
    ----------
    lat_rad, lon_rad : array_like
        Latitude / longitude in **radians** (lon in ``[0, 2*pi)`` or
        ``[-pi, pi)`` — only its degree value enters the local-time offset).
    doy : array_like or float
        Day-of-year (1..365, noleap); fractional values allowed.
    hour : array_like or float
        UTC hour-of-day in ``[0, 24)``.

    Returns
    -------
    array
        ``max(sin(phi) sin(delta) + cos(phi) cos(delta) cos(H), 0)``.
    """
    lat = jnp.asarray(lat_rad)
    lon_deg = jnp.asarray(lon_rad) / _RAD_PER_DEG
    decl = _AXIAL_TILT_DEG * _RAD_PER_DEG * jnp.sin(
        2.0 * jnp.pi * (jnp.asarray(doy) - _EQUINOX_DOY) / _DAYS_PER_YEAR
    )
    local_hour = jnp.asarray(hour) + lon_deg / _DEG_PER_HOUR
    ha = (local_hour - _SOLAR_NOON_HOUR) * _DEG_PER_HOUR * _RAD_PER_DEG
    return jnp.maximum(
        jnp.sin(lat) * jnp.sin(decl) + jnp.cos(lat) * jnp.cos(decl) * jnp.cos(ha),
        0.0,
    )


__all__ = ["cos_solar_zenith"]
