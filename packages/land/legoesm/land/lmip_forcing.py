"""Synthetic single-column atmospheric forcing for offline land runs (LMIP).

Shared by the offline single-point spin-up driver (``scripts/run/run_lmip.py``)
and the land-carbon equilibrium validator
(``scripts/validate/land_carbon_equilibrium.py``) so both drive the land model
with the *same* forcing generator — no duplicated solar-geometry / temperature
numerics.

The forcing is deliberately idealised (latitudinal temperature baseline +
seasonal + diurnal cycles, constant precipitation) so a single latitude fully
determines a repeating annual climate — exactly what an offline equilibrium
spin-up needs.  It is JAX-traceable: ``lat_rad`` / ``lon_rad`` are static
Python floats (one column) while ``day`` / ``hour`` may be traced scalars, so
the generator can be called inside a ``lax.scan`` time loop.

This is the latitude *preset* on top of the general climate-feature
generator: it derives ``(mat, t_seasonal, sw_mean)`` from ``lat_rad`` using
its own latitudinal baseline + solar-geometry daily-mean insolation, then
delegates the T/precip/humidity/solar body to
:func:`legoesm.land.climate_forcing.make_climatological_forcing` — no
duplicated T/precip/humidity numerics between the two.  The point-wise
declination/hour-angle solar geometry is
:func:`legoesm.land.forcing.solar.cos_solar_zenith` — the land package's
own canonical formula (reused here, not re-derived, per CLAUDE.md's
no-duplicated-solar-geometry rule); it is intentionally NOT imported from
``legoesm.atmosphere.physics.radiation.solar`` — the land and atmosphere
components must stay independent of each other (import-linter contract).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.climate_forcing import make_climatological_forcing
from legoesm.land.forcing.solar import cos_solar_zenith

# Latitude-preset shape constants (documented in make_synthetic_lmip_forcing).
_T_BASE_EQUATOR_K = 288.0     # latitudinal-mean surface-air T at the equator [K]
_T_BASE_POLE_DROP_K = 30.0    # equator->pole drop in the annual-mean baseline [K]
_T_SEASONAL_POLE_AMP_K = 15.0  # seasonal half-amplitude at the pole [K]
_DEG_PER_HOUR = 15.0          # Earth rotation: 360 deg / 24 h [deg/hour]
_HOURS_PER_DAY = 24.0         # [hour/day]
# Quadrature resolution (30-min steps) for the daily-mean cos(zenith) used to
# turn S_0 into a mean-annual-cycle downward-SW magnitude. A rectified-cosine
# average is smooth/AD-safe everywhere (unlike a closed-form sunset-hour-angle
# formula, which needs a delicate near-pole arccos guard); at 48 samples/day
# the quadrature error is far below the idealised-forcing precision this
# generator targets.
_N_DAILY_MEAN_HOUR_SAMPLES = 48
_DEFAULT_PRECIP_RATE = 2.0e-5  # default constant precipitation [kg/m2/s]


def _daily_mean_cos_sza(lat_rad, day):
    """Daily mean of the rectified (night-clipped) cosine solar zenith angle.

    Averages :func:`cos_solar_zenith` (already night-clipped) over
    ``_N_DAILY_MEAN_HOUR_SAMPLES`` hours spanning one rotation. Sampled at
    a fixed reference longitude (0.0) rather than the caller's actual
    ``lon_rad``: the mean over a full rotation is exactly longitude-
    invariant in the continuum limit (integrating out any phase shift), so
    anchoring the discretised sample grid at a fixed phase keeps this
    helper's result independent of the caller's site longitude (which
    instead shifts the *diurnal* phase passed separately to
    ``make_climatological_forcing``).
    """
    sample_hours = jnp.linspace(
        0.0, _HOURS_PER_DAY, _N_DAILY_MEAN_HOUR_SAMPLES, endpoint=False
    )
    return jnp.mean(cos_solar_zenith(lat_rad, 0.0, day, sample_hours))


# --- latitude -> climate-feature pieces (the "zonal derivation") --------------
# Factored out of make_synthetic_lmip_forcing so a zonal (latitude-only) monthly
# climatology builder reuses the IDENTICAL MAT / seasonal-amplitude / daily-mean
# insolation numerics instead of re-deriving them (CLAUDE.md: no duplicated
# solar-geometry / temperature numerics).  make_synthetic_lmip_forcing below is
# the single-column caller; a global carbon-IC driver is the batched caller.
def latitude_mean_annual_temp_k(lat_rad):
    """Mean-annual near-surface air temperature [K] from latitude.

    ``mat = _T_BASE_EQUATOR_K - _T_BASE_POLE_DROP_K * |phi| / (pi/2)`` -- the
    latitudinal annual-mean baseline (288 K at the equator, dropping 30 K to the
    pole).  Vectorised (accepts a scalar or an array of latitudes [rad]).
    """
    return (_T_BASE_EQUATOR_K
            - _T_BASE_POLE_DROP_K * jnp.abs(lat_rad) / (jnp.pi / 2.0))


def latitude_seasonal_amp_k(lat_rad):
    """Seasonal half-amplitude of the annual T cycle [K] from latitude.

    ``_T_SEASONAL_POLE_AMP_K * |phi| / (pi/2)`` -- 0 at the equator, max at the
    pole (NH-phased: peak near doy 200 in make_climatological_forcing).
    Vectorised over latitude [rad].
    """
    return _T_SEASONAL_POLE_AMP_K * jnp.abs(lat_rad) / (jnp.pi / 2.0)


def latitude_daily_mean_sw_w(lat_rad, day):
    """Daily-mean downward shortwave [W/m^2] from latitude and day-of-year.

    ``S_0 * <daily-mean cos(zenith)>`` at this latitude/day (see
    :func:`_daily_mean_cos_sza`); longitude-invariant.  ``lat_rad`` scalar,
    ``day`` scalar -- vmap over the batch axis for many (lat, day) pairs.
    """
    return constants.S_0 * _daily_mean_cos_sza(lat_rad, day)


def make_synthetic_lmip_forcing(
    lat_rad: float,
    lon_rad: float,
    day: float,
    hour: float,
    *,
    dtype=jnp.float64,
    precip_rate: float = _DEFAULT_PRECIP_RATE,
) -> AtmToSurface:
    """Construct synthetic single-column atmospheric forcing.

    Parameters
    ----------
    lat_rad : float
        Latitude in radians.
    lon_rad : float
        Longitude in radians (used to compute local solar hour angle).
    day : float
        Day of year [0, 365).
    hour : float
        UTC hour of day [0, 24).
    dtype :
        JAX dtype (default float64).
    precip_rate : float
        Constant precipitation rate [kg/m2/s].

    Returns
    -------
    AtmToSurface
        Forcing with shape (1,) for all fields.

    Notes
    -----
    Delegates to :func:`legoesm.land.climate_forcing.make_climatological_forcing`
    for the T/precip/humidity/solar body. This function only derives the
    climate features from latitude:

    1. Mean-annual T: ``mat = 288 - 30·|φ|/(π/2)`` [K]
    2. Seasonal half-amplitude: ``t_seasonal = 15 K × |φ|/(π/2)`` (NH peak
       at doy≈200, i.e. ``~July``)
    3. This ``day``'s daily-mean downward-SW: ``S_0 × <daily-mean cos(zenith)>``
       at this latitude/``day``, from this module's own solar geometry. The
       delegate has NO seasonal SW term, so passing the per-day daily-mean
       (not an annual mean) is what produces the seasonal SW cycle as ``day``
       advances.

    The diurnal T/SW cycles (±3 K / rectified-cosine shape) are then added
    by the delegate using *local* solar hour (``hour`` shifted by
    ``lon_rad``, computed here). This produces a physically realistic
    annual mean and seasonal cycle across latitudes.  At 45.5°N: T_atm ≈
    265 K (Jan) to 280 K (Jul).
    """
    # Local hour: UTC hour shifted by longitude (Earth rotates
    # _DEG_PER_HOUR degrees of longitude per hour). Drives the delegate's
    # diurnal T/SW phase so a nonzero lon_rad still shifts local solar noon
    # (production single-column runs select real (lat, lon) sites).
    local_hour = hour + jnp.rad2deg(lon_rad) / _DEG_PER_HOUR

    # --- Latitudinal baseline + seasonal amplitude + THIS day's daily-mean SW
    # (the shared latitude->feature pieces; longitude-invariant SW) ---
    mat = latitude_mean_annual_temp_k(lat_rad)
    t_seasonal = latitude_seasonal_amp_k(lat_rad)
    # Contract: make_climatological_forcing's third arg sets the daily mean of
    # the diurnal SW shape for the CURRENT `day` (no internal seasonal SW term),
    # so this per-day daily-mean insolation is what yields the seasonal SW cycle
    # as `day` advances -- it is NOT an annual mean (F9).
    daily_mean_sw_w = latitude_daily_mean_sw_w(lat_rad, day)

    return make_climatological_forcing(
        mat, t_seasonal, daily_mean_sw_w, precip_rate, day, local_hour,
        dtype=dtype,
    )
