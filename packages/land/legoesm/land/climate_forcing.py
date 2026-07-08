"""Climate-feature single-column forcing generator for offline land runs.

Generalises the idealised annual-cycle forcing used by LMIP: instead of
hard-wiring a latitude -> (mean T, seasonal amplitude, insolation) mapping,
this module takes those climate features directly as arguments, so any
climate archetype (a global carbon initial-condition map, a site
reanalysis-derived climatology, ...) can drive the same seasonal +
diurnal T/precip/humidity/solar machinery.

``legoesm.land.lmip_forcing.make_synthetic_lmip_forcing`` is the latitude
preset: it derives ``(mat_k, t_seasonal_amp_k, sw_mean_w)`` from latitude
using its own solar-geometry / latitudinal-baseline numerics, then
delegates here for the T/precip/humidity/solar body -- no duplicated
numerics between the two.

JAX-traceable: ``doy`` / ``hour`` may be traced scalars (safe inside a
``lax.scan`` time loop); ``mat_k`` / ``t_seasonal_amp_k`` / ``sw_mean_w`` /
``precip_rate`` are typically static per-column climate features.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.thermo import saturation_mixing_ratio

# --- idealised diurnal/seasonal shape constants ---
_T_DIURNAL_AMP_K = 3.0         # default diurnal half-amplitude [K]
_T_SEASONAL_PEAK_DAY = 200.0   # NH day-of-year of the seasonal T maximum (~July)
_YEAR_DAYS = 365.0             # calendar year length used by the cycles [days]
_HOURS_PER_DAY = 24.0          # [hour/day]
_NOON_HOUR = 12.0              # local solar noon [hour]
_DIURNAL_PEAK_HOUR = 14.0      # local hour of the diurnal T maximum

# --- radiative / humidity / precip constants ---
_LW_EFF_EMISSIVITY = 0.75      # effective clear-sky down-LW emissivity [-]
_RH_FRACTION = 0.6             # prescribed near-surface relative humidity [-]
_SNOW_RAIN_THRESHOLD_K = 275.0  # air T below which precipitation falls as snow [K]

# --- prescribed column state ---
_P_SURFACE_PA = 1.0e5          # prescribed surface pressure [Pa]
_P_LOWEST_PA = 9.5e4           # prescribed lowest-model-level pressure [Pa]
_RHO_LOWEST = 1.2              # prescribed near-surface air density [kg/m3]
_U_LOWEST = 3.0                # prescribed zonal wind [m/s]
_V_LOWEST = 2.0                # prescribed meridional wind [m/s]
_CO2_PPMV = 412.0              # prescribed atmospheric CO2 [ppmv]

# Peak-to-mean insolation shape factor for a sinusoidal daytime half-cycle:
# a rectified cosine over one day (positive half only) has mean = peak/pi,
# so peak = pi * mean.
_SW_PEAK_OVER_MEAN = jnp.pi


def make_climatological_forcing(
    mat_k,
    t_seasonal_amp_k,
    sw_mean_w,
    precip_rate,
    doy,
    hour,
    *,
    t_diurnal_amp_k=_T_DIURNAL_AMP_K,
    dtype=jnp.float64,
) -> AtmToSurface:
    """Construct single-column atmospheric forcing from climate features.

    Parameters
    ----------
    mat_k : float
        Mean-annual near-surface air temperature [K].
    t_seasonal_amp_k : float
        Half-amplitude of the annual T cycle [K] (NH-phased: peak at
        ``doy = _T_SEASONAL_PEAK_DAY``).
    sw_mean_w : float
        Daily-mean downward shortwave [W/m^2] for the forced ``doy``; scales the
        idealised diurnal insolation shape so its DAILY mean equals this value.
        There is no internal seasonal SW term, so a value held constant across
        the year is the annual mean, whereas a per-day daily-mean produces a
        seasonal SW cycle (see
        ``lmip_forcing.make_synthetic_lmip_forcing``) (F9).
    precip_rate : float
        Constant precipitation rate [kg/m2/s].
    doy : float
        Day of year in [0, 365); may be a traced scalar.
    hour : float
        Local solar hour of day in [0, 24); may be a traced scalar.
    t_diurnal_amp_k : float
        Diurnal T half-amplitude [K], peak at local hour
        ``_DIURNAL_PEAK_HOUR``.
    dtype :
        JAX dtype (default float64).

    Returns
    -------
    AtmToSurface
        Forcing with shape (1,) for all fields.

    Notes
    -----
    Atmospheric temperature is ``mat_k`` plus a seasonal term (amplitude
    ``t_seasonal_amp_k``, NH peak at ``doy~200``) plus a diurnal term
    (amplitude ``t_diurnal_amp_k``, peak at local hour 14). Downward
    shortwave follows a rectified-cosine daytime shape peaking at local
    solar noon, scaled so its daily mean equals ``sw_mean_w``.
    """
    # Daytime insolation shape: rectified cosine peaking at local solar
    # noon, scaled so its daily mean equals sw_mean_w.
    day_phase = jnp.cos(2.0 * jnp.pi * (hour - _NOON_HOUR) / _HOURS_PER_DAY)
    sw = jnp.maximum(_SW_PEAK_OVER_MEAN * sw_mean_w * day_phase, 0.0)
    sw_down = jnp.asarray([sw], dtype=dtype)

    T_season = t_seasonal_amp_k * jnp.cos(
        2.0 * jnp.pi * (doy - _T_SEASONAL_PEAK_DAY) / _YEAR_DAYS
    )
    T_diurnal = t_diurnal_amp_k * jnp.cos(
        2.0 * jnp.pi * (hour - _DIURNAL_PEAK_HOUR) / _HOURS_PER_DAY
    )
    T_atm = jnp.asarray([mat_k + T_season + T_diurnal], dtype=dtype)

    # --- LW down: effective emissivity ~0.75 of blackbody ---
    lw_down = jnp.asarray(
        [_LW_EFF_EMISSIVITY * constants.sigma_sb * T_atm[0] ** 4], dtype=dtype
    )

    # --- Humidity: prescribed RH using the model's saturation_mixing_ratio ---
    p_sfc = jnp.asarray([_P_SURFACE_PA], dtype=dtype)
    q_sat = saturation_mixing_ratio(T_atm, p_sfc)
    q_atm = (_RH_FRACTION * q_sat).astype(dtype)

    # --- Precipitation: rain below the snow threshold becomes snow ---
    precip_total = jnp.asarray([precip_rate], dtype=dtype)
    precip_snow = jnp.where(
        T_atm < _SNOW_RAIN_THRESHOLD_K,
        precip_total,
        jnp.zeros(1, dtype=dtype),
    )

    return AtmToSurface(
        sw_down=sw_down,
        lw_down=lw_down,
        precip_total=precip_total,
        precip_snow=precip_snow,
        T_lowest=T_atm,
        q_lowest=q_atm,
        u_lowest=jnp.asarray([_U_LOWEST], dtype=dtype),
        v_lowest=jnp.asarray([_V_LOWEST], dtype=dtype),
        p_lowest=jnp.asarray([_P_LOWEST_PA], dtype=dtype),
        p_surface=p_sfc,
        rho_lowest=jnp.asarray([_RHO_LOWEST], dtype=dtype),
        cos_zenith=jnp.asarray([jnp.maximum(day_phase, 0.0)], dtype=dtype),
        co2_ppmv=jnp.asarray([_CO2_PPMV], dtype=dtype),
        has_radiation=jnp.ones(1, dtype=dtype),
        has_precipitation=jnp.ones(1, dtype=dtype),
    )
