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
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.thermo import saturation_mixing_ratio

# Idealised-forcing shape constants (documented in make_synthetic_lmip_forcing).
_T_BASE_EQUATOR_K = 288.0     # latitudinal-mean surface-air T at the equator [K]
_T_BASE_POLE_DROP_K = 30.0    # equator->pole drop in the annual-mean baseline [K]
_T_SEASONAL_POLE_AMP_K = 15.0  # seasonal half-amplitude at the pole [K]
_T_SEASONAL_PEAK_DAY = 200.0  # NH day-of-year of the seasonal maximum (~July)
_T_DIURNAL_AMP_K = 3.0        # diurnal half-amplitude [K]
_T_DIURNAL_PEAK_HOUR = 14.0   # local hour of the diurnal maximum
_DECL_OBLIQUITY_DEG = 23.45   # Earth axial tilt [deg]
_EQUINOX_DAY = 80.0           # day-of-year of the vernal equinox
_YEAR_DAYS = 365.0            # calendar year length used by the cycles [days]
_DEG_PER_HOUR = 15.0          # Earth rotation: 360 deg / 24 h [deg/hour]
_NOON_HOUR = 12.0             # local solar noon [hour]
_HOURS_PER_DAY = 24.0         # [hour/day]
_LW_EFF_EMISSIVITY = 0.75     # effective clear-sky down-LW emissivity [-]
_RH_FRACTION = 0.6            # prescribed near-surface relative humidity [-]
_SNOW_RAIN_THRESHOLD_K = 275.0  # air T below which precipitation falls as snow [K]
_P_SURFACE_PA = 1.0e5         # prescribed surface pressure [Pa]
_P_LOWEST_PA = 9.5e4          # prescribed lowest-model-level pressure [Pa]
_RHO_LOWEST = 1.2             # prescribed near-surface air density [kg/m3]
_U_LOWEST = 3.0               # prescribed zonal wind [m/s]
_V_LOWEST = 2.0               # prescribed meridional wind [m/s]
_CO2_PPMV = 412.0             # prescribed atmospheric CO2 [ppmv]
_DEFAULT_PRECIP_RATE = 2.0e-5  # default constant precipitation [kg/m2/s]


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
    Atmospheric temperature includes three components:
    1. Latitudinal mean: ``T_base = 288 - 30·|φ|/(π/2)``
    2. Seasonal: amplitude ``~15 K × |φ|/(π/2)``, NH peak at doy≈200 (July)
    3. Diurnal: ±3 K, peak at local hour 14
    This produces a physically realistic annual mean and seasonal cycle
    across latitudes.  At 45.5°N: T_atm ≈ 265 K (Jan) to 280 K (Jul).
    """
    half_pi = jnp.pi / 2.0

    # --- Solar geometry ---
    decl_rad = jnp.deg2rad(_DECL_OBLIQUITY_DEG) * jnp.sin(
        2.0 * jnp.pi * (day - _EQUINOX_DAY) / _YEAR_DAYS
    )
    # Local hour angle: UTC hour shifted by longitude (Earth rotates
    # _DEG_PER_HOUR degrees of longitude per hour).
    local_hour = hour + jnp.rad2deg(lon_rad) / _DEG_PER_HOUR
    ha = jnp.deg2rad((local_hour - _NOON_HOUR) * _DEG_PER_HOUR)
    cos_sza = (
        jnp.sin(lat_rad) * jnp.sin(decl_rad)
        + jnp.cos(lat_rad) * jnp.cos(decl_rad) * jnp.cos(ha)
    )
    cos_sza = jnp.maximum(cos_sza, 0.0)

    sw_down = jnp.asarray([constants.S_0 * cos_sza], dtype=dtype)

    # --- Atmospheric temperature: latitudinal baseline + seasonal + diurnal ---
    T_base = _T_BASE_EQUATOR_K - _T_BASE_POLE_DROP_K * abs(lat_rad) / half_pi
    T_seasonal_amp = _T_SEASONAL_POLE_AMP_K * abs(lat_rad) / half_pi
    T_season = T_seasonal_amp * jnp.cos(
        2.0 * jnp.pi * (day - _T_SEASONAL_PEAK_DAY) / _YEAR_DAYS
    )
    T_atm = jnp.asarray(
        [T_base + T_season
         + _T_DIURNAL_AMP_K * jnp.cos(
             2.0 * jnp.pi * (local_hour - _T_DIURNAL_PEAK_HOUR)
             / _HOURS_PER_DAY)],
        dtype=dtype,
    )

    # --- LW down: effective emissivity ~0.75 of blackbody ---
    lw_down = jnp.asarray(
        [_LW_EFF_EMISSIVITY * constants.sigma_sb * T_atm[0] ** 4], dtype=dtype)

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
        cos_zenith=jnp.asarray([cos_sza], dtype=dtype),
        co2_ppmv=jnp.asarray([_CO2_PPMV], dtype=dtype),
        has_radiation=jnp.ones(1, dtype=dtype),
        has_precipitation=jnp.ones(1, dtype=dtype),
    )
