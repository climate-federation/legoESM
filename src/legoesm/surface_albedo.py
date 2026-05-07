"""Surface albedo parameterizations (Task 10).

Provides physically-based albedo for each surface tile:

1. **Land**: vegetation albedo varies with latitude (proxy for biome),
   blended with snow albedo via snow cover fraction.  Snow albedo
   decays exponentially with age.

2. **Sea ice**: albedo transitions from cold (dry) ice to warm (melting)
   ice as surface temperature approaches freezing.

3. **Ocean**: constant or solar-zenith-angle dependent (Briegleb 1992).

4. **Lake**: constant or zenith-dependent (same formula as ocean).

All functions are JAX-differentiable.

References
----------
- Briegleb, B. P., et al. (1992). NCAR Technical Note TN-373+STR,
  describing the zenith-angle dependent ocean albedo.
- Dickinson, R. E., et al. (1993). Biosphere-Atmosphere Transfer
  Scheme (BATS) for NCAR CCM.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


# =========================================================================
# Configuration
# =========================================================================

class LandAlbedoConfig(NamedTuple):
    """Land albedo configuration with snow feedback.

    Fields
    ------
    alpha_veg_tropics : float
        Vegetation albedo in the tropics (|lat| < 23.5).
    alpha_veg_midlat : float
        Vegetation albedo at mid-latitudes (23.5 < |lat| < 60).
    alpha_veg_highlat : float
        Vegetation albedo at high latitudes (|lat| > 60).
    alpha_snow_max : float
        Fresh snow albedo.
    alpha_snow_min : float
        Old/melting snow albedo.
    tau_snow_decay : float
        Snow albedo e-folding decay time [s].
    snow_depth_crit : float
        Critical snow depth for full snow cover [kg/m2].
    """
    alpha_veg_tropics: float = 0.15
    alpha_veg_midlat: float = 0.20
    alpha_veg_highlat: float = 0.25
    alpha_snow_max: float = 0.80
    alpha_snow_min: float = 0.50
    tau_snow_decay: float = 432000.0  # 5 days in seconds
    snow_depth_crit: float = 50.0     # kg/m2


class IceAlbedoConfig(NamedTuple):
    """Sea ice albedo configuration.

    Albedo transitions linearly between cold (dry) and warm (melting)
    values over a temperature window near the freezing point.

    Fields
    ------
    alpha_ice_cold : float
        Albedo of cold (dry) ice.
    alpha_ice_warm : float
        Albedo of warm (melting) ice.
    T_transition_width : float
        Temperature range [K] over which transition occurs below T_freeze.
    T_freeze : float
        Freezing point [K].
    """
    alpha_ice_cold: float = 0.65
    alpha_ice_warm: float = 0.45
    T_transition_width: float = 5.0
    T_freeze: float = 271.35       # = constants.T_freeze_ocean


class OceanAlbedoConfig(NamedTuple):
    """Ocean albedo configuration.

    Fields
    ------
    alpha_ocean_const : float
        Constant ocean albedo (used when method="constant").
    method : str
        "constant" or "zenith" (Briegleb 1992).
    """
    alpha_ocean_const: float = 0.06
    method: str = "constant"


# =========================================================================
# Land albedo
# =========================================================================

def land_vegetation_albedo(
    lat: jnp.ndarray,
    config: LandAlbedoConfig = LandAlbedoConfig(),
) -> jnp.ndarray:
    """Compute vegetation albedo as a function of latitude.

    Uses smooth transitions between tropical, midlatitude, and high-latitude
    albedo values via sigmoid blending.

    Parameters
    ----------
    lat : jnp.ndarray
        Latitude in radians.
    config : LandAlbedoConfig

    Returns
    -------
    alpha_veg : jnp.ndarray
        Vegetation albedo, same shape as lat.
    """
    abs_lat_deg = jnp.abs(lat) * 180.0 / jnp.pi

    # Smooth blending between regimes using sigmoid transitions
    # Tropics->midlat around 23.5 deg, midlat->highlat around 60 deg
    sharpness = 0.3  # degrees^-1 for smooth transition
    w_midlat = jnp.clip(
        (abs_lat_deg - 23.5) * sharpness, 0.0, 1.0
    )
    w_highlat = jnp.clip(
        (abs_lat_deg - 60.0) * sharpness, 0.0, 1.0
    )

    # Linear interpolation: tropics -> midlat -> highlat
    alpha = (
        config.alpha_veg_tropics * (1.0 - w_midlat)
        + config.alpha_veg_midlat * (w_midlat - w_highlat)
        + config.alpha_veg_highlat * w_highlat
    )
    return alpha


def snow_albedo(
    snow_age: jnp.ndarray,
    config: LandAlbedoConfig = LandAlbedoConfig(),
) -> jnp.ndarray:
    """Compute snow albedo as a function of snow age.

    alpha_snow = alpha_snow_min + (alpha_snow_max - alpha_snow_min) * exp(-t/tau)

    Parameters
    ----------
    snow_age : jnp.ndarray
        Time since last snowfall [s].
    config : LandAlbedoConfig

    Returns
    -------
    alpha_snow : jnp.ndarray
        Snow albedo.
    """
    return config.alpha_snow_min + (
        config.alpha_snow_max - config.alpha_snow_min
    ) * jnp.exp(-snow_age / config.tau_snow_decay)


def snow_cover_fraction(
    snow_depth: jnp.ndarray,
    config: LandAlbedoConfig = LandAlbedoConfig(),
) -> jnp.ndarray:
    """Compute snow cover fraction from snow depth.

    f_snow = min(1, snow_depth / snow_depth_crit)

    Parameters
    ----------
    snow_depth : jnp.ndarray
        Snow water equivalent [kg/m2].
    config : LandAlbedoConfig

    Returns
    -------
    f_snow : jnp.ndarray
        Snow cover fraction [0-1].
    """
    return jnp.clip(snow_depth / config.snow_depth_crit, 0.0, 1.0)


def land_albedo(
    lat: jnp.ndarray,
    snow_depth: jnp.ndarray,
    snow_age: jnp.ndarray,
    config: LandAlbedoConfig = LandAlbedoConfig(),
) -> jnp.ndarray:
    """Compute land surface albedo with snow feedback.

    alpha_land = alpha_veg * (1 - f_snow) + alpha_snow * f_snow

    Parameters
    ----------
    lat : jnp.ndarray
        Latitude in radians.
    snow_depth : jnp.ndarray
        Snow water equivalent [kg/m2].
    snow_age : jnp.ndarray
        Time since last snowfall [s].
    config : LandAlbedoConfig

    Returns
    -------
    alpha : jnp.ndarray
        Land surface albedo.
    """
    alpha_veg = land_vegetation_albedo(lat, config)
    alpha_snow = snow_albedo(snow_age, config)
    f_snow = snow_cover_fraction(snow_depth, config)
    return alpha_veg * (1.0 - f_snow) + alpha_snow * f_snow


# =========================================================================
# Sea ice albedo
# =========================================================================

def ice_albedo(
    T_ice: jnp.ndarray,
    config: IceAlbedoConfig = IceAlbedoConfig(),
) -> jnp.ndarray:
    """Compute sea ice albedo as a function of ice surface temperature.

    Linearly interpolates between cold and warm albedo values.
    When T_ice < T_freeze - T_transition_width: alpha = alpha_ice_cold.
    When T_ice >= T_freeze: alpha = alpha_ice_warm.

    Parameters
    ----------
    T_ice : jnp.ndarray
        Ice surface temperature [K].
    config : IceAlbedoConfig

    Returns
    -------
    alpha : jnp.ndarray
        Sea ice albedo.
    """
    T_cold = config.T_freeze - config.T_transition_width
    frac = jnp.clip(
        (T_ice - T_cold) / config.T_transition_width, 0.0, 1.0
    )
    return config.alpha_ice_cold * (1.0 - frac) + config.alpha_ice_warm * frac


# =========================================================================
# Ocean albedo
# =========================================================================

def ocean_albedo(
    cos_zenith: jnp.ndarray | None = None,
    config: OceanAlbedoConfig = OceanAlbedoConfig(),
) -> jnp.ndarray:
    """Compute ocean surface albedo.

    For method="constant": returns alpha_ocean_const.
    For method="zenith": Briegleb (1992) parameterization
        alpha = 0.026 / (cos_zenith^1.7 + 0.065)
        + 0.15 * (cos_zenith - 0.10) * (cos_zenith - 0.50) * (cos_zenith - 1.0)
    Clipped to [0.03, 0.40].

    Parameters
    ----------
    cos_zenith : jnp.ndarray or None
        Cosine of solar zenith angle. Required for method="zenith".
    config : OceanAlbedoConfig

    Returns
    -------
    alpha : jnp.ndarray or float
        Ocean surface albedo.
    """
    if config.method == "constant" or cos_zenith is None:
        if cos_zenith is not None:
            return jnp.broadcast_to(
                jnp.array(config.alpha_ocean_const), cos_zenith.shape
            )
        return config.alpha_ocean_const

    # Briegleb (1992) zenith-angle dependent albedo
    mu = jnp.clip(cos_zenith, 0.01, 1.0)
    alpha = (
        0.026 / (mu ** 1.7 + 0.065)
        + 0.15 * (mu - 0.10) * (mu - 0.50) * (mu - 1.0)
    )
    return jnp.clip(alpha, 0.03, 0.40)
