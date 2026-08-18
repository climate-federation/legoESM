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

from legoesm import constants


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
    veg_transition_sharpness_per_deg : float
        Sharpness [deg^-1] of the smooth tropics->midlat->highlat
        vegetation-albedo latitude blend (transition width = 1/sharpness).
    """
    alpha_veg_tropics: float = 0.15
    alpha_veg_midlat: float = 0.20
    alpha_veg_highlat: float = 0.25
    alpha_snow_max: float = 0.80
    alpha_snow_min: float = 0.50
    tau_snow_decay: float = 432000.0  # 5 days in seconds
    snow_depth_crit: float = 50.0     # kg/m2
    veg_transition_sharpness_per_deg: float = 0.3  # deg^-1 latitude blend
    # Solar-zenith snow brightening (BATS / Dickinson 1983, as in CLM): the direct-beam
    # snow albedo rises as the sun drops toward the horizon (high zenith angle), so
    # high-latitude / polar snow (Antarctica, Greenland) is brighter than its
    # overhead-sun value — a first-order control on the ice-sheet shortwave budget a
    # zenith-independent albedo misses.  alpha += factor * f(mu) * (1 - alpha), with
    # f(mu) = max((1/b) * ((1 + b) / (1 + 2 b mu) - 1), 0) and mu = cos(zenith).
    snow_zenith_factor: float = 0.2   # brightening weight (0 = no zenith dependence)
    snow_zenith_b: float = 2.0        # BATS shape parameter
    # Dry-soil brightening (Oleson et al. 2013, CLM): exposed soil brightens as the
    # top layer dries, so a DESERT (low soil moisture) is far brighter than moist bare
    # soil / tundra — a contrast a single per-PFT albedo cannot represent.  The
    # snow-free base albedo gains up to ``soil_dry_albedo_boost`` linearly as the
    # top-layer volumetric water falls below ``soil_dry_albedo_ref``.
    soil_dry_albedo_boost: float = 0.11   # max dry-soil albedo increment
    soil_dry_albedo_ref: float = 0.275    # theta [m3/m3] above which no brightening
    # Per-cell SCALE on the snow-cover fraction (CLM-style canopy snow masking):
    # a tall canopy stays exposed above the snowpack, so a forest cell's EFFECTIVE
    # snow-covered fraction (as seen by shortwave) is smaller than the Niu-Yang
    # ground cover (<1), while open tundra/grass may whiten faster than the global
    # snow_depth_crit implies (>1 allowed; the blended fraction is clipped to 1).
    # None = 1 everywhere (legacy, byte-identical).  Typically a PFT-weighted
    # per-cell array from a calibrated per-PFT masking table.
    snow_cover_scale: object = None


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
    T_freeze: float = constants.T_freeze_ocean


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

    # Smooth blending between regimes using sigmoid-like ramps.
    # Band EDGES 23.5 deg (tropics->midlat) and 60 deg (midlat->highlat)
    # are physical latitude boundaries; the transition SHARPNESS (a width,
    # forbidden as a body magic number per CLAUDE.md) is a config field.
    sharpness = config.veg_transition_sharpness_per_deg  # deg^-1
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
    cos_zenith: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute snow albedo as a function of snow age (and optionally solar zenith).

    Diffuse (age-decayed) part::

        alpha_diffuse = alpha_snow_min + (alpha_snow_max - alpha_snow_min) * exp(-t/tau)

    the exponential age decay standing in for grain-size metamorphism (fresh snow
    bright, aged/firn snow darker).  When ``cos_zenith`` is supplied, the BATS /
    Dickinson (1983) solar-zenith brightening (as in CLM) is added — snow is brighter
    at a low sun (high zenith), the leading control on the polar ice-sheet shortwave
    budget that an overhead-sun albedo misses::

        alpha = alpha_diffuse + factor * f(mu) * (1 - alpha_diffuse)
        f(mu) = max( (1/b) * ((1 + b) / (1 + 2 b mu) - 1), 0 ),  mu = clip(cos_zenith, 0, 1)

    ``f`` is 0 at overhead sun (mu = 1) and rises to 1 at grazing incidence (mu = 0),
    and the denominator ``1 + 2 b mu >= 1`` is never zero — differentiable everywhere.

    Parameters
    ----------
    snow_age : jnp.ndarray
        Time since last snowfall [s].
    config : LandAlbedoConfig
    cos_zenith : jnp.ndarray or None
        Cosine of the solar zenith angle.  ``None`` disables the zenith brightening
        (returns the diffuse albedo — the legacy behaviour).

    Returns
    -------
    alpha_snow : jnp.ndarray
        Snow albedo.
    """
    alpha_diffuse = config.alpha_snow_min + (
        config.alpha_snow_max - config.alpha_snow_min
    ) * jnp.exp(-snow_age / config.tau_snow_decay)
    if cos_zenith is None:
        return alpha_diffuse
    mu = jnp.clip(cos_zenith, 0.0, 1.0)
    b = config.snow_zenith_b
    f_zen = jnp.maximum((1.0 / b) * ((1.0 + b) / (1.0 + 2.0 * b * mu) - 1.0), 0.0)
    return alpha_diffuse + config.snow_zenith_factor * f_zen * (1.0 - alpha_diffuse)


def snow_cover_fraction(
    snow_depth: jnp.ndarray,
    config: LandAlbedoConfig = LandAlbedoConfig(),
) -> jnp.ndarray:
    """Compute snow cover fraction from snow water equivalent.

    Niu & Yang (2007) / CLM-family SATURATING form::

        f_snow = tanh(SWE / snow_depth_crit)

    A thin snowpack already masks most of the surface, so ``f_snow`` rises steeply and
    reaches ~1 by ~3x ``snow_depth_crit`` while still giving light transient snow a
    partial cover.  The previous linear ``min(1, SWE/crit)`` was too gradual: a
    perennial-snow cell (Antarctica, high latitudes, Tibet) with a modest offline SWE
    sat at ``f_snow ~ 0.5`` and blended in too much dark snow-free surface, leaving a
    large negative albedo bias vs ERA5.  tanh is monotonic, differentiable, and bounded
    in [0, 1) with no clip.

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
    return jnp.tanh(snow_depth / jnp.maximum(config.snow_depth_crit, 1e-6))


def dry_soil_brightening(
    theta_top: jnp.ndarray,
    config: LandAlbedoConfig = LandAlbedoConfig(),
) -> jnp.ndarray:
    """Dry-soil albedo increment (Oleson et al. 2013, CLM).

    ``delta = boost * clip(1 - theta_top / theta_ref, 0, 1)`` — the snow-free soil
    albedo rises by up to ``soil_dry_albedo_boost`` as the top-layer volumetric water
    ``theta_top`` falls to zero, and vanishes once ``theta_top >= soil_dry_albedo_ref``.
    This is what makes a dry desert (theta ~ 0.05) bright while moist bare soil / tundra
    (theta ~ 0.3) stays dark.

    Parameters
    ----------
    theta_top : jnp.ndarray
        Top soil-layer volumetric water content [m3/m3].
    config : LandAlbedoConfig

    Returns
    -------
    delta : jnp.ndarray
        Additive albedo increment in [0, soil_dry_albedo_boost].
    """
    ref = jnp.maximum(config.soil_dry_albedo_ref, 1e-6)
    return config.soil_dry_albedo_boost * jnp.clip(1.0 - theta_top / ref, 0.0, 1.0)


def land_albedo(
    lat: jnp.ndarray,
    snow_depth: jnp.ndarray,
    snow_age: jnp.ndarray,
    config: LandAlbedoConfig = LandAlbedoConfig(),
    base_albedo: jnp.ndarray | None = None,
    f_snow_override: jnp.ndarray | None = None,
    snow_contrib_override: jnp.ndarray | None = None,
    cos_zenith: jnp.ndarray | None = None,
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
    base_albedo : jnp.ndarray or None
        Per-cell snow-free base albedo (e.g. CLM PFT map); ``None`` uses the
        latitude-band vegetation albedo.
    f_snow_override : jnp.ndarray or None
        Pre-aggregated snow-cover fraction (e.g. the area-weighted sub-grid
        elevation-band cover from ``land.snow_bands``).  ``None`` computes the
        cover from ``snow_depth`` with the standard Niu & Yang curve.
    snow_contrib_override : jnp.ndarray or None
        Pre-aggregated snow-albedo contribution ``mean_k[alpha_snow(age_k)*f_k]``
        from the sub-grid elevation bands (``land.snow_bands.band_snow_albedo``),
        where each band uses its OWN age-decayed snow albedo.  When supplied (with
        ``f_snow_override``) the cell snow term is this pre-aggregate instead of a
        single ``alpha_snow(snow_age) * f_snow`` — so perennial firn and fresh snow
        contribute at their own ages.  ``None`` uses the single-age snow albedo.

    Returns
    -------
    alpha : jnp.ndarray
        Land surface albedo.
    """
    # Snow-free base: a per-cell map (e.g. CLM PFT albedo) if supplied, else the
    # latitude-band vegetation albedo.  Snow albedo always blends on top, so a real
    # albedo map and the snow feedback coexist (bright deserts AND bright ice sheets).
    alpha_veg = (land_vegetation_albedo(lat, config) if base_albedo is None
                 else base_albedo)
    f_snow = (snow_cover_fraction(snow_depth, config) if f_snow_override is None
              else f_snow_override)
    if config.snow_cover_scale is not None:
        # Canopy snow masking: scale the effective snow-covered fraction (forest
        # canopies hide ground snow, scale<1; open tundra whitens faster, scale>1)
        # and scale the banded pre-aggregate by the same factor (it is linear in
        # the per-band cover) so both paths stay consistent.
        scale = jnp.asarray(config.snow_cover_scale)
        f_snow = jnp.clip(f_snow * scale, 0.0, 1.0)
        if snow_contrib_override is not None:
            snow_contrib_override = snow_contrib_override * jnp.clip(scale, 0.0, None)
    if snow_contrib_override is not None:
        # Banded path: each elevation band already blended its own age-decayed snow
        # albedo; the aggregate snow contribution replaces alpha_snow * f_snow.
        return alpha_veg * (1.0 - f_snow) + snow_contrib_override
    alpha_snow = snow_albedo(snow_age, config, cos_zenith=cos_zenith)
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
    _valid_methods = ("constant", "zenith")
    if config.method not in _valid_methods:
        raise ValueError(
            f"Unknown ocean albedo method {config.method!r}; "
            f"expected one of {_valid_methods}."
        )
    if config.method == "constant" or cos_zenith is None:
        if cos_zenith is not None:
            return jnp.broadcast_to(
                jnp.array(config.alpha_ocean_const), cos_zenith.shape
            )
        return config.alpha_ocean_const

    # method == "zenith": Briegleb (1992) zenith-angle dependent albedo
    mu = jnp.clip(cos_zenith, 0.01, 1.0)
    alpha = (
        0.026 / (mu ** 1.7 + 0.065)
        + 0.15 * (mu - 0.10) * (mu - 0.50) * (mu - 1.0)
    )
    return jnp.clip(alpha, 0.03, 0.40)
