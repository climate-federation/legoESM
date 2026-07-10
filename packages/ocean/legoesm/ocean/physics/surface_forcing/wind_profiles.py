"""Grid-agnostic wind stress profile computation.

Computes (tau_x, tau_y) from latitude and PrescribedForcingConfig.
Used by both the structured-grid prescribed forcing (prescribed.py)
and the MPAS physics pipeline (mpas_physics.py) to avoid duplication.

All profiles operate on latitude in radians and return arrays of the
same shape as the input lat array — works with any grid layout
(cubed-sphere, latlon, MPAS cells, etc.).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.surface_forcing.config import PrescribedForcingConfig

__physics_contract__ = {
    "summary": (
        "Grid-agnostic analytic wind-stress profiles: map latitude to a "
        "(tau_x, tau_y) surface wind-stress vector for idealized ocean forcing "
        "(zonal jets, tropical easterlies, single/double-gyre and ACC-like "
        "patterns) selected by config."
    ),
    "inputs": {
        "lat": "rad", "cfg.tau_max": "N/m^2",
    },
    "outputs": {"tau_x": "N/m^2", "tau_y": "N/m^2"},
    "sign_convention": (
        "Returns the wind-stress vector [N/m^2] ON the ocean surface (a "
        "momentum boundary flux the caller applies), NOT an interior tendency; "
        "eastward tau_x > 0, northward tau_y > 0; the profile amplitude is set "
        "by tau_max. Nothing is conserved here (it is a boundary-flux producer)."
    ),
    # Surface momentum-flux (stress) producer; the caller applies the source.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Idealized zonal wind-stress profiles; Bryan (1987) JPO 17, Munk (1950) "
        "double-gyre / Southern-Ocean forcing"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_eps_div_and_wind_stress_convention.py — the "
        "zonal-jet profile gives mid-latitude westerlies and tropical "
        "easterlies; the sign matches the eastward tau_x > 0 convention; the "
        "'constant' profile returns cfg.tau_x/tau_y."
    ),
}


# Idealized analytic wind-stress profile constants (fixed scheme defaults).
_TAU_MAX_NORM = 0.1
_JET_LAT_DEG = 50.0
_JET_WIDTH_DEG = 12.0
_TROPICAL_WIDTH_DEG = 15.0
_TROPICAL_LAT_DEG = 15.0
_WP_C0, _WP_C1, _WP_C2, _WP_C3 = 0.08, 0.0397, 1.9487, 2.0397  # zonal-stress polynomial

def compute_wind_stress(
    lat: jnp.ndarray,
    cfg: PrescribedForcingConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute zonal and meridional wind stress from latitude.

    Parameters
    ----------
    lat : jnp.ndarray
        Latitude in radians. Any shape (grid-agnostic).
    cfg : PrescribedForcingConfig
        Wind forcing configuration.

    Returns
    -------
    tau_x, tau_y : jnp.ndarray
        Zonal and meridional wind stress [Pa], same shape as *lat*.

    Raises
    ------
    ValueError
        If ``cfg.wind_profile`` is not one of the supported profiles.
    """
    # Fail fast on an unknown profile rather than silently falling through
    # to the "constant" branch (slopbuster Pass 4: silent dispatch default
    # masks typos).  ``wind_profile`` is a static Python config string, so
    # this validation runs at trace time, not inside the JAX graph.
    _VALID_WIND_PROFILES = frozenset({
        "constant",
        "cosine_latitude",
        "single_gyre",
        "double_gyre",
        "double_gyre_sin2",
        "double_gyre_tapered",
        "channel_sine",
        "global_wind",
        "two_belt",
    })
    if cfg.wind_profile not in _VALID_WIND_PROFILES:
        raise ValueError(
            f"Unknown wind_profile {cfg.wind_profile!r}; expected one of "
            f"{sorted(_VALID_WIND_PROFILES)}"
        )

    if cfg.wind_profile == "cosine_latitude":
        lat_range = jnp.pi / 2.0  # 90 degrees
        tau_x = -cfg.tau_max * jnp.cos(jnp.pi * lat / lat_range)
        tau_y = jnp.zeros_like(tau_x)
    elif cfg.wind_profile == "single_gyre":
        # Basin-relative single-gyre wind stress (Stommel 1948, Munk 1950).
        # tau_x = -tau_max * cos(pi * (lat - lat_s) / (lat_n - lat_s))
        # Easterlies at southern boundary, westerlies at northern boundary.
        # One sign of curl → one anticyclonic (subtropical) gyre.
        lat_s = cfg.lat_south_deg * jnp.pi / 180.0
        lat_n = cfg.lat_north_deg * jnp.pi / 180.0
        basin_width = lat_n - lat_s
        tau_x = -cfg.tau_max * jnp.cos(jnp.pi * (lat - lat_s) / basin_width)
        tau_y = jnp.zeros_like(tau_x)
    elif cfg.wind_profile == "double_gyre":
        # Basin-relative double-gyre wind stress (Holland & Lin 1975).
        # tau_x = -tau_max * cos(2*pi * (lat - lat_s) / (lat_n - lat_s))
        # Easterlies at both boundaries, westerly jet at mid-basin.
        # Curl changes sign at mid-basin → subtropical gyre (south)
        # + subpolar gyre (north).
        lat_s = cfg.lat_south_deg * jnp.pi / 180.0
        lat_n = cfg.lat_north_deg * jnp.pi / 180.0
        basin_width = lat_n - lat_s
        tau_x = -cfg.tau_max * jnp.cos(
            2.0 * jnp.pi * (lat - lat_s) / basin_width)
        tau_y = jnp.zeros_like(tau_x)
    elif cfg.wind_profile == "double_gyre_sin2":
        # Mid-latitude westerly jet with sin^2 profile.
        # tau_x = +tau_max * sin^2(pi * (lat - lat_s') / (lat_n' - lat_s'))
        # where lat_s', lat_n' are inset by wind_buffer_deg from the basin
        # walls to ensure zero wind stress and zero Ekman transport at
        # the boundaries.
        #
        # Properties:
        #   - tau_x >= 0 everywhere (eastward, representing westerly jet)
        #   - tau_x = 0 at lat_s' and lat_n' (and in buffer zones)
        #   - Peak westerly at mid-basin
        #   - Curl changes sign at mid-basin → double gyre
        #   - Positive curl (Ekman suction) in southern half → subtropical gyre
        #   - Negative curl (Ekman pumping) in northern half → subpolar gyre
        buf = cfg.wind_buffer_deg * jnp.pi / 180.0
        lat_s = cfg.lat_south_deg * jnp.pi / 180.0 + buf
        lat_n = cfg.lat_north_deg * jnp.pi / 180.0 - buf
        wind_width = lat_n - lat_s
        y_norm = (lat - lat_s) / wind_width
        tau_x = cfg.tau_max * jnp.sin(jnp.pi * y_norm) ** 2
        tau_x = jnp.where((lat >= lat_s) & (lat <= lat_n), tau_x, 0.0)
        tau_y = jnp.zeros_like(tau_x)
    elif cfg.wind_profile == "double_gyre_tapered":
        # Cosine double-gyre wind with smooth taper to zero at walls.
        # Base profile: -tau_max * cos(2*pi*(lat-lat_s)/(lat_n-lat_s))
        # Taper: sin^2(pi/2 * distance_from_wall / buffer) in the buffer zone.
        #
        # Properties:
        #   - tau_x = 0 at lat_s and lat_n (smooth zero at walls)
        #   - Easterlies near walls, westerly jet at mid-basin (like cosine)
        #   - Basin-integrated wind ≈ 0 (small O(buffer/basin)^2 bias)
        #   - Curl changes sign at mid-basin → double gyre
        #   - No spurious Ekman transport at coastal walls
        buf = cfg.wind_buffer_deg * jnp.pi / 180.0
        lat_s = cfg.lat_south_deg * jnp.pi / 180.0
        lat_n = cfg.lat_north_deg * jnp.pi / 180.0
        basin_width = lat_n - lat_s
        y_norm = (lat - lat_s) / basin_width
        tau_base = -cfg.tau_max * jnp.cos(2.0 * jnp.pi * y_norm)
        dist_south = (lat - lat_s) / buf
        dist_north = (lat_n - lat) / buf
        taper_south = jnp.where(dist_south < 1.0,
                                jnp.sin(0.5 * jnp.pi * jnp.clip(dist_south, 0, 1))**2,
                                1.0)
        taper_north = jnp.where(dist_north < 1.0,
                                jnp.sin(0.5 * jnp.pi * jnp.clip(dist_north, 0, 1))**2,
                                1.0)
        taper = taper_south * taper_north
        tau_x = tau_base * taper
        tau_x = jnp.where((lat >= lat_s) & (lat <= lat_n), tau_x, 0.0)
        tau_y = jnp.zeros_like(tau_x)
    elif cfg.wind_profile == "channel_sine":
        # Half-sine zonal wind for channel experiments (Zhang et al. 2024).
        # tau_x = tau_max * sin(pi * (lat - lat_s) / (lat_n - lat_s))
        # Zero at both walls, peak westerly at channel center.
        # Positive everywhere (eastward) — drives ACC-like flow.
        lat_s = cfg.lat_south_deg * jnp.pi / 180.0
        lat_n = cfg.lat_north_deg * jnp.pi / 180.0
        width = lat_n - lat_s
        y_frac = (lat - lat_s) / width
        tau_x = cfg.tau_max * jnp.sin(jnp.pi * y_frac)
        tau_x = jnp.where((lat >= lat_s) & (lat <= lat_n), tau_x, 0.0)
        tau_y = jnp.zeros_like(tau_x)
    elif cfg.wind_profile == "global_wind":
        # Realistic 3-belt zonal wind stress following
        # Nikurashin & Vallis (2012, JPO) style profile.
        # Polynomial in sin^2(phi) with cos(phi) envelope:
        #   tau_x = tau_max * (a + b*s^2 + c*s^4 + d*s^6) * cos(phi)
        # where s = sin(phi). Coefficients tuned so that:
        #   phi=0:  tau_x = -0.08 Pa  (easterly trades)
        #   phi=30: tau_x = 0         (zero crossing)
        #   phi=50: tau_x = +0.10 Pa  (westerly peak)
        #   phi=70: tau_x = 0         (returns to zero)
        # Scaled by tau_max/0.1 so the default tau_max=0.1 gives
        # the reference amplitudes above.
        s2 = jnp.sin(lat) ** 2
        scale = cfg.tau_max / _TAU_MAX_NORM
        tau_x = scale * (
            -_WP_C0 - _WP_C1 * s2 + _WP_C2 * s2**2 - _WP_C3 * s2**3
        ) * jnp.cos(lat)
        tau_y = jnp.zeros_like(tau_x)
    elif cfg.wind_profile == "two_belt":
        # Two-belt zonal wind: equatorial easterlies + mid-latitude
        # westerlies, no polar easterlies.  Sum of two Gaussians:
        #   tau_x = -0.5*tau_max * exp(-phi^2/sigma_t^2)
        #         +     tau_max * exp(-( |phi| - phi_jet )^2/sigma_w^2)
        #
        # Default parameters (Option C):
        #   phi_jet  = 50 deg   westerly jet latitude
        #   sigma_w  = 12 deg   westerly jet width
        #   sigma_t  = 15 deg   trade wind width
        #
        # Properties:
        #   - Equatorial easterlies at half the westerly peak
        #   - Westerly peak ~0.1 Pa at 50 deg latitude
        #   - Smooth Gaussian decay toward poles (no polar easterlies)
        #   - Zero crossing at ~27 deg latitude
        #   - Meaningful wind stress over Drake Passage (55-80 deg S)
        phi_jet = jnp.radians(_JET_LAT_DEG)
        sigma_w = jnp.radians(_JET_WIDTH_DEG)
        sigma_t = jnp.radians(_TROPICAL_WIDTH_DEG)
        abs_lat = jnp.abs(lat)
        tau_trade = -0.5 * cfg.tau_max * jnp.exp(-(lat / sigma_t) ** 2)
        tau_west = cfg.tau_max * jnp.exp(
            -((abs_lat - phi_jet) / sigma_w) ** 2)
        tau_x = tau_trade + tau_west
        tau_y = jnp.zeros_like(tau_x)
    else:  # "constant"
        tau_x = jnp.full_like(lat, cfg.tau_x)
        tau_y = jnp.full_like(lat, cfg.tau_y)

    # Tropical wind reduction: scale wind stress near the equator.
    # Blends from tropical_wind_scale at lat=0 to 1.0 outside the band
    # using a Gaussian taper for smooth transition.
    _tw_scale = cfg.tropical_wind_scale
    if _tw_scale != 1.0:
        _tw_sigma = jnp.radians(cfg.tropical_wind_lat_deg)
        # Gaussian: 1 at equator → 0 at ±sigma.
        # scale_factor = 1 + (tropical_wind_scale - 1) * exp(-lat²/sigma²)
        # At lat=0: scale_factor = tropical_wind_scale
        # At |lat|>>sigma: scale_factor = 1.0 (unchanged)
        _gauss = jnp.exp(-0.5 * (lat / _tw_sigma) ** 2)
        _scale = 1.0 + (_tw_scale - 1.0) * _gauss
        tau_x = tau_x * _scale
        tau_y = tau_y * _scale

    return tau_x, tau_y
