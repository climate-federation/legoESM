"""Common utilities for DCMIP-2025 test cases.

Provides:
- Small-Earth scaling (reduced-radius sphere)
- Reference atmosphere profiles (isothermal, piecewise lapse rate)
- Mountain topography generators (Schaer-type, Gaussian, mountain chain)
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    CubedSphereGrid,
    create_cubed_sphere,
    apply_small_earth_scaling,
)


# ==============================================================================
# Reference atmosphere profiles
# ==============================================================================

def isothermal_theta_ref(T0: float = 300.0):
    """Return theta_0(z) for an isothermal atmosphere.

    For T=const, theta(z) = T * (p0/p)^kappa increases with height.
    We compute theta from T and the hydrostatic pressure profile.

    For simplicity, we return a function that computes theta at each z
    using the isothermal scale height: p(z) = p0 * exp(-z/H_s),
    H_s = R_d * T0 / g.
    """
    R_d = constants.R_d
    g = constants.g
    p_0 = constants.p_ref
    kappa = constants.kappa

    def theta_fn(z):
        H_s = R_d * T0 / g
        p = p_0 * jnp.exp(-z / H_s)
        return T0 * (p_0 / p) ** kappa

    return theta_fn


def piecewise_lapse_theta_ref(
    T_s: float = 300.0,
    lapse_tropo: float = -5.0e-3,
    lapse_strato: float = 5.0e-3,
    z_tropopause: float = 20000.0,
):
    """Return theta_0(z) for a piecewise linear temperature profile.

    Parameters
    ----------
    T_s : float
        Surface temperature [K].
    lapse_tropo : float
        Tropospheric lapse rate [K/m] (negative for decreasing T).
    lapse_strato : float
        Stratospheric lapse rate [K/m] (positive for increasing T).
    z_tropopause : float
        Tropopause height [m].

    Returns
    -------
    callable
        Function theta_0(z) -> potential temperature at height z.
    """
    R_d = constants.R_d
    g = constants.g
    p_0 = constants.p_ref
    kappa = constants.kappa
    c_p = constants.c_pd

    def theta_fn(z):
        # Temperature profile
        T_trop = T_s + lapse_tropo * jnp.minimum(z, z_tropopause)
        T_above = T_trop + lapse_strato * jnp.maximum(z - z_tropopause, 0.0)
        # Use stratospheric profile only above tropopause
        T = jnp.where(z <= z_tropopause, T_s + lapse_tropo * z, T_above)

        # Pressure from hydrostatic integration (approximate)
        # For piecewise linear T, we integrate dp/dz = -rho*g = -p*g/(R_d*T)
        # Below tropopause: if T = T_s + gamma*z, then
        #   p = p_0 * (T/T_s)^(-g/(R_d*gamma))
        # Above tropopause: isothermal-like with different lapse rate

        # Tropospheric pressure
        T_ratio = jnp.clip(T_s + lapse_tropo * jnp.minimum(z, z_tropopause), 100.0, None) / T_s
        exponent_tropo = -g / (R_d * lapse_tropo)
        p_tropo = p_0 * T_ratio ** exponent_tropo

        # At tropopause
        T_at_trop = T_s + lapse_tropo * z_tropopause
        T_ratio_trop = jnp.clip(T_at_trop, 100.0, None) / T_s
        p_at_trop = p_0 * T_ratio_trop ** exponent_tropo

        # Stratospheric pressure (above tropopause)
        dz_above = jnp.maximum(z - z_tropopause, 0.0)
        T_strato = T_at_trop + lapse_strato * dz_above
        # Handle potential zero lapse rate
        safe_lapse = jnp.where(
            jnp.abs(lapse_strato) > 1e-10,
            lapse_strato,
            1e-10,
        )
        exponent_strato = -g / (R_d * safe_lapse)
        T_ratio_strato = jnp.clip(T_strato, 100.0, None) / jnp.clip(T_at_trop, 100.0, None)
        p_strato = p_at_trop * T_ratio_strato ** exponent_strato

        p = jnp.where(z <= z_tropopause, p_tropo, p_strato)

        # Potential temperature
        theta = T * (p_0 / p) ** kappa

        return theta

    return theta_fn


# ==============================================================================
# Topography generators
# ==============================================================================

def schaer_mountain_profile(
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    radius: float,
    h0: float = 2000.0,
    halfwidth: float = 72.0e3,
    lat0: float = 0.349,  # ~20 deg N
    lon0: float = 0.0,
) -> jnp.ndarray:
    """Compute Schaer-type mountain topography from lat/lon arrays.

    z_s = h0 * exp(-((d/halfwidth)^2))

    where d is the great-circle distance from (lat0, lon0).
    Grid-agnostic: works with any lat/lon arrays of any shape.

    Parameters
    ----------
    lat : jnp.ndarray
        Latitude in radians, any shape.
    lon : jnp.ndarray
        Longitude in radians, same shape as lat.
    radius : float
        Sphere radius [m].
    h0 : float
        Mountain peak height [m].
    halfwidth : float
        Mountain half-width [m] (e-folding distance).
    lat0 : float
        Mountain center latitude [rad].
    lon0 : float
        Mountain center longitude [rad].

    Returns
    -------
    jnp.ndarray
        Surface elevation z_s, same shape as lat.
    """
    dlat = lat - lat0
    dlon = lon - lon0
    a = (jnp.sin(dlat / 2) ** 2
         + jnp.cos(lat) * jnp.cos(lat0) * jnp.sin(dlon / 2) ** 2)
    angular_dist = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0.0, 1.0)))
    dist = angular_dist * radius
    return h0 * jnp.exp(-(dist / halfwidth) ** 2)


def schaer_mountain(
    grid: CubedSphereGrid,
    h0: float = 2000.0,
    halfwidth: float = 72.0e3,
    lat0: float = 0.349,  # ~20 deg N
    lon0: float = 0.0,
) -> jnp.ndarray:
    """Generate Schaer-type mountain topography on a cubed-sphere grid.

    Thin wrapper around :func:`schaer_mountain_profile`.

    Parameters
    ----------
    grid : CubedSphereGrid
    h0 : float
        Mountain peak height [m].
    halfwidth : float
        Mountain half-width [m] (e-folding distance).
    lat0 : float
        Mountain center latitude [rad].
    lon0 : float
        Mountain center longitude [rad].

    Returns
    -------
    jnp.ndarray
        Surface elevation z_s, shape (6, n, n).
    """
    return schaer_mountain_profile(
        grid.lat, grid.lon, grid.radius,
        h0=h0, halfwidth=halfwidth, lat0=lat0, lon0=lon0,
    )


def gaussian_mountain(
    grid: CubedSphereGrid,
    h0: float = 2000.0,
    d: float = 50.0e3,
    lat0: float = 0.0,
    lon0: float = jnp.pi,
) -> jnp.ndarray:
    """Generate a Gaussian mountain (axisymmetric).

    z_s = h0 * exp(-((dist/d)^2))

    Parameters
    ----------
    grid : CubedSphereGrid
    h0 : float
        Mountain peak height [m].
    d : float
        Mountain half-width [m].
    lat0 : float
        Mountain center latitude [rad].
    lon0 : float
        Mountain center longitude [rad].

    Returns
    -------
    jnp.ndarray
        Surface elevation, shape (6, n, n).
    """
    lat = grid.lat
    lon = grid.lon

    dlat = lat - lat0
    dlon = lon - lon0
    a = jnp.sin(dlat / 2) ** 2 + jnp.cos(lat) * jnp.cos(lat0) * jnp.sin(dlon / 2) ** 2
    angular_dist = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0.0, 1.0)))
    dist = angular_dist * grid.radius

    return h0 * jnp.exp(-(dist / d) ** 2)


def mountain_chain_with_gap(
    grid: CubedSphereGrid,
    h0: float = 2000.0,
    chain_halfwidth_lon: float = 50.0e3,
    chain_halfwidth_lat: float = 500.0e3,
    gap_halfwidth: float = 50.0e3,
    lon0: float = jnp.pi,
    lat0: float = 0.0,
) -> jnp.ndarray:
    """Generate a mountain chain with a gap (for DCMIP-2025 TC2a).

    A north-south oriented mountain range centered at lon0 with a
    gap at lat0. The chain extends over chain_halfwidth_lat in
    latitude and chain_halfwidth_lon in longitude.

    Parameters
    ----------
    grid : CubedSphereGrid
    h0 : float
        Mountain peak height [m].
    chain_halfwidth_lon : float
        East-west half-width [m].
    chain_halfwidth_lat : float
        North-south half-extent [m].
    gap_halfwidth : float
        Half-width of the gap [m].
    lon0 : float
        Chain center longitude [rad].
    lat0 : float
        Gap center latitude [rad].

    Returns
    -------
    jnp.ndarray
        Surface elevation, shape (6, n, n).
    """
    lat = grid.lat
    lon = grid.lon

    # East-west distance from chain center
    dlon = lon - lon0
    # Wrap to [-pi, pi]
    dlon = jnp.mod(dlon + jnp.pi, 2 * jnp.pi) - jnp.pi
    x_dist = dlon * grid.radius * jnp.cos(lat)

    # North-south distance from gap center
    y_dist = (lat - lat0) * grid.radius

    # Chain envelope (east-west Gaussian)
    chain = jnp.exp(-(x_dist / chain_halfwidth_lon) ** 2)

    # Latitudinal extent (smooth cutoff)
    lat_envelope = jnp.exp(-(y_dist / chain_halfwidth_lat) ** 4)

    # Gap (remove mountain near lat0)
    gap = 1.0 - jnp.exp(-(y_dist / gap_halfwidth) ** 2)

    return h0 * chain * lat_envelope * gap
