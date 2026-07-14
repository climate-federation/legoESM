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

# Reference-atmosphere theta_0(z) profiles now live in the package (audit item
# 9: production spectral-NH initializers must not import from tests).  Re-export
# them here so existing test imports keep working with a single source of truth.
from legoesm.atmosphere.dynamics.gcm.dcmip2025_ic import (  # noqa: F401
    isothermal_theta_ref,
    piecewise_lapse_theta_ref,
)


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
