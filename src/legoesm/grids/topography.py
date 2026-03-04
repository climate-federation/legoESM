"""Analytic topography generators and land-mask utilities.

Provides functions to generate idealized topography fields on a
cubed-sphere grid (6, n, n), plus utilities to derive surface
geopotential and land/ocean masks.

All topography generators return surface elevation z_s in meters.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.cubed_sphere import CubedSphereGrid


def gaussian_mountain(
    grid: CubedSphereGrid,
    h0: float = 2500.0,
    lat0: float = 40.0 * jnp.pi / 180.0,
    lon0: float = 255.0 * jnp.pi / 180.0,
    sigma_lat: float = 15.0 * jnp.pi / 180.0,
    sigma_lon: float = 15.0 * jnp.pi / 180.0,
) -> jnp.ndarray:
    """Generate a single Gaussian mountain.

    z_s = h0 * exp(-((lat-lat0)^2/(2*sigma_lat^2) + (lon-lon0)^2/(2*sigma_lon^2)))

    Parameters
    ----------
    grid : CubedSphereGrid
        Cubed-sphere grid.
    h0 : float
        Peak height [m]. Default 2500 (Rocky Mountains scale).
    lat0 : float
        Center latitude [rad]. Default 40 deg N.
    lon0 : float
        Center longitude [rad]. Default 255 deg E.
    sigma_lat : float
        Latitudinal half-width [rad]. Default 15 deg.
    sigma_lon : float
        Longitudinal half-width [rad]. Default 15 deg.

    Returns
    -------
    jnp.ndarray
        Surface elevation z_s, shape (6, n, n).
    """
    lat = grid.lat
    lon = grid.lon

    dlat = lat - lat0

    # Periodic longitude wrapping to [-pi, pi]
    dlon = jnp.mod(lon - lon0 + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    z_s = h0 * jnp.exp(
        -(dlat ** 2 / (2.0 * sigma_lat ** 2) + dlon ** 2 / (2.0 * sigma_lon ** 2))
    )
    return z_s


def zonal_ridge(
    grid: CubedSphereGrid,
    h0: float = 2500.0,
    lat0: float = 45.0 * jnp.pi / 180.0,
    sigma_lat: float = 10.0 * jnp.pi / 180.0,
) -> jnp.ndarray:
    """Generate a zonally symmetric ridge.

    z_s = h0 * exp(-(lat-lat0)^2/(2*sigma_lat^2))

    Parameters
    ----------
    grid : CubedSphereGrid
        Cubed-sphere grid.
    h0 : float
        Peak height [m].
    lat0 : float
        Ridge center latitude [rad]. Default 45 deg N.
    sigma_lat : float
        Latitudinal half-width [rad]. Default 10 deg.

    Returns
    -------
    jnp.ndarray
        Surface elevation z_s, shape (6, n, n).
    """
    dlat = grid.lat - lat0
    return h0 * jnp.exp(-(dlat ** 2) / (2.0 * sigma_lat ** 2))


def schaer_mountain(
    grid: CubedSphereGrid,
    h0: float = 2000.0,
    lat0: float = 0.0,
    lon0: float = jnp.pi,
    half_width: float = 72.0e3,
) -> jnp.ndarray:
    """Generate Schaer-type cosine-bell mountain topography.

    Uses great-circle distance: z_s = h0/2 * (1 + cos(pi*r/a)) for r < a,
    where r is the distance from the mountain center and a is the half-width.

    Parameters
    ----------
    grid : CubedSphereGrid
        Cubed-sphere grid.
    h0 : float
        Peak height [m].
    lat0 : float
        Mountain center latitude [rad].
    lon0 : float
        Mountain center longitude [rad].
    half_width : float
        Mountain half-width [m].

    Returns
    -------
    jnp.ndarray
        Surface elevation z_s, shape (6, n, n).
    """
    lat = grid.lat
    lon = grid.lon

    # Great-circle distance via haversine
    dlat = lat - lat0
    dlon = lon - lon0
    a = (
        jnp.sin(dlat / 2.0) ** 2
        + jnp.cos(lat) * jnp.cos(lat0) * jnp.sin(dlon / 2.0) ** 2
    )
    angular_dist = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0.0, 1.0)))
    dist = angular_dist * grid.radius

    # Cosine bell: h0/2 * (1 + cos(pi*r/a)) for r < a, 0 otherwise
    z_s = jnp.where(
        dist < half_width,
        0.5 * h0 * (1.0 + jnp.cos(jnp.pi * dist / half_width)),
        0.0,
    )
    return z_s


def land_mask_from_topography(
    z_s: jnp.ndarray,
    threshold: float = 0.0,
) -> jnp.ndarray:
    """Derive a land/ocean mask from surface elevation.

    Parameters
    ----------
    z_s : jnp.ndarray
        Surface elevation [m], shape (6, n, n).
    threshold : float
        Elevation threshold [m]. Grid cells with z_s > threshold are land.

    Returns
    -------
    jnp.ndarray
        Land fraction f_land (1.0 = land, 0.0 = ocean), shape (6, n, n).
    """
    return jnp.where(z_s > threshold, 1.0, 0.0)


def phis_from_topography(z_s: jnp.ndarray) -> jnp.ndarray:
    """Convert surface elevation to surface geopotential.

    phis = g * z_s

    Parameters
    ----------
    z_s : jnp.ndarray
        Surface elevation [m].

    Returns
    -------
    jnp.ndarray
        Surface geopotential [m^2/s^2], same shape as z_s.
    """
    return constants.g * z_s
