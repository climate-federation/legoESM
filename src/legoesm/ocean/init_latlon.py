"""Initialization for the lat-lon FV ocean model."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.ocean.state import LatLonOceanState


def idealized_bathymetry_latlon(
    grid: LatLonGrid,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate idealized bathymetry and land mask on a lat-lon grid.

    Land is placed at high latitudes (|lat| > threshold).

    Parameters
    ----------
    grid : LatLonGrid
    H_max : float
        Maximum ocean depth [m].
    land_lat_threshold : float
        Latitude [degrees] above which cells are land.

    Returns
    -------
    H_bathy : array, shape (n_lat, n_lon)
    land_mask : array, shape (n_lat, n_lon)
    """
    if H_max <= 0.0:
        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    if land_lat_threshold < 0.0 or land_lat_threshold > 90.0:
        raise ValueError(
            f"land_lat_threshold must be in [0, 90], got {land_lat_threshold!r}",
        )

    lat_deg = jnp.abs(grid.lat2d) * (180.0 / jnp.pi)
    land_mask = jnp.where(lat_deg < land_lat_threshold, 1.0, 0.0).astype(jnp.float32)
    H_bathy = jnp.full_like(land_mask, H_max, dtype=jnp.float32)

    return H_bathy, land_mask


def rest_state_latlon_ocean(
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    T_surface: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
) -> LatLonOceanState:
    """Create a rest-state initial condition on a lat-lon grid.

    Temperature: exponential profile.
    Salinity: uniform.
    Velocity: zero.
    Eta: zero.

    Parameters
    ----------
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    T_surface, T_deep : float
        Surface and deep temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].
    H_max : float
        Maximum ocean depth [m].
    land_lat_threshold : float
        Latitude threshold for land [degrees].

    Returns
    -------
    LatLonOceanState
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels

    H_bathy, land_mask = idealized_bathymetry_latlon(grid, H_max, land_lat_threshold)

    # Exponential T stratification
    scale_depth = 1000.0
    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(
        z_coord.z_full_ref / scale_depth,
    )
    T_3d = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, :], (n_lat, n_lon, nlev),
    ).astype(jnp.float32)

    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=jnp.float32)

    zeros_3d = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float32)
    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=jnp.float32)

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    return LatLonOceanState(
        u=Field(data=zeros_3d, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=zeros_3d, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
    )


def wind_driven_gyre_latlon(
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    H_max: float = 5500.0,
) -> LatLonOceanState:
    """Create initial condition for a wind-driven double-gyre on lat-lon."""
    return rest_state_latlon_ocean(grid, z_coord, H_max=H_max)
