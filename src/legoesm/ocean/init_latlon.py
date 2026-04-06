"""Initialization for the lat-lon FV ocean model."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.precision import get_policy
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
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

    dtype = get_policy().storage
    lat_deg = jnp.abs(grid.lat2d) * (180.0 / jnp.pi)
    land_mask = jnp.where(lat_deg < land_lat_threshold, 1.0, 0.0).astype(dtype)
    H_bathy = jnp.full_like(land_mask, H_max)

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
    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(
        z_coord.z_full_ref / _SCALE_DEPTH,
    )
    dtype = get_policy().storage
    T_3d = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, :], (n_lat, n_lon, nlev),
    ).astype(dtype)

    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)

    zeros_3d = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)
    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)

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


def regional_rest_state_latlon(
    grid: LatLonGrid,
    wall_mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    H_max: float = 5500.0,
    T_surface: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
) -> LatLonOceanState:
    """Create a rest-state initial condition on a regional lat-lon grid.

    Use with :func:`~legoesm.grids.latlon.create_regional_latlon_grid`
    which provides both the grid and the wall mask.

    Parameters
    ----------
    grid : LatLonGrid
        Regional grid (includes 1-cell wall boundary).
    wall_mask : jax.Array, shape (n_lat, n_lon)
        1 = ocean interior, 0 = wall.
    z_coord : OceanZStarCoordinate
    H_max : float
        Maximum ocean depth [m].
    T_surface, T_deep : float
        Surface and deep temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].

    Returns
    -------
    LatLonOceanState
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels
    dtype = get_policy().storage

    H_bathy = jnp.full((n_lat, n_lon), H_max, dtype=dtype)

    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(
        z_coord.z_full_ref / _SCALE_DEPTH,
    )
    T_3d = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, :], (n_lat, n_lon, nlev),
    ).astype(dtype)

    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)

    zeros_3d = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)
    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    return LatLonOceanState(
        u=Field(data=zeros_3d, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=zeros_3d, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=wall_mask.astype(dtype), name="land_mask",
                        dims=dims_2d, units=""),
    )


def wind_driven_gyre_latlon(
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    H_max: float = 5500.0,
    lon_west: float = 0.0,
    lon_east: float = 120.0,
    lat_south: float = 15.0,
    lat_north: float = 75.0,
    T_uniform: float = 10.0,
    S_uniform: float = 35.0,
) -> LatLonOceanState:
    """Create initial condition for a wind-driven barotropic gyre on lat-lon.

    Uniform T and S inside a rectangular basin. Purely barotropic setup.
    Wind forcing is applied during integration via the prescribed
    surface forcing pipeline.
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels
    dtype = get_policy().storage

    # Basin land mask: ocean inside rectangle, land outside
    lon_deg = grid.lon2d * (180.0 / jnp.pi)
    lat_deg = grid.lat2d * (180.0 / jnp.pi)
    in_basin = (
        (lon_deg >= lon_west) & (lon_deg <= lon_east) &
        (lat_deg >= lat_south) & (lat_deg <= lat_north)
    )
    land_mask = jnp.where(in_basin, 1.0, 0.0).astype(dtype)

    H_bathy = jnp.full_like(land_mask, H_max)

    # Uniform T and S — purely barotropic setup
    T_3d = jnp.full((n_lat, n_lon, nlev), T_uniform, dtype=dtype)
    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)
    zeros_3d = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)
    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)

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
