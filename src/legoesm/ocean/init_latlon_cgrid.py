"""Initialization for the lat-lon C-grid FV ocean model."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.precision import get_policy
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.ocean.state import LatLonCGridOceanState
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks


def idealized_bathymetry_latlon_cgrid(
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


def rest_state_latlon_cgrid_ocean(
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    T_water_init_C: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
    land_mask_override: jnp.ndarray | None = None,
    H_bathy_override: jnp.ndarray | None = None,
) -> LatLonCGridOceanState:
    """Create a rest-state initial condition on a C-grid lat-lon grid.

    Temperature: exponential profile.
    Salinity: uniform.
    Velocity: zero.
    Eta: zero.

    Parameters
    ----------
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    T_water_init_C, T_deep : float
        Surface and deep temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].
    H_max : float
        Maximum ocean depth [m].
    land_lat_threshold : float
        Latitude threshold for land [degrees].  Ignored when
        *land_mask_override* is provided.
    land_mask_override : array (n_lat, n_lon), optional
        If provided, use this as the land mask (1=ocean, 0=land) instead
        of deriving one from *land_lat_threshold*.  Face masks (u_mask,
        v_mask) are computed from it automatically.
    H_bathy_override : array (n_lat, n_lon), optional
        If provided, use this as the per-cell bathymetry depth [m].
        When supplied together with *land_mask_override*, both are used
        as-is (caller is responsible for consistency between them).
        When supplied without *land_mask_override*, the land mask is
        derived from ``H_bathy_override > 0``.  When neither is given,
        a flat-bottom idealized bathymetry is constructed.

    Returns
    -------
    LatLonCGridOceanState
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels

    # Cast bathymetry/mask inputs to the active precision policy so that a
    # caller running under x32 does not silently get x64 fields (codex
    # adversarial review iter-1, bug #6).
    dtype = get_policy().storage
    if H_bathy_override is not None:
        H_bathy = jnp.asarray(H_bathy_override).astype(dtype)
        if land_mask_override is not None:
            land_mask = jnp.asarray(land_mask_override).astype(dtype)
        else:
            land_mask = (H_bathy > 0.0).astype(dtype)
    elif land_mask_override is not None:
        land_mask = jnp.asarray(land_mask_override).astype(dtype)
        H_bathy = jnp.full((n_lat, n_lon), H_max, dtype=dtype)
    else:
        H_bathy, land_mask = idealized_bathymetry_latlon_cgrid(
            grid, H_max, land_lat_threshold,
        )

    # Exponential T stratification
    T_profile = T_deep + (T_water_init_C - T_deep) * jnp.exp(
        z_coord.z_full_ref / _SCALE_DEPTH,
    )
    dtype = get_policy().storage
    T_3d = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, :], (n_lat, n_lon, nlev),
    ).astype(dtype)

    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)

    # C-grid velocity shapes
    u_zeros = jnp.zeros((n_lat, n_lon + 1, nlev), dtype=dtype)
    v_zeros = jnp.zeros((n_lat + 1, n_lon, nlev), dtype=dtype)
    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)

    # Face masks
    u_mask, v_mask = compute_face_masks(land_mask)

    # Initialize vertical velocity with zeros (will be computed during step)
    w_zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)

    dims_u = ("lat", "lon_u", "level")
    dims_v = ("lat_v", "lon", "level")
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    dims_u2d = ("lat", "lon_u")
    dims_v2d = ("lat_v", "lon")

    return LatLonCGridOceanState(
        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
                staggering="edge"),
        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
                staggering="edge"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
    )


def wind_driven_gyre_latlon_cgrid(
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    H_max: float = 5500.0,
    lon_west: float = 0.0,
    lon_east: float = 120.0,
    lat_south: float = 15.0,
    lat_north: float = 75.0,
    T_uniform: float = 10.0,
    S_uniform: float = 35.0,
) -> LatLonCGridOceanState:
    """Create initial condition for a wind-driven barotropic gyre on C-grid lat-lon.

    Uniform T and S inside a rectangular basin. Purely barotropic setup.
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels
    dtype = get_policy().storage

    lon_deg = grid.lon2d * (180.0 / jnp.pi)
    lat_deg = grid.lat2d * (180.0 / jnp.pi)
    in_basin = (
        (lon_deg >= lon_west) & (lon_deg <= lon_east) &
        (lat_deg >= lat_south) & (lat_deg <= lat_north)
    )
    land_mask = jnp.where(in_basin, 1.0, 0.0).astype(dtype)
    H_bathy = jnp.full_like(land_mask, H_max)

    T_3d = jnp.full((n_lat, n_lon, nlev), T_uniform, dtype=dtype)
    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)

    u_zeros = jnp.zeros((n_lat, n_lon + 1, nlev), dtype=dtype)
    v_zeros = jnp.zeros((n_lat + 1, n_lon, nlev), dtype=dtype)
    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)

    u_mask, v_mask = compute_face_masks(land_mask)

    # Initialize vertical velocity with zeros (will be computed during step)
    w_zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)

    dims_u = ("lat", "lon_u", "level")
    dims_v = ("lat_v", "lon", "level")
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    dims_u2d = ("lat", "lon_u")
    dims_v2d = ("lat_v", "lon")

    return LatLonCGridOceanState(
        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
                staggering="edge"),
        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
                staggering="edge"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
    )


def regional_rest_state_latlon_cgrid(
    grid: LatLonGrid,
    wall_mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    H_max: float = 5500.0,
    T_water_init_C: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
) -> LatLonCGridOceanState:
    """Create a rest-state initial condition on a regional C-grid lat-lon grid.

    Parameters
    ----------
    grid : LatLonGrid
        Regional grid (includes 1-cell wall boundary).
    wall_mask : jax.Array, shape (n_lat, n_lon)
        1 = ocean interior, 0 = wall.
    z_coord : OceanZStarCoordinate
    H_max : float
    T_water_init_C, T_deep : float
    S_uniform : float

    Returns
    -------
    LatLonCGridOceanState
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels
    dtype = get_policy().storage

    H_bathy = jnp.full((n_lat, n_lon), H_max, dtype=dtype)

    T_profile = T_deep + (T_water_init_C - T_deep) * jnp.exp(
        z_coord.z_full_ref / _SCALE_DEPTH,
    )
    T_3d = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, :], (n_lat, n_lon, nlev),
    ).astype(dtype)

    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)

    u_zeros = jnp.zeros((n_lat, n_lon + 1, nlev), dtype=dtype)
    v_zeros = jnp.zeros((n_lat + 1, n_lon, nlev), dtype=dtype)
    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)

    land_mask = wall_mask.astype(dtype)
    u_mask, v_mask = compute_face_masks(land_mask)

    # Initialize vertical velocity with zeros (will be computed during step)
    w_zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)

    dims_u = ("lat", "lon_u", "level")
    dims_v = ("lat_v", "lon", "level")
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    dims_u2d = ("lat", "lon_u")
    dims_v2d = ("lat_v", "lon")

    return LatLonCGridOceanState(
        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
                staggering="edge"),
        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
                staggering="edge"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
    )


def replace_land_mask(
    state: LatLonCGridOceanState,
    new_land_mask: jnp.ndarray,
) -> LatLonCGridOceanState:
    """Replace land_mask and recompute u_mask/v_mask atomically.

    Use this instead of ``state._replace(land_mask=...)`` to ensure
    face masks stay consistent with the cell mask.
    """
    new_land_mask = jnp.asarray(new_land_mask)
    u_mask, v_mask = compute_face_masks(new_land_mask)
    return state._replace(
        land_mask=Field(data=new_land_mask, name="land_mask",
                        dims=state.land_mask.dims, units=""),
        u_mask=Field(data=u_mask, name="u_mask",
                     dims=state.u_mask.dims, units=""),
        v_mask=Field(data=v_mask, name="v_mask",
                     dims=state.v_mask.dims, units=""),
    )
