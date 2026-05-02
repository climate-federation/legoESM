"""Initialization routines for MPAS ocean on Voronoi meshes.

Provides idealized bathymetry, rest-state initial conditions,
and velocity reconstruction utilities.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.precision import get_policy
from legoesm.core.state import MPASOceanState
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.vertical import OceanZStarCoordinate


def idealized_bathymetry_mpas(
    mesh: VoronoiMesh,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
):
    """Generate idealized bathymetry on Voronoi mesh.

    Parameters
    ----------
    mesh : VoronoiMesh
    H_max : float
        Uniform ocean depth [m].
    land_lat_threshold : float
        Latitude [deg] above which is land.

    Returns
    -------
    H_bathy : jnp.ndarray, shape (nCells,)
        Bathymetry depth, positive downward.
    land_mask : jnp.ndarray, shape (nCells,)
        1=ocean, 0=land.
    """
    dtype = get_policy().storage
    lat_deg = jnp.abs(jnp.degrees(mesh.latCell))
    land_mask = (lat_deg < land_lat_threshold).astype(dtype)
    # H_bathy = H_max everywhere (including land) so that the z-star
    # Jacobian (eta + H_bathy) / H_max is smooth across coastlines.
    # The land_mask prevents actual flow on land cells.
    H_bathy = jnp.full_like(land_mask, H_max)
    return H_bathy, land_mask


def rest_state_mpas_ocean(
    mesh: VoronoiMesh,
    z_coord: OceanZStarCoordinate,
    T_surface: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
) -> MPASOceanState:
    """Create a rest-state initial condition on Voronoi mesh.

    Temperature follows an exponential profile from surface to depth.
    Salinity is uniform. Velocity and SSH are zero.

    Parameters
    ----------
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    T_surface : float
        Surface temperature [degC].
    T_deep : float
        Deep temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].
    H_max : float
        Ocean depth [m].
    land_lat_threshold : float
        Land above this latitude [deg].

    Returns
    -------
    MPASOceanState
    """
    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = z_coord.n_levels

    H_bathy, land_mask = idealized_bathymetry_mpas(mesh, H_max, land_lat_threshold)

    dtype = get_policy().storage

    # Temperature: exponential profile
    z_full = z_coord.z_full_ref  # (nlev,), negative values
    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(z_full / _SCALE_DEPTH)
    T_data = jnp.broadcast_to(T_profile[jnp.newaxis, :], (nCells, nlev)).astype(dtype)

    # Salinity: uniform
    S_data = jnp.full((nCells, nlev), S_uniform, dtype=dtype)

    # Velocity: zero
    u_data = jnp.zeros((nEdges, nlev), dtype=dtype)

    # SSH: zero
    eta_data = jnp.zeros(nCells, dtype=dtype)

    w_data = jnp.zeros((nCells, nlev + 1), dtype=dtype)

    return MPASOceanState(
        u=Field(data=u_data, name="u", dims=("nEdges", "nlev"), units="m/s",
                staggering="edge"),
        T=Field(data=T_data, name="T", dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data, name="S", dims=("nCells", "nlev"), units="PSU"),
        eta=Field(data=eta_data, name="eta", dims=("nCells",), units="m"),
        w=Field(data=w_data, name="w", dims=("nCells", "nlev+1"), units="m/s"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=("nCells",), units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=("nCells",), units="1"),
    )


def wind_driven_gyre_mpas(
    mesh: VoronoiMesh,
    z_coord: OceanZStarCoordinate,
    H_max: float = 5500.0,
    lon_west: float = 0.0,
    lon_east: float = 120.0,
    lat_south: float = 15.0,
    lat_north: float = 75.0,
    T_uniform: float = 10.0,
    S_uniform: float = 35.0,
) -> MPASOceanState:
    """Create initial condition for a wind-driven barotropic gyre on MPAS.

    Uniform T and S inside a rectangular basin. Purely barotropic setup.
    Wind forcing is applied during integration via the MPAS physics pipeline.

    Parameters
    ----------
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    H_max : float
        Maximum ocean depth [m].
    lon_west, lon_east : float
        Basin longitude bounds [degrees].
    lat_south, lat_north : float
        Basin latitude bounds [degrees].
    T_uniform : float
        Uniform temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].

    Returns
    -------
    MPASOceanState
    """
    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = z_coord.n_levels
    dtype = get_policy().storage

    # Basin land mask: ocean inside rectangle, land outside
    lon_deg = jnp.degrees(mesh.lonCell)  # (nCells,)
    lat_deg = jnp.degrees(mesh.latCell)  # (nCells,)
    in_basin = (
        (lon_deg >= lon_west) & (lon_deg <= lon_east) &
        (lat_deg >= lat_south) & (lat_deg <= lat_north)
    )
    land_mask = jnp.where(in_basin, 1.0, 0.0).astype(dtype)

    # Uniform depth everywhere (smooth Jacobian at coastlines)
    H_bathy = jnp.full(nCells, H_max, dtype=dtype)

    # Uniform T and S - apply land masking to set land cells to 0.0°C
    T_data = jnp.full((nCells, nlev), T_uniform, dtype=dtype)
    S_data = jnp.full((nCells, nlev), S_uniform, dtype=dtype)

    # Apply land masking: land cells (mask <= 0.5) set to 0.0°C
    mask_3d = land_mask[:, jnp.newaxis]  # Broadcast to 3D
    T_data = jnp.where(mask_3d > 0.5, T_data, 0.0)
    S_data = jnp.where(mask_3d > 0.5, S_data, 0.0)

    # Zero velocity and SSH
    u_data = jnp.zeros((nEdges, nlev), dtype=dtype)
    eta_data = jnp.zeros(nCells, dtype=dtype)

    w_data = jnp.zeros((nCells, nlev + 1), dtype=dtype)

    return MPASOceanState(
        u=Field(data=u_data, name="u", dims=("nEdges", "nlev"), units="m/s",
                staggering="edge"),
        T=Field(data=T_data, name="T", dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data, name="S", dims=("nCells", "nlev"), units="PSU"),
        eta=Field(data=eta_data, name="eta", dims=("nCells",), units="m"),
        w=Field(data=w_data, name="w", dims=("nCells", "nlev+1"), units="m/s"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=("nCells",), units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=("nCells",), units="1"),
    )


def reconstruct_cell_velocity(u_edge, mesh):
    """Reconstruct (u_east, v_north) at cell centers from edge normals.

    Uses the Perot reconstruction: project edge-normal velocities onto
    zonal/meridional directions, weighted by ``dvEdge * dcEdge / (2 * areaCell)``.
    This formula is exact for uniform flow on any Voronoi mesh.

    Note: ``edgeSignOnCell`` is NOT used here — it is needed for the
    divergence operator (flux balance) but not for velocity reconstruction.
    The edge-normal velocity ``u_edge`` already follows the edge's own
    normal direction (defined by ``angleEdge``), so projecting with
    ``cos(angleEdge)`` / ``sin(angleEdge)`` directly gives the correct
    eastward/northward components.

    Parameters
    ----------
    u_edge : jax.Array, shape (nEdges,) or (nEdges, nlev)
        Normal velocity at edges.
    mesh : VoronoiMesh

    Returns
    -------
    u_east : jax.Array, shape (nCells,) or (nCells, nlev)
        Zonal velocity at cell centers.
    v_north : jax.Array, shape (nCells,) or (nCells, nlev)
        Meridional velocity at cell centers.
    """
    is_3d = u_edge.ndim == 2

    eoc = mesh.edgesOnCell  # (maxEdges, nCells)
    mask = (eoc >= 0).astype(u_edge.dtype)  # (maxEdges, nCells)
    eoc_safe = jnp.maximum(eoc, 0)

    # Reconstruction weight: dvEdge * dcEdge / (2 * areaCell)
    dv = mesh.dvEdge[eoc_safe] * mask  # (maxEdges, nCells)
    dc = mesh.dcEdge[eoc_safe] * mask  # (maxEdges, nCells)
    angle = mesh.angleEdge[eoc_safe]   # (maxEdges, nCells)
    weight = dv * dc / (2.0 * mesh.areaCell[jnp.newaxis, :])  # (maxEdges, nCells)

    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    if is_3d:
        u_gathered = u_edge[eoc_safe] * mask[..., jnp.newaxis]
        w3d = weight[..., jnp.newaxis]
        u_east = jnp.sum(u_gathered * w3d * cos_a[..., jnp.newaxis], axis=0)
        v_north = jnp.sum(u_gathered * w3d * sin_a[..., jnp.newaxis], axis=0)
    else:
        u_gathered = u_edge[eoc_safe] * mask
        u_east = jnp.sum(u_gathered * weight * cos_a, axis=0)
        v_north = jnp.sum(u_gathered * weight * sin_a, axis=0)

    return u_east, v_north
