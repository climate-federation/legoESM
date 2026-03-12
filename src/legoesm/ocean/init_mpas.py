"""Initialization routines for MPAS ocean on Voronoi meshes.

Provides idealized bathymetry, rest-state initial conditions,
and velocity reconstruction utilities.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASOceanState
from legoesm.grids.voronoi import VoronoiMesh
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
    lat_deg = jnp.abs(jnp.degrees(mesh.latCell))
    land_mask = (lat_deg < land_lat_threshold).astype(jnp.float64)
    H_bathy = H_max * land_mask
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

    # Temperature: exponential profile
    scale_depth = 1000.0  # meters
    z_full = z_coord.z_full_ref  # (nlev,), negative values
    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(z_full / scale_depth)
    T_data = jnp.broadcast_to(T_profile[jnp.newaxis, :], (nCells, nlev))

    # Salinity: uniform
    S_data = jnp.full((nCells, nlev), S_uniform)

    # Velocity: zero
    u_data = jnp.zeros((nEdges, nlev))

    # SSH: zero
    eta_data = jnp.zeros(nCells)

    return MPASOceanState(
        u=Field(data=u_data, name="u", dims=("nEdges", "nlev"), units="m/s",
                staggering="edge"),
        T=Field(data=T_data, name="T", dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data, name="S", dims=("nCells", "nlev"), units="PSU"),
        eta=Field(data=eta_data, name="eta", dims=("nCells",), units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=("nCells",), units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=("nCells",), units="1"),
    )


def reconstruct_cell_velocity(u_edge, mesh):
    """Reconstruct (u_east, v_north) at cell centers from edge normals.

    Uses area-weighted projection of edge normal velocities onto
    zonal/meridional directions.

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
    sign = mesh.edgeSignOnCell  # (maxEdges, nCells)
    mask = (eoc >= 0).astype(u_edge.dtype)  # (maxEdges, nCells)
    eoc_safe = jnp.maximum(eoc, 0)

    dv = mesh.dvEdge[eoc_safe] * mask  # (maxEdges, nCells)
    angle = mesh.angleEdge[eoc_safe]  # (maxEdges, nCells)

    if is_3d:
        # u_edge: (nEdges, nlev) -> u_gathered: (maxEdges, nCells, nlev)
        u_gathered = u_edge[eoc_safe]  # (maxEdges, nCells, nlev)
        u_signed = u_gathered * sign[..., jnp.newaxis] * mask[..., jnp.newaxis]
        dv_3d = dv[..., jnp.newaxis]
        cos_a = jnp.cos(angle)[..., jnp.newaxis]
        sin_a = jnp.sin(angle)[..., jnp.newaxis]
        u_east = jnp.sum(u_signed * dv_3d * cos_a, axis=0) / mesh.areaCell[:, jnp.newaxis]
        v_north = jnp.sum(u_signed * dv_3d * sin_a, axis=0) / mesh.areaCell[:, jnp.newaxis]
    else:
        u_gathered = u_edge[eoc_safe] * sign * mask  # (maxEdges, nCells)
        u_east = jnp.sum(u_gathered * dv * jnp.cos(angle), axis=0) / mesh.areaCell
        v_north = jnp.sum(u_gathered * dv * jnp.sin(angle), axis=0) / mesh.areaCell

    return u_east, v_north
