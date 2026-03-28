"""DCMIP-2025 Test Case 1 initialization for MPAS Voronoi mesh.

Mountain-triggered breaking gravity waves on an icosahedral grid.
Same physics as the cubed-sphere version but adapted for MPAS state types
with edge-normal velocity on a C-grid.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASNonHydrostaticState
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.grids.vertical import (
    HeightCoordinate,
    TerrainMetric,
    create_height_coordinate,
    compute_terrain_metric,
)
from .common import piecewise_lapse_theta_ref, schaer_mountain_profile
from legoesm import constants


# Default parameters (same as cubed-sphere TC1)
TC1_PARAMS = {
    "T_s": 300.0,
    "lapse_tropo": -5.0e-3,
    "lapse_strato": 5.0e-3,
    "z_tropopause": 20000.0,
    "H": 40000.0,
    "u0": 20.0,
    "mountain_lat": 20.0 * jnp.pi / 180.0,
    "mountain_lon": 0.0,
    "mountain_height": 2000.0,
    "mountain_halfwidth": 72.0e3,
}


def dcmip25_tc1_init_mpas(
    mesh: VoronoiMesh,
    n_levels: int = 88,
    params: dict | None = None,
) -> tuple[MPASNonHydrostaticState, HeightCoordinate, TerrainMetric]:
    """Initialize DCMIP-2025 Test Case 1 on MPAS Voronoi mesh.

    Parameters
    ----------
    mesh : VoronoiMesh
        MPAS mesh.
    n_levels : int
        Number of vertical levels.
    params : dict, optional
        Override default parameters.

    Returns
    -------
    state : MPASNonHydrostaticState
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    """
    p = {**TC1_PARAMS, **(params or {})}

    # Reference state
    theta_fn = piecewise_lapse_theta_ref(
        T_s=p["T_s"],
        lapse_tropo=p["lapse_tropo"],
        lapse_strato=p["lapse_strato"],
        z_tropopause=p["z_tropopause"],
    )

    # Vertical coordinate
    height_coord = create_height_coordinate(n_levels, p["H"], theta_fn)

    # Mountain topography (nCells,)
    z_s = schaer_mountain_profile(
        mesh.latCell, mesh.lonCell, mesh.radius,
        h0=p["mountain_height"],
        halfwidth=p["mountain_halfwidth"],
        lat0=p["mountain_lat"],
        lon0=p["mountain_lon"],
    )

    # Terrain metric
    terrain_metric = compute_terrain_metric(z_s, height_coord)

    # Initial conditions
    nCells = mesh.nCells
    nEdges = mesh.nEdges

    # Edge-normal velocity: u_n = u0 * cos(lat_edge) * cos(angleEdge)
    # (zonal wind projected onto edge normals)
    u0 = p["u0"]
    u_edge_2d = u0 * jnp.cos(mesh.latEdge) * jnp.cos(mesh.angleEdge)
    u_data = jnp.broadcast_to(u_edge_2d[:, None], (nEdges, n_levels))

    # Vertical velocity: w = 0 at half-levels
    w_data = jnp.zeros((nCells, n_levels + 1))

    # Perturbations: zero (atmosphere in balance)
    theta_prime_data = jnp.zeros((nCells, n_levels))
    rho_prime_data = jnp.zeros((nCells, n_levels))

    # Surface geopotential
    phis_data = constants.g * z_s

    # No tracers
    tracers_data = jnp.zeros((nCells, n_levels, 0))

    state = MPASNonHydrostaticState(
        u=Field(data=u_data, name="u", dims=("nEdges", "level"), units="m/s"),
        w=Field(data=w_data, name="w", dims=("nCells", "level_half"), units="m/s"),
        theta_prime=Field(
            data=theta_prime_data, name="theta_prime",
            dims=("nCells", "level"), units="K",
        ),
        rho_prime=Field(
            data=rho_prime_data, name="rho_prime",
            dims=("nCells", "level"), units="kg/m^3",
        ),
        phis=Field(data=phis_data, name="phis", dims=("nCells",), units="m^2/s^2"),
        tracers=Field(
            data=tracers_data, name="tracers",
            dims=("nCells", "level", "tracer"), units="kg/kg",
        ),
    )

    return state, height_coord, terrain_metric
