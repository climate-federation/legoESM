"""DCMIP-2025 Test Case 2 initialization for MPAS Voronoi mesh.

Mountain-triggered mesoscale flow on a small Earth (radius/20).
Subcases:
  2a: Gap flow through a mountain chain
  2b: Vortex shedding from an isolated mountain

Same physics as the cubed-sphere version but adapted for MPAS state types
with edge-normal velocity on a C-grid.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASNonHydrostaticState
from legoesm.grids.voronoi import VoronoiMesh, create_voronoi_mesh
from legoesm.grids.vertical import (
    HeightCoordinate,
    TerrainMetric,
    create_height_coordinate,
    compute_terrain_metric,
)
from .common import isothermal_theta_ref
from legoesm import constants


# Default parameters (same as cubed-sphere TC2)
TC2_PARAMS = {
    "small_earth_factor": 20.0,
    # Case rotation, read by the initialiser below so an explicit
    # override reaches the mesh (codex r3 P2: omega was hard-coded).
    "rotating": True,
    "T0": 250.0,
    "u0": 20.0,
    "H": 30000.0,
    "sponge_width": 15000.0,
    "sponge_coeff": 1.0 / (0.1 * 86400.0),
    # Gap flow (2a)
    "chain_h0": 2000.0,
    "chain_halfwidth_lon": 50.0e3,
    "chain_halfwidth_lat": 500.0e3,
    "gap_halfwidth": 50.0e3,
    "chain_lon": jnp.pi,
    "gap_lat": 0.0,
    # Vortex shedding (2b)
    "mountain_h0": 2000.0,
    "mountain_d": 50.0e3,
    "mountain_lat": 10.0 * jnp.pi / 180.0,
    "mountain_lon": jnp.pi,
}


def _mountain_chain_with_gap_mpas(
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    radius: float,
    h0: float,
    chain_halfwidth_lon: float,
    chain_halfwidth_lat: float,
    gap_halfwidth: float,
    lon0: float,
    lat0: float,
) -> jnp.ndarray:
    """Mountain chain with gap for TC2a on unstructured grid."""
    dlon = lon - lon0
    dlon = jnp.mod(dlon + jnp.pi, 2 * jnp.pi) - jnp.pi
    x_dist = dlon * radius * jnp.cos(lat)
    y_dist = (lat - lat0) * radius
    chain = jnp.exp(-(x_dist / chain_halfwidth_lon) ** 2)
    lat_envelope = jnp.exp(-(y_dist / chain_halfwidth_lat) ** 4)
    gap = 1.0 - jnp.exp(-(y_dist / gap_halfwidth) ** 2)
    return h0 * chain * lat_envelope * gap


def _gaussian_mountain_mpas(
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    radius: float,
    h0: float,
    d: float,
    lat0: float,
    lon0: float,
) -> jnp.ndarray:
    """Gaussian mountain for TC2b on unstructured grid."""
    dlat = lat - lat0
    dlon = lon - lon0
    a = (jnp.sin(dlat / 2) ** 2
         + jnp.cos(lat) * jnp.cos(lat0) * jnp.sin(dlon / 2) ** 2)
    angular_dist = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0.0, 1.0)))
    dist = angular_dist * radius
    return h0 * jnp.exp(-(dist / d) ** 2)


def dcmip25_tc2_init_mpas(
    mesh: VoronoiMesh,
    n_levels: int = 48,
    subcase: str = "a",
    params: dict | None = None,
) -> tuple[MPASNonHydrostaticState, HeightCoordinate, TerrainMetric, VoronoiMesh]:
    """Initialize DCMIP-2025 Test Case 2 on MPAS Voronoi mesh.

    Parameters
    ----------
    mesh : VoronoiMesh
        Original Earth-sized mesh.
    n_levels : int
        Number of vertical levels.
    subcase : str
        "a" for gap flow, "b" for vortex shedding.
    params : dict, optional
        Override default parameters.

    Returns
    -------
    state : MPASNonHydrostaticState
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    small_mesh : VoronoiMesh
        Mesh with small-Earth scaling applied.
    """
    p = {**TC2_PARAMS, **(params or {})}

    # Create small-Earth mesh
    factor = p["small_earth_factor"]
    level = round(jnp.log2(jnp.sqrt(mesh.nCells / 12.0)).item())
    small_mesh = create_voronoi_mesh(
        level,
        radius=constants.R_earth / factor,
        omega=(constants.Omega * factor
               if p.get("rotating", True) else 0.0),
    )

    # Isothermal reference state
    theta_fn = isothermal_theta_ref(T0=p["T0"])
    height_coord = create_height_coordinate(n_levels, p["H"], theta_fn)

    # Topography (nCells,)
    if subcase == "a":
        z_s = _mountain_chain_with_gap_mpas(
            small_mesh.latCell, small_mesh.lonCell, small_mesh.radius,
            h0=p["chain_h0"],
            chain_halfwidth_lon=p["chain_halfwidth_lon"],
            chain_halfwidth_lat=p["chain_halfwidth_lat"],
            gap_halfwidth=p["gap_halfwidth"],
            lon0=p["chain_lon"],
            lat0=p["gap_lat"],
        )
    elif subcase == "b":
        z_s = _gaussian_mountain_mpas(
            small_mesh.latCell, small_mesh.lonCell, small_mesh.radius,
            h0=p["mountain_h0"],
            d=p["mountain_d"],
            lat0=p["mountain_lat"],
            lon0=p["mountain_lon"],
        )
    else:
        raise ValueError(f"Unknown subcase: {subcase!r}. Use 'a' or 'b'.")

    terrain_metric = compute_terrain_metric(z_s, height_coord)

    # --- Initial conditions ---
    nCells = small_mesh.nCells
    nEdges = small_mesh.nEdges

    # Edge-normal velocity: u_n = u0 * cos(lat_edge) * cos(angleEdge)
    u0 = p["u0"]
    u_edge_2d = u0 * jnp.cos(small_mesh.latEdge) * jnp.cos(small_mesh.angleEdge)
    u_data = jnp.broadcast_to(u_edge_2d[:, None], (nEdges, n_levels))

    # Vertical velocity: w = 0 at half-levels
    w_data = jnp.zeros((nCells, n_levels + 1))

    # Perturbations: zero
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

    return state, height_coord, terrain_metric, small_mesh
