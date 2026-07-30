"""DCMIP-2025 Test Case 3 initialization for MPAS Voronoi mesh.

Squall line on a small Earth (radius/60) with warm bubbles.
Moist non-hydrostatic dynamics with 3 tracers: q_vapor, q_cloud, q_rain.

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
from legoesm.thermo import saturation_mixing_ratio
from legoesm import constants


# Default parameters (same as cubed-sphere TC3)
TC3_PARAMS = {
    "small_earth_factor": 60.0,
    # Case rotation, read by the initialiser below so an explicit
    # override reaches the mesh (codex r3 P2: omega was hard-coded).
    "rotating": False,
    "H": 20000.0,
    "n_levels": 40,
    "T_s": 302.0,
    "T_tropopause": 213.0,
    "z_tropopause": 12000.0,
    "p_s": 1.0e5,
    "RH_low": 0.95,
    "RH_high": 0.0,
    "RH_transition_z": 8000.0,
    "U_s": 30.0,
    "z_s": 5000.0,
    "U_c": 0.0,
    "n_bubbles": 9,
    "bubble_dtheta": 3.0,
    "bubble_rh": 5.0e3,
    "bubble_rz": 1500.0,
    "bubble_zc": 1500.0,
    "bubble_spacing": 10.0e3,
    "bubble_lon": jnp.pi,
    "sponge_width": 5000.0,
    "sponge_coeff": 0.05,
}


def _squall_line_sounding(z, params):
    """Compute thermodynamic sounding for the squall line."""
    T_s = params["T_s"]
    T_tr = params["T_tropopause"]
    z_tr = params["z_tropopause"]
    p_s = params["p_s"]
    g = constants.g
    R_d = constants.R_d
    kappa = constants.kappa
    p_0 = constants.p_ref

    gamma = (T_s - T_tr) / z_tr
    T_tropo = T_s - gamma * jnp.minimum(z, z_tr)
    T = jnp.where(z <= z_tr, T_tropo, T_tr)

    T_ratio = jnp.clip(T_tropo, 100.0, None) / T_s
    exponent = g / (R_d * gamma)
    p_tropo = p_s * T_ratio ** exponent

    T_at_tr = T_s - gamma * z_tr
    p_at_tr = p_s * (jnp.clip(T_at_tr, 100.0, None) / T_s) ** exponent

    dz_above = jnp.maximum(z - z_tr, 0.0)
    p_strato = p_at_tr * jnp.exp(-g * dz_above / (R_d * T_tr))

    p = jnp.where(z <= z_tr, p_tropo, p_strato)
    theta = T * (p_0 / p) ** kappa

    return T, theta, p


def _squall_line_theta_fn(params):
    """Return theta_0(z) function for the squall line sounding."""
    def theta_fn(z):
        _, theta, _ = _squall_line_sounding(z, params)
        return theta
    return theta_fn


def dcmip25_tc3_init_mpas(
    mesh: VoronoiMesh,
    n_levels: int = 40,
    params: dict | None = None,
) -> tuple[MPASNonHydrostaticState, HeightCoordinate, TerrainMetric, VoronoiMesh]:
    """Initialize DCMIP-2025 Test Case 3 (squall line) on MPAS Voronoi mesh.

    Parameters
    ----------
    mesh : VoronoiMesh
        Original Earth-sized mesh.
    n_levels : int
        Number of vertical levels.
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
    p = {**TC3_PARAMS, **(params or {})}

    # Create small-Earth mesh
    factor = p["small_earth_factor"]
    level = round(jnp.log2(jnp.sqrt(mesh.nCells / 12.0)).item())
    small_mesh = create_voronoi_mesh(
        level,
        radius=constants.R_earth / factor,
        # No Coriolis for the squall line (DCMIP TC3 spec); driven by
        # the case dict so an override is honoured, not hard-coded.
        omega=(constants.Omega * factor
               if p.get("rotating", True) else 0.0),
    )

    # Vertical coordinate with squall line sounding
    theta_fn = _squall_line_theta_fn(p)
    height_coord = create_height_coordinate(n_levels, p["H"], theta_fn)

    # Flat terrain
    z_s = jnp.zeros(small_mesh.nCells)
    terrain_metric = compute_terrain_metric(z_s, height_coord)

    # --- Initial conditions ---
    nCells = small_mesh.nCells
    nEdges = small_mesh.nEdges

    # Compute sounding at each level
    z_full = height_coord.z_full  # (nlev,)
    T_sounding, _, p_sounding = _squall_line_sounding(z_full, p)

    # Background wind shear: u_east(z) = U_c + U_s * min(z/z_s, 1)
    u_profile = p["U_c"] + p["U_s"] * jnp.minimum(z_full / p["z_s"], 1.0)

    # Edge-normal velocity: project zonal wind onto edge normals
    # u_n = u_east(z) * cos(angleEdge) (since v_north = 0)
    cos_angle = jnp.cos(small_mesh.angleEdge)  # (nEdges,)
    u_data = cos_angle[:, None] * u_profile[None, :]  # (nEdges, nlev)

    # Vertical velocity: w = 0
    w_data = jnp.zeros((nCells, n_levels + 1))

    # Moisture: relative humidity profile
    RH_profile = p["RH_low"] * jnp.exp(-z_full / p["RH_transition_z"])
    RH_profile = jnp.clip(RH_profile, p["RH_high"], p["RH_low"])
    q_sat_profile = saturation_mixing_ratio(T_sounding, p_sounding)
    q_v_profile = RH_profile * q_sat_profile

    # 3D moisture fields (uniform horizontally)
    q_v_data = jnp.ones((nCells, n_levels)) * q_v_profile[None, :]
    q_c_data = jnp.zeros((nCells, n_levels))
    q_r_data = jnp.zeros((nCells, n_levels))

    # --- Warm bubbles ---
    n_bubbles = p["n_bubbles"]
    bubble_dtheta = p["bubble_dtheta"]
    r_h = p["bubble_rh"]
    r_z = p["bubble_rz"]
    z_c = p["bubble_zc"]
    spacing = p["bubble_spacing"]
    lon_c = p["bubble_lon"]

    lat = small_mesh.latCell  # (nCells,)
    lon = small_mesh.lonCell

    theta_pert = jnp.zeros((nCells, n_levels))
    for i in range(n_bubbles):
        lat_c = (i - n_bubbles // 2) * spacing / small_mesh.radius

        dlat = lat - lat_c
        dlon = lon - lon_c
        dlon = jnp.mod(dlon + jnp.pi, 2 * jnp.pi) - jnp.pi
        x_dist = dlon * small_mesh.radius * jnp.cos(lat)
        y_dist = dlat * small_mesh.radius
        r_horiz = jnp.sqrt(x_dist**2 + y_dist**2)

        z_dist = z_full[None, :] - z_c  # (1, nlev)

        r_norm = jnp.sqrt(
            (r_horiz[:, None] / r_h) ** 2 + (z_dist / r_z) ** 2
        )
        bubble = bubble_dtheta * jnp.where(
            r_norm <= 1.0,
            jnp.cos(0.5 * jnp.pi * r_norm) ** 2,
            0.0,
        )
        theta_pert = theta_pert + bubble

    theta_prime_data = theta_pert
    rho_prime_data = jnp.zeros((nCells, n_levels))

    # Stack tracers: vapor, cloud, rain
    tracers_data = jnp.stack([q_v_data, q_c_data, q_r_data], axis=-1)

    phis_data = jnp.zeros(nCells)

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
