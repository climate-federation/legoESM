"""DCMIP-2025 Test Case 3: Squall line.

Moist non-hydrostatic dynamics on a small Earth (radius/60) with
Kessler warm-rain microphysics. A line of warm bubbles triggers
convection that organizes into a squall line.

Setup:
- Small Earth: R = R_earth / 60
- Background thermodynamic sounding with CAPE ~2000 J/kg
- Background wind shear
- Line of warm thermal bubbles along constant longitude
- Cyclostrophic balance (no Coriolis forcing)
- 3 tracers: q_vapor, q_cloud, q_rain

References
----------
- DCMIP-2025: https://sites.google.com/umich.edu/dcmip-2025/
- Klemp et al. (2015): Idealized Global Nonhydrostatic Atmospheric
  Test Cases on a Reduced-Radius Sphere.
- Zarzycki et al. (2019): DCMIP2016 Splitting Supercell Test Case.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import (
    HeightCoordinate,
    TerrainMetric,
    create_height_coordinate,
    compute_terrain_metric,
)
from legoesm.atmosphere.dynamics.dcmip2025.common import (
    apply_small_earth_scaling,
)
from legoesm.atmosphere.physics.kessler import saturation_mixing_ratio
from legoesm import constants


# Default parameters
TC3_PARAMS = {
    "small_earth_factor": 60.0,
    "H": 20000.0,                     # Model top [m]
    "n_levels": 40,
    # Sounding parameters
    "T_s": 302.0,                     # Surface temperature [K]
    "T_tropopause": 213.0,            # Tropopause temperature [K]
    "z_tropopause": 12000.0,          # Tropopause height [m]
    "p_s": 1.0e5,                     # Surface pressure [Pa]
    "RH_low": 0.95,                   # Surface relative humidity
    "RH_high": 0.0,                   # Stratospheric relative humidity
    "RH_transition_z": 8000.0,        # RH transition height [m]
    # Wind shear
    "U_s": 30.0,                      # Shear magnitude [m/s]
    "z_s": 5000.0,                    # Shear layer height [m]
    "U_c": 0.0,                       # Surface velocity [m/s]
    # Bubbles
    "n_bubbles": 9,
    "bubble_dtheta": 3.0,             # Perturbation amplitude [K]
    "bubble_rh": 5.0e3,               # Horizontal half-width [m]
    "bubble_rz": 1500.0,              # Vertical half-width [m]
    "bubble_zc": 1500.0,              # Bubble center height [m]
    "bubble_spacing": 10.0e3,         # Spacing between bubbles [m]
    "bubble_lon": jnp.pi,             # Bubble line longitude [rad]
    # Sponge
    "sponge_width": 5000.0,
    "sponge_coeff": 0.05,
}


def _squall_line_sounding(z, params):
    """Compute thermodynamic sounding for the squall line.

    Returns temperature, potential temperature, and pressure at height z.
    Uses a simple tropospheric lapse rate transitioning to isothermal
    stratosphere.
    """
    T_s = params["T_s"]
    T_tr = params["T_tropopause"]
    z_tr = params["z_tropopause"]
    p_s = params["p_s"]
    g = constants.g
    R_d = constants.R_d
    kappa = constants.kappa
    p_0 = constants.p_ref

    # Lapse rate in troposphere
    gamma = (T_s - T_tr) / z_tr

    # Temperature profile
    T_tropo = T_s - gamma * jnp.minimum(z, z_tr)
    T = jnp.where(z <= z_tr, T_tropo, T_tr)

    # Pressure from hydrostatic integration
    # Tropospheric: p = p_s * (T/T_s)^(g/(R_d*gamma))
    T_ratio = jnp.clip(T_tropo, 100.0, None) / T_s
    exponent = g / (R_d * gamma)
    p_tropo = p_s * T_ratio ** exponent

    # At tropopause
    T_at_tr = T_s - gamma * z_tr
    p_at_tr = p_s * (jnp.clip(T_at_tr, 100.0, None) / T_s) ** exponent

    # Stratospheric: isothermal
    dz_above = jnp.maximum(z - z_tr, 0.0)
    p_strato = p_at_tr * jnp.exp(-g * dz_above / (R_d * T_tr))

    p = jnp.where(z <= z_tr, p_tropo, p_strato)

    # Potential temperature
    theta = T * (p_0 / p) ** kappa

    return T, theta, p


def _squall_line_theta_fn(params):
    """Return theta_0(z) function for the squall line sounding."""
    def theta_fn(z):
        _, theta, _ = _squall_line_sounding(z, params)
        return theta
    return theta_fn


def dcmip25_tc3_init(
    grid: CubedSphereGrid,
    n_levels: int = 40,
    params: dict | None = None,
) -> tuple[NonHydrostaticState, HeightCoordinate, TerrainMetric, CubedSphereGrid]:
    """Initialize DCMIP-2025 Test Case 3 (squall line).

    Parameters
    ----------
    grid : CubedSphereGrid
        Original Earth-sized grid.
    n_levels : int
        Number of vertical levels.
    params : dict, optional
        Override default parameters.

    Returns
    -------
    state : NonHydrostaticState
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    small_grid : CubedSphereGrid
        Grid with small-Earth scaling applied.
    """
    p = {**TC3_PARAMS, **(params or {})}

    # Small-Earth scaling
    small_grid = apply_small_earth_scaling(grid, p["small_earth_factor"])

    # Vertical coordinate with squall line sounding
    theta_fn = _squall_line_theta_fn(p)
    height_coord = create_height_coordinate(n_levels, p["H"], theta_fn)

    # Flat terrain (no topography for squall line)
    z_s = jnp.zeros((6, small_grid.n, small_grid.n))
    terrain_metric = compute_terrain_metric(z_s, height_coord)

    # --- Initial conditions ---
    shape_3d = (6, small_grid.n, small_grid.n, n_levels)
    shape_w = (6, small_grid.n, small_grid.n, n_levels + 1)
    shape_2d = (6, small_grid.n, small_grid.n)

    # Compute sounding at each level
    z_full = height_coord.z_full  # (nlev,)
    T_sounding, theta_sounding, p_sounding = _squall_line_sounding(z_full, p)

    # Background wind shear: u(z) = U_c + U_s * min(z/z_s, 1)
    u_profile = p["U_c"] + p["U_s"] * jnp.minimum(z_full / p["z_s"], 1.0)
    u_data = jnp.ones(shape_3d) * u_profile[None, None, None, :]
    v_data = jnp.zeros(shape_3d)
    w_data = jnp.zeros(shape_w)

    # Moisture: relative humidity profile
    RH_profile = p["RH_low"] * jnp.exp(-z_full / p["RH_transition_z"])
    RH_profile = jnp.clip(RH_profile, p["RH_high"], p["RH_low"])

    # Saturation mixing ratio at each level
    q_sat_profile = saturation_mixing_ratio(T_sounding, p_sounding)
    q_v_profile = RH_profile * q_sat_profile

    # 3D moisture fields
    q_v_data = jnp.ones(shape_3d) * q_v_profile[None, None, None, :]
    q_c_data = jnp.zeros(shape_3d)
    q_r_data = jnp.zeros(shape_3d)

    # --- Warm bubbles ---
    # Line of bubbles along constant longitude at equator
    n_bubbles = p["n_bubbles"]
    bubble_dtheta = p["bubble_dtheta"]
    r_h = p["bubble_rh"]
    r_z = p["bubble_rz"]
    z_c = p["bubble_zc"]
    spacing = p["bubble_spacing"]
    lon_c = p["bubble_lon"]

    lat = small_grid.lat  # (6, n, n)
    lon = small_grid.lon

    # Total theta perturbation from all bubbles
    theta_pert = jnp.zeros(shape_3d)
    for i in range(n_bubbles):
        # Bubble center latitude: evenly spaced around equator
        lat_c = (i - n_bubbles // 2) * spacing / small_grid.radius

        # Horizontal distance (great circle)
        dlat = lat - lat_c
        dlon = lon - lon_c
        dlon = jnp.mod(dlon + jnp.pi, 2 * jnp.pi) - jnp.pi
        x_dist = dlon * small_grid.radius * jnp.cos(lat)
        y_dist = dlat * small_grid.radius
        r_horiz = jnp.sqrt(x_dist**2 + y_dist**2)

        # Vertical distance
        z_dist = z_full[None, None, None, :] - z_c

        # Bubble shape: cos^2 envelope
        r_norm = jnp.sqrt((r_horiz[..., None] / r_h) ** 2 + (z_dist / r_z) ** 2)
        bubble = bubble_dtheta * jnp.where(
            r_norm <= 1.0,
            jnp.cos(0.5 * jnp.pi * r_norm) ** 2,
            0.0,
        )
        theta_pert = theta_pert + bubble

    # The bubble is a perturbation on top of the reference theta
    theta_prime_data = theta_pert
    rho_prime_data = jnp.zeros(shape_3d)

    # Stack tracers: vapor, cloud, rain
    tracers_data = jnp.stack([q_v_data, q_c_data, q_r_data], axis=-1)

    phis_data = jnp.zeros(shape_2d)  # Flat terrain

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=u_data, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_data, name="v", dims=dims_3d, units="m/s"),
        w=Field(data=w_data, name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(
            data=theta_prime_data, name="theta_prime", dims=dims_3d, units="K",
        ),
        rho_prime=Field(
            data=rho_prime_data, name="rho_prime", dims=dims_3d, units="kg/m^3",
        ),
        phis=Field(data=phis_data, name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(
            data=tracers_data, name="tracers",
            dims=("face", "x", "y", "level", "tracer"), units="kg/kg",
        ),
    )

    return state, height_coord, terrain_metric, small_grid
