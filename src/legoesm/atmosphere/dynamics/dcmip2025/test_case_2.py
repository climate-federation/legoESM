"""DCMIP-2025 Test Case 2: Mountain-triggered mesoscale flow phenomena.

Dry non-hydrostatic dynamics on a small Earth (radius/20) with two subcases:

2a: Gap flow through a mountain chain
    - Mountain chain with a gap, producing accelerated flow through the gap
    - Flowaround regime (inverse Froude number > 1)

2b: Vortex shedding from an isolated mountain
    - Von Karman vortex streets in the lee
    - Mountain positioned off-equator for asymmetry

Setup:
- Small Earth: R = R_earth / 20, Omega = Omega_earth * 20
- Isothermal atmosphere with constant Brunt-Vaisala frequency
- Background flow: u = u0 * cos(lat), solid body rotation
- Rayleigh friction sponge layer

References
----------
- DCMIP-2025: https://sites.google.com/umich.edu/dcmip-2025/
- Klemp et al. (2015): Idealized Global Nonhydrostatic Atmospheric
  Test Cases on a Reduced-Radius Sphere.
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
    isothermal_theta_ref,
    gaussian_mountain,
    mountain_chain_with_gap,
)
from legoesm import constants


# Default parameters
TC2_PARAMS = {
    "small_earth_factor": 20.0,      # Radius reduction factor
    "T0": 250.0,                     # Isothermal temperature [K]
    "u0": 20.0,                      # Background zonal wind [m/s]
    "H": 30000.0,                    # Model top [m]
    "sponge_width": 15000.0,         # Sponge layer width [m]
    "sponge_coeff": 1.0 / (0.1 * 86400.0),  # 0.1-day timescale [1/s]
    # Gap flow (2a) specific
    "chain_h0": 2000.0,              # Mountain chain height [m]
    "chain_halfwidth_lon": 50.0e3,   # E-W half-width [m] (on small Earth)
    "chain_halfwidth_lat": 500.0e3,  # N-S half-extent [m]
    "gap_halfwidth": 50.0e3,         # Gap half-width [m]
    "chain_lon": jnp.pi,             # Chain center longitude [rad]
    "gap_lat": 0.0,                  # Gap latitude [rad]
    # Vortex shedding (2b) specific
    "mountain_h0": 2000.0,           # Mountain height [m]
    "mountain_d": 50.0e3,            # Mountain half-width [m]
    "mountain_lat": 10.0 * jnp.pi / 180.0,  # Off-equator for asymmetry [rad]
    "mountain_lon": jnp.pi,
}


def dcmip25_tc2a_topography(
    grid: CubedSphereGrid,
    params: dict | None = None,
) -> jnp.ndarray:
    """Mountain chain with gap for Test Case 2a.

    Returns shape (6, n, n).
    """
    p = {**TC2_PARAMS, **(params or {})}
    return mountain_chain_with_gap(
        grid,
        h0=p["chain_h0"],
        chain_halfwidth_lon=p["chain_halfwidth_lon"],
        chain_halfwidth_lat=p["chain_halfwidth_lat"],
        gap_halfwidth=p["gap_halfwidth"],
        lon0=p["chain_lon"],
        lat0=p["gap_lat"],
    )


def dcmip25_tc2b_topography(
    grid: CubedSphereGrid,
    params: dict | None = None,
) -> jnp.ndarray:
    """Isolated Gaussian mountain for Test Case 2b (vortex shedding).

    Returns shape (6, n, n).
    """
    p = {**TC2_PARAMS, **(params or {})}
    return gaussian_mountain(
        grid,
        h0=p["mountain_h0"],
        d=p["mountain_d"],
        lat0=p["mountain_lat"],
        lon0=p["mountain_lon"],
    )


def dcmip25_tc2_init(
    grid: CubedSphereGrid,
    n_levels: int = 48,
    subcase: str = "a",
    params: dict | None = None,
) -> tuple[NonHydrostaticState, HeightCoordinate, TerrainMetric, CubedSphereGrid]:
    """Initialize DCMIP-2025 Test Case 2.

    Parameters
    ----------
    grid : CubedSphereGrid
        Original Earth-sized grid.
    n_levels : int
        Number of vertical levels.
    subcase : str
        "a" for gap flow, "b" for vortex shedding.
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
    p = {**TC2_PARAMS, **(params or {})}

    # Apply small-Earth scaling
    small_grid = apply_small_earth_scaling(grid, p["small_earth_factor"])

    # Isothermal reference state
    theta_fn = isothermal_theta_ref(T0=p["T0"])
    height_coord = create_height_coordinate(n_levels, p["H"], theta_fn)

    # Topography (on small-Earth grid)
    if subcase == "a":
        z_s = dcmip25_tc2a_topography(small_grid, p)
    elif subcase == "b":
        z_s = dcmip25_tc2b_topography(small_grid, p)
    else:
        raise ValueError(f"Unknown subcase: {subcase!r}. Use 'a' or 'b'.")

    terrain_metric = compute_terrain_metric(z_s, height_coord)

    # --- Initial conditions ---
    shape_3d = (6, small_grid.n, small_grid.n, n_levels)
    shape_w = (6, small_grid.n, small_grid.n, n_levels + 1)

    # Solid-body rotation: u = u0 * cos(lat)
    u_data = jnp.ones(shape_3d) * (p["u0"] * jnp.cos(small_grid.lat))[..., None]
    v_data = jnp.zeros(shape_3d)
    w_data = jnp.zeros(shape_w)
    theta_prime_data = jnp.zeros(shape_3d)
    rho_prime_data = jnp.zeros(shape_3d)
    phis_data = constants.g * z_s
    tracers_data = jnp.zeros((*shape_3d, 0))

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
