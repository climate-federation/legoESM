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
from legoesm.grids.cubed_sphere import (
    CubedSphereGrid,
    rotate_winds_geo_to_grid,
)
from legoesm.grids.vertical import (
    HeightCoordinate,
    TerrainMetric,
    create_height_coordinate,
    compute_terrain_metric,
)
from .common import (
    apply_small_earth_scaling,
    isothermal_theta_ref,
    gaussian_mountain,
    mountain_chain_with_gap,
)
from legoesm import constants


# Default parameters
# Default parameters (single source of truth in the package; audit item 9).
from legoesm.atmosphere.dynamics.gcm.dcmip2025_ic import TC2_PARAMS  # noqa: E402,F401


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


def _gradient_wind_balanced_rho_prime(
    lat: jnp.ndarray,
    u0: float,
    height_coord: HeightCoordinate,
) -> jnp.ndarray:
    """Compute density perturbation for gradient-wind balance with u = u0·cos(lat).

    For barotropic solid-body rotation u = u0·cos(lat) on a sphere, the
    gradient-wind balance equation (momentum equation in steady state with v=0)
    gives the required Exner function perturbation:

        π'(lat, z) = (u₀² + 2·R_earth·Ω·u₀) / (2·cₚ·θ₀(z)) · cos²(lat)

    The density perturbation is then recovered from the equation of state:

        ρ' = ρ₀ · ((1 + π'/π₀)^(cᵥ/R_d) − 1)

    Note: the formula is independent of the small-Earth factor because the
    Coriolis (∝ X) and curvature metric (∝ 1/X) scaling cancel exactly.

    Parameters
    ----------
    lat : jnp.ndarray
        Latitude [rad], shape (6, n, n).
    u0 : float
        Solid-body rotation speed [m/s].
    height_coord : HeightCoordinate
        Vertical coordinate with reference state (rho_0, theta_0, pi_0).

    Returns
    -------
    rho_prime : jnp.ndarray
        Balanced density perturbation, shape (6, n, n, nlev).
    """
    c_p = constants.c_pd
    c_v = constants.c_vd
    R_d = constants.R_d
    R_earth = constants.R_earth
    Omega = constants.Omega

    theta_0 = height_coord.theta_ref    # (nlev,)
    pi_0 = height_coord.exner_ref       # (nlev,)
    rho_0 = height_coord.rho_ref        # (nlev,)

    cos_lat = jnp.cos(lat)  # (6, n, n)

    # Exner perturbation from gradient-wind balance
    # π' = (u0² + 2·a·Ω·u0) / (2·cp·θ0) · cos²(lat)
    coeff = (u0**2 + 2.0 * R_earth * Omega * u0) / (2.0 * c_p)
    pi_prime = coeff * cos_lat[..., None]**2 / theta_0  # (6,n,n,nlev)

    # Invert equation of state: ρ' = ρ0·((1 + π'/π0)^(cv/Rd) − 1)
    exponent = c_v / R_d
    rho_prime = rho_0 * ((1.0 + pi_prime / pi_0) ** exponent - 1.0)

    return rho_prime


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
    small_grid = apply_small_earth_scaling(
        grid, p["small_earth_factor"],
        rotating=p.get("rotating", True))

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

    # Solid-body rotation: u_east = u0 * cos(lat), v_north = 0
    # Must rotate from geographic to grid-aligned coordinates on cubed sphere
    u_east = p["u0"] * jnp.cos(small_grid.lat)  # (6, n, n)
    v_north = jnp.zeros_like(u_east)

    u_grid_2d, v_grid_2d = rotate_winds_geo_to_grid(
        u_east, v_north, small_grid.angle,
    )
    u_data = jnp.broadcast_to(u_grid_2d[..., None], shape_3d).copy()
    v_data = jnp.broadcast_to(v_grid_2d[..., None], shape_3d).copy()
    w_data = jnp.zeros(shape_w)
    theta_prime_data = jnp.zeros(shape_3d)

    # Density perturbation: start at rest (rho'=0).
    # Note: _gradient_wind_balanced_rho_prime() is available for
    # gradient-wind balanced initialization, but the explicit split-explicit
    # time integrator does not maintain this balance well with isothermal
    # reference states, leading to faster blowup than rho'=0. A semi-implicit
    # time integrator would be needed for balanced long integrations.
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
