"""DCMIP-2025 Test Case 1: Mountain-triggered breaking gravity waves.

Dry non-hydrostatic dynamics with a piecewise lapse rate atmosphere.
Gravity waves are triggered by a mountain at 20N and propagate vertically,
breaking near the tropopause (20 km) due to the sharp stability transition.

Setup:
- Model top: 40 km
- Troposphere (0-20 km): lapse rate -5 K/km
- Stratosphere (20-40 km): lapse rate +5 K/km
- Mountain: Schaer-type at 20N, peak 2 km, half-width 72 km
- Background flow: u = u0 * cos(lat), v = 0
- Vertical grids: L88, L120, L207
- Sponge layer near model top to absorb upward-propagating waves

References
----------
- DCMIP-2025: https://sites.google.com/umich.edu/dcmip-2025/
- Skamarock et al. (2019): Vertical Resolution Requirements in
  Atmospheric Simulation.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import CubedSphereGrid, rotate_winds_geo_to_grid
from legoesm.grids.vertical import (
    HeightCoordinate,
    TerrainMetric,
    create_height_coordinate,
    compute_terrain_metric,
)
from .common import (
    piecewise_lapse_theta_ref,
    schaer_mountain,
)
from legoesm import constants


# Default parameters (single source of truth in the package; audit item 9).
from legoesm.atmosphere.dynamics.gcm.dcmip2025_ic import TC1_PARAMS  # noqa: E402,F401


def dcmip25_tc1_topography(
    grid: CubedSphereGrid,
    params: dict | None = None,
) -> jnp.ndarray:
    """Compute mountain topography for Test Case 1.

    Returns
    -------
    jnp.ndarray
        Surface elevation z_s, shape (6, n, n).
    """
    p = {**TC1_PARAMS, **(params or {})}
    return schaer_mountain(
        grid,
        h0=p["mountain_height"],
        halfwidth=p["mountain_halfwidth"],
        lat0=p["mountain_lat"],
        lon0=p["mountain_lon"],
    )


def dcmip25_tc1_init(
    grid: CubedSphereGrid,
    n_levels: int = 88,
    params: dict | None = None,
) -> tuple[NonHydrostaticState, HeightCoordinate, TerrainMetric]:
    """Initialize DCMIP-2025 Test Case 1.

    Parameters
    ----------
    grid : CubedSphereGrid
        Horizontal grid.
    n_levels : int
        Number of vertical levels (88, 120, or 207).
    params : dict, optional
        Override default parameters.

    Returns
    -------
    state : NonHydrostaticState
        Initial state.
    height_coord : HeightCoordinate
        Vertical coordinate.
    terrain_metric : TerrainMetric
        Terrain metric terms.
    """
    p = {**TC1_PARAMS, **(params or {})}

    # Reference state with piecewise lapse rate
    theta_fn = piecewise_lapse_theta_ref(
        T_s=p["T_s"],
        lapse_tropo=p["lapse_tropo"],
        lapse_strato=p["lapse_strato"],
        z_tropopause=p["z_tropopause"],
    )

    # Vertical coordinate
    height_coord = create_height_coordinate(n_levels, p["H"], theta_fn)

    # Topography
    z_s = dcmip25_tc1_topography(grid, p)

    # Terrain metric
    terrain_metric = compute_terrain_metric(z_s, height_coord)

    # --- Initial conditions ---
    shape_3d = (6, grid.n, grid.n, n_levels)
    shape_w = (6, grid.n, grid.n, n_levels + 1)
    shape_2d = (6, grid.n, grid.n)

    # Horizontal wind: u_east = u0 * cos(lat), v_north = 0
    # Rotate from geographic to grid-aligned coordinates on cubed sphere
    u0 = p["u0"]
    u_east = u0 * jnp.cos(grid.lat)  # (6, n, n)
    v_north = jnp.zeros_like(u_east)
    u_grid_2d, v_grid_2d = rotate_winds_geo_to_grid(
        u_east, v_north, grid.angle,
    )
    u_data = jnp.broadcast_to(u_grid_2d[..., None], shape_3d).copy()
    v_data = jnp.broadcast_to(v_grid_2d[..., None], shape_3d).copy()

    # Vertical velocity: w = 0
    w_data = jnp.zeros(shape_w)

    # Perturbations are zero initially (atmosphere is in balance)
    theta_prime_data = jnp.zeros(shape_3d)
    rho_prime_data = jnp.zeros(shape_3d)

    # Surface geopotential
    phis_data = constants.g * z_s

    # No tracers for dry dynamics
    tracers_data = jnp.zeros((*shape_3d, 0))

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=u_data, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_data, name="v", dims=dims_3d, units="m/s"),
        w=Field(data=w_data, name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(
            data=theta_prime_data, name="theta_prime",
            dims=dims_3d, units="K",
        ),
        rho_prime=Field(
            data=rho_prime_data, name="rho_prime",
            dims=dims_3d, units="kg/m^3",
        ),
        phis=Field(data=phis_data, name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(
            data=tracers_data, name="tracers",
            dims=("face", "x", "y", "level", "tracer"), units="kg/kg",
        ),
    )

    return state, height_coord, terrain_metric
