"""Williamson et al. (1992) standard test cases for the shallow water equations.

These test cases validate the numerical accuracy and conservation properties
of the shallow water solver on the sphere.

References
----------
- Williamson, D. L., Drake, J. B., Hack, J. J., Jakob, R., & Swarztrauber, P. N.
  (1992). A standard test set for numerical approximations to the shallow water
  equations in spherical geometry. J. Comput. Phys., 102(1), 211-224.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.core.williamson_sw_analytic import (
    W2_GH0,
    W5_H0_M,
    W5_UBAR_MS,
    solid_body_geopotential,
    solid_body_rotation_speed,
    solid_body_winds,
    williamson_5_mountain_height,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm import constants


def williamson_test2(grid: CubedSphereGrid) -> ShallowWaterState:
    """Williamson Test Case 2: Global steady-state nonlinear zonal geostrophic flow.

    A solid body rotation in geostrophic balance. The analytic solution
    is the initial condition for all time, so any deviation is numerical error.

    Parameters
    ----------
    grid : CubedSphereGrid
        The computational grid.

    Returns
    -------
    ShallowWaterState : Initial condition for Test Case 2.

    Notes
    -----
    The flow is a solid-body rotation with:
        u = u_0 * cos(lat) * cos(alpha) + cos(lon) * sin(lat) * sin(alpha)
        v = -u_0 * sin(lon) * sin(alpha)
    where alpha is the angle of rotation axis relative to the polar axis.
    For alpha=0 (standard case), this simplifies to u = u_0*cos(lat), v = 0.

    The height field is in exact geostrophic balance:
        g*h = g*h_0 - (R*Omega*u_0 + u_0^2/2) * sin^2(lat)
    """
    R = grid.radius
    g = constants.g

    u_0 = solid_body_rotation_speed(R)          # ~38.6 m/s, 12-day rotation
    lat = grid.lat
    lon = grid.lon

    # Analytic fields: ONE shared definition (williamson_sw_analytic).
    u_east, v_north = solid_body_winds(lon, lat, u0=u_0, xp=jnp)

    # Rotate to grid-aligned coordinates (grid-specific; stays here)
    cos_a = jnp.cos(grid.angle)
    sin_a = jnp.sin(grid.angle)
    u_grid = cos_a * u_east + sin_a * v_north
    v_grid = -sin_a * u_east + cos_a * v_north

    h_data = solid_body_geopotential(lon, lat, radius=R,
                                     omega=constants.Omega, u0=u_0,
                                     gh0=W2_GH0, xp=jnp) / g

    # No topography
    h_s_data = jnp.zeros_like(h_data)

    dims = ("face", "x", "y")

    return ShallowWaterState(
        h=Field(data=h_data, name="h", dims=dims, units="m", long_name="Fluid depth"),
        u=Field(data=u_grid, name="u", dims=dims, units="m/s", long_name="Zonal velocity (grid)"),
        v=Field(data=v_grid, name="v", dims=dims, units="m/s", long_name="Meridional velocity (grid)"),
        h_s=Field(data=h_s_data, name="h_s", dims=dims, units="m", long_name="Surface topography"),
    )


def williamson_test2_exact(grid: CubedSphereGrid, t: float) -> ShallowWaterState:
    """Exact solution for Williamson Test 2 at time t.

    Since Test 2 is a steady-state solution, the exact solution
    is identical to the initial condition for all time.
    """
    return williamson_test2(grid)


def williamson_test5(grid: CubedSphereGrid) -> ShallowWaterState:
    """Williamson Test Case 5: Zonal flow over an isolated mountain.

    Same initial flow as Test 2 but with an isolated mountain at (3pi/2, pi/6).
    This tests the model's ability to handle topography and generate
    a Rossby wave train.

    Parameters
    ----------
    grid : CubedSphereGrid
        The computational grid.

    Returns
    -------
    ShallowWaterState : Initial condition for Test Case 5.

    Notes
    -----
    Mountain specification:
        h_s = h_s0 * (1 - r/R_m)
    where:
        h_s0 = 2000 m
        R_m = pi/9 (20 degrees)
        center: (lon_c, lat_c) = (3*pi/2, pi/6) = (270E, 30N)
        r = min(R_m, sqrt((lon - lon_c)^2 + (lat - lat_c)^2))
    """
    R = grid.radius
    g = constants.g

    u_0 = W5_UBAR_MS          # note: different from Test 2
    lat = grid.lat
    lon = grid.lon

    # Analytic fields: ONE shared definition (williamson_sw_analytic).
    u_east, v_north = solid_body_winds(lon, lat, u0=u_0, xp=jnp)

    # Rotate to grid-aligned coordinates (grid-specific; stays here)
    cos_a = jnp.cos(grid.angle)
    sin_a = jnp.sin(grid.angle)
    u_grid = cos_a * u_east + sin_a * v_north
    v_grid = -sin_a * u_east + cos_a * v_north

    h_s_data = williamson_5_mountain_height(lon, lat, xp=jnp)

    # h is fluid DEPTH (column above topography): the solver computes
    # B = KE + g*(h + h_s), so h = h_free - h_s.
    h_free = solid_body_geopotential(lon, lat, radius=R,
                                     omega=constants.Omega, u0=u_0,
                                     gh0=W5_H0_M * g, xp=jnp) / g
    h_data = h_free - h_s_data

    dims = ("face", "x", "y")

    return ShallowWaterState(
        h=Field(data=h_data, name="h", dims=dims, units="m", long_name="Fluid depth"),
        u=Field(data=u_grid, name="u", dims=dims, units="m/s", long_name="Zonal velocity (grid)"),
        v=Field(data=v_grid, name="v", dims=dims, units="m/s", long_name="Meridional velocity (grid)"),
        h_s=Field(data=h_s_data, name="h_s", dims=dims, units="m", long_name="Mountain topography"),
    )


def compute_error_norms(
    state: ShallowWaterState,
    reference: ShallowWaterState,
    grid: CubedSphereGrid,
) -> dict[str, float]:
    """Compute L1, L2, and Linf error norms for height field.

    Parameters
    ----------
    state : ShallowWaterState
        Computed state.
    reference : ShallowWaterState
        Reference (exact) state.
    grid : CubedSphereGrid
        The grid.

    Returns
    -------
    dict : Error norms {l1, l2, linf} normalized by the reference field norm.
    """
    err = state.h.data - reference.h.data
    ref = reference.h.data

    area = grid.area

    # Normalized L1 error
    l1 = jnp.sum(jnp.abs(err) * area) / jnp.sum(jnp.abs(ref) * area)

    # Normalized L2 error
    l2 = jnp.sqrt(jnp.sum(err**2 * area) / jnp.sum(ref**2 * area))

    # Normalized Linf error
    linf = jnp.max(jnp.abs(err)) / jnp.max(jnp.abs(ref))

    return {"l1": float(l1), "l2": float(l2), "linf": float(linf)}


def williamson2_cdgrid_initial_condition(
    cdgrid, u_0: float = 38.61068276698372, h_0: float = 29400.0 / constants.g
):
    """Williamson TC2 initial condition on the cubed-sphere C-D grid.

    Steady solid-body rotation in geostrophic balance: height at cell centres,
    velocity at D-grid corners (grid-aligned via the corner rotation angle).
    Side-effect-free shared helper (no global jax.config mutation) used by both
    ``test_williamson2_cdgrid`` and the Stage-A2 Williamson Experiment rung.
    """
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterState,
    )

    g = constants.g
    omega = constants.Omega
    R = cdgrid.radius

    lat_c = cdgrid.base.lat  # (6, n, n)
    h = h_0 - (R * omega * u_0 + 0.5 * u_0 ** 2) * jnp.sin(lat_c) ** 2 / g

    lat_corner = cdgrid.lat_corner
    u_geo = u_0 * jnp.cos(lat_corner)
    u_d = u_geo * cdgrid.cos_angle_corner
    v_d = -u_geo * cdgrid.sin_angle_corner

    h_s = jnp.zeros_like(h)
    return CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)
