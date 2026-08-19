"""Williamson et al. (1992) test cases for the lat-lon shallow water solver.

Adapted from williamson.py (cubed-sphere) for the lat-lon grid.
Key simplification: no angle rotation needed (u,v are already east/north).

References
----------
- Williamson, D. L., Drake, J. B., Hack, J. J., Jakob, R., & Swarztrauber, P. N.
  (1992). A standard test set for numerical approximations to the shallow water
  equations in spherical geometry. J. Comput. Phys., 102(1), 211-224.
"""

from __future__ import annotations

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
from legoesm.grids.latlon import LatLonGrid
from legoesm import constants


def williamson_test2_latlon(grid: LatLonGrid) -> ShallowWaterState:
    """Williamson Test Case 2: Global steady-state nonlinear zonal geostrophic flow.

    A solid body rotation in geostrophic balance on a lat-lon grid.
    No angle rotation needed: u = u_0*cos(lat), v = 0 directly.

    Parameters
    ----------
    grid : LatLonGrid

    Returns
    -------
    ShallowWaterState
    """
    R = grid.radius
    g = constants.g

    u_0 = solid_body_rotation_speed(R)          # 12-day rotation
    lat = grid.lat2d
    lon = grid.lon2d

    # Analytic fields: ONE shared definition (williamson_sw_analytic).
    # On lat-lon the geographic components ARE the grid components.
    u_data, v_data = solid_body_winds(lon, lat, u0=u_0, xp=jnp)
    h_data = solid_body_geopotential(lon, lat, radius=R,
                                     omega=constants.Omega, u0=u_0,
                                     gh0=W2_GH0, xp=jnp) / g

    h_s_data = jnp.zeros_like(h_data)

    dims = ("lat", "lon")

    return ShallowWaterState(
        h=Field(data=h_data, name="h", dims=dims, units="m"),
        u=Field(data=u_data, name="u", dims=dims, units="m/s"),
        v=Field(data=v_data, name="v", dims=dims, units="m/s"),
        h_s=Field(data=h_s_data, name="h_s", dims=dims, units="m"),
    )


def williamson_test2_exact_latlon(grid: LatLonGrid, t: float) -> ShallowWaterState:
    """Exact solution for Williamson Test 2 at time t (steady-state)."""
    return williamson_test2_latlon(grid)


def williamson_test5_latlon(grid: LatLonGrid) -> ShallowWaterState:
    """Williamson Test Case 5: Zonal flow over an isolated mountain.

    Same initial flow as Test 2 but with an isolated mountain at (3pi/2, pi/6).

    Parameters
    ----------
    grid : LatLonGrid

    Returns
    -------
    ShallowWaterState
    """
    R = grid.radius
    g = constants.g

    u_0 = W5_UBAR_MS
    lat = grid.lat2d
    lon = grid.lon2d

    # Analytic fields: ONE shared definition (williamson_sw_analytic).
    u_data, v_data = solid_body_winds(lon, lat, u0=u_0, xp=jnp)

    # ★ CHANGED: this used a GREAT-CIRCLE radius.  Williamson et al.
    # (1992) case 5 and test_cases.F90:1185 both specify the clipped
    # PLANAR (lon, lat) radius, which is what the shared helper applies
    # and what the cubed-sphere sibling already used -- so W5 was not
    # the same mountain across grids.
    h_s_data = williamson_5_mountain_height(lon, lat, xp=jnp)

    # h is fluid DEPTH: the solver computes B = KE + g*(h + h_s).
    h_free = solid_body_geopotential(lon, lat, radius=R,
                                     omega=constants.Omega, u0=u_0,
                                     gh0=W5_H0_M * g, xp=jnp) / g
    h_data = h_free - h_s_data

    dims = ("lat", "lon")

    return ShallowWaterState(
        h=Field(data=h_data, name="h", dims=dims, units="m"),
        u=Field(data=u_data, name="u", dims=dims, units="m/s"),
        v=Field(data=v_data, name="v", dims=dims, units="m/s"),
        h_s=Field(data=h_s_data, name="h_s", dims=dims, units="m"),
    )


def compute_error_norms_latlon(
    state: ShallowWaterState,
    reference: ShallowWaterState,
    grid: LatLonGrid,
) -> dict[str, float]:
    """Compute L1, L2, and Linf error norms for height field.

    Parameters
    ----------
    state : ShallowWaterState
        Computed state.
    reference : ShallowWaterState
        Reference (exact) state.
    grid : LatLonGrid

    Returns
    -------
    dict : Error norms {l1, l2, linf} normalized by the reference field norm.
    """
    err = state.h.data - reference.h.data
    ref = reference.h.data

    area = grid.area

    l1 = jnp.sum(jnp.abs(err) * area) / jnp.sum(jnp.abs(ref) * area)
    l2 = jnp.sqrt(jnp.sum(err**2 * area) / jnp.sum(ref**2 * area))
    linf = jnp.max(jnp.abs(err)) / jnp.max(jnp.abs(ref))

    return {"l1": float(l1), "l2": float(l2), "linf": float(linf)}
