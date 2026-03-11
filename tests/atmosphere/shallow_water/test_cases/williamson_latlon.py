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
    Omega = constants.Omega
    g = constants.g

    u_0 = 2.0 * jnp.pi * R / (12.0 * 86400.0)  # 12-day rotation
    gh_0 = 2.94e4
    h_0 = gh_0 / g

    lat = grid.lat2d
    lon = grid.lon2d

    # Velocity (solid body rotation, alpha=0)
    u_data = u_0 * jnp.cos(lat)
    v_data = jnp.zeros_like(lat)

    # Height field (geostrophic balance)
    h_data = h_0 - (R * Omega * u_0 + u_0**2 / 2.0) * jnp.sin(lat)**2 / g

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
    Omega = constants.Omega
    g = constants.g

    u_0 = 20.0  # m/s
    gh_0 = 5960.0 * g

    lat = grid.lat2d
    lon = grid.lon2d

    # Velocity
    u_data = u_0 * jnp.cos(lat)
    v_data = jnp.zeros_like(lat)

    # Height field
    h_data = (gh_0 - (R * Omega * u_0 + u_0**2 / 2.0) * jnp.sin(lat)**2) / g

    # Mountain topography
    lon_c = 3.0 * jnp.pi / 2.0
    lat_c = jnp.pi / 6.0
    R_m = jnp.pi / 9.0
    h_s0 = 2000.0

    # Great-circle angular distance
    r = jnp.arccos(jnp.clip(
        jnp.sin(lat_c) * jnp.sin(lat) +
        jnp.cos(lat_c) * jnp.cos(lat) * jnp.cos(lon - lon_c),
        -1.0, 1.0,
    ))

    h_s_data = jnp.where(r < R_m, h_s0 * (1.0 - r / R_m), 0.0)

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
