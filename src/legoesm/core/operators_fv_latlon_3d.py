"""3D FV operator wrappers for lat-lon grids.

vmap of 2D lat-lon FV operators over vertical levels via vmap_over_levels.
All functions operate on raw jax.Array data with shape
(n_lat, n_lon, nlev), where the level axis is last.
"""

from __future__ import annotations

import jax

from legoesm.core.vmap_levels import vmap_over_levels
from legoesm.core.operators_fv_latlon import (
    fv_flux_divergence_latlon as _fv_flux_divergence_2d,
    fv_scalar_advection_latlon as _fv_scalar_advection_2d,
)
from legoesm.grids.latlon import LatLonGrid


def fv_flux_divergence_latlon_3d(
    q_3d: jax.Array, u_3d: jax.Array, v_3d: jax.Array,
    grid: LatLonGrid, limiter: bool = True,
) -> jax.Array:
    """Conservative FV flux divergence at all levels via vmap.

    Parameters
    ----------
    q_3d : jax.Array, shape (n_lat, n_lon, nlev)
    u_3d, v_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid
    limiter : bool

    Returns
    -------
    jax.Array : shape (n_lat, n_lon, nlev)
    """
    return vmap_over_levels(_fv_flux_divergence_2d)(
        q_3d, u_3d, v_3d, grid=grid, limiter=limiter)


def fv_scalar_advection_latlon_3d(
    q_3d: jax.Array, u_3d: jax.Array, v_3d: jax.Array,
    grid: LatLonGrid, limiter: bool = True,
) -> jax.Array:
    """PPM advection of scalar at all levels via vmap.

    Parameters
    ----------
    q_3d : jax.Array, shape (n_lat, n_lon, nlev)
    u_3d, v_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid
    limiter : bool

    Returns
    -------
    jax.Array : shape (n_lat, n_lon, nlev)
    """
    return vmap_over_levels(_fv_scalar_advection_2d)(
        q_3d, u_3d, v_3d, grid=grid, limiter=limiter)
