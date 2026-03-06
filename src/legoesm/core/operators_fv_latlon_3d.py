"""3D FV operator wrappers for lat-lon grids.

vmap-over-levels pattern using the 2D lat-lon FV operators.
All functions operate on raw jax.Array data with shape
(n_lat, n_lon, nlev), where the level axis is last.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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
    def single_level(q_k, u_k, v_k):
        return _fv_flux_divergence_2d(q_k, u_k, v_k, grid, limiter)

    q_t = jnp.moveaxis(q_3d, -1, 0)
    u_t = jnp.moveaxis(u_3d, -1, 0)
    v_t = jnp.moveaxis(v_3d, -1, 0)
    result = jax.vmap(single_level)(q_t, u_t, v_t)
    return jnp.moveaxis(result, 0, -1)


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
    def single_level(q_k, u_k, v_k):
        return _fv_scalar_advection_2d(q_k, u_k, v_k, grid, limiter)

    q_t = jnp.moveaxis(q_3d, -1, 0)
    u_t = jnp.moveaxis(u_3d, -1, 0)
    v_t = jnp.moveaxis(v_3d, -1, 0)
    result = jax.vmap(single_level)(q_t, u_t, v_t)
    return jnp.moveaxis(result, 0, -1)
