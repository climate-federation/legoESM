"""3D operator wrappers for lat-lon grids.

Same vmap-over-levels pattern as operators_3d.py but using lat-lon
2D operators. All functions operate on raw jax.Array data with shape
(n_lat, n_lon, nlev), where the level axis is last.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators_latlon import (
    gradient_x,
    gradient_y,
    divergence,
    curl_z,
    hyperdiffusion,
)
from legoesm.grids.latlon import LatLonGrid


def vorticity_3d(
    u_3d: jax.Array, v_3d: jax.Array, grid: LatLonGrid,
) -> jax.Array:
    """Compute vorticity at all levels via vmap of 2D curl_z.

    Parameters
    ----------
    u_3d, v_3d : jax.Array
        Wind components, shape (n_lat, n_lon, nlev).
    grid : LatLonGrid

    Returns
    -------
    jax.Array : Vorticity, shape (n_lat, n_lon, nlev).
    """
    def single_level(u_k, v_k):
        u_f = Field(data=u_k, name="u", dims=("lat", "lon"), units="m/s")
        v_f = Field(data=v_k, name="v", dims=("lat", "lon"), units="m/s")
        return curl_z(u_f, v_f, grid).data

    u_t = jnp.moveaxis(u_3d, -1, 0)   # (nlev, n_lat, n_lon)
    v_t = jnp.moveaxis(v_3d, -1, 0)
    result = jax.vmap(single_level)(u_t, v_t)  # (nlev, n_lat, n_lon)
    return jnp.moveaxis(result, 0, -1)  # (n_lat, n_lon, nlev)


def gradient_x_3d(
    field_3d: jax.Array, grid: LatLonGrid,
) -> jax.Array:
    """Compute x-gradient at all levels via vmap.

    Parameters
    ----------
    field_3d : jax.Array
        Scalar field, shape (n_lat, n_lon, nlev).
    grid : LatLonGrid

    Returns
    -------
    jax.Array : d(field)/dx, shape (n_lat, n_lon, nlev).
    """
    def single_level(f_k):
        f_field = Field(data=f_k, name="f", dims=("lat", "lon"),
                        units="", staggering="cell")
        return gradient_x(f_field, grid).data

    f_t = jnp.moveaxis(field_3d, -1, 0)
    result = jax.vmap(single_level)(f_t)
    return jnp.moveaxis(result, 0, -1)


def gradient_y_3d(
    field_3d: jax.Array, grid: LatLonGrid,
) -> jax.Array:
    """Compute y-gradient at all levels via vmap.

    Parameters
    ----------
    field_3d : jax.Array
        Scalar field, shape (n_lat, n_lon, nlev).
    grid : LatLonGrid

    Returns
    -------
    jax.Array : d(field)/dy, shape (n_lat, n_lon, nlev).
    """
    def single_level(f_k):
        f_field = Field(data=f_k, name="f", dims=("lat", "lon"),
                        units="", staggering="cell")
        return gradient_y(f_field, grid).data

    f_t = jnp.moveaxis(field_3d, -1, 0)
    result = jax.vmap(single_level)(f_t)
    return jnp.moveaxis(result, 0, -1)


def divergence_3d(
    u_3d: jax.Array, v_3d: jax.Array, grid: LatLonGrid,
) -> jax.Array:
    """Compute divergence at all levels via vmap.

    Parameters
    ----------
    u_3d, v_3d : jax.Array
        Vector field components, shape (n_lat, n_lon, nlev).
    grid : LatLonGrid

    Returns
    -------
    jax.Array : Divergence, shape (n_lat, n_lon, nlev).
    """
    def single_level(u_k, v_k):
        u_f = Field(data=u_k, name="u", dims=("lat", "lon"), units="m/s")
        v_f = Field(data=v_k, name="v", dims=("lat", "lon"), units="m/s")
        return divergence(u_f, v_f, grid).data

    u_t = jnp.moveaxis(u_3d, -1, 0)
    v_t = jnp.moveaxis(v_3d, -1, 0)
    result = jax.vmap(single_level)(u_t, v_t)
    return jnp.moveaxis(result, 0, -1)


def hyperdiffusion_3d(
    field_3d: jax.Array, grid: LatLonGrid, coeff: float,
) -> jax.Array:
    """Compute hyperdiffusion at all levels via vmap.

    Parameters
    ----------
    field_3d : jax.Array
        Scalar field, shape (n_lat, n_lon, nlev).
    grid : LatLonGrid
    coeff : float
        Hyperdiffusion coefficient.

    Returns
    -------
    jax.Array : Hyperdiffusion tendency, shape (n_lat, n_lon, nlev).
    """
    def single_level(f_k):
        f_field = Field(data=f_k, name="f", dims=("lat", "lon"), units="")
        return hyperdiffusion(f_field, grid, coeff).data

    f_t = jnp.moveaxis(field_3d, -1, 0)
    result = jax.vmap(single_level)(f_t)
    return jnp.moveaxis(result, 0, -1)
