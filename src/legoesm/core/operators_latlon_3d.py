"""3D operator wrappers for lat-lon grids.

vmap of 2D lat-lon operators over vertical levels via vmap_over_levels.
All functions operate on raw jax.Array data with shape
(n_lat, n_lon, nlev), where the level axis is last.
"""

from __future__ import annotations

import jax

from legoesm.core.vmap_levels import vmap_over_levels
from legoesm.core.field import Field
from legoesm.core.operators_latlon import (
    gradient_x,
    gradient_y,
    divergence,
    curl_z,
    laplacian,
    hyperdiffusion,
)
from legoesm.grids.latlon import LatLonGrid


def _curl_z_raw(u_k, v_k, *, grid):
    """curl_z wrapper: raw arrays -> raw array."""
    u_f = Field(data=u_k, name="u", dims=("lat", "lon"), units="m/s")
    v_f = Field(data=v_k, name="v", dims=("lat", "lon"), units="m/s")
    return curl_z(u_f, v_f, grid).data


def _gradient_x_raw(f_k, *, grid):
    """gradient_x wrapper: raw array -> raw array."""
    f_field = Field(data=f_k, name="f", dims=("lat", "lon"),
                    units="", staggering="cell")
    return gradient_x(f_field, grid).data


def _gradient_y_raw(f_k, *, grid):
    """gradient_y wrapper: raw array -> raw array."""
    f_field = Field(data=f_k, name="f", dims=("lat", "lon"),
                    units="", staggering="cell")
    return gradient_y(f_field, grid).data


def _divergence_raw(u_k, v_k, *, grid):
    """divergence wrapper: raw arrays -> raw array."""
    u_f = Field(data=u_k, name="u", dims=("lat", "lon"), units="m/s")
    v_f = Field(data=v_k, name="v", dims=("lat", "lon"), units="m/s")
    return divergence(u_f, v_f, grid).data


def _laplacian_raw(f_k, *, grid):
    """laplacian wrapper: raw array -> raw array."""
    f_field = Field(data=f_k, name="f", dims=("lat", "lon"), units="")
    return laplacian(f_field, grid).data


def _hyperdiffusion_raw(f_k, *, grid, coeff):
    """hyperdiffusion wrapper: raw array -> raw array."""
    f_field = Field(data=f_k, name="f", dims=("lat", "lon"), units="")
    return hyperdiffusion(f_field, grid, coeff).data


def vorticity_3d(u_3d: jax.Array, v_3d: jax.Array, grid: LatLonGrid) -> jax.Array:
    """Compute vorticity at all levels via vmap of 2D curl_z.

    Parameters
    ----------
    u_3d, v_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    jax.Array : Vorticity, shape (n_lat, n_lon, nlev).
    """
    return vmap_over_levels(_curl_z_raw)(u_3d, v_3d, grid=grid)


def gradient_x_3d(field_3d: jax.Array, grid: LatLonGrid) -> jax.Array:
    """Compute x-gradient at all levels via vmap.

    Parameters
    ----------
    field_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    jax.Array : d(field)/dx, shape (n_lat, n_lon, nlev).
    """
    return vmap_over_levels(_gradient_x_raw)(field_3d, grid=grid)


def gradient_y_3d(field_3d: jax.Array, grid: LatLonGrid) -> jax.Array:
    """Compute y-gradient at all levels via vmap.

    Parameters
    ----------
    field_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    jax.Array : d(field)/dy, shape (n_lat, n_lon, nlev).
    """
    return vmap_over_levels(_gradient_y_raw)(field_3d, grid=grid)


def divergence_3d(u_3d: jax.Array, v_3d: jax.Array, grid: LatLonGrid) -> jax.Array:
    """Compute divergence at all levels via vmap.

    Parameters
    ----------
    u_3d, v_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    jax.Array : Divergence, shape (n_lat, n_lon, nlev).
    """
    return vmap_over_levels(_divergence_raw)(u_3d, v_3d, grid=grid)


def hyperdiffusion_3d(field_3d: jax.Array, grid: LatLonGrid, coeff: float) -> jax.Array:
    """Compute hyperdiffusion at all levels via vmap.

    Parameters
    ----------
    field_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid
    coeff : float

    Returns
    -------
    jax.Array : Hyperdiffusion tendency, shape (n_lat, n_lon, nlev).
    """
    return vmap_over_levels(_hyperdiffusion_raw)(field_3d, grid=grid, coeff=coeff)


def laplacian_3d(field_3d: jax.Array, grid: LatLonGrid) -> jax.Array:
    """Compute Laplacian at all levels via vmap.

    Parameters
    ----------
    field_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    jax.Array : Laplacian, shape (n_lat, n_lon, nlev).
    """
    return vmap_over_levels(_laplacian_raw)(field_3d, grid=grid)
