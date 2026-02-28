"""3D operator wrappers: vmap 2D cubed-sphere operators over vertical levels.

All functions operate on raw ``jax.Array`` data with shape ``(6, n, n, nlev)``.
They use ``jax.vmap`` of the 2D :class:`~legoesm.core.field.Field`-based
operators from :mod:`legoesm.core.operators`.

These are shared between the hydrostatic primitive equation model and the
prescribed-wind tracer transport model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators import (
    gradient_x,
    gradient_y,
    divergence,
    curl_z,
    hyperdiffusion,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid


def vorticity_3d(
    u_3d: jax.Array, v_3d: jax.Array, grid: CubedSphereGrid,
) -> jax.Array:
    """Compute vorticity at all levels via vmap of 2D curl_z.

    Parameters
    ----------
    u_3d, v_3d : jax.Array
        Wind components, shape (6, n, n, nlev).
    grid : CubedSphereGrid
        Horizontal grid.

    Returns
    -------
    jax.Array : Vorticity, shape (6, n, n, nlev).
    """
    def single_level(u_k, v_k):
        u_f = Field(data=u_k, name="u", dims=("face", "x", "y"), units="m/s")
        v_f = Field(data=v_k, name="v", dims=("face", "x", "y"), units="m/s")
        return curl_z(u_f, v_f, grid).data

    u_t = jnp.moveaxis(u_3d, -1, 0)   # (nlev, 6, n, n)
    v_t = jnp.moveaxis(v_3d, -1, 0)
    result = jax.vmap(single_level)(u_t, v_t)  # (nlev, 6, n, n)
    return jnp.moveaxis(result, 0, -1)  # (6, n, n, nlev)


def gradient_x_3d(
    field_3d: jax.Array, grid: CubedSphereGrid,
) -> jax.Array:
    """Compute x-gradient at all levels via vmap.

    Parameters
    ----------
    field_3d : jax.Array
        Scalar field, shape (6, n, n, nlev).
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array : d(field)/dx, shape (6, n, n, nlev).
    """
    def single_level(f_k):
        f_field = Field(data=f_k, name="f", dims=("face", "x", "y"),
                        units="", staggering="cell")
        return gradient_x(f_field, grid).data

    f_t = jnp.moveaxis(field_3d, -1, 0)
    result = jax.vmap(single_level)(f_t)
    return jnp.moveaxis(result, 0, -1)


def gradient_y_3d(
    field_3d: jax.Array, grid: CubedSphereGrid,
) -> jax.Array:
    """Compute y-gradient at all levels via vmap.

    Parameters
    ----------
    field_3d : jax.Array
        Scalar field, shape (6, n, n, nlev).
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array : d(field)/dy, shape (6, n, n, nlev).
    """
    def single_level(f_k):
        f_field = Field(data=f_k, name="f", dims=("face", "x", "y"),
                        units="", staggering="cell")
        return gradient_y(f_field, grid).data

    f_t = jnp.moveaxis(field_3d, -1, 0)
    result = jax.vmap(single_level)(f_t)
    return jnp.moveaxis(result, 0, -1)


def divergence_3d(
    u_3d: jax.Array, v_3d: jax.Array, grid: CubedSphereGrid,
) -> jax.Array:
    """Compute divergence at all levels via vmap.

    Parameters
    ----------
    u_3d, v_3d : jax.Array
        Vector field components, shape (6, n, n, nlev).
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array : Divergence, shape (6, n, n, nlev).
    """
    def single_level(u_k, v_k):
        u_f = Field(data=u_k, name="u", dims=("face", "x", "y"), units="m/s")
        v_f = Field(data=v_k, name="v", dims=("face", "x", "y"), units="m/s")
        return divergence(u_f, v_f, grid).data

    u_t = jnp.moveaxis(u_3d, -1, 0)
    v_t = jnp.moveaxis(v_3d, -1, 0)
    result = jax.vmap(single_level)(u_t, v_t)
    return jnp.moveaxis(result, 0, -1)


def hyperdiffusion_3d(
    field_3d: jax.Array, grid: CubedSphereGrid, coeff: float,
) -> jax.Array:
    """Compute hyperdiffusion at all levels via vmap.

    Parameters
    ----------
    field_3d : jax.Array
        Scalar field, shape (6, n, n, nlev).
    grid : CubedSphereGrid
    coeff : float
        Hyperdiffusion coefficient.

    Returns
    -------
    jax.Array : Hyperdiffusion tendency, shape (6, n, n, nlev).
    """
    def single_level(f_k):
        f_field = Field(data=f_k, name="f", dims=("face", "x", "y"), units="")
        return hyperdiffusion(f_field, grid, coeff).data

    f_t = jnp.moveaxis(field_3d, -1, 0)
    result = jax.vmap(single_level)(f_t)
    return jnp.moveaxis(result, 0, -1)
