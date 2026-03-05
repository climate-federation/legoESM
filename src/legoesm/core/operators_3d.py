"""3D operator wrappers for cubed-sphere grids.

Provides two sets of operators:

1. **Horizontal operators** (vmap of 2D operators over vertical levels):
   vorticity_3d, gradient_x_3d, gradient_y_3d, divergence_3d, hyperdiffusion_3d.

2. **Vertical operators for height coordinates** (non-hydrostatic):
   vertical_gradient_full_to_half, vertical_gradient_half_to_full,
   vertical_advection_height, vertical_divergence_height.

All functions operate on raw ``jax.Array`` data.
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


# ==============================================================================
# Vertical operators for height-based coordinates (non-hydrostatic)
# ==============================================================================

def vertical_gradient_full_to_half(
    field_full: jax.Array,
    dz: jax.Array,
) -> jax.Array:
    """Compute vertical gradient from full levels to half (interface) levels.

    Uses centered difference: d(f)/dz*|_{k+1/2} = (f_k - f_{k+1}) / dz_avg

    Maps fields at full levels (nlev) to gradients at interior
    half levels (nlev-1). Top and bottom boundary gradients are not
    included -- the caller handles boundary conditions.

    Parameters
    ----------
    field_full : jax.Array
        Field at full levels, shape (..., nlev).
    dz : jax.Array
        Layer thickness dz* [m], shape (nlev,). Positive.

    Returns
    -------
    jax.Array
        Gradient at interior half levels, shape (..., nlev-1).
    """
    df = field_full[..., :-1] - field_full[..., 1:]
    dz_interface = 0.5 * (dz[:-1] + dz[1:])
    return df / dz_interface


def vertical_gradient_half_to_full(
    field_half: jax.Array,
    dz: jax.Array,
) -> jax.Array:
    """Compute vertical gradient from half (interface) levels to full levels.

    Uses centered difference: d(f)/dz*|_k = (f_{k-1/2} - f_{k+1/2}) / dz_k

    Maps fields at half levels (nlev+1) to gradients at full levels (nlev).

    Parameters
    ----------
    field_half : jax.Array
        Field at half (interface) levels, shape (..., nlev+1).
    dz : jax.Array
        Layer thickness dz* [m], shape (nlev,). Positive.

    Returns
    -------
    jax.Array
        Gradient at full levels, shape (..., nlev).
    """
    df = field_half[..., :-1] - field_half[..., 1:]
    return df / dz


def vertical_advection_height(
    field_full: jax.Array,
    w_half: jax.Array,
    dz: jax.Array,
    dz_half: jax.Array,
    jacobian: jax.Array,
) -> jax.Array:
    """Compute vertical advection in height coordinates with upwind scheme.

    Computes: -w · d(field)/dz = -(w/J) · d(field)/dz*

    where J = dz/dz* is the terrain-following Jacobian.

    Parameters
    ----------
    field_full : jax.Array
        Field at full levels, shape (6, n, n, nlev).
    w_half : jax.Array
        Vertical velocity at half levels [m/s], shape (6, n, n, nlev+1).
    dz : jax.Array
        Layer thickness dz* [m], shape (nlev,).
    dz_half : jax.Array
        Distance between full levels [m], shape (nlev-1,).
    jacobian : jax.Array
        Terrain Jacobian dz/dz*, shape (6, n, n).

    Returns
    -------
    jax.Array
        Vertical advection tendency, shape (6, n, n, nlev).
    """
    w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
    w_star = w_full / jacobian[..., None]

    # Backward difference (upward)
    df_bwd = field_full[..., :-1] - field_full[..., 1:]
    grad_bwd = jnp.concatenate(
        [jnp.zeros((*field_full.shape[:-1], 1)),
         df_bwd / dz_half],
        axis=-1,
    )

    # Forward difference (downward)
    grad_fwd = jnp.concatenate(
        [df_bwd / dz_half,
         jnp.zeros((*field_full.shape[:-1], 1))],
        axis=-1,
    )

    # Upwind: w* > 0 = upward => backward; w* < 0 = downward => forward
    grad = jnp.where(w_star > 0, grad_bwd, grad_fwd)
    return -w_star * grad


def vertical_divergence_height(
    rho_w_half: jax.Array,
    dz: jax.Array,
    jacobian: jax.Array,
) -> jax.Array:
    """Compute vertical divergence d(rho*w)/dz for continuity equation.

    Computes (1/J) · d(rho*w)/dz* at full levels from flux at half levels.

    Parameters
    ----------
    rho_w_half : jax.Array
        Mass flux rho*w at half levels [kg/(m^2 s)], shape (6, n, n, nlev+1).
    dz : jax.Array
        Layer thickness dz* [m], shape (nlev,).
    jacobian : jax.Array
        Terrain Jacobian dz/dz*, shape (6, n, n).

    Returns
    -------
    jax.Array
        Vertical divergence at full levels, shape (6, n, n, nlev).
    """
    d_flux = rho_w_half[..., :-1] - rho_w_half[..., 1:]
    div_z_star = d_flux / dz
    return div_z_star / jacobian[..., None]
