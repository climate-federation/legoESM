"""3D operator wrappers for cubed-sphere grids.

Provides two sets of operators:

1. **Horizontal operators** using native 4D halo exchange (one
   communication for all vertical levels):
   vorticity_3d, gradient_x_3d, gradient_y_3d, divergence_3d,
   hyperdiffusion_3d, laplacian_compact_3d.

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
    laplacian_compact,
)
from legoesm.core.operators_fv import (
    fv_flux_divergence as _fv_flux_divergence_2d,
    fv_scalar_advection as _fv_scalar_advection_2d,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo_4d, pad_halo_vector_4d


def vorticity_3d(
    u_3d: jax.Array, v_3d: jax.Array, grid: CubedSphereGrid,
) -> jax.Array:
    """Compute vorticity at all levels using native 4D halo exchange.

    Parameters
    ----------
    u_3d, v_3d : jax.Array
        Wind components, shape (6, n, n, nlev).
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array : Vorticity, shape (6, n, n, nlev).
    """
    # One 4D vector halo exchange = 2 MPI messages (instead of 2*nlev)
    u_pad, v_pad = pad_halo_vector_4d(
        u_3d, v_3d,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )

    # Stencil identical to 2D curl_z but with trailing level axis
    vort_x = v_pad * grid.hy_ext[..., None]
    vort_y = u_pad * grid.hx_ext[..., None]

    d_vort_x = vort_x[:, 2:, 1:-1, :] - vort_x[:, :-2, 1:-1, :]
    d_vort_y = vort_y[:, 1:-1, 2:, :] - vort_y[:, 1:-1, :-2, :]

    return (d_vort_x - d_vort_y) / (2.0 * grid.area[..., None])


def gradient_x_3d(
    field_3d: jax.Array, grid: CubedSphereGrid,
) -> jax.Array:
    """Compute x-gradient at all levels using native 4D halo exchange.

    Parameters
    ----------
    field_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array : d(field)/dx, shape (6, n, n, nlev).
    """
    # One 4D halo exchange = 1 MPI message set (instead of nlev)
    padded = pad_halo_4d(field_3d, interp_offsets=grid.halo_interp_offsets)
    return (padded[:, 2:, 1:-1, :] - padded[:, :-2, 1:-1, :]) / grid.dx[..., None]


def gradient_y_3d(
    field_3d: jax.Array, grid: CubedSphereGrid,
) -> jax.Array:
    """Compute y-gradient at all levels using native 4D halo exchange.

    Parameters
    ----------
    field_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array : d(field)/dy, shape (6, n, n, nlev).
    """
    padded = pad_halo_4d(field_3d, interp_offsets=grid.halo_interp_offsets)
    return (padded[:, 1:-1, 2:, :] - padded[:, 1:-1, :-2, :]) / grid.dy[..., None]


def divergence_3d(
    u_3d: jax.Array, v_3d: jax.Array, grid: CubedSphereGrid,
) -> jax.Array:
    """Compute divergence at all levels using native 4D halo exchange.

    Parameters
    ----------
    u_3d, v_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array : Divergence, shape (6, n, n, nlev).
    """
    # One 4D vector halo exchange = 2 MPI messages (instead of 2*nlev)
    u_pad, v_pad = pad_halo_vector_4d(
        u_3d, v_3d,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )

    flux_x_pad = u_pad * grid.hy_ext[..., None]
    flux_y_pad = v_pad * grid.hx_ext[..., None]

    d_flux_x = flux_x_pad[:, 2:, 1:-1, :] - flux_x_pad[:, :-2, 1:-1, :]
    d_flux_y = flux_y_pad[:, 1:-1, 2:, :] - flux_y_pad[:, 1:-1, :-2, :]

    return (d_flux_x + d_flux_y) / (2.0 * grid.area[..., None])


def hyperdiffusion_3d(
    field_3d: jax.Array, grid: CubedSphereGrid, coeff: float,
) -> jax.Array:
    """Compute hyperdiffusion at all levels using native 4D halo.

    -coeff * nabla^4(field) where the inner Laplacian is the compact
    stencil and the outer is the standard div(grad) form.

    Total halo exchanges: 5 (regardless of nlev), down from 5*nlev.

    Parameters
    ----------
    field_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid
    coeff : float

    Returns
    -------
    jax.Array : Hyperdiffusion tendency, shape (6, n, n, nlev).
    """
    # Inner ∇² (compact): 1 halo exchange
    lap1 = laplacian_compact_3d(field_3d, grid)
    # Outer ∇² = div(grad): gradient_x + gradient_y + divergence = 1+1+2 = 4 halo exchanges
    gx = gradient_x_3d(lap1, grid)
    gy = gradient_y_3d(lap1, grid)
    lap2 = divergence_3d(gx, gy, grid)
    return -coeff * lap2


def laplacian_compact_3d(
    field_3d: jax.Array, grid: CubedSphereGrid,
) -> jax.Array:
    """Compact-stencil Laplacian at all levels using native 4D halo.

    Uses adjacent-cell second differences and resolves the 2Δx
    checkerboard mode.  One halo exchange for all levels.

    Parameters
    ----------
    field_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array : ∇²f, shape (6, n, n, nlev)
    """
    padded = pad_halo_4d(field_3d, interp_offsets=grid.halo_interp_offsets)
    interior = padded[:, 1:-1, 1:-1, :]
    hx_sq = (grid.dx / 2.0) ** 2
    hy_sq = (grid.dy / 2.0) ** 2

    d2f_dx2 = (padded[:, 2:, 1:-1, :] - 2.0 * interior + padded[:, :-2, 1:-1, :]) / hx_sq[..., None]
    d2f_dy2 = (padded[:, 1:-1, 2:, :] - 2.0 * interior + padded[:, 1:-1, :-2, :]) / hy_sq[..., None]

    return d2f_dx2 + d2f_dy2


def fv_flux_divergence_3d(
    q_3d: jax.Array, u_3d: jax.Array, v_3d: jax.Array,
    grid: CubedSphereGrid, limiter: bool = True,
) -> jax.Array:
    """Conservative FV flux divergence at all levels via vmap.

    Parameters
    ----------
    q_3d : jax.Array, shape (6, n, n, nlev)
    u_3d, v_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid
    limiter : bool

    Returns
    -------
    jax.Array : shape (6, n, n, nlev)
    """
    def single_level(q_k, u_k, v_k):
        return _fv_flux_divergence_2d(q_k, u_k, v_k, grid, limiter)

    q_t = jnp.moveaxis(q_3d, -1, 0)
    u_t = jnp.moveaxis(u_3d, -1, 0)
    v_t = jnp.moveaxis(v_3d, -1, 0)
    result = jax.vmap(single_level)(q_t, u_t, v_t)
    return jnp.moveaxis(result, 0, -1)


def fv_scalar_advection_3d(
    q_3d: jax.Array, u_3d: jax.Array, v_3d: jax.Array,
    grid: CubedSphereGrid, limiter: bool = True,
) -> jax.Array:
    """PPM advection of scalar at all levels via vmap.

    Parameters
    ----------
    q_3d : jax.Array, shape (6, n, n, nlev)
    u_3d, v_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid
    limiter : bool

    Returns
    -------
    jax.Array : shape (6, n, n, nlev)
    """
    def single_level(q_k, u_k, v_k):
        return _fv_scalar_advection_2d(q_k, u_k, v_k, grid, limiter)

    q_t = jnp.moveaxis(q_3d, -1, 0)
    u_t = jnp.moveaxis(u_3d, -1, 0)
    v_t = jnp.moveaxis(v_3d, -1, 0)
    result = jax.vmap(single_level)(q_t, u_t, v_t)
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
