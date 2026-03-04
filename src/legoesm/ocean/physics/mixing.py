"""Ocean mixing parameterizations.

Horizontal: Laplacian viscosity/diffusivity (reuses core operators).
Vertical: Explicit second-order diffusion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators import laplacian
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.vertical import OceanZStarCoordinate


def laplacian_viscosity_3d(
    field_3d: jnp.ndarray,
    grid: CubedSphereGrid,
    coeff: float,
) -> jnp.ndarray:
    """Compute A_h * nabla^2(field) vmapped over levels.

    Reuses the 2D laplacian operator from core/operators.py,
    same vmap-over-levels pattern as operators_3d.hyperdiffusion_3d.

    Parameters
    ----------
    field_3d : array
        3D field, shape (6, n, n, nlev).
    grid : CubedSphereGrid
        Horizontal grid.
    coeff : float
        Viscosity/diffusivity coefficient [m^2/s].

    Returns
    -------
    array : Laplacian tendency, shape (6, n, n, nlev).
    """
    def single_level(f_k):
        f_field = Field(data=f_k, name="f", dims=("face", "x", "y"), units="")
        return laplacian(f_field, grid).data

    f_t = jnp.moveaxis(field_3d, -1, 0)   # (nlev, 6, n, n)
    result = jax.vmap(single_level)(f_t)   # (nlev, 6, n, n)
    return coeff * jnp.moveaxis(result, 0, -1)


def vertical_diffusion(
    field: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    coeff: float,
) -> jnp.ndarray:
    """Compute d/dz(coeff * d(field)/dz) using 2nd-order centered differences.

    Explicit vertical diffusion with zero-flux boundary conditions
    at surface and bottom.

    Parameters
    ----------
    field : array
        3D field, shape (..., nlev).
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    jacobian : array
        Dynamic Jacobian, shape (...).
    coeff : float
        Diffusivity [m^2/s].

    Returns
    -------
    array : Vertical diffusion tendency, shape (..., nlev).
    """
    dtype = field.dtype
    jacobian = jacobian.astype(dtype)
    coeff = jnp.asarray(coeff, dtype=dtype)

    # Actual layer thickness and interface spacing
    dz = z_coord.dz_ref * jacobian[..., jnp.newaxis]         # (..., nlev)
    dz_half = 0.5 * (dz[..., :-1] + dz[..., 1:])  # (..., nlev-1)

    # Diffusive flux at interior interfaces: coeff * d(field)/dz
    df_dz = (field[..., :-1] - field[..., 1:]) / dz_half
    flux = coeff * df_dz  # (..., nlev-1)

    # Tendency at full levels: d(flux)/dz with zero-flux BCs.
    # Using concatenate avoids scatter updates (better JIT lowering and
    # no mixed-dtype scatter edge cases on strict x64 runs).
    top = -flux[..., :1] / dz[..., :1]  # surface: flux_above = 0
    interior = (flux[..., :-1] - flux[..., 1:]) / dz[..., 1:-1]
    bottom = flux[..., -1:] / dz[..., -1:]  # bottom: flux_below = 0
    return jnp.concatenate([top, interior, bottom], axis=-1)


def vertical_diffusion_variable_K(
    field: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    K_half: jnp.ndarray,
) -> jnp.ndarray:
    """Compute d/dz(K(z) * d(field)/dz) with spatially varying diffusivity.

    Same algorithm as ``vertical_diffusion`` but accepts a 3-D diffusivity
    array at interior interfaces instead of a scalar.

    Parameters
    ----------
    field : array
        3D field, shape (..., nlev).
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    jacobian : array
        Dynamic Jacobian, shape (...).
    K_half : array
        Diffusivity at interior interfaces [m^2/s], shape (..., nlev-1).

    Returns
    -------
    array : Vertical diffusion tendency, shape (..., nlev).
    """
    dtype = field.dtype
    jacobian = jacobian.astype(dtype)
    K_half = K_half.astype(dtype)

    dz = z_coord.dz_ref * jacobian[..., jnp.newaxis]         # (..., nlev)
    dz_half = 0.5 * (dz[..., :-1] + dz[..., 1:])  # (..., nlev-1)

    df_dz = (field[..., :-1] - field[..., 1:]) / dz_half
    flux = K_half * df_dz  # (..., nlev-1)

    top = -flux[..., :1] / dz[..., :1]
    interior = (flux[..., :-1] - flux[..., 1:]) / dz[..., 1:-1]
    bottom = flux[..., -1:] / dz[..., -1:]
    return jnp.concatenate([top, interior, bottom], axis=-1)
