"""Ocean mixing parameterizations.

Horizontal: Laplacian viscosity/diffusivity (reuses core operators).
Vertical: Explicit second-order diffusion.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.operators_3d import (
    divergence_3d,
    gradient_x_3d,
    gradient_y_3d,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo_4d
from legoesm.ocean.vertical import OceanZStarCoordinate


def laplacian_viscosity_3d(
    field_3d: jnp.ndarray,
    grid: CubedSphereGrid,
    coeff: float,
    padded: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute ``coeff · ∇² f`` for all levels using the native 4D path.

    Mathematically identical to the previous ``vmap(laplacian)`` over
    the level axis (``div(grad(f))`` with the same centred metric-aware
    operators), but runs the cubed-sphere halo exchange once over all
    levels through ``pad_halo_4d`` / ``pad_halo_vector_4d`` instead of
    once per level.  Under MPI this collapses ``nlev`` separate messages
    into a constant number, which is the dominant cost on multi-GPU
    runs (CLAUDE.md ``Parallel and HPC Rules`` flag the per-level
    ``vmap(pad_halo)`` pattern explicitly).

    Parameters
    ----------
    field_3d : array
        3D field, shape (6, n, n, nlev).
    grid : CubedSphereGrid
        Horizontal grid.
    coeff : float
        Viscosity/diffusivity coefficient [m^2/s].
    padded : array or None
        Pre-padded field, shape (6, n+2, n+2, nlev).  When provided,
        the internal halo exchange is skipped — used by callers that
        share the same input across multiple operators (e.g. an
        explicit Laplacian alongside a biharmonic hyperdiffusion).

    Returns
    -------
    array : Laplacian tendency, shape (6, n, n, nlev).
    """
    # Pre-pad the input field once so both ``gradient_x_3d`` and
    # ``gradient_y_3d`` skip their internal halo exchange — saves one
    # MPI message in distributed runs.  ``divergence_3d`` still issues
    # its own vector halo exchange on the gradient outputs.
    if padded is None:
        dg = getattr(grid, 'duogrid', None)
        offsets = None if dg is not None else grid.halo_interp_offsets
        padded = pad_halo_4d(field_3d, interp_offsets=offsets, duogrid=dg)
    gx = gradient_x_3d(field_3d, grid, padded=padded)
    gy = gradient_y_3d(field_3d, grid, padded=padded)
    return coeff * divergence_3d(gx, gy, grid)


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
    if field.shape[-1] < 2:
        return jnp.zeros_like(field)

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
    if field.shape[-1] < 2:
        return jnp.zeros_like(field)

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
