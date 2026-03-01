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
    # Actual layer thickness and interface spacing
    dz = z_coord.dz_ref * jacobian[..., jnp.newaxis]         # (..., nlev)
    dz_half = 0.5 * (dz[..., :-1] + dz[..., 1:])  # (..., nlev-1)

    # Diffusive flux at interior interfaces: coeff * d(field)/dz
    df_dz = (field[..., :-1] - field[..., 1:]) / dz_half
    flux = coeff * df_dz  # (..., nlev-1)

    # Tendency at full levels: d(flux)/dz
    # Zero-flux BCs: flux = 0 at surface (above k=0) and bottom (below k=nlev-1)
    tendency = jnp.zeros_like(field)
    # Interior levels: (flux_above - flux_below) / dz
    tendency = tendency.at[..., 0].set(
        -flux[..., 0] / dz[..., 0]                    # surface: flux_above=0
    )
    tendency = tendency.at[..., 1:-1].set(
        (flux[..., :-1] - flux[..., 1:]) / dz[..., 1:-1]  # interior
    )
    tendency = tendency.at[..., -1].set(
        flux[..., -1] / dz[..., -1]                    # bottom: flux_below=0
    )

    return tendency
