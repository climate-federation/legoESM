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


# Default explicit-diffusion CFL safety factor (numerics).
_CFL_SAFETY_DEFAULT = 0.45

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
    dt: float | None = None,
    cfl_safety: float = _CFL_SAFETY_DEFAULT,
) -> jnp.ndarray:
    """Compute d/dz(coeff * d(field)/dz) using 2nd-order centered differences.

    Explicit vertical diffusion with zero-flux boundary conditions
    at surface and bottom.  When ``dt`` is supplied the diffusivity is
    capped by the local explicit-Euler CFL bound
    ``K ≤ cfl_safety · dz_half² / dt`` (default safety 0.45 ≤ 0.5
    leaves a small stability margin).  Without the cap a moderate
    ``coeff`` over thin upper layers will explode on an ocean physics
    step.  Codex narrow review iter-4 #1.

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
    dt : float, optional
        Physics step [s].  When provided, enforces explicit-Euler CFL.
    cfl_safety : float
        Stability margin (must be ≤ 0.5 for explicit Euler).

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

    if dt is not None:
        coeff_cap = cfl_safety * jnp.minimum(
            dz[..., :-1], dz[..., 1:],
        ) ** 2 / jnp.maximum(dt, 1.0e-12)
        coeff_eff = jnp.minimum(coeff, coeff_cap)
    else:
        coeff_eff = coeff

    # Diffusive flux at interior interfaces: coeff * d(field)/dz.
    # Dry / land columns have ``dz = 0`` (jacobian = 0).  Safe
    # denominators avoid 0/0 = NaN in both the flux and tendency
    # divisions; the result is gated to zero on dry columns at the
    # end so wet-cell output is bitwise identical.  Field is also
    # substituted to 0 on dry cells so an upstream NaN sentinel in T
    # / S / u / v cannot leak NaN through the backward pass of the
    # subtraction (``NaN − NaN`` has NaN gradients).
    dz_half_safe = jnp.where(dz_half > 0.0, dz_half, 1.0)
    dz_safe = jnp.where(dz > 0.0, dz, 1.0)
    field_safe = jnp.where(dz > 0.0, field, 0.0)
    df_dz = (field_safe[..., :-1] - field_safe[..., 1:]) / dz_half_safe
    flux = coeff_eff * df_dz  # (..., nlev-1)

    # Tendency at full levels: d(flux)/dz with zero-flux BCs.
    # Using concatenate avoids scatter updates (better JIT lowering and
    # no mixed-dtype scatter edge cases on strict x64 runs).
    top = -flux[..., :1] / dz_safe[..., :1]  # surface: flux_above = 0
    interior = (flux[..., :-1] - flux[..., 1:]) / dz_safe[..., 1:-1]
    bottom = flux[..., -1:] / dz_safe[..., -1:]  # bottom: flux_below = 0
    tend = jnp.concatenate([top, interior, bottom], axis=-1)
    return jnp.where(dz > 0.0, tend, 0.0)


def vertical_diffusion_variable_K(
    field: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    K_half: jnp.ndarray,
    dt: float | None = None,
    cfl_safety: float = _CFL_SAFETY_DEFAULT,
) -> jnp.ndarray:
    """Compute d/dz(K(z) * d(field)/dz) with spatially varying diffusivity.

    Same algorithm as ``vertical_diffusion`` but accepts a 3-D
    diffusivity array at interior interfaces instead of a scalar.
    When ``dt`` is supplied, ``K_half`` is capped per-interface by
    ``cfl_safety · min(dz_k, dz_{k+1})² / dt`` so Richardson/KPP
    callers cannot violate the explicit-Euler CFL on thin upper
    layers.  Codex narrow review iter-4 #2.

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
    dt : float, optional
        Physics step [s].  When provided, enforces explicit-Euler CFL.
    cfl_safety : float
        Stability margin (must be ≤ 0.5 for explicit Euler).

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

    # Substitute K_half with 0 on dry interfaces BEFORE any arithmetic.
    # Upstream callers (KPP, Richardson) compute K_half from
    # ``h_bl·w_s·G(σ)`` / ``N²``-dependent profiles that may carry
    # NaN/Inf into the dry columns.  ``K_half * df_dz`` then propagates
    # NaN forward, and the gradient ∂(NaN·x)/∂x = NaN — leaking NaN
    # through the backward pass even though my iter-58 ``dz_safe`` /
    # output mask scrub the forward value.  ``jnp.where(wet, K_half, 0)``
    # SUBSTITUTES the dry-cell value with a clean 0 so neither forward
    # nor backward sees NaN.  Codex iter-58 stop-time review.
    dry_iface = (dz[..., :-1] <= 0.0) | (dz[..., 1:] <= 0.0)
    K_half = jnp.where(dry_iface, 0.0, K_half)

    if dt is not None:
        K_cap = cfl_safety * jnp.minimum(
            dz[..., :-1], dz[..., 1:],
        ) ** 2 / jnp.maximum(dt, 1.0e-12)
        K_half = jnp.minimum(K_half, K_cap)

    # Safe denominators for dry / land columns (dz = 0 there).  See
    # the matching comment in ``vertical_diffusion`` — wet-cell output
    # is bitwise identical.
    dz_half_safe = jnp.where(dz_half > 0.0, dz_half, 1.0)
    dz_safe = jnp.where(dz > 0.0, dz, 1.0)
    # Also scrub field on dry cells before differencing so an upstream
    # NaN sentinel in T/S/u/v cannot leak through (``field[..., :-1]
    # − field[..., 1:]`` is otherwise ``NaN − NaN`` on a fully-dry
    # column).
    field_safe = jnp.where(dz > 0.0, field, 0.0)
    df_dz = (field_safe[..., :-1] - field_safe[..., 1:]) / dz_half_safe
    flux = K_half * df_dz  # (..., nlev-1)

    top = -flux[..., :1] / dz_safe[..., :1]
    interior = (flux[..., :-1] - flux[..., 1:]) / dz_safe[..., 1:-1]
    bottom = flux[..., -1:] / dz_safe[..., -1:]
    tend = jnp.concatenate([top, interior, bottom], axis=-1)
    return jnp.where(dz > 0.0, tend, 0.0)
