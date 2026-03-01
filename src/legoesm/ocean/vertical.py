"""Ocean z-star vertical coordinate.

z* = H_max * (z + H) / (eta + H)

where H is the local ocean depth (bathymetry) and eta is the
time-varying sea surface height.

Unlike the atmosphere's z-star (static terrain Jacobian), the ocean
z-star has a DYNAMIC Jacobian J = (eta + H) / H that is recomputed
at every timestep as eta evolves.

Level convention: k=0 is surface, k=nlev-1 is deepest.
Reference z values are negative (below sea level).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class OceanZStarCoordinate(NamedTuple):
    """Static vertical grid definition (independent of eta).

    Levels indexed surface-to-bottom: k=0 is surface, k=nlev-1 is deepest.
    Reference layer thicknesses assume eta=0 and flat bottom H_max.

    Fields
    ------
    n_levels : int
        Number of vertical levels.
    H_max : float
        Maximum ocean depth [m] (positive).
    z_full_ref : array
        Reference z* at full (cell center) levels [m], shape (nlev,).
        Negative values (below sea level). z_full_ref[0] is shallowest.
    z_half_ref : array
        Reference z* at half (interface) levels [m], shape (nlev+1,).
        z_half_ref[0] = 0 (surface), z_half_ref[-1] = -H_max (bottom).
    dz_ref : array
        Reference layer thickness [m], shape (nlev,). Positive.
    dz_half_ref : array
        Distance between adjacent full levels [m], shape (nlev-1,).
    """
    n_levels: int
    H_max: float
    z_full_ref: jnp.ndarray
    z_half_ref: jnp.ndarray
    dz_ref: jnp.ndarray
    dz_half_ref: jnp.ndarray


def create_ocean_z_star(
    n_levels: int = 50,
    H_max: float = 5500.0,
    dz_surface: float = 10.0,
    dz_deep: float = 200.0,
) -> OceanZStarCoordinate:
    """Create a stretched ocean z-star coordinate.

    Uses hyperbolic tangent stretching: fine resolution near surface
    (~dz_surface m), coarse at depth (~dz_deep m).

    Parameters
    ----------
    n_levels : int
        Number of vertical levels.
    H_max : float
        Maximum ocean depth [m].
    dz_surface : float
        Target layer thickness near surface [m].
    dz_deep : float
        Target layer thickness at depth [m].

    Returns
    -------
    OceanZStarCoordinate : The vertical coordinate.
    """
    if n_levels < 2:
        raise ValueError(
            f"n_levels must be >= 2 for finite-difference operators, got {n_levels!r}",
        )
    if H_max <= 0.0:
        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    if dz_surface <= 0.0:
        raise ValueError(f"dz_surface must be > 0, got {dz_surface!r}")
    if dz_deep <= 0.0:
        raise ValueError(f"dz_deep must be > 0, got {dz_deep!r}")

    # Stretched grid: dz grows smoothly from dz_surface to dz_deep.
    # Use a normalized distribution then scale to match H_max.
    k = jnp.arange(n_levels, dtype=jnp.float32)

    # Layer thickness profile: linear growth from dz_surface to dz_deep
    dz_raw = dz_surface + k * (dz_deep - dz_surface) / jnp.maximum(n_levels - 1.0, 1.0)

    # Normalize so total thickness matches H_max
    scale = H_max / jnp.sum(dz_raw)
    dz_ref = dz_raw * scale

    # Interface depths from cumulative sum (surface=0, bottom=-H_max)
    z_half_ref = jnp.concatenate([
        jnp.array([0.0]),
        -jnp.cumsum(dz_ref),
    ])

    # Full level depths (cell centers)
    z_full_ref = 0.5 * (z_half_ref[:-1] + z_half_ref[1:])

    # Layer thicknesses (positive)
    dz_ref = z_half_ref[:-1] - z_half_ref[1:]  # positive since z[k] > z[k+1]

    # Distance between full levels
    dz_half_ref = z_full_ref[:-1] - z_full_ref[1:]  # positive

    return OceanZStarCoordinate(
        n_levels=n_levels,
        H_max=H_max,
        z_full_ref=z_full_ref,
        z_half_ref=z_half_ref,
        dz_ref=dz_ref,
        dz_half_ref=dz_half_ref,
    )


def compute_layer_thickness(
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
) -> jnp.ndarray:
    """Compute actual layer thickness incorporating eta and bathymetry.

    h_k = dz_ref[k] * (eta + H_bathy) / H_max

    The dynamic Jacobian J = (eta + H_bathy) / H_max modifies
    reference thicknesses to account for the actual water column.

    Parameters
    ----------
    eta : array
        Sea surface height [m], shape (...).
    H_bathy : array
        Local bathymetry depth [m], shape (...). Positive.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.

    Returns
    -------
    array : Layer thickness [m], shape (..., nlev). Positive.
    """
    J = compute_ocean_jacobian(eta, H_bathy, z_coord)
    return z_coord.dz_ref * J[..., jnp.newaxis]


def compute_ocean_jacobian(
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
) -> jnp.ndarray:
    """Compute the dynamic z-star Jacobian.

    J = (eta + H_bathy) / H_max

    This is recomputed at every timestep as eta evolves.

    Parameters
    ----------
    eta : array
        Sea surface height [m], shape (...).
    H_bathy : array
        Local bathymetry depth [m], shape (...). Positive.
    z_coord : OceanZStarCoordinate
        Vertical coordinate (provides H_max).

    Returns
    -------
    array : Jacobian, shape (...).
    """
    return (eta + H_bathy) / z_coord.H_max


def upwind_vertical_gradient(
    field: jnp.ndarray,
    dz_half: jnp.ndarray,
    w_star: jnp.ndarray,
    *,
    eps: float = 1.0e-12,
) -> jnp.ndarray:
    """Compute first-order upwind d(field)/dz at full levels.

    Assumes levels are indexed surface-to-bottom (k=0 at surface).

    Parameters
    ----------
    field : array
        Field at full levels, shape (..., nlev).
    dz_half : array
        Full-level spacing, shape (..., nlev-1). Positive.
    w_star : array
        Vertical velocity in transformed coordinates, shape (..., nlev).
        Positive means upward.
    eps : float
        Small denominator guard for spacing.

    Returns
    -------
    array : Upwind vertical gradient d(field)/dz, shape (..., nlev).
    """
    inv_dz_half = 1.0 / jnp.maximum(dz_half, eps)
    df = (field[..., :-1] - field[..., 1:]) * inv_dz_half

    zeros = jnp.zeros((*field.shape[:-1], 1), dtype=field.dtype)

    # Upward flow (w>0): donor is deeper cell -> (f[k] - f[k+1]) / dz.
    grad_up = jnp.concatenate([df, zeros], axis=-1)
    # Downward flow (w<0): donor is shallower cell -> (f[k-1] - f[k]) / dz.
    grad_down = jnp.concatenate([zeros, df], axis=-1)

    return jnp.where(w_star > 0.0, grad_up, grad_down)
