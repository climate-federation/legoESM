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
    if n_levels < 1:
        raise ValueError(
            f"n_levels must be >= 1, got {n_levels!r}",
        )
    if H_max <= 0.0:
        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    if dz_surface <= 0.0:
        raise ValueError(f"dz_surface must be > 0, got {dz_surface!r}")
    if dz_deep <= 0.0:
        raise ValueError(f"dz_deep must be > 0, got {dz_deep!r}")

    # Stretched grid: dz grows smoothly from dz_surface to dz_deep.
    # Use a normalized distribution then scale to match H_max.
    from legoesm.core.precision import get_policy
    k = jnp.arange(n_levels, dtype=get_policy().control)

    # Layer thickness profile: linear growth from dz_surface to dz_deep
    dz_raw = dz_surface + k * (dz_deep - dz_surface) / jnp.maximum(n_levels - 1.0, 1.0)

    # Normalize so total thickness matches H_max
    scale = H_max / jnp.sum(dz_raw)
    dz_ref = dz_raw * scale

    # Interface depths from cumulative sum (surface=0, bottom=-H_max)
    z_half_ref = jnp.concatenate([
        jnp.array([0.0], dtype=dz_ref.dtype),
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
    min_water_column_m: float | None = None,
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
    min_water_column_m : float or None
        Optional lower bound for local water-column thickness
        ``eta + H_bathy`` [m]. When set, Jacobian/thickness values are
        clipped to avoid dry or negative columns.

    Returns
    -------
    array : Layer thickness [m], shape (..., nlev). Positive.
    """
    J = compute_ocean_jacobian(
        eta, H_bathy, z_coord, min_water_column_m=min_water_column_m,
    )
    return z_coord.dz_ref * J[..., jnp.newaxis]


def compute_ocean_jacobian(
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    min_water_column_m: float | None = None,
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
    min_water_column_m : float or None
        Optional lower bound for local water-column thickness
        ``eta + H_bathy`` [m].

    Returns
    -------
    array : Jacobian, shape (...).
    """
    water_col = eta + H_bathy
    if min_water_column_m is not None:
        min_col = jnp.asarray(min_water_column_m, dtype=water_col.dtype)
        water_col = jnp.maximum(water_col, min_col)
    return water_col / z_coord.H_max


def upwind_vertical_gradient(
    field: jnp.ndarray,
    dz_half: jnp.ndarray,
    w: jnp.ndarray,
    *,
    eps: float = 1.0e-12,
) -> jnp.ndarray:
    """Compute first-order upwind d(field)/dz at full levels.

    Assumes levels are indexed surface-to-bottom (k=0 at surface).
    Caller must use consistent coordinates: both ``dz_half`` and ``w``
    should be in the same vertical coordinate (physical z or z*).

    Parameters
    ----------
    field : array
        Field at full levels, shape (..., nlev).
    dz_half : array
        Full-level spacing [m], shape (..., nlev-1). Positive.
    w : array
        Vertical velocity [m/s], shape (..., nlev).
        Only its sign is used (upwind direction). Positive = upward.
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

    return jnp.where(w > 0.0, grad_up, grad_down)


# ---------------------------------------------------------------------------
# Vertical velocity diagnosis and advection (shared across ocean dycores)
# ---------------------------------------------------------------------------

def diagnose_w_from_flux_div(flux_div_k, z_coord=None,
                              thickness_weighted=False):
    """Diagnose z-star transport velocity from flux divergence.

    Performs a bottom-up cumulative sum of the horizontal flux divergence
    and optionally applies the z-star sigma correction so that
    ẇ = 0 at both surface and bottom.

    Parameters
    ----------
    flux_div_k : array, shape (..., nlev)
        Horizontal flux divergence at each layer.
        If ``thickness_weighted=False`` (legacy), this is ``div(u)`` and
        will be multiplied by ``dz_ref`` before integration.
        If ``thickness_weighted=True``, this is ``div(h*u)`` [m/s] and
        already has layer thickness folded in; no dz multiplication.
    z_coord : OceanZStarCoordinate or None
        When provided, applies the z-star correction.
    thickness_weighted : bool
        If True, ``flux_div_k`` already includes layer thickness
        (i.e. it was computed from thickness-weighted velocity).
        Default False for backward compatibility.

    Returns
    -------
    w : array, shape (..., nlev+1)
        Vertical velocity on half levels (surface first, bottom last = 0).
    """
    # From continuity: w(k) = w(k+1) + div_h(h_k * u_k)
    # If flux_div_k already includes layer thickness (thickness_weighted=True),
    # we cumsum directly. Otherwise, multiply by dz_ref first.
    if thickness_weighted:
        fd_integrated = flux_div_k
    elif z_coord is not None:
        dz_ref = z_coord.dz_ref  # Layer thicknesses
        fd_integrated = flux_div_k * dz_ref[jnp.newaxis, jnp.newaxis, :]
    else:
        # Fallback for testing (assume unit thickness)
        fd_integrated = flux_div_k

    fd_rev = fd_integrated[..., ::-1]
    cumsum_rev = jnp.cumsum(fd_rev, axis=-1)
    w_inner = -cumsum_rev[..., ::-1]
    zeros_bottom = jnp.zeros((*flux_div_k.shape[:-1], 1), dtype=flux_div_k.dtype)
    w_euler = jnp.concatenate([w_inner, zeros_bottom], axis=-1)

    if z_coord is None:
        return w_euler

    sigma = (z_coord.z_half_ref + z_coord.H_max) / z_coord.H_max
    deta_dt = w_euler[..., 0:1]
    return w_euler - sigma * deta_dt


def vertical_advection_ocean(field, w_half, z_coord, jacobian):
    """Vertical advection ``-w * d(field)/dz`` with upwind scheme.

    Parameters
    ----------
    field : array, shape (..., nlev)
        Quantity being advected.
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half levels.
    z_coord : OceanZStarCoordinate
        Vertical coordinate (provides ``dz_half_ref``).
    jacobian : array, shape (...)
        Dynamic z-star Jacobian.

    Returns
    -------
    tendency : array, shape (..., nlev)
    """
    w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
    jac_safe = jnp.maximum(jacobian[..., jnp.newaxis], 1.0e-10)
    dz_half = z_coord.dz_half_ref * jac_safe
    grad = upwind_vertical_gradient(field, dz_half, w_full)
    
    # Vertical advection calculation
    
    return -w_full * grad


def compute_depth_mean(
    u: jnp.ndarray,
    h: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    min_h: float = 1e-10,
) -> jnp.ndarray:
    """Thickness-weighted depth-mean of a 3D field (#172).

    Parameters
    ----------
    u : (..., nlev)
    h : (..., nlev)
    mask : (...) or None
    min_h : float

    Returns
    -------
    u_bar : (...)
    """
    H = jnp.maximum(jnp.sum(h, axis=-1), min_h)
    u_bar = jnp.sum(u * h, axis=-1) / H
    if mask is not None:
        u_bar = u_bar * mask
    return u_bar


def flux_form_vertical_momentum_advection(
    u: jnp.ndarray,
    w_half: jnp.ndarray,
    h_u: jnp.ndarray,
) -> jnp.ndarray:
    """Flux-form vertical momentum advection as a per-thickness tendency.

    Computes the interface-upwind vertical momentum flux
    ``F[k] = w_half[k] * u_face[k]`` with

    - ``F[0] = F[nlev] = 0``  (rigid-lid / no-flux boundary)
    - ``u_face[k] = u[k]``     when ``w_half[k] > 0``  (upward, from below)
    - ``u_face[k] = u[k-1]``   when ``w_half[k] <= 0`` (downward, from above)

    and returns ``-(F_top - F_bot) / h_u`` at each level.  This is the
    tracer-path pattern (``flux_form_vertical_tracer_advection``)
    converted back to a per-thickness advective tendency so that
    callers can add it directly to ``du/dt``.

    Properties
    ----------
    1. Interior interface-upwind (consistent with the tracer path).
    2. Rigid-lid boundary by construction: ``F[0] = F[nlev] = 0``.
       No artificial momentum injection from the boundary via the
       "hard zero at k=0 and k=nlev-1" pathology that the old
       ``vertical_advection_ocean`` cell-upwind gradient has.
    3. Column momentum flux identity: for any ``w_half`` with
       ``w_half[0] = w_half[nlev] = 0`` (closed column), the sum of
       ``(tendency * h_u)`` over the column is exactly zero.

    Partial-fix status (issue #171)
    -------------------------------
    This is a **Level-1** fix.  Dividing by ``h_u_old`` instead of
    doing a full ``(h·u)_new = (h·u)_old - dt * flux_div`` / ``u_new =
    (h·u)_new / h_u_new`` update leaves a residual
    ``O(dt · u · dh_u/dt / h_u)`` error under dynamic z-star.  A full
    flux-form momentum update requires restructuring the model step
    function (Level 2 in the #171 discussion) and is still open.

    Parameters
    ----------
    u : array, shape (..., nlev)
        Velocity at full levels at the momentum points (u-face, v-face,
        or edge — caller's choice, as long as ``w_half`` and ``h_u``
        are interpolated to the same points).
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels, at the same
        momentum points as ``u``.  Positive = upward.  Must be zero
        at the surface and bottom interfaces.
    h_u : array, shape (..., nlev)
        Layer thickness at the momentum points.  Used only as the
        advective-form denominator.

    Returns
    -------
    tendency : array, shape (..., nlev)
        ``-(F_top - F_bot) / h_u`` — a per-thickness momentum tendency
        ready to add to ``du/dt``.
    """
    vert_flux_div = flux_form_vertical_tracer_advection(u, w_half)
    h_u_safe = jnp.maximum(h_u, 1.0e-10)
    return -vert_flux_div / h_u_safe


def flux_form_vertical_tracer_advection(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
) -> jnp.ndarray:
    """Flux-form vertical tracer advection with first-order upwind.

    Computes the vertical flux divergence  F_top[k] - F_bot[k]  for each
    level k, where F = w * T_face is the upward tracer flux on interfaces.

    Level convention
    ----------------
    k = 0 is surface, k = nlev-1 is bottom.
    Interface k sits ABOVE level k:
      - interface 0  = sea surface  (top of level 0)
      - interface k  = between level k-1 (above) and level k (below), k=1..nlev-1
      - interface nlev = ocean bottom (below level nlev-1)
    w positive = upward.

    Upwind at interior interface k (k = 1 .. nlev-1):
      - w[k] > 0  (upward):  fluid from level k  (below) → T_face = field[k]
      - w[k] <= 0 (downward): fluid from level k-1 (above) → T_face = field[k-1]

    Surface and bottom fluxes are zero (w[0] = w[nlev] = 0 by construction).

    Parameters
    ----------
    field : array, shape (..., nlev)
        Tracer at full levels (e.g. temperature [degC]).
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels [m/s].

    Returns
    -------
    vert_flux_div : array, shape (..., nlev)
        Vertical flux divergence  F_top[k] - F_bot[k]  for each level.
        Units are [tracer] * [m/s]  (NOT divided by layer thickness).
        The caller uses:  h_new*T_new = h_old*T_old - dt*vert_flux_div - dt*horiz_flux_div
    """
    nlev = field.shape[-1]

    # --- Compute upwind tracer flux at each interface ---
    # F has shape (..., nlev+1).  F[..., 0] = 0, F[..., nlev] = 0.
    # For interior interface k (1 <= k <= nlev-1):
    #   F[k] = w[k] * T_face[k]
    #   where T_face[k] = field[k]   if w[k] > 0   (upward, from below)
    #                    = field[k-1] if w[k] <= 0  (downward, from above)

    # Interior w values: w_half[..., 1:nlev] has shape (..., nlev-1)
    w_interior = w_half[..., 1:nlev]  # (..., nlev-1)

    # Upwind selection at interior interfaces
    # Interface k (1-indexed) is between level k-1 (above) and level k (below)
    T_below = field[..., 1:]    # field[k]   for k=1..nlev-1 → (..., nlev-1)
    T_above = field[..., :-1]   # field[k-1] for k=1..nlev-1 → (..., nlev-1)

    T_face_interior = jnp.where(w_interior > 0.0, T_below, T_above)
    F_interior = w_interior * T_face_interior  # (..., nlev-1)

    # Full flux array with zero boundaries
    zeros = jnp.zeros((*field.shape[:-1], 1), dtype=field.dtype)
    F = jnp.concatenate([zeros, F_interior, zeros], axis=-1)  # (..., nlev+1)

    # Flux divergence: F_top[k] - F_bot[k] = F[k] - F[k+1]
    vert_flux_div = F[..., :-1] - F[..., 1:]  # (..., nlev)

    return vert_flux_div
