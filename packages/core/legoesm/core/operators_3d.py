"""3D operator wrappers for cubed-sphere grids.

Provides two sets of operators:

1. **Horizontal operators** using native 4D halo exchange (one
   communication for all vertical levels):
   vorticity_3d, gradient_x_3d, gradient_y_3d, divergence_3d,
   hyperdiffusion_3d, laplacian_compact_3d.

2. **Vertical operators for height coordinates** (non-hydrostatic):
   vertical_advection_height.

All functions operate on raw ``jax.Array`` data.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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
    dg = getattr(grid, 'duogrid', None)
    offsets = None if dg is not None else grid.halo_interp_offsets
    u_pad, v_pad = pad_halo_vector_4d(
        u_3d, v_3d,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )

    # Stencil identical to 2D curl_z but with trailing level axis
    vort_x = v_pad * grid.hy_ext[..., None]
    vort_y = u_pad * grid.hx_ext[..., None]

    d_vort_x = vort_x[:, 2:, 1:-1, :] - vort_x[:, :-2, 1:-1, :]
    d_vort_y = vort_y[:, 1:-1, 2:, :] - vort_y[:, 1:-1, :-2, :]

    return (d_vort_x - d_vort_y) / (2.0 * grid.area[..., None])


def gradient_x_3d_core(padded: jax.Array, dx: jax.Array) -> jax.Array:
    """Centred cc x-gradient from a 1-cell-padded field and the cc ``dx`` metric.

    ``padded`` (F, m+2, m+2, nlev) — the field haloed by one cell; ``dx``
    (F, m, m) the cell-centre x-spacing.  Returns d(field)/dx (F, m, m, nlev).

    Factored out of :func:`gradient_x_3d` so the SAME centred-difference stencil
    serves both the global op (``dx = grid.dx``, full face extent) and the tiled
    production stage (``dx`` = per-tile slice of ``grid.dx``).  The tiled sub-face
    shard_map gradient is therefore bit-identical to the global op with no
    duplicated stencil arithmetic (CLAUDE.md: no re-derived numerics).
    """
    return (padded[:, 2:, 1:-1, :] - padded[:, :-2, 1:-1, :]) / dx[..., None]


def gradient_x_3d(
    field_3d: jax.Array, grid: CubedSphereGrid,
    padded: jax.Array | None = None,
) -> jax.Array:
    """Compute x-gradient at all levels using native 4D halo exchange.

    Parameters
    ----------
    field_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid
    padded : jax.Array, optional
        Pre-padded field (6, n+2, n+2, nlev).  Skips internal halo
        exchange when provided (stage-level packing).

    Returns
    -------
    jax.Array : d(field)/dx, shape (6, n, n, nlev).
    """
    if padded is None:
        dg = getattr(grid, 'duogrid', None)
        offsets = None if dg is not None else grid.halo_interp_offsets
        padded = pad_halo_4d(field_3d, interp_offsets=offsets, duogrid=dg)
    return gradient_x_3d_core(padded, grid.dx)


def gradient_y_3d_core(padded: jax.Array, dy: jax.Array) -> jax.Array:
    """Centred cc y-gradient from a 1-cell-padded field and the cc ``dy`` metric.

    ``padded`` (F, m+2, m+2, nlev); ``dy`` (F, m, m).  Returns d(field)/dy
    (F, m, m, nlev).  The y-axis companion of :func:`gradient_x_3d_core` —
    shared by the global :func:`gradient_y_3d` and the tiled production stage.
    """
    return (padded[:, 1:-1, 2:, :] - padded[:, 1:-1, :-2, :]) / dy[..., None]


def gradient_y_3d(
    field_3d: jax.Array, grid: CubedSphereGrid,
    padded: jax.Array | None = None,
) -> jax.Array:
    """Compute y-gradient at all levels using native 4D halo exchange.

    Parameters
    ----------
    field_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid
    padded : jax.Array, optional
        Pre-padded field (6, n+2, n+2, nlev).  Skips internal halo
        exchange when provided (stage-level packing).

    Returns
    -------
    jax.Array : d(field)/dy, shape (6, n, n, nlev).
    """
    if padded is None:
        dg = getattr(grid, 'duogrid', None)
        offsets = None if dg is not None else grid.halo_interp_offsets
        padded = pad_halo_4d(field_3d, interp_offsets=offsets, duogrid=dg)
    return gradient_y_3d_core(padded, grid.dy)


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
    dg = getattr(grid, 'duogrid', None)
    offsets = None if dg is not None else grid.halo_interp_offsets
    u_pad, v_pad = pad_halo_vector_4d(
        u_3d, v_3d,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )

    flux_x_pad = u_pad * grid.hy_ext[..., None]
    flux_y_pad = v_pad * grid.hx_ext[..., None]

    d_flux_x = flux_x_pad[:, 2:, 1:-1, :] - flux_x_pad[:, :-2, 1:-1, :]
    d_flux_y = flux_y_pad[:, 1:-1, 2:, :] - flux_y_pad[:, 1:-1, :-2, :]

    return (d_flux_x + d_flux_y) / (2.0 * grid.area[..., None])


def hyperdiffusion_3d(
    field_3d: jax.Array, grid: CubedSphereGrid, coeff: float,
    padded: jax.Array | None = None,
    inner_lap: jax.Array | None = None,
    compact_outer: bool = False,
) -> jax.Array:
    """Compute hyperdiffusion at all levels using native 4D halo.

    ``-coeff * nabla^4(field)``.  The inner Laplacian is always the
    compact (reach-1) stencil; the OUTER Laplacian is selected by
    ``compact_outer``:

    - ``compact_outer=False`` (default): the outer ``∇²`` is the wide
      ``div(grad)`` form.  ``gradient_x/y_3d`` use the centred reach-2
      difference ``(f[i+1]-f[i-1])/dx``, which is IDENTICALLY ZERO on a
      ``(-1)^i`` grid mode, so the composite ``∇⁴`` has an exact 2Δx
      NULL — it does not damp the grid-scale checkerboard the biharmonic
      exists to remove (contract's "grid-scale noise damped far faster"
      is vacuous on this path).  Kept as the default only so that dycore
      coefficients tuned against this (grid-scale-blind) operator stay
      bit-identical; flipping the global default needs a coefficient
      re-tune + atm/ocean matrix + visual-regression campaign.
    - ``compact_outer=True``: the outer ``∇²`` is ALSO the compact
      stencil, so ``∇⁴ = ∇²_compact(∇²_compact(field))`` — the
      ``(1,-4,6,-4,1)`` stencil whose transfer symbol ``16 sin⁴(k dx/2)``
      is MAXIMAL at 2Δx (matches MOM5/MOM6 ``delsq`` twice, MPAS-O
      ``del4=del2(del2)``, Griffies 2004).  On the cubed sphere its
      area-integral non-conservation (~1e-3) is bounded by the same
      halo-interp limit as the wide form (in practice slightly SMALLER),
      and it damps the 2Δx mode by ~1024·coeff/dx⁴ instead of ~0.

    Parameters
    ----------
    field_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid
    coeff : float
    padded : jax.Array or None
        Pre-padded field, shape (6, n+2, n+2, nlev).  When provided,
        the inner Laplacian skips its own halo exchange (saves 1 msg).
    inner_lap : jax.Array or None
        Pre-computed inner ``∇²(field_3d)`` (compact stencil), shape
        ``(6, n, n, nlev)``.  When provided, the inner Laplacian
        computation is skipped entirely — useful when the caller has
        already evaluated the same ∇² for an explicit ``A_h``
        Laplacian on the same input and wants to reuse it for the
        biharmonic.  ``padded`` is then ignored for the inner stage
        (still does not affect the outer halo).
    compact_outer : bool
        Select the compact (2Δx-damping) outer Laplacian.  Default
        ``False`` preserves the legacy wide ``div(grad)`` outer stage
        bit-identically.

    Returns
    -------
    jax.Array : Hyperdiffusion tendency, shape (6, n, n, nlev).
    """
    # Inner ∇² (compact): uses pre-padded if available, or skip the
    # whole computation when the caller already has the result.
    if inner_lap is not None:
        lap1 = inner_lap
    else:
        lap1 = laplacian_compact_3d(field_3d, grid, padded=padded)
    # Pad lap1 once and reuse for the outer stage — otherwise each
    # gradient / compact-Laplacian call would emit its own pad_halo_4d
    # MPI exchange on the same lap1 (saves 1 halo MPI call per hyperdiff).
    dg = getattr(grid, 'duogrid', None)
    offsets = None if dg is not None else grid.halo_interp_offsets
    lap1_pad = pad_halo_4d(lap1, interp_offsets=offsets, duogrid=dg)
    if compact_outer:
        # Compact outer ∇² -> ∇⁴ = compact(compact): maximal 2Δx damping.
        lap2 = laplacian_compact_3d(lap1, grid, padded=lap1_pad)
    else:
        # Wide outer ∇² = div(grad): legacy path (2Δx null, see docstring).
        gx = gradient_x_3d(lap1, grid, padded=lap1_pad)
        gy = gradient_y_3d(lap1, grid, padded=lap1_pad)
        lap2 = divergence_3d(gx, gy, grid)
    return -coeff * lap2


def laplacian_compact_3d(
    field_3d: jax.Array, grid: CubedSphereGrid,
    padded: jax.Array | None = None,
) -> jax.Array:
    """Compact-stencil Laplacian at all levels using native 4D halo.

    Uses adjacent-cell second differences and resolves the 2Δx
    checkerboard mode.  One halo exchange for all levels.

    Parameters
    ----------
    field_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid
    padded : jax.Array or None
        Pre-padded field, shape (6, n+2, n+2, nlev).  When provided,
        the internal halo exchange is skipped (saves 1 MPI message).

    Returns
    -------
    jax.Array : ∇²f, shape (6, n, n, nlev)
    """
    if padded is None:
        dg = getattr(grid, 'duogrid', None)
        offsets = None if dg is not None else grid.halo_interp_offsets
        padded = pad_halo_4d(field_3d, interp_offsets=offsets, duogrid=dg)
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
    """Conservative FV flux-divergence TENDENCY at all levels via vmap.

    **Sign convention** (load-bearing, audit cycle iter-35):
    Returns the tracer tendency form ``dq/dt = -div(q · v)``, NOT
    the raw divergence ``+div(q · v)``.  Callers that want moisture
    convergence should use this output directly without further
    negation; callers that want the divergence quantity itself should
    explicitly negate the output.  See the underlying ``_fv_flux_
    divergence_2d`` (operators_fv.py) for the formula.

    Parameters
    ----------
    q_3d : jax.Array, shape (6, n, n, nlev)
    u_3d, v_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid
    limiter : bool

    Returns
    -------
    jax.Array : shape (6, n, n, nlev)
        ``dq/dt = -div(q · v)`` — the tracer-advection tendency.
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

    # Backward / forward differences share ``df_bwd / dz_half`` — pad
    # along the trailing axis instead of allocating two fresh
    # ``jnp.zeros`` buffers and concatenating.  Single Pad HLO op
    # each, no zero-buffer allocation.
    df = (field_full[..., :-1] - field_full[..., 1:]) / dz_half
    pad_axes = ((0, 0),) * (df.ndim - 1)
    # Backward difference (upward): zero at the surface boundary.
    grad_bwd = jnp.pad(df, (*pad_axes, (1, 0)))
    # Forward difference (downward): zero at the top boundary.
    grad_fwd = jnp.pad(df, (*pad_axes, (0, 1)))

    # Upwind: w* > 0 = upward => backward; w* < 0 = downward => forward
    grad = jnp.where(w_star > 0, grad_bwd, grad_fwd)
    return -w_star * grad
