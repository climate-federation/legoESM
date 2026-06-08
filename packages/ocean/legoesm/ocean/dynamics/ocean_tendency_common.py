"""Grid-agnostic baroclinic tendency helpers shared by ocean_pe_*.py files.

This module factors logic that was previously duplicated across the
A-grid (``ocean_pe_cdgrid.py``), lat-lon C-grid
(``ocean_pe_latlon_cgrid.py``), and MPAS Voronoi (``ocean_pe_mpas.py``)
baroclinic tendency entry points.  Grid-specific operators (gradient,
divergence, vorticity, fill, ...) are passed in as callables so this
module never imports from a particular grid package.

Closes #214 (Phase 1).

Functions
---------
``iterate_eos_and_pressure_anomaly``
    EOS iteration (T_filled, S_filled, p) → ρ, ρ', p' using a
    reference-thickness hydrostatic integral.  The cubed-sphere caller
    can request high-precision arithmetic for the cumsum so that the
    halo-exchanged p' gradient stays clean.

``apply_sponge_tracer_relaxation``
    Linear restoring of T, S towards reference fields with rate
    ``γ``.  Used identically by the lat-lon C-grid and MPAS callers.

``apply_freshwater_virtual_salt_top``
    Top-layer salinity tendency from the freshwater volume flux.

``implicit_bottom_drag_factor``
    Returns ``1 - dt * r / max(H, eps)`` — the per-substep multiplicative
    factor used by both C-grid lat-lon and MPAS barotropic substeps.

These helpers are pure and pytree-friendly: they accept and return
``jax.Array`` values and never mutate inputs.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax.numpy as jnp

from legoesm.ocean.eos import compute_hydrostatic_pressure
from legoesm.ocean.freshwater import (
    virtual_salt_flux,
    normalized_virtual_salt_flux,
)


def iterate_eos_and_pressure_anomaly(
    T: jnp.ndarray,
    S: jnp.ndarray,
    mask: jnp.ndarray,
    fill_fn: Callable[[jnp.ndarray], jnp.ndarray],
    eos_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    dz_ref: jnp.ndarray,
    rho_0: float,
    g: float,
    *,
    n_iter: int = 2,
    hi_precision_pressure: bool = False,
    h_actual: jnp.ndarray | None = None,
    use_depth_dependent_ref: bool = False,
    is_active_3d: jnp.ndarray | None = None,
    rho_ref_z_static: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the standard 2-pass EOS iteration and form ``p_prime``.

    Replicates the identical iteration that previously lived inline in
    every ``ocean_pe_*.py`` file:

    1. Fill land-cell ``T``, ``S`` with ocean-neighbour values via
       ``fill_fn`` so the EOS does not produce spurious ``ρ'`` values
       on land that contaminate the gradient at coastlines.
    2. Iterate ``ρ ← EOS(T, S, p_hydro(ρ))`` ``n_iter`` times against
       the **reference** thickness profile ``dz_ref`` (i.e. ``J=1``,
       ``η=0``).  Using the actual Jacobian here would double-count
       the ``-g·∇η`` forcing handled by the barotropic solver.
    3. Build the layer-centred baroclinic pressure anomaly

       ``p'(k) = g · Σ_{j<k} ρ'(j) · dz_ref(j) + 0.5 · g · ρ'(k) · dz_ref(k)``

       which equals the half-trapezoidal cumulative integral of
       ``g·ρ'`` from the surface to the layer mid-point.

    Parameters
    ----------
    T, S : jax.Array
        Tracer fields with a trailing vertical axis (``..., nlev``).  All
        upstream callers store T and S with identical shapes.
    mask : jax.Array
        Land mask with the same horizontal shape as ``T[..., 0]`` (used
        only by ``fill_fn``; passed back to the caller for any post-
        processing it needs).
    fill_fn : Callable[[jax.Array], jax.Array]
        Grid-specific land-cell filler.  Must accept and return arrays
        with the same shape as ``T``.  Typical implementations:
        ``jax.vmap(fill_land_cells_cubed_sphere, in_axes=-1)``,
        ``_neumann_fill_cgrid`` (lat-lon C-grid), or
        ``fill_land_cells_mpas`` (MPAS).
    eos_fn : Callable
        Equation of state ``(T, S, p) → ρ``.  Same signature used by
        every grid (built via ``make_eos_fn``).
    dz_ref : jax.Array
        Reference layer thickness (``z_coord.dz_ref``), shape
        ``(nlev,)``.
    rho_0, g : float
    n_iter : int, default 2
        Number of EOS iterations *before* the final pressure update.
        All current callers use 2.
    hi_precision_pressure : bool, default False
        If True, perform the pressure cumsum in float64 so that
        halo-exchange interpolation errors do not contaminate the
        downstream gradient.  Used by the cubed-sphere C-D path; on
        lat-lon and MPAS the compact 2-cell stencils are well-behaved
        enough that the working precision is sufficient.

    Returns
    -------
    rho : jax.Array
        In-situ density (same shape as ``T``).
    rho_prime : jax.Array
        ``rho - rho_0`` (same shape as ``T``).
    p_prime : jax.Array
        Baroclinic pressure anomaly (same shape as ``T``).  Returned in
        whatever precision was used for the cumulative sum.
    """
    del mask  # currently unused (passed to fill_fn by the caller); kept
              # in signature for clarity at call sites.

    T_filled = fill_fn(T)
    S_filled = fill_fn(S)

    # Reference Jacobian (J=1, η=0).  Both have the horizontal shape of
    # T (i.e. no vertical axis).  Match dtype to the working state so we
    # never accidentally promote the EOS iteration to float64.
    horiz_shape = T.shape[:-1]
    J_ref = jnp.ones(horiz_shape, dtype=T.dtype)
    eta_ref = jnp.zeros(horiz_shape, dtype=T.dtype)

    rho = eos_fn(T_filled, S_filled, jnp.zeros_like(T))
    for _ in range(n_iter):
        p_hydro = compute_hydrostatic_pressure(
            rho, eta_ref, dz_ref, J_ref, rho_0, g,
            h_actual=h_actual,
        )
        rho = eos_fn(T_filled, S_filled, p_hydro)

    if rho_ref_z_static is not None:
        # STATIC reference profile (preferred): a frozen-at-init
        # ρ_ref(z) computed from the initial T, S over wet cells.
        # ``ρ' = ρ − ρ_ref_z_static`` cuts the partial-cell PGF residual
        # (the seed of the bottom-trapped rotational mode) by ~24× in
        # offline probes without the positive-feedback drift that broke
        # the dynamic recomputed-mean version (the dynamic version
        # NaN'd at day 60 because ρ_ref_z chases T,S drift; see
        # project_mpas_etopo_instability.md §8c).  Broadcasts on the
        # trailing axis.
        rho_ref_static = jnp.asarray(rho_ref_z_static, dtype=rho.dtype)
        rho_prime = rho - rho_ref_static
    elif use_depth_dependent_ref:
        # DYNAMIC reference profile (legacy / discouraged): recompute
        # the wet-cell mean every call.  Subtracts horizontally-uniform
        # per-level ρ_ref(z) = mean over wet cells at each level.  In
        # principle reduces the PGF residual; in practice creates a
        # positive feedback as T,S drift over a long run, leading to
        # NaN around day 60 on ETOPO + ico-4.  Kept for back-compat
        # comparison only.
        if is_active_3d is not None:
            wet = is_active_3d.astype(rho.dtype)
        else:
            wet = jnp.broadcast_to(
                mask[..., None].astype(rho.dtype), rho.shape,
            )
        # Reduce over all axes except the trailing vertical one.
        horiz_axes = tuple(range(rho.ndim - 1))
        wet_count = jnp.maximum(jnp.sum(wet, axis=horiz_axes), 1.0)
        rho_ref_z = jnp.sum(rho * wet, axis=horiz_axes) / wet_count
        # Broadcast back across horizontal axes.
        rho_prime = rho - rho_ref_z
    else:
        rho_prime = rho - rho_0

    # Layer-thickness array used in the baroclinic-anomaly cumsum.
    # When ``h_actual`` is None, use the reference ``dz_ref`` (legacy z*
    # path).  When provided (partial-cell path), use per-cell thickness
    # so the anomaly integrates to each cell's actual centroid depth.
    # Cells below the seafloor have h_actual=0 and contribute zero.
    if h_actual is None:
        h_for_cumsum = dz_ref
    else:
        h_for_cumsum = h_actual

    if hi_precision_pressure:
        rho_prime_hi = rho_prime.astype(jnp.float64)
        h_hi = jnp.asarray(h_for_cumsum, dtype=jnp.float64)
        dp_layer = rho_prime_hi * g * h_hi
    else:
        dp_layer = rho_prime * g * h_for_cumsum

    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer

    return rho, rho_prime, p_prime


def compute_static_rho_ref_z(
    T: jnp.ndarray,
    S: jnp.ndarray,
    mask: jnp.ndarray,
    fill_fn: Callable[[jnp.ndarray], jnp.ndarray],
    eos_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    dz_ref: jnp.ndarray,
    rho_0: float,
    g: float,
    *,
    n_iter: int = 2,
    h_actual: jnp.ndarray | None = None,
    is_active_3d: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute a frozen horizontally-uniform reference density profile.

    Runs the same 2-pass EOS+hydrostatic-pressure iteration that
    :func:`iterate_eos_and_pressure_anomaly` uses, then averages the
    resulting in-situ density over wet cells at each level to produce a
    ``(nlev,)`` profile.  Used at init time on ``T_init``, ``S_init`` to
    build a STATIC ``ρ_ref(z)`` that is then passed to every call of
    :func:`iterate_eos_and_pressure_anomaly` via the
    ``rho_ref_z_static`` argument.

    Why a separate helper rather than reusing the dynamic path: the
    runtime path mixes two distinct concerns — computing ``ρ`` and
    forming ``ρ' = ρ − ρ_ref``.  At init we want only the first plus the
    horizontal mean.  Reusing the same EOS iteration here guarantees
    that ``ρ_ref_z(T_init, S_init)`` is identical to what the dynamic
    path would have computed on the initial state, so the static and
    dynamic versions agree at ``t = 0`` and any divergence later is
    purely from the recomputed-mean drift in the dynamic path.

    Parameters mirror :func:`iterate_eos_and_pressure_anomaly`.
    ``is_active_3d`` (when provided, e.g. by the partial-cell path)
    takes precedence over the 2-D ``mask``: cells flagged inactive by
    the partial-cell coordinate are excluded from the per-level mean
    even when their column-mask says ``ocean``.

    Returns
    -------
    rho_ref_z : jax.Array
        Shape ``(nlev,)`` (always, regardless of T's horizontal shape).
        Wet-cell mean of ``ρ`` after the 2-pass iteration.  Cast to
        ``T.dtype``.
    """
    T_filled = fill_fn(T)
    S_filled = fill_fn(S)

    horiz_shape = T.shape[:-1]
    J_ref = jnp.ones(horiz_shape, dtype=T.dtype)
    eta_ref = jnp.zeros(horiz_shape, dtype=T.dtype)

    rho = eos_fn(T_filled, S_filled, jnp.zeros_like(T))
    for _ in range(n_iter):
        p_hydro = compute_hydrostatic_pressure(
            rho, eta_ref, dz_ref, J_ref, rho_0, g,
            h_actual=h_actual,
        )
        rho = eos_fn(T_filled, S_filled, p_hydro)

    if is_active_3d is not None:
        wet = is_active_3d.astype(rho.dtype)
    else:
        wet = jnp.broadcast_to(
            mask[..., None].astype(rho.dtype), rho.shape,
        )
    horiz_axes = tuple(range(rho.ndim - 1))
    wet_count = jnp.maximum(jnp.sum(wet, axis=horiz_axes), 1.0)
    rho_ref_z = jnp.sum(rho * wet, axis=horiz_axes) / wet_count
    return rho_ref_z.astype(T.dtype)


def apply_sponge_tracer_relaxation(
    dT_dt: jnp.ndarray,
    dS_dt: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    sponge,
    mask: Optional[jnp.ndarray] = None,
    *,
    expand_gamma_axis: int = -1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply tracer sponge relaxation ``+γ·(ref - q)``.

    Casts ``sponge`` arrays to ``T.dtype`` so the precision policy that
    holds the state in float32 is not silently promoted to float64
    (which previously crashed the barotropic scan; see latlon C-grid).

    Parameters
    ----------
    dT_dt, dS_dt : jax.Array
        Existing tendency arrays (modified by addition).
    T, S : jax.Array
        Current tracer state.
    sponge : SpongeForcing
        Must expose ``gamma``, ``T_ref``, ``S_ref``.  ``gamma`` is the
        relaxation rate per cell (1 / s), broadcast along the vertical
        axis via ``expand_gamma_axis``.
    mask : jax.Array, optional
        Ocean mask.  When provided, the relaxation tendency is
        multiplied by ``mask`` (with the same axis expansion as
        ``gamma``) so land cells stay quiescent.  When ``None`` no
        masking is applied (the caller masks downstream).
    expand_gamma_axis : int, default -1
        Axis on which to insert a singleton in ``sponge.gamma`` so that
        it broadcasts against the (..., nlev) tracer arrays.  Use ``-1``
        for both lat-lon C-grid (axis after lat/lon) and MPAS (axis
        after nCells).

    Returns
    -------
    (dT_dt_new, dS_dt_new) : tuple of jax.Array
    """
    dtype = T.dtype
    gamma = sponge.gamma.astype(dtype)
    gamma_b = jnp.expand_dims(gamma, expand_gamma_axis)
    dT = gamma_b * (sponge.T_ref.astype(dtype) - T)
    dS = gamma_b * (sponge.S_ref.astype(dtype) - S)
    if mask is not None:
        mask_b = jnp.expand_dims(mask, expand_gamma_axis)
        dT = dT * mask_b
        dS = dS * mask_b
    return dT_dt + dT, dS_dt + dS


def apply_freshwater_virtual_salt_top(
    dS_dt: jnp.ndarray,
    freshwater,
    S_ref: float,
    h_top: jnp.ndarray,
    rho_0: float,
    mask: jnp.ndarray,
    *,
    area: jnp.ndarray | None = None,
    normalize: bool = False,
) -> jnp.ndarray:
    """Add the surface virtual-salt tendency to the top tracer level.

    Wraps ``freshwater.virtual_salt_flux`` and assigns the result to
    ``dS_dt[..., 0]`` (or the equivalent leading-axis slice for MPAS).
    All callers operate on cell-centred salinity, so the trailing
    ``nlev`` axis is the vertical axis.

    Parameters
    ----------
    dS_dt : jax.Array
        Salinity tendency, modified at index ``[..., 0]``.
    freshwater : FreshwaterForcing
    S_ref : float
        Reference salinity used for the virtual-flux closure.
    h_top : jax.Array
        Top-layer thickness with the same horizontal shape as ``mask``.
    rho_0 : float
    mask : jax.Array
        Ocean mask (1 = ocean) with the same horizontal shape as the
        leading axes of ``dS_dt``.
    area : jax.Array, optional
        Cell area (same shape as ``mask``).  REQUIRED when ``normalize``.
    normalize : bool
        When True, remove the ocean-area-weighted global mean of the net
        freshwater flux BEFORE the virtual-salt closure so the surface flux
        conserves GLOBAL SALT (the OMIP global freshwater correction; otherwise
        an unbalanced ∮(P-E+R) drifts the mean salinity).  Mirrors the free
        surface (eta) normalization so volume and salt stay consistent.

    Returns
    -------
    jax.Array
        ``dS_dt`` with the virtual-salt flux added to the top layer.
    """
    if normalize:
        if area is None:
            raise ValueError(
                "apply_freshwater_virtual_salt_top: normalize=True requires `area`")
        dS_top = normalized_virtual_salt_flux(
            freshwater, S_ref, h_top, rho_0, area, mask)
    else:
        dS_top = virtual_salt_flux(freshwater, S_ref, h_top, rho_0)
    # Cast the freshwater contribution to dS_dt's dtype so the scatter
    # add does not silently widen on x64 mode (the freshwater struct
    # is built at JAX-default precision in init helpers, which can be
    # f64 while the salinity tendency runs at the storage policy's
    # f32).
    return dS_dt.at[..., 0].add((dS_top * mask).astype(dS_dt.dtype))


def implicit_bottom_drag_factor(
    dt: jnp.ndarray | float,
    drag_r: jnp.ndarray | float,
    H: jnp.ndarray,
    *,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """Per-substep bottom-drag multiplier ``1 / (1 + dt·r / max(H, eps))``.

    Backward-Euler implicit form that, *as a standalone update* of
    ``dU/dt = -r·U/H``, solves ``U_new = U_old - dt·r·U_new / H`` for
    ``U_new / U_old``.  The result is in ``(0, 1]`` for any positive
    ``dt``, ``r``, ``H`` — unconditionally stable, never flips velocity
    sign.  Equivalent to the explicit form ``1 - dt·r/H`` to first
    order; finite and bounded for arbitrary ``dt·r/H`` (the explicit
    form would diverge for ``dt·r/H > 2``, a hazard in shallow-shelf
    and inundation configurations).

    Known limitation — combined drag application
    --------------------------------------------
    The explicit barotropic substep loops in this codebase apply this
    factor *in addition to* a depth-mean bottom drag carried by
    ``F_slow_u`` / ``F_slow_v`` (the depth-average of the 3D PE solver's
    ``du_dt``, which already contains a bottom-cell drag of magnitude
    ``-r·u_bot / dz_bot`` whose depth-average is ``-r·u_bot / H``).
    The Crank-Nicolson implicit barotropic solver
    (``barotropic_implicit_*``) intentionally relies on ``F_slow``
    alone and does *not* apply this factor.  Effective barotropic-mode
    drag in the explicit path is therefore ``≈ 2·r/H`` rather than
    ``r/H`` (codex adversarial review iter-2 finding #1).  Resolving
    this requires single-owner drag plumbing: either subtract the
    depth-mean bottom drag from ``F_slow_u`` before the barotropic
    substep, or remove the bottom-drag contribution from the 3D
    solver's ``du_dt`` for the barotropic-explicit path.  Tracked as
    open architectural debt; do not silently change call-site
    semantics without a paired update to ``F_slow`` construction in
    ``ocean_model_*.py``.

    Parameters
    ----------
    dt : float or jax.Array
        Substep size [s].
    drag_r : float or jax.Array
        Linear drag coefficient [m/s].
    H : jax.Array
        Total water column depth at the velocity location [m].
    eps : float
        Floor on ``H`` for numerical safety.

    Returns
    -------
    jax.Array, same shape as ``H``.
    """
    return 1.0 / (1.0 + dt * drag_r / jnp.maximum(H, eps))


def bbl_distributed_drag_face_column(
    u_field: jnp.ndarray,
    h_face: jnp.ndarray,
    drag_r: float,
    H_BBL: float,
    *,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """Killworth & Edwards (1999) / MOM6 ``BBL_thick_min`` distributed
    bottom drag, per face-column.

    On a partial-cell coordinate the deepest active cell can be O(1) m
    thick.  Applying a linear drag of the form ``-r · u / h`` to that
    single cell makes the bottom-cell drag tendency 100× the deep-ocean
    value, blows up the explicit-CFL criterion ``r·dt < h``, and (per
    the 2026-05-03 MPAS+ETOPO diagnostic) sign-reverses ``u`` for any
    realistic ``dt``.  This helper instead spreads the same total drag
    stress over a fixed Ekman-thickness BBL ``H_BBL`` near the
    seafloor, distributing the drag tendency across whichever cells
    overlap the BBL band.

    Algorithm (per face column, vectorised over all face indices):

    1. Build interface depths from ``cumsum(h_face)`` along the level
       axis, with ``z = 0`` at the surface and depths *negative-downward*
       (lat-lon convention; sign cancels in the ``min/max`` below).
    2. Define the BBL band as ``[z_seafloor, z_seafloor + H_BBL]``.
    3. For each cell ``k``, compute its overlap with the BBL band:
       ``overlap_k = max(0, min(z_top_k, bbl_top) − max(z_bot_k,
       z_seafloor))``.
    4. Distribute drag stress proportionally:
       ``dudt_k = −r · u_k · overlap_k / (h_k · H_BBL)``.

    Limits:

    - ``h_bot ≥ H_BBL`` (deep ocean): bottom-cell overlap = ``H_BBL``,
      cell drag = ``−r·u/h_bot`` (recovers legacy single-cell form).
      Cells above the bottom: zero overlap, zero drag contribution.
    - ``h_bot < H_BBL`` (thin partial cell): bottom-cell drag =
      ``−r·u/H_BBL`` (much weaker than ``−r·u/h_bot``; bounded by the
      BBL thickness so explicit-CFL is dt-stable for any reasonable
      ``r·dt < H_BBL``).  Cells above absorb the rest of the BBL band
      with overlap-weighted drag.

    The function is grid-agnostic — it operates per-face-column on
    arrays of shape ``(face_dim..., nlev)``.  Same logic used by both
    the lat-lon C-grid and MPAS Voronoi PE entries.

    Parameters
    ----------
    u_field : array, shape (face_dim..., nlev)
        Edge-normal velocity at the face for each level [m/s].
    h_face : array, shape (face_dim..., nlev)
        Per-level layer thickness at the face [m].  Inactive cells
        have ``h = 0`` and contribute zero overlap.
    drag_r : float
        Linear drag coefficient [m/s].
    H_BBL : float
        BBL thickness [m] (typical ocean: 10–100 m; ``MOM6 BBL_thick_min``
        defaults to 10 m).  Must be > 0.
    eps : float
        Safety floor on ``h_face`` to keep the divide finite at fully
        dry cells; ``h = 0`` inactive cells get ``overlap = 0`` so the
        choice of ``eps`` does not affect the answer.

    Returns
    -------
    array, shape (face_dim..., nlev)
        Drag tendency [m/s²] at each face-level.

    References
    ----------
    Killworth & Edwards (1999), JPO 29, 1221–1238.
    MOM6 ``BBL_thick_min`` (Adcroft et al. 2019, JAMES).

    See ``docs/ocean_experiments/density_jacobian_pgf_mpas.md`` §8a
    for the MPAS+ETOPO diagnostic that motivated the cross-grid port.
    """
    pad_axes = ((0, 0),) * (h_face.ndim - 1)  # noqa: F841 (parity with lat-lon)
    z_half = jnp.concatenate(
        [
            jnp.zeros(h_face.shape[:-1] + (1,), dtype=h_face.dtype),
            -jnp.cumsum(h_face, axis=-1),
        ],
        axis=-1,
    )
    z_top = z_half[..., :-1]                          # (..., nlev)
    z_bot = z_half[..., 1:]                           # (..., nlev)
    z_seafloor = z_half[..., -1:]                     # (..., 1)
    bbl_top = z_seafloor + H_BBL
    overlap = jnp.maximum(
        0.0,
        jnp.minimum(z_top, bbl_top) - jnp.maximum(z_bot, z_seafloor),
    )
    h_safe = jnp.maximum(h_face, eps)
    # Effective BBL thickness for the per-layer rate: when the total
    # wet column is shallower than ``H_BBL`` the bottom-boundary layer
    # cannot extend to its full nominal thickness, so divide by the
    # ACTUAL total overlap (which equals the wet depth in that case)
    # rather than the nominal ``H_BBL``.  Otherwise shelf/coastal
    # cells with total depth < H_BBL see a BBL drag rate that is too
    # weak by ``total_wet_depth / H_BBL`` (codex iter-39 #2).
    total_overlap = jnp.sum(overlap, axis=-1, keepdims=True)
    h_bbl_eff = jnp.minimum(jnp.maximum(total_overlap, eps), H_BBL)
    return -drag_r * u_field * overlap / (h_safe * h_bbl_eff)
