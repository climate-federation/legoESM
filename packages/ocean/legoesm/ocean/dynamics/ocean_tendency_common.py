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

import os
from typing import Callable, Optional, Tuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import compute_hydrostatic_pressure
from legoesm.ocean.freshwater import (
    virtual_salt_flux,
    normalized_virtual_salt_flux,
)


def _baroclinic_f32_enabled(dtype) -> bool:
    """Mixed-precision opt-in (``LEGOESM_BAROCLINIC_F32=1``, f64-state-gated): pass
    ``compute_dtype=float32`` to the EOS in the baroclinic anomaly iteration so the
    EOS polynomial runs in f32 WORK while rho is RETURNED in the f64 STATE dtype and
    the downstream rho' / pressure / PGF stay f64 — the Oceananigans / NeuralGCM
    "f32 work, f64 state" pattern, mirroring the vmix ``LEGOESM_VMIX_F32_SOLVE``
    lever and SCOPED to the baroclinic EOS call ONLY (N^2 / diagnostics keep the
    policy dtype, since their full-density vertical differences are NOT anomaly-
    safe in f32).  Precision-safe here via the density ANOMALY rho'~O(1) (offline
    experiment 8520588: PGF relRMS ~8e-5, spurious |v| ~1 mm/s/day).  GPU-only
    benefit (RTX8000 f64 = 1/32 f32); CPU f32 ~flat/slowdown — OPT-IN, default
    OFF.  Requires a ``make_eos_fn``-built eos_fn (accepts ``compute_dtype``)."""
    return (dtype == jnp.float64
            and os.environ.get("LEGOESM_BAROCLINIC_F32", "0") == "1")


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
    allow_baroclinic_f32: bool = False,
    quadrature: str = "cell_integral",
    trapezoid_t_depth_1d: jnp.ndarray | None = None,
    eos_depth: str = "insitu",
    eos_geometric_depth_1d: jnp.ndarray | None = None,
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
    3. Build the layer-centred baroclinic pressure anomaly (see
       ``quadrature``):

       - ``"cell_integral"`` (legacy default, bit-identical):
         ``p'(k) = g · Σ_{j<k} ρ'(j) · dz(j) + 0.5 · g · ρ'(k) · dz(k)``
         — the half-cell cumulative integral of ``g·ρ'`` to the layer
         mid-point using the CELL value over each cell.
       - ``"nemo_trapezoid"`` — NEMO ``dynhpg`` vertical quadrature
         (dynhpg.F90 hpg_sco/zco recurrence): trapezoid between cell
         centres on the w-spacing ``e3w(k) = (dz(k)+dz(k−1))/2`` with
         surface half-cell ``e3w(1) = dz(1)``:
         ``p'(1) = (g/2)·dz(1)·ρ'(1)``,
         ``p'(k) = p'(k−1) + (g/2)·e3w(k)·(ρ'(k)+ρ'(k−1))``.
         ``ρ'`` is zeroed below the seafloor (NEMO's masked ``rhd``)
         when ``is_active_3d`` is provided.  The two rules agree on a
         UNIFORM grid; on stretched levels they differ per interface by
         ``(g/4)·(dz(k)−dz(k−1))·(ρ'(k−1)−ρ'(k))``.
         ``trapezoid_t_depth_1d`` (positive t-depths, shape (nlev,)):
         when given, the w-spacings come from the ACTUAL t-depth ladder
         — ``e3w(1) = 2·gdept(1)``, ``e3w(k) = gdept(k)−gdept(k−1)`` —
         exactly NEMO ``depth_to_e3``.  For interface-midpoint centres
         this is algebraically identical to the h-derived form; for
         analytic (mi96) centres it is the exact NEMO quadrature.

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
        ``neumann_fill_cgrid`` (lat-lon C-grid), or
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
    eos_depth : str, default ``"insitu"``
        Depth the EOS pressure term sees during the density iteration.
        ``"insitu"`` (default, BYTE-IDENTICAL): iterate ``rho <- EOS(T, S,
        p_hydro(rho))`` so the EOS depth is the in-situ hydrostatic integral
        (recovers ~(rho_bar/rho0)*gdept, a ~0.5% stretch vs geometric).
        ``"geometric"``: feed ``p_eos = rho_0*g*gdept`` from
        ``eos_geometric_depth_1d`` ONCE (no iteration — the depth no longer
        depends on rho), so the EOS reconstructs geometric depth exactly.
        Matches NEMO ``eos_insitu`` (uses geometric ``gdept`` directly).  The
        *eos_fn* must be built with the SAME ``rho_0`` (``make_eos_fn(rho0=
        rho_0)``) so the value cancels.  Only affects the density fed to the
        EOS; the p' baroclinic anomaly (the PGF) is still the rho' integral.
    eos_geometric_depth_1d : array or None
        Geometric T-depth ladder (positive down, shape ``(nlev,)``; NEMO
        ``gdept_1d``).  Required when ``eos_depth="geometric"``.

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

    # f32-EOS lever (default OFF -> byte-identical): pass compute_dtype=f32 so the
    # EOS polynomial runs in f32 WORK; rho is returned in the f64 STATE dtype
    # (T_filled is f64), so the hydrostatic iteration, rho', cumsum and PGF below
    # all stay f64 — only the polynomial is f32.  Gated by BOTH the env AND the
    # caller's explicit ``allow_baroclinic_f32`` so it fires ONLY for the true
    # baroclinic-PGF dynamics callers (where the anomaly rho'~O(1) makes f32 safe)
    # — NOT for the GM/Redi/MLE/N^2/init callers that also use this helper but
    # whose density gradients are not anomaly-safe in f32 (codex review).
    eos_kw = ({"compute_dtype": jnp.float32}
              if (allow_baroclinic_f32 and _baroclinic_f32_enabled(T.dtype))
              else {})
    if eos_depth not in ("insitu", "geometric"):
        raise ValueError(
            f"Unknown eos_depth {eos_depth!r}; expected 'insitu' or 'geometric'")
    if eos_depth == "geometric":
        # NEMO eos_insitu: feed p = rho_0*g*gdept so the EOS depth term
        # reconstructs the GEOMETRIC gdept exactly (no in-situ stretch).  The
        # depth no longer depends on rho, so a single evaluation suffices.
        if eos_geometric_depth_1d is None:
            raise ValueError(
                "eos_depth='geometric' requires eos_geometric_depth_1d "
                "(the geometric gdept ladder)")
        t_depth = jnp.asarray(eos_geometric_depth_1d, dtype=T.dtype)
        # Use constants.g (NOT the passed config g): the EOS reconstructs depth as
        # zh = p/(rho0*constants.g), so p_eos MUST use the same constants.g for the
        # g to cancel and zh to equal gdept exactly (independent of the config g).
        # This matches compute_ocean_rho's geometric path; using config.g here
        # would leave a zh = gdept*(config.g/constants.g) stretch when they differ.
        p_eos = (rho_0 * constants.g) * t_depth
        rho = eos_fn(T_filled, S_filled, p_eos, **eos_kw)
    else:
        rho = eos_fn(T_filled, S_filled, jnp.zeros_like(T), **eos_kw)
        for _ in range(n_iter):
            p_hydro = compute_hydrostatic_pressure(
                rho, eta_ref, dz_ref, J_ref, rho_0, g,
                h_actual=h_actual,
            )
            rho = eos_fn(T_filled, S_filled, p_hydro, **eos_kw)

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

    if quadrature not in ("cell_integral", "nemo_trapezoid"):
        raise ValueError(
            f"Unknown p' quadrature {quadrature!r}; expected "
            "'cell_integral' or 'nemo_trapezoid'")

    if quadrature == "nemo_trapezoid":
        # NEMO dynhpg recurrence (see docstring).  Mask ρ' below the
        # seafloor first — NEMO's rhd is masked, so a column's cumsum
        # stays constant past its own bottom (h may still be the full
        # dz_ref there on the pure-z* path).
        if hi_precision_pressure:
            rho_q = rho_prime.astype(jnp.float64)
            h_q = jnp.asarray(h_for_cumsum, dtype=jnp.float64)
        else:
            rho_q = rho_prime
            h_q = jnp.asarray(h_for_cumsum)
        if is_active_3d is not None:
            rho_q = jnp.where(is_active_3d, rho_q, jnp.zeros_like(rho_q))
        pair = rho_q[..., 1:] + rho_q[..., :-1]          # (..., nlev-1)
        if trapezoid_t_depth_1d is not None:
            t_q = jnp.asarray(trapezoid_t_depth_1d,
                              dtype=jnp.float64 if hi_precision_pressure
                              else None)
            e3w_int = t_q[1:] - t_q[:-1]                 # (nlev-1,)
            e3w_1 = 2.0 * t_q[:1]                        # NEMO depth_to_e3
            inc = jnp.concatenate(
                [e3w_1 * rho_q[..., :1],
                 jnp.broadcast_to(e3w_int, pair.shape) * pair], axis=-1)
        else:
            h_b = jnp.broadcast_to(h_q, rho_q.shape)
            e3w_int = 0.5 * (h_b[..., 1:] + h_b[..., :-1])
            inc = jnp.concatenate(
                [h_b[..., :1] * rho_q[..., :1], e3w_int * pair], axis=-1)
        p_prime = (0.5 * g) * jnp.cumsum(inc, axis=-1)
        return rho, rho_prime, p_prime

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


def ab2_blend(
    f_new: jnp.ndarray,
    f_old: jnp.ndarray,
    eps: jnp.ndarray | float,
) -> jnp.ndarray:
    """Adams-Bashforth-2 time blend ``(1.5 + eps)·f_new − (0.5 + eps)·f_old``.

    The standard AB2 extrapolation of a tendency to the half-step, with
    the Robert-Asselin-style stabilising offset ``eps`` (``config.
    ab2_epsilon``) that shifts the weights ``(3/2, −1/2)`` towards a more
    damped ``(3/2 + ε, −1/2 − ε)`` to suppress the AB2 weak instability.

    ``f_new`` is the tendency at the current step ``F^n``; ``f_old`` is
    the previous step ``F^{n−1}``.  Open-coded identically across the
    rigid-lid streamfunction update and the split / unsplit baroclinic
    tracer + momentum predictors — factored here (#517 item 8).

    The arithmetic is written EXACTLY as the call sites had it
    (``(1.5 + eps) * f_new - (0.5 + eps) * f_old``, including those that
    pre-bound ``a_n = 1.5 + eps``, ``a_p = 0.5 + eps``) so the result is
    BIT-IDENTICAL.  Pure elementwise; broadcasts over any shape; fully
    differentiable.

    Parameters
    ----------
    f_new : jax.Array
        Tendency at the current step ``F^n``.
    f_old : jax.Array
        Tendency at the previous step ``F^{n−1}``.
    eps : float or jax.Array
        AB2 stabilising offset (``config.ab2_epsilon``).

    Returns
    -------
    jax.Array
        ``(1.5 + eps)·f_new − (0.5 + eps)·f_old``.
    """
    return (1.5 + eps) * f_new - (0.5 + eps) * f_old


def column_depth(
    h_face: jnp.ndarray,
    min_water_column_m: jnp.ndarray | float,
    *,
    axis: int = -1,
    keepdims: bool = False,
) -> jnp.ndarray:
    """Floored water-column depth ``max(Σ_k h_face, min_water_column_m)``.

    The vertically-summed thickness at a velocity/face/cell location,
    floored to a positive ``min_water_column_m`` so it is safe to divide
    by (the barotropic depth-average denominator).

    NOTE on the floor: call sites historically use DIVERGENT floors —
    e.g. ``config.min_water_column_m`` (≈ a physical wet-cell minimum,
    O(0.1–1 m)) in the barotropic-mean and PE-flux paths, but a bare
    numerical ``1e-10`` in the diagnostic depth-mean paths.  This helper
    takes the floor as an explicit argument so every call site passes
    its OWN current value and the result is BIT-IDENTICAL.  Do NOT unify
    these into one floor — that would change answers.

    Parameters
    ----------
    h_face : jax.Array
        Per-level layer thickness at the location, with the vertical
        axis at ``axis`` [m].
    min_water_column_m : float or jax.Array
        Lower bound on the column depth.  ``max`` is exact (no rounding)
        so passing each site's own floor reproduces it bit-for-bit.
    axis : int, default -1
        Vertical (level) axis to sum over.  Lat-lon C-grid uses ``-1``;
        MPAS edge columns use ``1``.
    keepdims : bool, default False
        Forwarded to ``jnp.sum`` so a site that needs the summed axis
        retained for broadcasting gets a bit-identical result.

    Returns
    -------
    jax.Array
        ``max(Σ_k h_face, min_water_column_m)``.
    """
    return jnp.maximum(
        jnp.sum(h_face, axis=axis, keepdims=keepdims), min_water_column_m
    )


def depth_mean(
    field: jnp.ndarray,
    h_face: jnp.ndarray,
    min_water_column_m: jnp.ndarray | float,
    *,
    axis: int = -1,
    keepdims: bool = False,
    fused: bool = True,
) -> jnp.ndarray:
    """Thickness-weighted column mean ``Σ_k(field·h_face) / max(Σ_k h_face, floor)``.

    The barotropic / depth-averaged component of a 3-D field: the
    thickness-weighted vertical average, with the denominator floored
    via :func:`column_depth`.

    NOTE on the floor: see :func:`column_depth`.  Call sites pass
    DIVERGENT floors (``config.min_water_column_m`` vs ``1e-10``); each
    passes its OWN value here so the result is BIT-IDENTICAL.  No mask is
    applied — use :func:`depth_average_to_faces` when a face mask is
    required (it multiplies the mean by the mask afterwards, matching the
    original ``... * u_mask`` call sites).

    NOTE on ``fused`` (BYTE-IDENTITY — reduction topology):  the original
    call sites split into two families that XLA can compile to *bit-
    different* results in the full step (it reassociates the level-axis
    accumulation differently depending on the surrounding fused kernel —
    a ~1e-10 drift observed on the MPAS seamount path):

    * ``fused=True`` (default): ONE stacked column reduction
      ``jnp.sum(jnp.stack([field·h, h]), axis)`` — reproduces the sites
      that were already written with the fused stack (the
      ``ocean_model_latlon_cgrid`` barotropic split, ``barotropic_implicit``
      ``V_bar``).
    * ``fused=False``: TWO independent ``jnp.sum`` calls — reproduces the
      sites that were open-coded as separate sums (``barotropic_implicit``
      ``U_bar``, ``rigid_lid`` ``U_old``/``V_old``, the ``_split`` baroclinic
      decomposition).

    Each call site selects the flag matching its ORIGINAL form so the
    result is bit-identical on every backend — do NOT change a site's flag
    to "tidy up", it changes answers.

    Parameters
    ----------
    field : jax.Array
        3-D field to depth-average (velocity, momentum tendency, …),
        with its vertical axis at ``axis``.
    h_face : jax.Array
        Per-level layer thickness at the same location.
    min_water_column_m : float or jax.Array
        Column-depth floor passed verbatim to :func:`column_depth`.
    axis : int, default -1
        Vertical (level) axis.
    keepdims : bool, default False
        Retain the reduced axis (for broadcasting the mean back against
        the 3-D field, e.g. ``u - U_bar[..., None]`` callers that keep
        the axis).

    Returns
    -------
    jax.Array
        Thickness-weighted depth mean of ``field``.
    """
    # Fuse the numerator (Σ field·h) and denominator (Σ h) into ONE
    # stacked column reduction.  This is the exact form the call sites
    # had (``jnp.sum(jnp.stack([h, field·h], axis=-1), axis=-2)``).  It
    # MATTERS for byte-identity: splitting the fused stack into two
    # separate ``jnp.sum`` calls changes XLA's fusion in the full
    # compiled step and drifts results at ~1e-10 on some grids (observed
    # in the MPAS seamount path).  Stacking on a NEW trailing axis and
    # reducing the level axis ``axis`` reproduces the original exactly;
    # the stack ORDER does not matter (both slices reduce independently).
    if fused:
        stacked = jnp.stack([field * h_face, h_face], axis=-1)
        pair = jnp.sum(stacked, axis=axis if axis >= 0 else axis - 1,
                       keepdims=keepdims)
        num = pair[..., 0]
        denom = jnp.maximum(pair[..., 1], min_water_column_m)
        return num / denom
    # Split form: two independent reductions (matches the open-coded
    # ``jnp.sum(field·h)/max(jnp.sum(h), floor)`` sites bit-for-bit).
    num = jnp.sum(field * h_face, axis=axis, keepdims=keepdims)
    denom = column_depth(h_face, min_water_column_m, axis=axis,
                         keepdims=keepdims)
    return num / denom


def depth_average_to_faces(
    field: jnp.ndarray,
    h_face: jnp.ndarray,
    mask: jnp.ndarray,
    min_water_column_m: jnp.ndarray | float,
    *,
    axis: int = -1,
    fused: bool = True,
) -> jnp.ndarray:
    """Masked thickness-weighted depth mean ``depth_mean(field, h_face) · mask``.

    The barotropic face velocity ``U_bar`` used in the baroclinic /
    barotropic split: the thickness-weighted vertical average of an
    edge-normal velocity at a face, multiplied by the face mask so land
    faces stay zero.  Equivalent to the open-coded
    ``Σ(u·h)/max(Σh, floor) · u_mask`` that previously lived in every
    ``barotropic_*`` and ``ocean_pe_*`` / ``ocean_model_*`` split.

    NOTE on the floor: see :func:`column_depth`.  Each call site passes
    its OWN ``min_water_column_m`` (``config.min_water_column_m`` for the
    barotropic-mean paths, ``1e-10`` for the PE diagnostic paths) so the
    output is BIT-IDENTICAL.

    Parameters
    ----------
    field : jax.Array
        Edge-normal velocity with its vertical axis at ``axis``.
    h_face : jax.Array
        Per-level face thickness (same shape as ``field``).
    mask : jax.Array
        Face mask (1 = wet) with the horizontal shape of the depth mean.
    min_water_column_m : float or jax.Array
        Column-depth floor passed verbatim to :func:`column_depth`.
    axis : int, default -1
        Vertical (level) axis.
    fused : bool, default True
        Reduction topology — forwarded to :func:`depth_mean`.  ``True`` for
        the already-fused-stack sites (``ocean_model`` split, ``V_bar``);
        ``False`` for the open-coded separate-sum sites (``U_bar``,
        ``U_old``/``V_old``).  See :func:`depth_mean` for the byte-identity
        rationale.

    Returns
    -------
    jax.Array
        ``Σ_k(field·h_face) / max(Σ_k h_face, floor) · mask``.
    """
    return depth_mean(field, h_face, min_water_column_m, axis=axis,
                      fused=fused) * mask


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
        relaxation rate (1 / s), either HORIZONTAL (rank ``T.ndim - 1``:
        ``(n_lat, n_lon)`` lat-lon / ``(nCells,)`` MPAS), broadcast along
        the vertical axis via ``expand_gamma_axis`` — the original path,
        bit-identical — or FULL-RANK per-cell (rank ``T.ndim``: a
        z-varying / partial-column rate, e.g. the Veros north_atlantic
        ``rest_tscl(x, y, z)`` field, ``north_atlantic.py:102/257-261``,
        consumed as ``temp_source = maskT · rest_tscl · (t* − T)``,
        ``:346-356``), used as-is.  Any other rank raises (trace-time:
        ``ndim`` is static under JIT).
    mask : jax.Array, optional
        Ocean mask.  When provided, the relaxation tendency is
        multiplied by ``mask`` (with the same axis expansion as
        ``gamma``) so land cells stay quiescent.  When ``None`` no
        masking is applied (the caller masks downstream).
    expand_gamma_axis : int, default -1
        Axis on which to insert a singleton in a HORIZONTAL
        ``sponge.gamma`` so that it broadcasts against the (..., nlev)
        tracer arrays.  Use ``-1`` for both lat-lon C-grid (axis after
        lat/lon) and MPAS (axis after nCells).  Ignored for a full-rank
        per-cell gamma.

    Returns
    -------
    (dT_dt_new, dS_dt_new) : tuple of jax.Array
    """
    dtype = T.dtype
    gamma = sponge.gamma.astype(dtype)
    if gamma.ndim == T.ndim:
        # Full-rank per-cell rate (EXT-N1: Veros north_atlantic rest_tscl).
        gamma_b = gamma
    elif gamma.ndim == T.ndim - 1:
        gamma_b = jnp.expand_dims(gamma, expand_gamma_axis)
    else:
        raise ValueError(
            f"SpongeForcing.gamma must have rank T.ndim - 1 (horizontal, "
            f"vertically broadcast) or T.ndim (full per-cell rate); got "
            f"gamma.ndim={gamma.ndim} with T.ndim={T.ndim}.")
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
    S_ref: float | jnp.ndarray,
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

    Single-owner bottom drag (RESOLVED — was codex iter-2 finding #1)
    ----------------------------------------------------------------
    The 3D PE solvers already apply the full bottom drag ``-r·u_bot/h_bot``
    (with BBL / partial-cell handling) to ``du_dt``; its depth-mean
    ``-r·u_bot/H`` is carried into the barotropic mode through ``F_slow_u`` /
    ``F_slow_v`` and applied at every substep.  The explicit barotropic substep
    loops therefore NO LONGER multiply by this factor (it would double-count to
    an effective ``≈ 2·r/H``) — drag is owned exclusively by the 3D tendency /
    F_slow, matching the Crank-Nicolson implicit solver
    (``barotropic_implicit_*``), which always relied on ``F_slow`` alone.  This
    helper is retained for the implicit-vertical-drag use cases that legitimately
    DO want an unconditionally-stable standalone ``dU/dt = -r·U/H`` update; any
    new caller MUST confirm the drag is not already in its ``F_slow`` to avoid
    re-introducing the double count.

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

    See ``docs/ocean/experiments/density_jacobian_pgf_mpas.md`` §8a
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


# --- NEMO zdfdrg non-linear bottom drag (np_non_lin / np_loglayer) -----------
#
# Shared, grid-agnostic core for the NEMO drag-law bottom friction
# (nemo_5.0.1/src/OCE/ZDF/zdfdrg.F90).  The grid adapters average the
# bottom-cell velocity components to the tracer point, call
# :func:`nemo_effective_bottom_drag_r`, and average the resulting
# coefficient back to the velocity faces (NEMO's dynzdf 2-point average
# of ``rCdU_bot``).

BOTTOM_DRAG_SCHEMES = ("legacy", "nemo_quadratic", "nemo_loglayer")


def validate_bottom_drag_scheme(scheme: str) -> str:
    """Fn-entry guard for the nested ``bottom_drag_scheme`` literal.

    A nested dynamics-config field never reaches ``validate_strict``, so a
    typo would silently run different physics — raise here instead (see
    CLAUDE.md "Dispatch": nested scheme-Config dispatch raises too).
    """
    if scheme not in BOTTOM_DRAG_SCHEMES:
        raise ValueError(
            f"unknown bottom_drag_scheme={scheme!r}; expected one of "
            f"{BOTTOM_DRAG_SCHEMES}"
        )
    return scheme


def nemo_loglayer_cd(
    h_bot: jnp.ndarray,
    *,
    z0: float,
    cd_min: float,
    cd_max: float,
    von_karman: float,
) -> jnp.ndarray:
    """Log-layer drag coefficient ``Cd(h_bot)`` (zdfdrg.F90 np_loglayer).

    NEMO (zdf_drg_nonlin, zdfdrg.F90:176-180)::

        zzz = 0.5 * e3t(bottom)                 ! altitude above the boundary
        zcd = ( vkarmn / LOG( zzz / z0 ) )**2
        zcd = MIN( MAX( rn_Cd0, zcd ), rn_Cdmax )   ! rn_Cd0 <= Cd <= rn_Cdmax

    i.e. the drag coefficient of a logarithmic boundary layer whose
    velocity point sits half a bottom-cell above the seafloor, clipped
    below by ``rn_Cd0`` (the quadratic-case coefficient acting as the
    smooth-wall minimum) and above by ``rn_Cdmax``.

    Numerical guards (F90-EXACT wherever the raw expression is finite —
    codex r1 #4): NEMO evaluates the raw ``LOG`` and relies on the clip;
    this port only (a) clamps the ``log = 0`` pole at ``½h = z0`` (raw
    Cd → +inf there, which the clip already maps to ``cd_max`` — the
    clamp reproduces exactly that limit), and (b) floors ``log`` at −30
    (h below ~1e-10 m, i.e. only dry/masked cells) so reverse-mode AD
    through masked columns stays finite.  For every representable wet
    thickness the returned value equals ``clip((κ/ln(½h/z0))², cd0,
    cdmax)`` bit-for-bit.

    Parameters
    ----------
    h_bot : array
        Bottom-cell thickness at the tracer point [m].
    z0 : float
        Bottom roughness length [m] (NEMO ``rn_z0``, ORCA1: 3e-3).
    cd_min, cd_max : float
        Clip bounds [-] (NEMO ``rn_Cd0`` = 1e-3, ``rn_Cdmax`` = 0.1).
    von_karman : float
        Von Kármán constant (``legoesm.constants.kappa_von_karman``).
    """
    ln_raw = jnp.log(jnp.maximum(0.5 * h_bot, jnp.exp(-30.0) * z0) / z0)
    # log = 0 pole (½h = z0): raw Cd → +inf → the clip maps it to cd_max;
    # substitute a tiny magnitude of the SAME sign structure so the
    # division reproduces that limit without inf/NaN (AD-safe).
    ln_safe = jnp.where(jnp.abs(ln_raw) < 1.0e-12, 1.0e-12, ln_raw)
    cd = (von_karman / ln_safe) ** 2
    return jnp.clip(cd, cd_min, cd_max)


def nemo_effective_bottom_drag_r(
    u_bot: jnp.ndarray,
    v_bot: jnp.ndarray,
    h_bot: jnp.ndarray,
    *,
    scheme: str,
    cd0: float,
    cd_max: float,
    z0: float,
    ke0: float,
    von_karman: float,
) -> jnp.ndarray:
    """NEMO non-linear bottom-drag coefficient ``r = Cd·|U|`` at tracer points.

    Transliterates ``zdf_drg_nonlin`` (zdfdrg.F90:171-191)::

        ! np_loglayer:  zcd = clip( (vkarmn/LOG(0.5*e3t/z0))**2, Cd0, Cdmax )
        ! np_non_lin:   zcd = Cd0
        pCdU = - zcd * SQRT( 0.25*(zut**2 + zvt**2) + rn_ke0 )

    where ``zut/zvt`` are 2x the tracer-point velocity components, so
    ``0.25*(zut² + zvt²) = ū² + v̄²`` — the FULL speed at the tracer
    point, with the background tidal kinetic energy ``rn_ke0`` [m²/s²]
    combined in quadrature (NOT a max-floor: the NEMO form keeps the
    drag quadratic through |U| → 0 with a smooth ``√ke0`` minimum
    speed).

    Sign convention: NEMO stores ``rCdU <= 0`` and applies it as an
    implicit friction; legoESM's drag machinery uses a POSITIVE linear
    coefficient ``r`` [m/s] with the minus sign applied in the tendency
    (``du/dt = -r·u/h``).  This helper therefore returns ``+Cd·|U|``
    (= ``-pCdU``), always >= 0.

    Time-discretization note (deliberate deviation): ORCA1 runs
    ``ln_drgimp = .true.`` — NEMO folds ``rCdU`` into the BACKWARD-EULER
    vertical momentum matrix (dynzdf.F90).  legoESM applies the SAME
    coefficient through its established explicit-tendency + ``F_slow``
    path (single-owner drag doctrine, shared with every legacy scheme).
    The discretization difference is O(dt·r/h) per step — ~5e-6 at OMIP
    dt = 3600 s, r ~ 3e-4 m/s, h_bot ~ 200 m — and unconditionally
    stable at those scales; the fidelity content of the port is the
    COEFFICIENT.  An implicit placement would restructure the vertical
    solve's bottom BC for all drag schemes and is tracked as follow-up.

    Parameters
    ----------
    u_bot, v_bot : array
        Bottom-cell velocity components AVERAGED TO THE TRACER POINT
        [m/s] (the caller owns the grid-specific averaging).
    h_bot : array
        Bottom-cell thickness at the tracer point [m] (used by
        ``nemo_loglayer`` only).
    scheme : str
        ``"nemo_quadratic"`` (zdfdrg np_non_lin) or ``"nemo_loglayer"``
        (np_loglayer).  ``"legacy"`` is rejected — callers keep the
        historical MOM6-style path for it and must not route here.
    cd0, cd_max, z0, ke0 : float
        NEMO ``rn_Cd0``, ``rn_Cdmax``, ``rn_z0``, ``rn_ke0`` (ORCA1:
        1e-3, 0.1, 3e-3, 2.5e-3).
    von_karman : float
        Von Kármán constant.

    Returns
    -------
    array, shape of ``u_bot``
        ``r = Cd·√(ū² + v̄² + ke0)`` [m/s], >= 0, at tracer points.
    """
    return nemo_drag_r_from_speed_sq(
        u_bot * u_bot + v_bot * v_bot, h_bot,
        scheme=scheme, cd0=cd0, cd_max=cd_max, z0=z0, ke0=ke0,
        von_karman=von_karman,
    )


def nemo_drag_r_from_speed_sq(
    speed_sq: jnp.ndarray,
    h_bot: jnp.ndarray,
    *,
    scheme: str,
    cd0: float,
    cd_max: float,
    z0: float,
    ke0: float,
    von_karman: float,
) -> jnp.ndarray:
    """Speed-squared form of :func:`nemo_effective_bottom_drag_r`.

    ``r = Cd · √(speed_sq + ke0)`` — the entry point for grids whose
    natural bottom-speed diagnostic is already a squared speed (MPAS
    Voronoi: 2x the Ringler discrete kinetic energy at cell centres)
    rather than separate velocity components.  Same Cd selection and
    sign convention as the component form (which delegates here).
    """
    validate_bottom_drag_scheme(scheme)
    if scheme == "nemo_loglayer":
        cd = nemo_loglayer_cd(
            h_bot, z0=z0, cd_min=cd0, cd_max=cd_max, von_karman=von_karman,
        )
    elif scheme == "nemo_quadratic":
        cd = cd0
    else:
        raise ValueError(
            "nemo_drag_r_from_speed_sq handles the NEMO drag laws only; "
            f"got scheme={scheme!r} (the 'legacy' path stays in the grid "
            "adapters)."
        )
    return cd * jnp.sqrt(speed_sq + ke0)


def masked_background_vmix_coefficient(
    background: jnp.ndarray | float,
    bottom_level: jnp.ndarray,
    n_half: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-column background vertical-mixing coefficient zeroed below seafloor.

    Builds the constant background diffusivity / viscosity floor on the
    HALF-LEVELS (interfaces) of an MPAS partial-cell column and zeros it at
    every interface that lies below the deepest active full level.  Interface
    ``k`` couples full levels ``k`` and ``k + 1``; it is *active* only when
    ``k < bottom_level`` (so both coupled cells are at or above the seafloor).
    Interfaces at/below the seafloor must carry EXACTLY zero coefficient, or
    the backward-Euler tridiagonal solver — fed the floored ``dz = 1e-10`` of
    a dry cell — sees ``dt·K/dz² ~ 1e20`` coefficients, goes singular, and
    produces NaN (see ``ocean_model_mpas`` §2a/§3a).

    This is the genuinely-shared composition between the MPAS tracer (per
    cell, ``bottom_level = z_coord.bottom_level``) and momentum (per edge,
    ``bottom_level = compute_max_level_edge_bot(...)``) implicit-mixing
    coefficient builds — the only byte-identical-mergeable sub-part of the
    two assemblies (the surrounding KPP/convection composition and the
    lat-lon C-grid analogue differ structurally; see the #517-item-6 report).

    Parameters
    ----------
    background : float or jax.Array
        Scalar background coefficient (``config.K_v`` for tracers,
        ``config.A_v`` for momentum) [m²/s].
    bottom_level : jax.Array, shape (n_col,)
        Index of the deepest active full level per column.
    n_half : int
        Number of half-levels (interfaces), ``= nlev - 1``.

    Returns
    -------
    coeff : jax.Array, shape (n_col, n_half)
        ``background`` on active interfaces, ``0.0`` below the seafloor.
    active_half : jax.Array (bool), shape (n_col, n_half)
        The active-interface mask ``k < bottom_level`` — returned so the
        caller can reuse it for the convection / KPP profile masking with
        the SAME seafloor definition (cast to the working dtype as needed).
    """
    k_half = jnp.arange(n_half, dtype=jnp.int32)
    active_half = k_half[None, :] < bottom_level[:, None].astype(jnp.int32)
    coeff = jnp.where(active_half, background, 0.0)
    return coeff, active_half
