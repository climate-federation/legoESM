"""Backward-Euler implicit vertical diffusion for the ocean.

Solves the 1-D diffusion equation per column

    ∂φ/∂t = ∂/∂z [K · ∂φ/∂z]

for each grid column using a backward-Euler discretisation that is
unconditionally stable.  This is the key ingredient for issue #204:
without it, the ocean dycore must keep first-order-upwind vertical
momentum advection (whose numerical viscosity ~|w|·dz/2 silently damps
baroclinic shear); with it, the physical ``A_v`` / ``K_v`` and any
Richardson-number or KPP-based enhancement can do that job directly
and the resolved advection can be upgraded to higher-order schemes.

Discrete form (no-flux top + bottom):

    ρ · ∂φ/∂t = ∂/∂z ( K ∂φ/∂z )       →     backward-Euler
    (1 + α_k + β_k) φ^{n+1}_k
        − α_k φ^{n+1}_{k-1}
        − β_k φ^{n+1}_{k+1}
      = φ^{n}_k
with
    α_k = dt · K_{k-1/2} / ( dz_k · dz_half_{k-1/2} )
    β_k = dt · K_{k+1/2} / ( dz_k · dz_half_{k+1/2} )

``K`` lives on interfaces (``nlev-1`` values per column), ``dz`` on
layer centres (``nlev``), ``dz_half`` on interfaces (``nlev-1``).
No-flux boundaries are enforced by setting ``K_{-1/2} = K_{N-1/2} = 0``
(implicit via α_0 = 0 and β_{N-1} = 0).

References
----------
- Thomas, L.H. (1949) — the Thomas algorithm is already implemented in
  :mod:`legoesm.timestepping.tridiagonal`; this module just builds the
  per-column system.
- MOM6 Technical Manual §7 (implicit vertical viscosity).
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from legoesm.timestepping.tridiagonal import (
    thomas_solve,
    thomas_solve_batched,
    thomas_solve_shared,
)

__physics_contract__ = {
    "summary": (
        "Backward-Euler implicit vertical diffusion of a generic column field: "
        "(1 - dt*d_z K d_z) phi^{n+1} = phi^n solved with the Thomas algorithm, "
        "with zero-flux top and bottom boundaries (unconditionally stable)."
    ),
    "inputs": {
        "field": "generic phi (m/s for u,v; degC for T; psu for S)",
        "K": "m^2/s", "dz": "m", "dz_half": "m", "dt": "s",
    },
    "outputs": {"field_new": "same units as input phi"},
    "sign_convention": (
        "K >= 0; z positive up; zero-flux top and bottom BC so the dz-weighted "
        "column integral of phi is invariant (the optional f32 work-precision "
        "path adds an explicit column-mean correction to keep it exact); "
        "surface/bottom fluxes are applied externally, not here; single-level "
        "columns are a no-op."
    ),
    # Generic-phi conservative solver: the same operator conserves the column
    # integral of heat (energy) for T, salt for S, and momentum for u/v.
    "conserves": ["tracer"],
    "differentiable": True,
    "reference": (
        "Backward-Euler tridiagonal (Thomas 1949) implicit vertical diffusion; "
        "MOM6 Technical Manual sec. 7 (implicit vertical viscosity)"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_implicit_solver.py — a step profile relaxes "
        "toward the column mean with the dz-weighted column integral conserved "
        "to machine precision; a single-level column returns unchanged."
    ),
}

_EPS = float(jnp.finfo(jnp.float32).eps)  # ~1.19e-7


def _vmix_f32_solve_enabled(dtype) -> bool:
    """Mixed-precision opt-in (``LEGOESM_VMIX_F32_SOLVE=1``): run the
    backward-Euler vertical-diffusion Thomas solve in f32 WORK precision while
    keeping the f64 STATE — the codex / Oceananigans / NeuralGCM "f32 work,
    f64 state/reductions" pattern.  The implicit ``(1 - dt·∂zK∂z)`` matrix is
    diagonally dominant; its 2-norm condition is ``≈1+4r`` with the diffusion
    number ``r = dt·K/dz²`` (codex emulation: rel/mass err ``~6e-7`` at hourly
    dt with stiff ``K=0.1, dz=1`` ``r≈360 cond≈1.4e3``; degrades to ``~9e-5`` at
    ``dt=86400`` ``cond≈3.4e4`` — DD does NOT by itself prove well-conditioned at
    large dt, so f32 is acceptable at sub-daily ocean dt but the column-mass
    correction below makes conservation exact regardless).  GPU-only BENEFIT
    (RTX8000 f64 = 1/32 f32 throughput); on CPU/x86 f32 is a measured SLOWDOWN
    (dtype churn, job 8500033), so OPT-IN, default OFF, f64-state-gated.  Covers
    the single + pair (shared-factor T+S) paths; the BATCHED variant
    (``LEGOESM_VMIX_BATCHED=1``) is NOT covered — f32 has no effect there."""
    return (dtype == jnp.float64
            and os.environ.get("LEGOESM_VMIX_F32_SOLVE", "0") == "1")


def _restore_column_mass(x: jax.Array, field: jax.Array,
                         dz: jax.Array) -> jax.Array:
    """f64 column-mean conservation correction after an f32 WORK solve.

    The f32 Thomas solve leaves a tiny (``~1e-6`` relative) ``dz``-weighted
    mass residual that the cast-back to f64 does NOT remove (codex finding 3).
    Add the uniform per-column shift that restores ``Σ(x·dz) == Σ(field·dz)``
    exactly in f64 (the diffusion has zero-flux BCs, so the column integral is
    invariant).  Cheap (one f64 column sum) and mass-exact; the shift is
    ``~1e-6`` of the field so it is physically negligible.  All inputs f64."""
    w = jnp.broadcast_to(dz, x.shape)
    deficit = jnp.sum((field - x) * w, axis=-1, keepdims=True)
    return x + deficit / jnp.sum(w, axis=-1, keepdims=True)


def implicit_vertical_diffusion_ocean(
    field: jax.Array,
    K: jax.Array | float,
    dz: jax.Array,
    dz_half: jax.Array,
    dt: float,
    *,
    extra_diag: jax.Array | float = 0.0,
) -> jax.Array:
    """Backward-Euler implicit vertical diffusion for a column field.

    Applies one implicit step of

        (1 - dt · ∂_z K ∂_z) φ^{n+1} = φ^{n}

    per column with *zero-flux* boundary conditions at the top and the
    bottom (the ocean's natural choice for momentum and tracers in the
    absence of a prescribed surface flux).  Non-zero surface / bottom
    fluxes can be applied *externally* before or after this call — the
    helper is deliberately boundary-condition-simple so callers don't
    have to thread flux arrays through when they don't need them.

    Parameters
    ----------
    field : jax.Array, shape ``(..., nlev)``
        Field to diffuse.  The vertical axis is the last axis; any
        number of leading horizontal axes is allowed.  Examples:
        ``(n_lat, n_lon, nlev)`` for lat-lon, ``(nCells, nlev)`` for
        MPAS, ``(n_lat, n_lon+1, nlev)`` for u on a C-grid.
    K : jax.Array or float, shape ``(..., nlev-1)``
        Vertical viscosity / diffusivity at interior interfaces.
        Must be ≥ 0.  A scalar is broadcast to every interface and
        every column.
    dz : jax.Array, shape ``(..., nlev)`` or ``(nlev,)``
        Layer thickness at full levels.  Must match ``field`` along
        the last axis (a 1-D ``dz`` broadcasts across all columns).
    dz_half : jax.Array, shape ``(..., nlev-1)`` or ``(nlev-1,)``
        Distance between adjacent full-level centres
        (``dz_half_k = 0.5 (dz_k + dz_{k+1})`` is the standard choice).
    dt : float
        Time step [s].  Must be positive.
    extra_diag : jax.Array or float, shape ``(..., nlev)``, default 0.0
        Additional POSITIVE term added to the diagonal ``b`` (e.g. an
        implicit bottom-drag rate ``dt·r/h`` at the bottom cell, zero
        elsewhere — NEMO dynzdf.F90 ``ln_drgimp``: ``zwd(iku) -=
        zDt_2*(rCdU_bot(i+1)+rCdU_bot(i))/e3u(iku)`` where ``rCdU_bot<=0``
        so the subtraction ADDS positive definiteness).  Sign convention:
        MUST be ``>= 0`` — it represents a damping (sink) term; a negative
        value would remove diagonal dominance and could destabilise the
        solve.  Default 0.0 ⇒ bit-identical to the pre-existing diagonal.

    Returns
    -------
    jax.Array
        Updated field with the same shape as ``field``.
    """
    # Only enforce the positivity check when ``dt`` is a concrete Python
    # scalar — under ``jax.jit`` it may be a traced argument, and a
    # Python-level ``if`` would raise ``TracerBoolConversionError``.
    if not isinstance(dt, jax.core.Tracer):
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt!r}")

    nlev = field.shape[-1]
    if nlev < 2:
        # One-level columns have no vertical gradient ⇒ no-op.
        return field

    a, b, c, d = _build_implicit_tridiag(field, K, dz, dz_half, dt,
                                          extra_diag=extra_diag)
    if _vmix_f32_solve_enabled(field.dtype):
        f32 = jnp.float32
        x = thomas_solve(a.astype(f32), b.astype(f32), c.astype(f32),
                         d.astype(f32)).astype(field.dtype)
        return _restore_column_mass(x, field, dz)
    return thomas_solve(a, b, c, d)


def implicit_vertical_diffusion_ocean_pair(
    field_1: jax.Array,
    field_2: jax.Array,
    K: jax.Array | float,
    dz: jax.Array,
    dz_half: jax.Array,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """Backward-Euler vertical diffusion of TWO fields sharing ONE matrix.

    ``field_1`` and ``field_2`` (same shape, e.g. T and S) diffuse
    against the IDENTICAL tridiagonal system — same ``K``, ``dz``,
    ``dz_half``, ``dt`` — so the coefficient build and the Thomas
    forward-elimination factors are computed ONCE
    (:func:`legoesm.timestepping.tridiagonal.thomas_solve_shared`)
    instead of once per field.  The implicit vmix is memory-bandwidth
    bound (phase split 8458934; field-batching measured a NON-win,
    jobs 8459136/8459145, because it ADDS traffic — this variant
    REMOVES a full duplicate coefficient build + factor sweep).

    Outputs are BIT-IDENTICAL to two
    :func:`implicit_vertical_diffusion_ocean` calls (the shared solver
    keeps per-RHS arithmetic operation-identical; on mixed dtypes it
    falls back to per-field solves).
    """
    if not isinstance(dt, jax.core.Tracer):
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt!r}")
    if field_1.shape != field_2.shape:
        raise ValueError(
            "implicit_vertical_diffusion_ocean_pair: fields must share a "
            f"shape; got {field_1.shape} vs {field_2.shape}"
        )

    nlev = field_1.shape[-1]
    if nlev < 2:
        return field_1, field_2

    a, b, c, d1 = _build_implicit_tridiag(field_1, K, dz, dz_half, dt)
    # BOTH fields must be f64 (codex finding 1): keying only off field_1 would
    # force a f32 field_2 through the f32-work path, violating the helper's
    # "no-op if state already f32" contract + the shared-solver mixed-dtype
    # faithfulness tests.
    if (_vmix_f32_solve_enabled(field_1.dtype)
            and field_2.dtype == jnp.float64):
        f32 = jnp.float32
        x1, x2 = thomas_solve_shared(
            a.astype(f32), b.astype(f32), c.astype(f32),
            (d1.astype(f32), field_2.astype(f32)))
        x1 = _restore_column_mass(x1.astype(field_1.dtype), field_1, dz)
        x2 = _restore_column_mass(x2.astype(field_2.dtype), field_2, dz)
        return x1, x2
    x1, x2 = thomas_solve_shared(a, b, c, (d1, field_2))
    return x1, x2


def _build_implicit_tridiag(
    field: jax.Array,
    K: jax.Array | float,
    dz: jax.Array,
    dz_half: jax.Array,
    dt: float,
    *,
    extra_diag: jax.Array | float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Assemble the backward-Euler tridiagonal ``(a, b, c, d)`` for one field.

    This is the *exact* coefficient construction that
    :func:`implicit_vertical_diffusion_ocean` feeds to :func:`thomas_solve`,
    factored out so the field-batched entry point
    (:func:`implicit_vertical_diffusion_ocean_batched`) can build each field's
    system with byte-identical arithmetic and then solve them all in ONE
    batched Thomas call.  ``a, b, c, d`` are returned with the system (level)
    on the LAST axis and ``field``'s leading shape preserved (no flattening).

    ``extra_diag`` (default 0.0, see :func:`implicit_vertical_diffusion_ocean`)
    is added directly to the diagonal ``b`` — a POSITIVE damping term (e.g.
    NEMO's implicit bottom drag) localized to whichever level the caller
    already zeroed elsewhere (typically via a bottom-level indicator mask).

    Caller guarantees ``field.shape[-1] = nlev >= 2`` (the ``nlev < 2`` no-op
    is handled by the public wrappers before this is invoked).
    """
    nlev = field.shape[-1]

    # --- Promote K, dz, dz_half to match the field's leading shape ---
    K_arr = jnp.asarray(K)
    if K_arr.ndim == 0:
        K_arr = jnp.broadcast_to(K_arr, field.shape[:-1] + (nlev - 1,))
    elif K_arr.shape[-1] != nlev - 1:
        raise ValueError(
            f"K last dim {K_arr.shape[-1]} must equal nlev-1 = {nlev - 1}")

    dz_arr = jnp.asarray(dz)
    if dz_arr.ndim == 1:
        dz_arr = jnp.broadcast_to(dz_arr, field.shape)
    elif dz_arr.shape[-1] != nlev:
        raise ValueError(
            f"dz last dim {dz_arr.shape[-1]} must equal nlev = {nlev}")

    dzh_arr = jnp.asarray(dz_half)
    if dzh_arr.ndim == 1:
        dzh_arr = jnp.broadcast_to(dzh_arr, field.shape[:-1] + (nlev - 1,))
    elif dzh_arr.shape[-1] != nlev - 1:
        raise ValueError(
            f"dz_half last dim {dzh_arr.shape[-1]} must equal nlev-1 = "
            f"{nlev - 1}")

    # --- Build α and β at every cell (last axis = level) ---
    # α_k uses the (k-1/2) interface, β_k the (k+1/2) interface.
    # We pad K with an extra zero on each side so indexing is uniform;
    # the zeros naturally encode the no-flux BCs.
    K_safe = jnp.maximum(K_arr, 0.0)
    dzh_safe = jnp.maximum(dzh_arr, _EPS)

    # K / dz_half at interfaces (nlev-1)
    flux_coeff = K_safe / dzh_safe                   # (..., nlev-1)

    # Pad top and bottom with zero (no-flux).  Two Pad HLO ops replace
    # alloc-zeros + two concatenate-of-two.
    pad_axes = ((0, 0),) * (flux_coeff.ndim - 1)
    flux_top = jnp.pad(flux_coeff, (*pad_axes, (1, 0)))  # (..., nlev)
    flux_bot = jnp.pad(flux_coeff, (*pad_axes, (0, 1)))  # (..., nlev)

    inv_dz = 1.0 / jnp.maximum(dz_arr, _EPS)          # (..., nlev)
    alpha = dt * flux_top * inv_dz                    # (..., nlev)
    beta = dt * flux_bot * inv_dz                     # (..., nlev)

    # Tridiagonal coefficients:
    #   a_k = -α_k   (sub-diagonal, a_0 = 0)
    #   b_k = 1 + α_k + β_k + extra_diag_k
    #   c_k = -β_k   (super-diagonal, c_{N-1} = 0)
    #   d_k = φ^n_k
    # extra_diag defaults to 0.0 -> b unchanged -> bit-identical.
    a = -alpha
    b = 1.0 + alpha + beta + extra_diag
    c = -beta
    d = field
    return a, b, c, d


def implicit_vertical_diffusion_ocean_batched(
    systems: "list[tuple[jax.Array, jax.Array | float, jax.Array, jax.Array, float]]",
) -> "list[jax.Array]":
    """Backward-Euler vertical diffusion for SEVERAL fields in ONE solve.

    Each entry of ``systems`` is ``(field, K, dz, dz_half, dt)`` with the SAME
    per-field semantics as :func:`implicit_vertical_diffusion_ocean` — every
    field may have a different leading (column) shape and its OWN ``K`` /
    ``dz`` / ``dz_half`` / ``dt``.  The only shared requirement is that all
    fields have the SAME number of vertical levels ``nlev`` on the last axis
    (the tridiagonal system size).

    Why batch
    ---------
    The ocean step solves four independent backward-Euler systems per step —
    ``T``, ``S`` (cell centres, tracer ``dt``, diffusivity ``K_v``) and
    ``u``, ``v`` (staggered faces, momentum ``dt_mom``, viscosity ``A_v``).
    Each separate :func:`thomas_solve` lowers to TWO ``fori_loop`` while-loops
    (forward + backward sweep) that XLA cannot fuse across, so the four calls
    serialise into eight while-loops.  Here we build each field's tridiagonal
    with the *identical* :func:`_build_implicit_tridiag` arithmetic, flatten
    each to ``(n_cols_field, nlev)``, CONCATENATE along the column axis into a
    single ``(Σ n_cols, nlev)`` batch, and issue ONE
    :func:`thomas_solve_batched`.  On CPU that collapses 8 while-loops → 2; on
    a CUDA backend ``thomas_solve_batched`` uses the while-loop-free PCR
    (or cuSPARSE) path, so the per-device dispatch cost drops further.

    Bit-faithfulness
    ----------------
    The per-field coefficients are byte-identical to the four separate
    :func:`implicit_vertical_diffusion_ocean` calls (same
    ``_build_implicit_tridiag``).  ``thomas_solve_batched``'s default CPU
    backend is ``vmap(thomas_solve)`` over the concatenated columns, so the
    batched result equals the four-separate result to the last bit on the
    DEFAULT/LEGACY CPU backend.  (On CUDA — or with a forced
    ``LEGOESM_TRIDIAG=pcr``/``cusparse`` on any backend — PCR/cuSPARSE are
    different but equally exact direct solvers, agreeing to machine epsilon,
    NOT bit-for-bit; the bit-faithfulness tests force ``LEGOESM_TRIDIAG=legacy``
    so the exact-equality gate is meaningful regardless of where they run.)  The dtype contract of
    :func:`thomas_solve` (solve in ``result_type(a,b,c,d)``, return in
    ``d.dtype``) is reproduced exactly PER FIELD: each system uses its OWN
    work dtype, and fields are GROUPED by work dtype so a f32-only system is
    never silently promoted to f64 just because it shares the batch with a
    f64 system (that would change its f32 result bits) — each group is one
    dtype-uniform :func:`thomas_solve_batched`, and each field's output is
    cast back to its own ``d.dtype``.  Production passes four dtype-uniform
    fields ⇒ a single group ⇒ one batched solve; mixed dtypes ⇒ one solve
    per dtype, each still bit-faithful to the scalar path.

    Returns
    -------
    list[jax.Array]
        One updated field per input entry, in the same order, each with the
        same shape and dtype as its input ``field``.
    """
    if not systems:
        return []

    nlev = systems[0][0].shape[-1]
    for idx, (field, _K, _dz, _dzh, dt) in enumerate(systems):
        if field.shape[-1] != nlev:
            raise ValueError(
                f"implicit_vertical_diffusion_ocean_batched requires every "
                f"field to share nlev on the last axis; system 0 has "
                f"nlev={nlev} but system {idx} has nlev={field.shape[-1]}")
        # Mirror the public wrapper's concrete-scalar positivity guard.
        if not isinstance(dt, jax.core.Tracer):
            if dt <= 0.0:
                raise ValueError(
                    f"dt must be > 0, got {dt!r} (system {idx})")

    if nlev < 2:
        # One-level columns have no vertical gradient ⇒ no-op (same as the
        # scalar wrapper); return the inputs unchanged.
        return [field for (field, _K, _dz, _dzh, _dt) in systems]

    # Build each field's tridiagonal with the IDENTICAL per-field arithmetic
    # the scalar solve uses, flatten the leading (column) axes, and record its
    # OWN work dtype + output dtype.  thomas_solve solves field i in
    # ``result_type(a_i, b_i, c_i, d_i)`` and returns in ``d_i.dtype``; to stay
    # bit-faithful per field we reproduce BOTH exactly — fields are GROUPED BY
    # work dtype so a f32-only system is NEVER promoted to f64 just because it
    # shares the batch with a f64 system (that would change its f32 result
    # bits).  Within a group all systems share the work dtype, so they
    # concatenate into one dtype-uniform thomas_solve_batched call (which
    # requires a/b/c/d to share dtype).  Production passes four dtype-uniform
    # fields ⇒ a single group ⇒ one solve; mixed dtypes ⇒ one solve per dtype.
    built = []  # per system: (a2d, b2d, c2d, d2d, work_dtype, out_dtype, shape)
    for (field, K, dz, dz_half, dt) in systems:
        a, b, c, d = _build_implicit_tridiag(field, K, dz, dz_half, dt)
        a2d = a.reshape(-1, nlev)
        b2d = b.reshape(-1, nlev)
        c2d = c.reshape(-1, nlev)
        d2d = d.reshape(-1, nlev)
        work_dtype = jnp.result_type(a2d, b2d, c2d, d2d)
        built.append((a2d, b2d, c2d, d2d, work_dtype, d.dtype, field.shape))

    # Group system indices by their work dtype, preserving first-seen order so
    # the (rare) multi-group path is deterministic.
    groups: dict = {}
    for i, item in enumerate(built):
        groups.setdefault(item[4], []).append(i)

    outs: list = [None] * len(systems)
    for work_dtype, idxs in groups.items():
        a_cat = jnp.concatenate(
            [built[i][0].astype(work_dtype) for i in idxs], axis=0)
        b_cat = jnp.concatenate(
            [built[i][1].astype(work_dtype) for i in idxs], axis=0)
        c_cat = jnp.concatenate(
            [built[i][2].astype(work_dtype) for i in idxs], axis=0)
        d_cat = jnp.concatenate(
            [built[i][3].astype(work_dtype) for i in idxs], axis=0)
        x_cat = thomas_solve_batched(a_cat, b_cat, c_cat, d_cat)
        start = 0
        for i in idxs:
            n_i = built[i][0].shape[0]
            out_dtype = built[i][5]
            shape = built[i][6]
            outs[i] = jax.lax.convert_element_type(
                x_cat[start:start + n_i], out_dtype,
            ).reshape(shape)
            start += n_i
    return outs


# ---------------------------------------------------------------------------
# Convenience: build dz_half from dz with the standard midpoint rule.
# ---------------------------------------------------------------------------


def build_dz_half(dz: jax.Array) -> jax.Array:
    """Midpoint distance between adjacent full-level centres.

    ``dz_half_k = 0.5 · (dz_k + dz_{k+1})`` with shape ``(..., nlev-1)``.
    Provided as a helper because most callers don't carry ``dz_half``
    separately from ``dz_ref`` but do need it to build the tridiagonal
    system.
    """
    return 0.5 * (dz[..., :-1] + dz[..., 1:])
