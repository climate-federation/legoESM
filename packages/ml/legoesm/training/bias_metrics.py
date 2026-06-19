"""Quantify whether the LES-informed parameter update reduced the AMIP/CMIP bias.

The success criterion of ``docs/COMPARE_REANALYSIS.md`` is that *updating the
parameters improves the biases*.  This module measures it: it collapses the
per-column model-vs-ERA5 ``combined_score`` field
(:func:`legoesm.training.column_era5_metrics.score_columns`) to a single
area-weighted **global bias**, and compares a baseline run to an updated run —
globally and at the targeted worst columns.

The per-column scores come from the already-tested comparison metric.  The
area-weighted mean is computed with an explicit zero-weight guard — the shared
:func:`legoesm.core.precision.weighted_mean` forms ``sum(xw)/sum(w)`` directly,
which is ``0/0`` (a leaking NaN reverse-mode gradient) when every column is
masked out; the masked aggregation here needs that guard.  Pure-JAX, AD-safe,
jit-friendly.

Precondition: with ``valid_mask=None`` the scores must be finite (the comparison
metric :func:`score_columns` guarantees this — it NaN-masks per level); a
``valid_mask`` additionally sanitises masked-out scores so a NaN there cannot
contaminate the sum.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

# Floor protecting the weighted-mean + fractional-improvement denominators.
_BIAS_FLOOR = 1.0e-30


class BiasImprovement(NamedTuple):
    """Comparison of a baseline vs updated run's global bias."""

    baseline_bias: jax.Array       # area-weighted mean combined score, baseline
    updated_bias: jax.Array        # area-weighted mean combined score, updated
    absolute_reduction: jax.Array  # baseline − updated (> 0 ⇒ improved)
    fractional_improvement: jax.Array  # (baseline − updated) / baseline
    improved: jax.Array            # bool scalar: updated < baseline


def _area_weighted_mean(
    field: jax.Array,
    area_weights: jax.Array,
    *,
    valid_mask: jax.Array | None = None,
    global_reduce: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    """Area-weighted global mean ``Σ field·w / Σ w`` of a per-column field.

    The shared, guarded reduction core behind :func:`aggregate_combined_bias`
    (combined-score bias) and :func:`aggregate_per_variable_bias` (per-variable
    bias).  ``area_weights`` (broadcastable to ``field``) is the per-column
    quadrature weight; ``valid_mask`` (same shape) excludes columns by zeroing
    their weight AND sanitising any non-finite value there (so a masked NaN cannot
    leak into the sum).  An all-excluded field yields ``0`` rather than ``0/0``.

    ``global_reduce`` (DISTRIBUTED, optional): a SUM reduction across MPI ranks
    (e.g. :func:`legoesm.parallel.reductions.global_sum_mpi`).  When given, the
    NUMERATOR ``Σ field·w`` and DENOMINATOR ``Σ w`` are each reduced across ranks
    BEFORE the division, so a rank-local slice yields the SAME GLOBAL mean on every
    rank (the basis for a deadlock-free accept/reject + line-search, iter 88).
    AD-safe: the denominator is masked before division.
    """
    field = jnp.asarray(field)
    weights = jnp.asarray(area_weights)
    dtype = jnp.result_type(field, weights, jnp.float32)
    field = field.astype(dtype)
    w = jnp.broadcast_to(weights.astype(dtype), field.shape)
    if valid_mask is not None:
        m = jnp.asarray(valid_mask, dtype=bool)
        w = jnp.where(m, w, jnp.zeros_like(w))
        field = jnp.where(m, field, jnp.zeros_like(field))
    numerator = jnp.sum(field * w)
    total = jnp.sum(w)
    if global_reduce is not None:
        numerator = global_reduce(numerator)
        total = global_reduce(total)
    floor = jnp.asarray(_BIAS_FLOOR, dtype=dtype)
    has_weight = total > floor
    # Guarded denominator: never 0/0 in either where-branch, so the reverse-mode
    # gradient stays finite even when every column is excluded.
    safe_total = jnp.where(has_weight, total, jnp.ones_like(total))
    mean = numerator / safe_total
    return jnp.where(has_weight, mean, jnp.zeros_like(mean))


def aggregate_combined_bias(
    combined_score: jax.Array,
    area_weights: jax.Array,
    *,
    valid_mask: jax.Array | None = None,
    global_reduce: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    """Area-weighted global mean of the per-column ``combined_score``.

    ``area_weights`` is the per-column quadrature weight (e.g. ``cos(lat)`` or
    ``grid.area``), broadcastable to ``combined_score``.  ``valid_mask`` (same
    shape) excludes columns by zeroing their weight (and sanitising any
    non-finite score there, so a masked NaN cannot leak into the sum).  Returns
    a scalar; an all-excluded field yields ``0`` rather than ``0/0``.

    ``global_reduce`` (DISTRIBUTED, optional): a SUM reduction across MPI ranks
    yields the SAME GLOBAL area-weighted bias on every rank — the basis for an
    identical (deadlock-free) accept/reject and line-search decision across ranks
    (iter 88).  ``None`` (single-process default) keeps the purely-local sum.
    """
    return _area_weighted_mean(
        combined_score, area_weights,
        valid_mask=valid_mask, global_reduce=global_reduce)


def bias_improvement(
    baseline_score: jax.Array,
    updated_score: jax.Array,
    area_weights: jax.Array,
    *,
    valid_mask: jax.Array | None = None,
    global_reduce: Callable[[jax.Array], jax.Array] | None = None,
) -> BiasImprovement:
    """Compare the global bias of a baseline vs an updated AMIP/CMIP run.

    Both ``*_score`` fields are per-column ``combined_score`` on the same grid;
    ``improved`` is true when the updated run's area-weighted global bias is
    lower.  ``fractional_improvement`` is the relative reduction (floored
    denominator for a near-perfect baseline).  ``global_reduce`` (DISTRIBUTED,
    optional) reduces the weighted bias across MPI ranks so ``improved`` is the
    GLOBAL verdict — identical on every rank, so the campaign's accept/reject gate
    and the line-search step choice cannot diverge between ranks (iter 88); the
    SAME reducer MUST feed both the baseline and updated aggregation (it does here).
    """
    base = aggregate_combined_bias(
        baseline_score, area_weights, valid_mask=valid_mask, global_reduce=global_reduce)
    upd = aggregate_combined_bias(
        updated_score, area_weights, valid_mask=valid_mask, global_reduce=global_reduce)
    reduction = base - upd
    frac = reduction / jnp.maximum(base, jnp.asarray(_BIAS_FLOOR, dtype=base.dtype))
    return BiasImprovement(
        baseline_bias=base,
        updated_bias=upd,
        absolute_reduction=reduction,
        fractional_improvement=frac,
        improved=upd < base,
    )


class PerVariableBias(NamedTuple):
    """Per-VARIABLE area-weighted global bias (physical units), the interpretable
    companion to the single :func:`aggregate_combined_bias` dimensionless score.

    ``global_T_rmse_K`` / ``global_qv_rmse_kg_kg`` / ``global_wind_rmse_m_s`` are
    the GLOBAL RMSE — ``sqrt(area_weighted_mean(per_column_rmse²))`` (the per-column
    fields are ALREADY vertical RMSEs, so they combine in QUADRATURE, not linearly,
    to the true global RMSE).  ``global_precip_err_mm_day`` is the area-weighted
    mean ABSOLUTE precip error (precip is an absolute, not RMS, error); it is
    ``NaN`` when precipitation was not compared (``have_precip=False``), so a
    not-compared precip is never reported as a spurious ``0``.  These names are
    GLOBAL scalars — distinct from the per-column ``ColumnErrorFields``.
    """

    global_T_rmse_K: jax.Array
    global_qv_rmse_kg_kg: jax.Array
    global_wind_rmse_m_s: jax.Array
    global_precip_err_mm_day: jax.Array


def aggregate_per_variable_bias(
    error_fields: Any,
    area_weights: jax.Array,
    *,
    have_precip: bool = False,
    valid_mask: jax.Array | None = None,
    global_reduce: Callable[[jax.Array], jax.Array] | None = None,
) -> PerVariableBias:
    """Per-variable global bias from a :class:`ColumnErrorFields`.

    The combined score can improve via a TRADE-OFF (better T, worse wind); this
    exposes each physical variable so a correction's effect is interpretable
    per-variable (the done-criterion: "improve the biases", plural).  T / q_v /
    wind aggregate in MSE-space — ``sqrt(area_weighted_mean(rmse²))`` (the true
    GLOBAL RMSE, since the inputs are per-column RMSEs); precip aggregates linearly
    (mean absolute error) and is ``NaN`` unless ``have_precip`` (a flat-zero precip
    field for a not-compared run would otherwise read as a perfect ``0``).
    ``valid_mask`` / ``global_reduce`` behave as in :func:`aggregate_combined_bias`
    (the same shared :func:`_area_weighted_mean` core).
    """
    from legoesm.training.scm_rce_metrics import safe_sqrt

    def _global_rmse(field):
        ms = _area_weighted_mean(
            jnp.asarray(field) ** 2, area_weights,
            valid_mask=valid_mask, global_reduce=global_reduce)
        return safe_sqrt(ms)

    precip = _area_weighted_mean(
        error_fields.precip_err_mm_day, area_weights,
        valid_mask=valid_mask, global_reduce=global_reduce)
    if not have_precip:
        precip = jnp.full_like(jnp.asarray(precip), jnp.nan)
    return PerVariableBias(
        global_T_rmse_K=_global_rmse(error_fields.T_rmse_K),
        global_qv_rmse_kg_kg=_global_rmse(error_fields.qv_rmse_kg_kg),
        global_wind_rmse_m_s=_global_rmse(error_fields.wind_rmse_m_s),
        global_precip_err_mm_day=precip,
    )


def worst_column_bias_change(
    baseline_score: jax.Array,
    updated_score: jax.Array,
    flat_indices: jax.Array,
    *,
    valid: jax.Array | None = None,
) -> jax.Array:
    """Mean bias reduction at the targeted worst columns (``> 0`` ⇒ improved).

    ``flat_indices`` are the worst-column flat indices; the LES correction targets
    these, so they should improve most.  Returns the mean of ``baseline − updated``
    over those columns; an empty (or all-invalid) index set returns ``0``.

    **Precondition — only ACTUAL worst columns must be passed.**
    :func:`legoesm.training.column_era5_metrics.rank_worst_columns` returns a
    STATIC-length array that PADS with invalid (``valid=False``) slots when ``n``
    exceeds the valid-column count, and those padded indices are still IN RANGE
    (real column ids), so the bounds check below CANNOT catch them — a padded,
    non-worst column would silently DILUTE the mean.  So EITHER pass the
    manifest-filtered indices (:func:`build_worst_column_manifest` drops invalid
    slots — what the campaign does) OR pass ``rank_worst_columns``'s ``valid``
    mask here (same shape as ``flat_indices``) and the padded slots are excluded
    from the mean.
    """
    base_flat = jnp.asarray(baseline_score).reshape(-1)
    upd_flat = jnp.asarray(updated_score).reshape(-1)
    idx = jnp.asarray(flat_indices)
    if int(idx.shape[0]) == 0:
        return jnp.asarray(0.0, dtype=base_flat.dtype)
    # Host-side bounds check: JAX gather silently clamps out-of-range indices, so
    # validate here (indices are concrete worst-column ids) to fail loudly.  When a
    # ``valid`` mask is given, only the indices that will actually contribute are
    # checked (a padded slot's index is irrelevant — it is masked out of the mean).
    import numpy as _np

    idx_np = _np.asarray(idx)
    n_cols = base_flat.shape[0]
    if valid is not None:
        valid_mask = jnp.asarray(valid, dtype=bool)
        if valid_mask.shape != idx.shape:
            raise ValueError(
                f"valid mask shape {tuple(valid_mask.shape)} != flat_indices "
                f"shape {tuple(idx.shape)} (must align on the same worst-column axis)."
            )
        check_idx = idx_np[_np.asarray(valid_mask)]
    else:
        valid_mask = None
        check_idx = idx_np
    if check_idx.size and (check_idx.min() < 0 or check_idx.max() >= n_cols):
        raise ValueError(
            f"flat_indices out of range [0, {n_cols}); got "
            f"[{int(check_idx.min())}, {int(check_idx.max())}]."
        )
    diff = base_flat[idx] - upd_flat[idx]
    if valid_mask is None:
        return jnp.mean(diff)
    # Mean over the valid (actual worst) slots only; all-invalid ⇒ 0.
    n_valid = jnp.sum(valid_mask.astype(diff.dtype))
    total = jnp.sum(jnp.where(valid_mask, diff, jnp.zeros_like(diff)))
    return jnp.where(n_valid > 0, total / jnp.maximum(n_valid, 1.0), jnp.zeros((), diff.dtype))
