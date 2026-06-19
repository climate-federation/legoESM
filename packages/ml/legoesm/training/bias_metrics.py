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
from typing import NamedTuple

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
    (e.g. :func:`legoesm.parallel.reductions.global_sum_mpi`).  When given, the
    weighted NUMERATOR ``Σ score·w`` and the DENOMINATOR ``Σ w`` are each reduced
    across ranks BEFORE the division, so a rank-local slice (e.g. an MPAS rank's
    owned cells under the iter-86 owned mask) yields the SAME GLOBAL area-weighted
    bias on every rank — the basis for an identical (deadlock-free) accept/reject
    and line-search decision across ranks (iter 88).  ``None`` (single-process
    default) keeps the purely-local sum, byte-identical to the prior behaviour.
    """
    score = jnp.asarray(combined_score)
    weights = jnp.asarray(area_weights)
    dtype = jnp.result_type(score, weights, jnp.float32)
    score = score.astype(dtype)
    w = jnp.broadcast_to(weights.astype(dtype), score.shape)
    if valid_mask is not None:
        m = jnp.asarray(valid_mask, dtype=bool)
        w = jnp.where(m, w, jnp.zeros_like(w))
        score = jnp.where(m, score, jnp.zeros_like(score))
    numerator = jnp.sum(score * w)
    total = jnp.sum(w)
    if global_reduce is not None:
        # Reduce the SUMS (not the ratio) across ranks, then divide — the global
        # weighted mean of a partitioned field.  Both reductions share the dtype.
        numerator = global_reduce(numerator)
        total = global_reduce(total)
    floor = jnp.asarray(_BIAS_FLOOR, dtype=dtype)
    has_weight = total > floor
    # Guarded denominator: never 0/0 in either where-branch, so the reverse-mode
    # gradient stays finite even when every column is excluded.
    safe_total = jnp.where(has_weight, total, jnp.ones_like(total))
    mean = numerator / safe_total
    return jnp.where(has_weight, mean, jnp.zeros_like(mean))


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
