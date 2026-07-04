"""Per-column model-vs-ERA5 comparison metrics.

Reanalysis-comparison companion to :mod:`legoesm.training.scm_rce_metrics`.
Where the SCM/CRM path scores a *single* RCE column against a CRM reference,
this module scores **every** GCM column against ERA5 on the same horizontal
grid and vertical levels, producing per-column error fields that rank the
worst-performing columns for LES spin-off (see ``docs/COMPARE_REANALYSIS.md``,
stage 2 / gap #1).

Design constraints (CLAUDE.md):

* **No re-derived numerics.**  The mass-weighted vertical RMSE arithmetic and
  the precipitation normalization are imported from
  :mod:`legoesm.training.scm_rce_metrics` so the SCM/CRM and AMIP/ERA5 paths
  cannot drift.
* **Pure pytree / differentiable.**  Every public function is a pure JAX
  function over arrays with stable shapes (no Python control flow on traced
  values), so it composes with ``jax.grad``/``jit`` and ``vmap`` over arbitrary
  leading column dimensions (lat-lon ``(n_lat, n_lon)`` or cubed-sphere
  ``(6, n, n)``).
* **No hardcoded tunable params.**  Diagnostic normalization scales are
  module-level ``UPPER_SNAKE`` constants (mirroring
  ``PRECIP_NORMALIZATION_MM_DAY``) and overridable through
  :class:`ColumnErrorConfig`; they are *diagnostic normalizers*, not trainable
  physics coefficients, so they carry no ``__param_spec__``.

The per-column ``combined_score`` is a dimensionless RMS of the per-variable
errors, each divided by its physical normalization scale, so that an order-1
score corresponds to a substantial AMIP bias and no single variable dominates
the ranking.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.training.scm_rce_metrics import (
    PRECIP_NORMALIZATION_MM_DAY,
    precip_score_jax,
    safe_sqrt,
    weighted_rmse,
)

# --- diagnostic normalization scales (per-variable, physical units) ---------
# Chosen so an order-1 normalized component corresponds to a large AMIP-vs-ERA5
# column bias.  These are *ranking normalizers*, not tunable physics: they set
# the relative emphasis of T / humidity / wind / precip in the combined score.
# Values follow typical free-running AMIP RMSE magnitudes at 6-hourly cadence.
T_NORMALIZATION_K = 3.0  # column-mean temperature error scale [K]
QV_NORMALIZATION_KG_KG = 1.5e-3  # water-vapour mixing-ratio error scale [kg/kg]
WIND_NORMALIZATION_M_S = 5.0  # vector-wind error scale [m/s]
# precip reuses scm_rce_metrics.PRECIP_NORMALIZATION_MM_DAY (3.0 mm/day)

# Floor protecting weight renormalization for an all-invalid column.
_WEIGHT_SUM_FLOOR = 1.0e-30


class ColumnErrorConfig(NamedTuple):
    """Normalization scales and combined-score weights for column scoring.

    Defaults reference the module-level normalization constants.  ``*_weight``
    fields set the relative contribution of each variable to the combined RMS
    score; set a weight to ``0.0`` to drop a variable (e.g. ``precip_weight=0``
    when an ERA5 precipitation field is unavailable).
    """

    T_norm_K: float = T_NORMALIZATION_K
    qv_norm_kg_kg: float = QV_NORMALIZATION_KG_KG
    wind_norm_m_s: float = WIND_NORMALIZATION_M_S
    precip_norm_mm_day: float = PRECIP_NORMALIZATION_MM_DAY
    T_weight: float = 1.0
    qv_weight: float = 1.0
    wind_weight: float = 1.0
    precip_weight: float = 1.0


class ColumnErrorFields(NamedTuple):
    """Per-column error fields (physical units) plus the combined score.

    Every field has the model's horizontal column shape (leading dims), e.g.
    ``(n_lat, n_lon)`` or ``(6, n, n)``.  ``combined_score`` is dimensionless.
    """

    T_rmse_K: jax.Array
    qv_rmse_kg_kg: jax.Array
    wind_rmse_m_s: jax.Array
    precip_err_mm_day: jax.Array
    combined_score: jax.Array


def normalized_mass_weights(layer_thickness: jax.Array) -> jax.Array:
    """Return per-level mass weights that sum to one along the LAST (vertical) axis.

    The layer mass weight is the layer thickness divided by the column total.  The
    thickness can be:

    * a ``[nlev]`` sigma-thickness ``dsigma`` (pure-sigma: ``Δp = p_s·dsigma`` and the
      ``p_s`` factor cancels in the per-column normalization, so this reduces to
      ``dsigma / Σ dsigma``), or
    * a ``[..., nlev]`` per-column PRESSURE thickness ``dp = diff(p_half)`` — correct
      for a HYBRID coordinate (``Δp = dA·p_ref + dB·p_s``), where ``p_s`` does NOT cancel.

    Normalization is along ``axis=-1`` so a ``[nlev]`` vector and a ``[ncol, nlev]`` field
    both normalize per column (the 1-D case is unchanged: ``Σ`` over the only axis).
    """
    layer_thickness = jnp.asarray(layer_thickness)
    total = jnp.maximum(
        jnp.sum(layer_thickness, axis=-1, keepdims=True),
        jnp.asarray(_WEIGHT_SUM_FLOOR, dtype=layer_thickness.dtype),
    )
    return layer_thickness / total


def _masked_weights(
    weights: jax.Array, valid_level: jax.Array
) -> jax.Array:
    """Zero invalid levels and renormalize per column so weights sum to one.

    ``weights`` broadcasts to ``valid_level`` (``[..., nlev]`` boolean).  Levels
    that are non-finite in either field carry zero weight; the remaining valid
    levels are renormalized so each column's weights still sum to one (a column
    with no valid level gets all-zero weights → zero RMSE, and is expected to be
    excluded via ``valid_mask``).
    """
    w = jnp.broadcast_to(weights, valid_level.shape)
    w = jnp.where(valid_level, w, jnp.zeros_like(w))
    total = jnp.sum(w, axis=-1, keepdims=True)
    total = jnp.maximum(total, jnp.asarray(_WEIGHT_SUM_FLOOR, dtype=w.dtype))
    return w / total


def per_column_weighted_rmse(
    model_profile: jax.Array,
    ref_profile: jax.Array,
    mass_weights: jax.Array,
) -> jax.Array:
    """Mass-weighted vertical RMSE per column, in the field's physical units.

    ``model_profile`` / ``ref_profile`` are ``[..., nlev]`` (any leading column
    shape); ``mass_weights`` is ``[nlev]`` (or broadcastable to the column
    shape).  Non-finite levels in either field are masked out and the remaining
    valid weights renormalized per column, so missing ERA5 levels never poison
    a column's score.  Reuses :func:`weighted_rmse` (no re-derived RMSE).
    """
    model = jnp.asarray(model_profile)
    ref = jnp.asarray(ref_profile, dtype=model.dtype)
    weights = jnp.asarray(mass_weights, dtype=model.dtype)

    valid_level = jnp.isfinite(model) & jnp.isfinite(ref)
    diff = jnp.where(valid_level, model - ref, jnp.zeros_like(model))
    w = _masked_weights(weights, valid_level)

    nlev = diff.shape[-1]
    flat_diff = diff.reshape(-1, nlev)
    flat_w = w.reshape(-1, nlev)
    rmse_flat = jax.vmap(weighted_rmse)(flat_diff, flat_w)
    return rmse_flat.reshape(diff.shape[:-1])


def per_column_vector_wind_rmse(
    u_model: jax.Array,
    v_model: jax.Array,
    u_ref: jax.Array,
    v_ref: jax.Array,
    mass_weights: jax.Array,
) -> jax.Array:
    """Mass-weighted vertical RMS of the **vector** wind error per column [m/s].

    Computes ``sqrt(Σ_k w_k ((Δu_k)^2 + (Δv_k)^2))`` so a rotated-but-equal-speed
    wind still registers as an error.  Reuses :func:`weighted_rmse` by stacking
    the two components along the vertical axis with duplicated weights (the RMS
    over the doubled axis equals the vector RMS).
    """
    u_model = jnp.asarray(u_model)
    v_model = jnp.asarray(v_model, dtype=u_model.dtype)
    u_ref = jnp.asarray(u_ref, dtype=u_model.dtype)
    v_ref = jnp.asarray(v_ref, dtype=u_model.dtype)
    weights = jnp.asarray(mass_weights, dtype=u_model.dtype)

    valid_u = jnp.isfinite(u_model) & jnp.isfinite(u_ref)
    valid_v = jnp.isfinite(v_model) & jnp.isfinite(v_ref)
    valid = valid_u & valid_v
    du = jnp.where(valid, u_model - u_ref, jnp.zeros_like(u_model))
    dv = jnp.where(valid, v_model - v_ref, jnp.zeros_like(v_model))

    # Per-column weights renormalized over valid levels, applied at FULL weight
    # to each component so weighted_rmse yields sqrt(Σ_k w_k (Δu_k² + Δv_k²)) —
    # the vector wind RMS (the stacked weights sum to two, by construction).
    w = _masked_weights(weights, valid)
    stacked_diff = jnp.concatenate([du, dv], axis=-1)
    stacked_w = jnp.concatenate([w, w], axis=-1)

    n2 = stacked_diff.shape[-1]
    flat_diff = stacked_diff.reshape(-1, n2)
    flat_w = stacked_w.reshape(-1, n2)
    rmse_flat = jax.vmap(weighted_rmse)(flat_diff, flat_w)
    return rmse_flat.reshape(du.shape[:-1])


def score_columns(
    *,
    T_model: jax.Array,
    qv_model: jax.Array,
    u_model: jax.Array,
    v_model: jax.Array,
    T_ref: jax.Array,
    qv_ref: jax.Array,
    u_ref: jax.Array,
    v_ref: jax.Array,
    mass_weights: jax.Array,
    precip_model_mm_day: jax.Array | None = None,
    precip_ref_mm_day: jax.Array | None = None,
    config: ColumnErrorConfig = ColumnErrorConfig(),
) -> ColumnErrorFields:
    """Score every GCM column against ERA5 on a shared grid + vertical levels.

    All ``*_model`` / ``*_ref`` profile arrays are ``[..., nlev]`` on the same
    grid; ``precip_*`` (optional) are ``[...]`` surface fields in mm/day.
    ``mass_weights`` is the ``[nlev]`` vector from :func:`normalized_mass_weights`.

    Returns a :class:`ColumnErrorFields` of per-column physical-unit errors plus
    a dimensionless ``combined_score`` (RMS of weighted, normalized components).
    Higher score = worse column.
    """
    T_rmse = per_column_weighted_rmse(T_model, T_ref, mass_weights)
    qv_rmse = per_column_weighted_rmse(qv_model, qv_ref, mass_weights)
    wind_rmse = per_column_vector_wind_rmse(
        u_model, v_model, u_ref, v_ref, mass_weights
    )

    dtype = T_rmse.dtype
    have_precip = precip_model_mm_day is not None and precip_ref_mm_day is not None
    if (precip_model_mm_day is None) != (precip_ref_mm_day is None):
        raise ValueError(
            "score_columns: pass both precip_model_mm_day and "
            "precip_ref_mm_day, or neither (got exactly one)."
        )
    if have_precip:
        precip_err = precip_score_jax(
            precip_model_mm_day,
            precip_ref_mm_day,
            normalization_mm_day=config.precip_norm_mm_day,
        ) * jnp.asarray(config.precip_norm_mm_day, dtype=dtype)
        precip_norm_term = precip_err / jnp.asarray(
            config.precip_norm_mm_day, dtype=dtype
        )
        precip_weight = jnp.asarray(config.precip_weight, dtype=dtype)
    else:
        precip_err = jnp.zeros_like(T_rmse)
        precip_norm_term = jnp.zeros_like(T_rmse)
        precip_weight = jnp.asarray(0.0, dtype=dtype)

    T_term = T_rmse / jnp.asarray(config.T_norm_K, dtype=dtype)
    qv_term = qv_rmse / jnp.asarray(config.qv_norm_kg_kg, dtype=dtype)
    wind_term = wind_rmse / jnp.asarray(config.wind_norm_m_s, dtype=dtype)

    w_T = jnp.asarray(config.T_weight, dtype=dtype)
    w_qv = jnp.asarray(config.qv_weight, dtype=dtype)
    w_wind = jnp.asarray(config.wind_weight, dtype=dtype)

    weight_sum = jnp.maximum(
        w_T + w_qv + w_wind + precip_weight,
        jnp.asarray(_WEIGHT_SUM_FLOOR, dtype=dtype),
    )
    combined = safe_sqrt(
        (
            w_T * T_term**2
            + w_qv * qv_term**2
            + w_wind * wind_term**2
            + precip_weight * precip_norm_term**2
        )
        / weight_sum
    )
    return ColumnErrorFields(
        T_rmse_K=T_rmse,
        qv_rmse_kg_kg=qv_rmse,
        wind_rmse_m_s=wind_rmse,
        precip_err_mm_day=precip_err,
        combined_score=combined,
    )


def rank_worst_columns(
    combined_score: jax.Array,
    n: int,
    *,
    valid_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the flat indices, scores, and validity of the ``n`` worst columns.

    ``combined_score`` has the horizontal column shape; it is flattened and the
    ``n`` highest scores selected with :func:`jax.lax.top_k` (descending).
    ``valid_mask`` (same shape, boolean) excludes columns (e.g. land when a
    metric is ocean-only) by forcing their score to ``-inf`` before selection.

    Returns ``(flat_indices, scores, valid)`` each of length
    ``min(n, n_columns)``.  ``valid`` is ``False`` for any slot that fell back
    to a masked (``-inf``) score because ``n`` exceeded the number of valid
    columns — callers MUST filter on it before trusting an index (the shape is
    kept static for JIT).  Use :func:`jax.numpy.unravel_index` with the original
    shape to recover grid coordinates.
    """
    score = jnp.asarray(combined_score)
    flat = score.reshape(-1)
    neg_inf = jnp.asarray(-jnp.inf, dtype=flat.dtype)
    # A non-finite score (e.g. NaN) has undefined top_k ordering and cannot be
    # trusted as "worst": route it to -inf so it is never selected and is
    # flagged invalid below.
    flat = jnp.where(jnp.isfinite(flat), flat, neg_inf)
    if valid_mask is not None:
        valid_flat = jnp.asarray(valid_mask, dtype=bool).reshape(-1)
        flat = jnp.where(valid_flat, flat, neg_inf)
    k = min(int(n), flat.shape[0])
    top_scores, top_idx = jax.lax.top_k(flat, k)
    top_valid = top_scores > neg_inf
    return top_idx, top_scores, top_valid
