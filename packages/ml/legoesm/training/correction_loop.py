"""The closed LES-informed correction loop: diagnose → correct → verify bias↓.

Capstone orchestration of ``docs/COMPARE_REANALYSIS.md``: compose every stage of
the pipeline into one iteration of the offline correction loop —

  baseline config
    → run AMIP + compare to ERA5      (compare_fn → per-column score + manifest)
    → LES-diagnose each worst column   (diagnose_fn → K / w_e)
    → assemble the feedback field       (:func:`assemble_feedback_field`, iter 16)
    → apply it to the target scheme     (:func:`apply_feedback_to_scheme`, iter 18)
    → re-run AMIP + compare             (compare_fn on the updated config)
    → measure the bias change           (:func:`bias_improvement`, iter 17).

The two heavy, data-bound steps are **injected** so the loop logic is unit-
testable end-to-end and the production driver swaps in the real AMIP/LES runs:

* ``compare_fn(config) -> CompareResult`` runs an AMIP/CMIP simulation with that
  config and scores it against ERA5 (``scripts/validate/compare_amip_era5.py`` +
  :func:`legoesm.training.column_era5_metrics.score_columns`).
* ``diagnose_fn(record, model_ctx) -> diagnosis`` spins off + diagnoses the
  column LES (``legoesm.atmosphere.dynamics.column_les.process_column``).

The loop is the offline / iterative correction of §1 — restartable, with the LES
batch embarrassingly parallel across columns.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from legoesm.training.bias_metrics import (
    BiasImprovement,
    bias_improvement,
    worst_column_bias_change,
)
from legoesm.training.column_clustering import cluster_columns_by_environment
from legoesm.training.compare_reanalysis import (
    ColumnState,
    compare_state_to_reference,
)
from legoesm.training.feedback_assembly import assemble_feedback_field
from legoesm.training.promotable_params import (
    apply_feedback_to_scheme,
    clip_field_to_promotable_bounds,
)


class CompareResult(NamedTuple):
    """One AMIP-vs-ERA5 comparison: the score field, manifest, and weights."""

    combined_score: jax.Array          # per-column combined score (grid-shaped)
    manifest: Sequence[Any]            # worst-column ColumnRecords
    area_weights: jax.Array            # per-column quadrature weights
    model_ctx: Any = None              # opaque state handed to diagnose_fn
    valid_mask: jax.Array | None = None  # optional column validity for aggregation


class CorrectionResult(NamedTuple):
    """Outcome of one correction iteration."""

    updated_config: Any
    bias: BiasImprovement              # baseline vs updated global bias
    worst_column_change: jax.Array     # mean bias reduction at the worst columns
    feedback_field: jax.Array          # the assembled per-column field
    n_corrected: int                   # number of worst columns corrected (all flagged)
    n_diagnosed: int = 0               # number of LES diagnoses run (= n_corrected, or
    #                                    K representatives when clustering w/ les_budget)
    step_fraction: float = 1.0         # the accepted line-search step toward the raw
    #                                    diagnosis (1.0 = full; 0.0 = no-op round)


class CampaignResult(NamedTuple):
    """Outcome of a multi-iteration correction campaign."""

    final_config: Any
    iterations: tuple                  # the per-iteration CorrectionResults
    final_field: jax.Array             # the accumulated per-column field
    accepted: tuple = ()               # per-round bool: was the round kept?
                                       # (all True unless accept_only_if_improved)


def run_correction_campaign(
    initial_config: Any,
    n_iterations: int,
    *,
    compare_fn: Callable[[Any], CompareResult],
    diagnose_fn: Callable[[Any, Any], Any],
    promotion_key: str,
    grid_shape: tuple[int, ...],
    diagnosis_method: str = "eddy_diffusivity",
    background: float = 0.0,
    expected_ncol: int | None = None,
    les_budget: int | None = None,
    env_scales: Sequence[float] | None = None,
    initial_field: jax.Array | None = None,
    start_round: int = 0,
    checkpoint_callback: Callable[[int, CorrectionResult, jax.Array], None] | None = None,
    feedback_strategy: str = "static",
    env_grid_fn: Callable[[Any], tuple] | None = None,
    accept_only_if_improved: bool = False,
    step_fractions: Sequence[float] | None = None,
    clip_to_bounds: bool = False,
) -> CampaignResult:
    """Run the offline iterative correction loop for ``n_iterations`` rounds.

    The §1 offline / iterative loop: each round re-runs the model, re-diagnoses
    the (now different) worst columns, and folds their correction into a
    persistent per-column field — so a column corrected in an earlier round
    keeps its value while newly-flagged columns are added (the feedback field
    of round *k* becomes the **background** of round *k+1*, accumulating).
    ``background`` is the production scalar default used as the round-0 base.

    **Restartable** (§1, for the HPC campaign that may outlive a job): pass
    ``initial_field`` (the accumulated feedback field of the last completed round)
    + ``initial_config`` (its corrected config) to RESUME — that field is the
    round-0 base instead of ``background``, so the accumulation continues
    seamlessly.  ``checkpoint_callback(round_index, result, accumulated_field)`` is
    invoked after EVERY round; ``start_round`` offsets ``round_index`` for the
    checkpoint label on resume.  **The ``accumulated_field`` is the ACCEPTED
    restart state** (post-gate base) and is the single source of truth — under the
    monotonic gate a rejected round's ``result.updated_config`` is the DISCARDED
    update, so persist/resume the config FROM ``accumulated_field`` (its flattened
    column field), never from ``result.updated_config``.  ``result`` is for logging
    (bias, ``n_corrected``).

    **Monotonic acceptance** (``accept_only_if_improved``, the done-criterion
    "updating these parameters IMPROVE the biases"): when true, a round is KEPT
    only if its correction lowered the area-weighted global bias
    (``result.bias.improved``); a worsening round is REJECTED — its config + field
    are discarded and the next round restarts from the prior accepted (best-so-far)
    base, so the accumulated field can never regress.  A no-op round (no flagged
    columns) changes nothing and is vacuously accepted.  ``CampaignResult.accepted``
    records the per-round decision.  Default false preserves the unconditional
    accumulation used by the existing callers/tests.

    ``step_fractions`` is forwarded to each round's
    :func:`run_correction_iteration` line search (backtracking the correction
    magnitude toward the raw LES diagnosis); ``None`` ⇒ the full single-step
    correction.  It composes with the gate: the line search first finds the largest
    improving sub-step, and a round that cannot improve at any fraction is then
    rejected by ``accept_only_if_improved``.

    Returns the final config + every round's :class:`CorrectionResult`.
    ``final_config``/``final_field`` are the LAST ACCEPTED state (not necessarily
    ``iterations[-1]``, which is retained for logging even if rejected).  With the
    gate off every round is accepted, so the overall bias change is
    ``iterations[0].bias.baseline_bias`` (first run) vs
    ``iterations[-1].bias.updated_bias`` (final run); with the gate on, use the last
    ACCEPTED round's ``updated_bias`` for the achieved bias.
    """
    if int(n_iterations) < 0:
        raise ValueError(f"n_iterations must be >= 0, got {n_iterations}.")
    config = initial_config
    base = background if initial_field is None else initial_field
    iterations: list = []
    accepted_flags: list = []
    for i in range(int(n_iterations)):
        result = run_correction_iteration(
            config,
            compare_fn=compare_fn, diagnose_fn=diagnose_fn,
            promotion_key=promotion_key, grid_shape=grid_shape,
            diagnosis_method=diagnosis_method, background=base,
            expected_ncol=expected_ncol,
            les_budget=les_budget, env_scales=env_scales,
            feedback_strategy=feedback_strategy, env_grid_fn=env_grid_fn,
            step_fractions=step_fractions, clip_to_bounds=clip_to_bounds,
        )
        # Monotonic acceptance: keep the round only if it lowered the global bias.
        # A no-op round (no flagged columns) changes nothing → vacuously accepted.
        # A rejected round's worsening config + field are discarded; the next
        # round restarts from the prior accepted base, so the field never regresses.
        accepted = not (
            accept_only_if_improved
            and result.n_corrected > 0
            and not bool(result.bias.improved)
        )
        if accepted:
            config = result.updated_config
            base = result.feedback_field   # accumulate into the next round's base
        iterations.append(result)
        accepted_flags.append(accepted)
        if checkpoint_callback is not None:
            # Pass the ACCEPTED accumulated field so a checkpoint never persists a
            # rejected (worsening) correction.
            checkpoint_callback(int(start_round) + i, result, base)
    # ``base`` is the last ACCEPTED accumulated field (or the round-0 base when no
    # round was accepted / no iterations ran); reshape it to the grid uniformly.
    bg = jnp.asarray(base)
    final_field = (
        jnp.full(grid_shape, bg) if bg.ndim == 0 else bg.reshape(grid_shape)
    )
    return CampaignResult(
        final_config=config, iterations=tuple(iterations),
        final_field=final_field, accepted=tuple(accepted_flags),
    )


def make_compare_fn(
    *,
    reference: ColumnState,
    sigma_full: jax.Array,
    sigma_half: jax.Array,
    lat_deg: jax.Array,
    lon_deg: jax.Array,
    area_weights: jax.Array,
    n_worst: int,
    run_amip_fn: Callable[[Any], ColumnState],
    time_index: int = 0,
    valid_mask: jax.Array | None = None,
    **compare_kwargs: Any,
) -> Callable[[Any], CompareResult]:
    """Build a ``compare_fn`` for :func:`run_correction_iteration` over the REAL
    comparison (:func:`legoesm.training.compare_reanalysis.compare_state_to_reference`).

    Only the AMIP/CMIP *run* is injected (``run_amip_fn(config) -> ColumnState``,
    a model state on the model grid + sigma levels); the scoring + worst-column
    manifest reuse the already-tested compare logic.  ``reference`` is the ERA5
    state regridded to the model grid; ``area_weights`` feed the bias
    aggregation.  ``run_amip_fn`` MUST be deterministic except for ``config``
    (same ERA5 reference, grid, masks across calls) so the measured change
    reflects only the parameter update.  Extra ``compare_kwargs`` (e.g.
    ``error_config`` / ``env_config``) forward to ``compare_state_to_reference``.
    ``model`` is the one comparison argument this adapter injects that is NOT an
    explicit parameter here, so a duplicate would silently shadow it rather than
    raise; it is therefore rejected.  The other injected args (``reference`` /
    ``sigma_full`` / ``lat_deg`` / ``time_index`` / ``n_worst`` / ``valid_mask``
    …) are explicit parameters, so a duplicate is already a Python ``TypeError``.
    """
    if "model" in compare_kwargs:
        raise ValueError(
            "compare_kwargs may not override 'model' — it is the per-config "
            "model state produced by run_amip_fn inside make_compare_fn."
        )

    def compare_fn(config: Any) -> CompareResult:
        model = run_amip_fn(config)
        comparison = compare_state_to_reference(
            model=model, reference=reference,
            sigma_full=sigma_full, sigma_half=sigma_half,
            lat_deg=lat_deg, lon_deg=lon_deg,
            time_index=time_index, n_worst=n_worst,
            valid_mask=valid_mask, **compare_kwargs,
        )
        return CompareResult(
            combined_score=comparison.error_fields.combined_score,
            manifest=comparison.manifest,
            area_weights=area_weights,
            model_ctx=model,
            valid_mask=valid_mask,
        )

    return compare_fn


def _run_line_search(compare_fn, baseline, fractions, make_candidate):
    """Backtracking line search shared by the single- and (future) multi-coefficient
    iterations: try ``fractions`` (already descending) and keep the LARGEST whose
    re-run lowers the global bias (first improving, Armijo-style); if none improve
    report the ``k==0`` (largest) candidate so the monotonic gate rejects it.

    ``make_candidate(frac) -> (config, field_state)`` builds the injected config +
    the field state to store for that step (the per-coefficient blend/clamp/apply
    lives in the caller's closure — including the ``frac == 1.0`` short-circuit — so
    the single-coefficient path stays byte-identical).  Returns
    ``(frac, field_state, config, updated_compare, bias)``.
    """
    chosen = None
    fallback = None
    for k, frac in enumerate(fractions):           # descending: largest step first
        cfg, field_state = make_candidate(frac)
        upd = compare_fn(cfg)
        imp = bias_improvement(
            baseline.combined_score, upd.combined_score,
            baseline.area_weights, valid_mask=baseline.valid_mask,
        )
        candidate = (float(frac), field_state, cfg, upd, imp)
        if k == 0:
            fallback = candidate                   # largest configured step
        if bool(imp.improved):
            chosen = candidate                     # first (largest) improving step
            break
    return chosen if chosen is not None else fallback


def _validate_step_fractions(
    step_fractions: Sequence[float] | None,
) -> tuple[float, ...]:
    """Normalize ``step_fractions`` for the line search: ``None`` ⇒ ``(1.0,)``;
    else each must be in ``(0, 1]``, deduplicated and sorted DESCENDING so the
    backtracking search tries the largest correction first.  Raises on an empty
    sequence or an out-of-range fraction (dispatch hardening)."""
    if step_fractions is None:
        return (1.0,)
    fracs = [float(s) for s in step_fractions]
    if not fracs:
        raise ValueError(
            "step_fractions must be non-empty (or None for the full step)."
        )
    for s in fracs:
        if not (0.0 < s <= 1.0):
            raise ValueError(f"step_fractions must lie in (0, 1]; got {s}.")
    return tuple(sorted(set(fracs), reverse=True))


def run_correction_iteration(
    baseline_config: Any,
    *,
    compare_fn: Callable[[Any], CompareResult],
    diagnose_fn: Callable[[Any, Any], Any],
    promotion_key: str,
    grid_shape: tuple[int, ...],
    diagnosis_method: str = "eddy_diffusivity",
    background: float = 0.0,
    expected_ncol: int | None = None,
    les_budget: int | None = None,
    env_scales: Sequence[float] | None = None,
    feedback_strategy: str = "static",
    env_grid_fn: Callable[[Any], tuple] | None = None,
    step_fractions: Sequence[float] | None = None,
    clip_to_bounds: bool = False,
) -> CorrectionResult:
    """Run one diagnose→correct→verify iteration; report the bias change.

    ``promotion_key`` selects the target coefficient
    (:mod:`legoesm.training.promotable_params`); ``diagnosis_method`` selects the
    LES closure-coefficient (eddy diffusivity / entrainment).  ``background`` is
    the coefficient's production default (columns with no valid diagnosis keep
    it).  Returns the updated config + the :class:`BiasImprovement` (its
    ``improved`` field is the success test) + the worst-column change.

    **LES-cost reduction** (``les_budget``): the LES is the loop's dominant cost
    (§7), so when ``les_budget = K < len(manifest)`` is set, the worst columns are
    environment-clustered (:func:`cluster_columns_by_environment`, ``env_scales``
    forwarded) into ``K`` groups and the LES ``diagnose_fn`` runs ONLY on the
    ``K`` representative columns; each representative's coefficient is then mapped
    to every column in its cluster before the feedback field is assembled.  So
    ``n_corrected`` columns are still corrected but only ``n_diagnosed = K`` LES
    run.  ``les_budget=None`` (default) diagnoses every worst column.

    **Line search** (``step_fractions``): the LES-diagnosed coefficient is an
    estimate from a *different* model (the LES↔GCM gap is the method's central
    challenge, §9), so the raw value can overshoot the bias-minimising GCM
    coefficient.  ``step_fractions`` (e.g. ``(1.0, 0.5, 0.25)``) backtracks the
    correction magnitude: the injected field is ``base + s·(raw − base)`` for the
    LARGEST ``s`` (tried in descending order) whose re-run lowers the global bias;
    the search stops at the first improving ``s`` (Armijo-style, ≤ one re-run per
    fraction).  If none improve, the largest ``s`` is reported with
    ``improved=False`` (so the monotonic gate rejects it, exactly as the single-shot
    path would).  ``step_fractions=None`` (default) ⇒ ``(1.0,)`` — the full-step,
    single-re-run behaviour.  The accepted ``s`` is ``CorrectionResult.step_fraction``.

    **Physical bounds** (``clip_to_bounds``): the LES diagnosis is trusted only
    within the coefficient's registered ``__param_spec__`` range — when true, the
    diagnosed field is clamped to ``(lo, hi)`` (the calibratable bounds, e.g.
    ``C_K ∈ (0.1, 1.2)``) BEFORE the line search, so a degenerate column cannot
    inject an unphysical / destabilizing value and every sub-step stays in range.
    A non-finite reduced diagnosis is ALWAYS treated as invalid (keeps the
    background), independent of this flag.  Default false (library compat); the
    production campaign enables it.

    A grid with no flagged worst columns short-circuits to a no-op correction
    (baseline config unchanged, zero feedback, ``improved=False``, ``step_fraction=0``)
    without a second AMIP run or a config change.

    ``compare_fn`` MUST be deterministic across the two calls — same ERA5
    reference, grid, masks and (seeded) stochastic state — so the measured
    change reflects only the config update, not run-to-run noise.  This is
    host-side (non-jitted) orchestration: it drives AMIP/LES callbacks and loops
    over the manifest in Python by design.  ``expected_ncol`` defaults to
    ``prod(grid_shape)`` (the field IS the grid), so a field/grid length
    mismatch is always caught at the splice; an explicit value that disagrees
    raises.
    """
    ncol = 1
    for d in grid_shape:
        ncol *= int(d)
    if expected_ncol is not None and int(expected_ncol) != ncol:
        raise ValueError(
            f"expected_ncol {expected_ncol} != prod(grid_shape) {ncol}."
        )

    baseline = compare_fn(baseline_config)
    records = list(baseline.manifest)

    if not records:
        # No flagged columns → no-op correction (no second run, no config edit).
        dtype = jnp.asarray(baseline.combined_score).dtype
        noop_bias = bias_improvement(
            baseline.combined_score, baseline.combined_score,
            baseline.area_weights, valid_mask=baseline.valid_mask,
        )
        # No flagged columns ⇒ the feedback field is the unchanged background
        # (scalar → uniform at the score dtype; array → the accumulated field
        # reshaped to the grid, dtype PRESERVED so it cannot diverge from config).
        bg = jnp.asarray(background)
        noop_field = (
            jnp.full(grid_shape, bg.astype(dtype))
            if bg.ndim == 0
            else bg.reshape(grid_shape)
        )
        if clip_to_bounds:
            # Keep the bounds invariant even on a no-op round: the campaign
            # accumulates feedback_field as the next round's base, so an
            # out-of-range carried background must not survive a no-op unclamped.
            noop_field = clip_field_to_promotable_bounds(
                baseline_config, promotion_key, noop_field
            )
        return CorrectionResult(
            updated_config=baseline_config,
            bias=noop_bias,
            worst_column_change=jnp.asarray(0.0, dtype=dtype),
            feedback_field=noop_field,
            n_corrected=0,
            n_diagnosed=0,
            step_fraction=0.0,
        )

    if les_budget is not None and int(les_budget) < len(records):
        if int(les_budget) <= 0:
            raise ValueError(f"les_budget must be > 0, got {les_budget}.")
        # Diagnose only the K environment-representative columns, then map each
        # representative's coefficient to every column in its cluster.
        clusters = cluster_columns_by_environment(
            records, int(les_budget), env_scales=env_scales
        )
        rep_diagnoses = [
            diagnose_fn(records[ri], baseline.model_ctx)
            for ri in clusters.representative_indices
        ]
        diagnoses = [rep_diagnoses[label] for label in clusters.labels]
        n_diagnosed = len(clusters.representative_indices)
    else:
        diagnoses = [diagnose_fn(rec, baseline.model_ctx) for rec in records]
        n_diagnosed = len(records)
    # Feedback field: scatter at the worst columns ("static"), or generalise the
    # diagnosed coefficients to ALL environmentally-similar columns ("environment"
    # — `env_grid_fn(model_ctx)` supplies the full-grid env predictors).
    grid_env, length_scales = (None, None)
    if feedback_strategy == "environment":
        if env_grid_fn is None:
            raise ValueError(
                "feedback_strategy='environment' requires env_grid_fn(model_ctx) "
                "-> (grid_env, length_scales)."
            )
        grid_env, length_scales = env_grid_fn(baseline.model_ctx)
    raw_field = assemble_feedback_field(
        records, diagnoses, grid_shape,
        method=diagnosis_method, background=background,
        strategy=feedback_strategy, grid_env=grid_env, length_scales=length_scales,
    )
    if clip_to_bounds:
        # Clamp the diagnosed field to the coefficient's registered physical bounds
        # BEFORE the line search, so a degenerate LES cannot inject an unphysical
        # value and every convex sub-step (base + s·(clamped − base)) stays in range
        # (base is in-bounds by induction: prior accepted clamped field / default).
        raw_field = clip_field_to_promotable_bounds(
            baseline_config, promotion_key, raw_field
        )
    # Line search over the correction magnitude. ``base`` (the prior accepted
    # field) is broadcast to the grid so the blend ``base + s·(raw − base)`` is a
    # partial step from the current state toward the raw diagnosis; at columns the
    # round did not touch ``raw == base`` so the blend is the identity for any ``s``.
    fractions = _validate_step_fractions(step_fractions)
    # Build base_field at raw_field's dtype: assemble_feedback_field writes the
    # SAME background into raw_field's untouched columns at that dtype, so this
    # makes raw_field − base_field EXACTLY zero there (the blend is the identity
    # for any step) and keeps the full-step (frac==1) and partial paths in one dtype.
    bg = jnp.asarray(background, dtype=raw_field.dtype)
    base_field = (
        jnp.full(grid_shape, bg) if bg.ndim == 0 else bg.reshape(grid_shape)
    )

    def _make_candidate(frac):
        # frac==1.0 short-circuit (NOT a generic blend) keeps the full-step path
        # FP-identical to the pre-refactor behavior; at untouched columns the blend
        # is the identity (raw==base) for any frac.
        field = raw_field if frac == 1.0 else base_field + frac * (raw_field - base_field)
        if clip_to_bounds:
            # Pre-clamping raw_field gives a MEANINGFUL line search (distinct in-range
            # sub-steps); this post-blend clamp is the actual invariant — it also
            # bounds the result when ``base`` itself is out of range (e.g. the guard
            # toggled on mid-campaign, or an out-of-range initial_field), so the
            # injected field can NEVER leave the registered bounds.
            field = clip_field_to_promotable_bounds(
                baseline_config, promotion_key, field
            )
        cfg = apply_feedback_to_scheme(
            baseline_config, promotion_key, field, expected_ncol=ncol
        )
        return cfg, field

    frac, field, updated_config, updated, improvement = _run_line_search(
        compare_fn, baseline, fractions, _make_candidate
    )
    worst_idx = jnp.asarray([int(r.flat_index) for r in records], dtype=jnp.int32)
    worst_change = worst_column_bias_change(
        baseline.combined_score, updated.combined_score, worst_idx
    )
    return CorrectionResult(
        updated_config=updated_config,
        bias=improvement,
        worst_column_change=worst_change,
        feedback_field=field,
        n_corrected=len(records),
        n_diagnosed=n_diagnosed,
        step_fraction=frac,
    )
