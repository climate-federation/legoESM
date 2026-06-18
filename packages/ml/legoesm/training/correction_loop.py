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
from legoesm.training.promotable_params import apply_feedback_to_scheme


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


class CampaignResult(NamedTuple):
    """Outcome of a multi-iteration correction campaign."""

    final_config: Any
    iterations: tuple                  # the per-iteration CorrectionResults
    final_field: jax.Array             # the accumulated per-column field


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
    invoked after EVERY round so the caller can persist the corrected config + the
    field; ``start_round`` offsets ``round_index`` for the checkpoint label on
    resume.

    Returns the final config + every round's :class:`CorrectionResult`.  The
    overall bias change is ``iterations[0].bias.baseline_bias`` (first run) vs
    ``iterations[-1].bias.updated_bias`` (final run).
    """
    if int(n_iterations) < 0:
        raise ValueError(f"n_iterations must be >= 0, got {n_iterations}.")
    config = initial_config
    base = background if initial_field is None else initial_field
    iterations: list = []
    for i in range(int(n_iterations)):
        result = run_correction_iteration(
            config,
            compare_fn=compare_fn, diagnose_fn=diagnose_fn,
            promotion_key=promotion_key, grid_shape=grid_shape,
            diagnosis_method=diagnosis_method, background=base,
            expected_ncol=expected_ncol,
            les_budget=les_budget, env_scales=env_scales,
            feedback_strategy=feedback_strategy, env_grid_fn=env_grid_fn,
        )
        config = result.updated_config
        base = result.feedback_field   # accumulate into the next round's base
        iterations.append(result)
        if checkpoint_callback is not None:
            checkpoint_callback(int(start_round) + i, result, base)
    if iterations:
        final_field = iterations[-1].feedback_field
    elif initial_field is not None:
        final_field = jnp.asarray(initial_field).reshape(grid_shape)
    else:
        bg = jnp.asarray(background)
        final_field = (
            jnp.full(grid_shape, bg) if bg.ndim == 0 else bg.reshape(grid_shape)
        )
    return CampaignResult(
        final_config=config, iterations=tuple(iterations), final_field=final_field
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

    A grid with no flagged worst columns short-circuits to a no-op correction
    (baseline config unchanged, zero feedback, ``improved=False``) without a
    second AMIP run or a config change.

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
        return CorrectionResult(
            updated_config=baseline_config,
            bias=noop_bias,
            worst_column_change=jnp.asarray(0.0, dtype=dtype),
            feedback_field=noop_field,
            n_corrected=0,
            n_diagnosed=0,
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
    field = assemble_feedback_field(
        records, diagnoses, grid_shape,
        method=diagnosis_method, background=background,
        strategy=feedback_strategy, grid_env=grid_env, length_scales=length_scales,
    )
    updated_config = apply_feedback_to_scheme(
        baseline_config, promotion_key, field, expected_ncol=ncol
    )

    updated = compare_fn(updated_config)
    improvement = bias_improvement(
        baseline.combined_score, updated.combined_score,
        baseline.area_weights, valid_mask=baseline.valid_mask,
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
    )
