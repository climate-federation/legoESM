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
  column LES (``legoesm.atmosphere.dynamics.les.column_les.process_column``).

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
    assert_model_state_finite,
    compare_state_to_reference,
    model_state_is_finite,
)
from legoesm.training.feedback_assembly import (
    assemble_feedback_field,
    count_valid_diagnoses,
    count_valid_multi_diagnoses,
)
from legoesm.training.promotable_params import (
    PROMOTABLE_FIELDS,
    apply_feedback_to_scheme,
    clip_field_to_promotable_bounds,
)


def _clubb_field_for_promotion_key(promotion_key: str) -> str | None:
    """Resolve a promotion key to its CLUBB-lite coefficient field for env-kernel
    export, or ``None`` when the key targets a NON-CLUBB scheme.

    The cross-resolution environment kernel
    (:class:`legoesm.training.deploy_correction.EnvKernel`) is a CLUBB-turbulence
    deploy artifact, so only ``clubb_lite_*`` keys produce one.  Resolves through
    the :data:`promotable_params.PROMOTABLE_FIELDS` registry (NOT string-stripping),
    so a malformed / unregistered key resolves to ``None`` rather than silently
    fabricating a field name.
    """
    pf = PROMOTABLE_FIELDS.get(promotion_key)
    if pf is None or not pf.scheme.startswith("CLUBB"):
        return None
    return pf.field


def _assert_resume_config_carries_field(
    initial_config: Any, promotion_key: str, initial_field: Any
) -> None:
    """RESUME CONTRACT (fail-loud): when a campaign is RESUMED with an accumulated
    ``initial_field``, ``initial_config`` MUST already carry that field in its
    promotable leaf — because the FIRST resumed round's ``compare_fn`` reads the
    CONFIG (not the feedback ``base``).  A scalar / stale ``initial_config`` paired
    with an array ``initial_field`` would silently RE-RANK that round against the
    un-promoted (uniform) field and mis-select the worst columns — a SILENT resume
    desync, the exact class of past 'checkpoint resume desync' bug.  The CLI's
    ``_load_single_resume`` rebuilds the config FROM the field (so a correctly-resumed
    campaign passes); this guards a DIRECT ``run_*_campaign`` caller — or a future
    regression in that rebuild — that forgets to.  No-op for a fresh run
    (``initial_field is None``) or an unregistered key (validated elsewhere)."""
    if initial_field is None:
        return
    pf = PROMOTABLE_FIELDS.get(promotion_key)
    if pf is None:
        return
    leaf = getattr(initial_config, pf.field, None)
    field_arr = jnp.asarray(initial_field)
    field_flat = field_arr.reshape(-1)
    if leaf is None:
        bad, got = True, "absent"
    else:
        leaf_arr = jnp.asarray(leaf)
        # A SCALAR (un-promoted) leaf can NEVER carry a per-column field — even at
        # ncol=1, where a (1,)-vs-(1,) reshape-equality would FALSELY pass, the first
        # resumed round's compare reads config.<field>'s NDIM to branch (scalar ⇒ the
        # round-0 uniform path), so reject the ndim mismatch explicitly (codex iter 401).
        if leaf_arr.ndim == 0 and field_arr.ndim > 0:
            bad, got = True, "scalar (un-promoted)"
        elif leaf_arr.reshape(-1).shape != field_flat.shape:
            bad, got = True, f"shape {tuple(leaf_arr.shape)}"
        elif not bool(jnp.array_equal(leaf_arr.reshape(-1), field_flat)):
            bad, got = True, f"shape {tuple(leaf_arr.shape)}, values differ"
        else:
            bad, got = False, ""
    if bad:
        raise ValueError(
            f"run_correction_campaign resume contract violated: the accumulated "
            f"initial_field for {promotion_key!r} {tuple(field_flat.shape)} is NOT "
            f"carried by initial_config.{pf.field} ({got}). The first resumed round's "
            f"compare_fn reads the CONFIG, so resume with the config promoted FROM the "
            f"field — replace the scheme leaf with initial_field.reshape(-1), as the "
            f"CLI's _load_single_resume does. Pass a consistent (config, field) pair.")


class CompareResult(NamedTuple):
    """One AMIP-vs-ERA5 comparison: the score field, manifest, and weights."""

    combined_score: jax.Array          # per-column combined score (grid-shaped)
    manifest: Sequence[Any]            # worst-column ColumnRecords
    area_weights: jax.Array            # per-column quadrature weights
    model_ctx: Any = None              # opaque state handed to diagnose_fn
    valid_mask: jax.Array | None = None  # optional column validity for aggregation
    error_fields: Any = None           # per-variable ColumnErrorFields (None for mocks)
    have_precip: bool = False          # both states carried precip (precip in the score)


class CorrectionResult(NamedTuple):
    """Outcome of one correction iteration."""

    updated_config: Any
    bias: BiasImprovement              # baseline vs updated global bias
    worst_column_change: jax.Array     # mean bias reduction at the worst columns
    feedback_field: jax.Array          # the assembled per-column field
    n_corrected: int                   # number of worst columns corrected (all flagged)
    n_diagnosed: int = 0               # number of LES diagnoses run (= n_corrected, or
    #                                    K representatives when clustering w/ les_budget)
    n_diagnoses_valid: int = 0         # of the n_corrected columns, how many received a
    #                                    VALID LES diagnosis (≥1 valid level + finite);
    #                                    the rest kept the background. 0 with
    #                                    n_corrected>0 ⇒ every LES spin-off was rejected
    #                                    by the realism gate (e.g. too short to develop
    #                                    turbulence) → no correction this round.
    step_fraction: float = 1.0         # the accepted line-search step toward the raw
    #                                    diagnosis (1.0 = full; 0.0 = no-op round)
    env_kernel: Any = None             # the exported RAW environment kernel
    #   (deploy_correction.EnvKernel) for an "environment"-strategy CLUBB round —
    #   the grid-AGNOSTIC env→coefficient regression for cross-resolution deploy.
    #   None for static/non-CLUBB/no-op rounds. RAW = full-step, unclipped, scalar
    #   background: deploy re-applies it fresh and runs its OWN line search + gate,
    #   so it is NOT this round's post-line-search/clipped/accumulated feedback_field.
    per_variable_bias: Any = None      # PerVariableBiasImprovement (T/q_v/wind/precip
    #   baseline→updated in physical units + per-variable improved flags) — exposes a
    #   round that lowered the COMBINED score by trading variables off. None when the
    #   compare_fn carried no error_fields (a mock).


class CampaignResult(NamedTuple):
    """Outcome of a multi-iteration correction campaign."""

    final_config: Any
    iterations: tuple                  # the per-iteration CorrectionResults
    final_field: jax.Array             # the accumulated per-column field
    accepted: tuple = ()               # per-round bool: was the round kept?
                                       # (all True unless accept_only_if_improved)
    stop_reason: str = "max_iterations"  # "max_iterations" | "converged" |
    #                                    "no_valid_diagnoses" (early stop)


def last_accepted_env_kernel(campaign_result):
    """The ``env_kernel`` of the LAST ACCEPTED round that produced one, or ``None``.

    The cross-resolution deploy artifact (the RAW env→coefficient ``EnvKernel``,
    iter 70).  Under the monotonic gate a REJECTED final round's kernel is
    inconsistent with the accepted final field/config, so the transferable kernel
    is the last ACCEPTED round's.  ``accepted`` is populated EVERY round by
    :func:`run_correction_campaign` (kept rounds flagged True); an EMPTY tuple — a
    hand-built or zero-round :class:`CampaignResult` — is treated as all-kept.
    Returns ``None`` for a static / non-CLUBB / no-op campaign (no round produced a
    kernel).  Shared by the campaign CLI and the cross-resolution OSSE.
    """
    accepted = list(campaign_result.accepted)
    kernel = None
    for i, it in enumerate(campaign_result.iterations):
        kept = accepted[i] if i < len(accepted) else True
        if kept and getattr(it, "env_kernel", None) is not None:
            kernel = it.env_kernel
    return kernel


class CorrectionSpec(NamedTuple):
    """One coefficient to correct in a SIMULTANEOUS multi-coefficient round.

    ``promotion_key`` selects the promotable config field
    (:mod:`legoesm.training.promotable_params`); ``diagnosis_method`` selects the
    LES diagnosis (a key of the ``{method: diagnosis}`` dict the multi diagnose_fn
    returns); ``background`` is the coefficient's production default / accumulated
    base.  All specs in a round share one LES run and one line-search fraction.
    """

    promotion_key: str
    diagnosis_method: str
    background: Any


class MultiCorrectionResult(NamedTuple):
    """Outcome of one SIMULTANEOUS multi-coefficient correction iteration."""

    updated_config: Any
    bias: BiasImprovement              # baseline vs combined-update global bias
    worst_column_change: jax.Array
    feedback_fields: dict              # {promotion_key: per-column field}
    n_corrected: int
    n_diagnosed: int = 0
    n_diagnoses_valid: int = 0         # columns valid for ≥1 requested coefficient (0
    #                                    with n_corrected>0 ⇒ all LES rejected this round)
    step_fraction: float = 1.0         # combined: the shared fraction; sequential:
    #                                    the largest accepted per-coefficient fraction
    step_fractions_by_key: dict | None = None  # sequential mode: per-coefficient
    #                                            accepted fraction (0.0 = rejected)
    per_variable_bias: Any = None      # PerVariableBiasImprovement (see CorrectionResult)


class MultiCampaignResult(NamedTuple):
    """Outcome of a multi-iteration SIMULTANEOUS multi-coefficient campaign."""

    final_config: Any
    iterations: tuple                  # the per-round MultiCorrectionResults
    final_fields: dict                 # {promotion_key: accumulated per-column field}
    accepted: tuple = ()
    stop_reason: str = "max_iterations"  # "max_iterations" | "converged" |
    #                                    "no_valid_diagnoses" (early stop)


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
    bias_tol: float | None = None,
    patience: int = 2,
    stop_on_no_valid_diagnoses: bool = True,
    global_reduce: Callable[[jax.Array], jax.Array] | None = None,
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

    **Convergence early-stopping** (``bias_tol``): when set, the campaign STOPS once
    ``patience`` (default 2) consecutive rounds make no meaningful progress — a
    rejected round, or an accepted round whose area-weighted bias reduction is below
    ``bias_tol`` — so the multi-day HPC campaign does not waste expensive rounds once
    the bias has plateaued.  ``CampaignResult.stop_reason`` is ``"converged"`` then,
    else ``"max_iterations"``.  ``None`` (default) runs all ``n_iterations`` rounds.

    **Dry-LES early-abort** (``stop_on_no_valid_diagnoses``, default ``True``): the
    campaign also STOPS (``stop_reason="no_valid_diagnoses"``) once ``patience``
    consecutive rounds FLAG worst columns but produce ZERO globally-valid LES
    diagnoses — the spin-off LES is developing no turbulence (too short / unforced),
    so every further multi-day round is wasted; this is independent of ``bias_tol``
    and is checked FIRST so the cause is reported correctly.  The dry test is on the
    GLOBAL corrected + valid counts (collective-safe under MPI).

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
    if bias_tol is not None and int(patience) < 1:
        raise ValueError(
            f"patience must be >= 1 when bias_tol is set, got {patience}.")
    config = initial_config
    # The contract only bites once a round actually runs (its compare reads the
    # config); n_iterations=0 just reads the field back, so skip the guard there.
    if int(n_iterations) >= 1:
        _assert_resume_config_carries_field(initial_config, promotion_key, initial_field)
    base = background if initial_field is None else initial_field
    iterations: list = []
    accepted_flags: list = []
    no_progress = 0
    no_valid_streak = 0
    stop_reason = "max_iterations"
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
            # Forward the ORIGINAL scalar production default as the env-kernel
            # out-of-hull fallback even once ``base`` is the accumulated array,
            # so each round's exported EnvKernel has a scalar (not array) fallback.
            kernel_background=background,
            global_reduce=global_reduce,
        )
        # Monotonic acceptance: keep the round only if it lowered the global bias.
        # A no-op round (no flagged columns) changes nothing → vacuously accepted.
        # A rejected round's worsening config + field are discarded; the next
        # round restarts from the prior accepted base, so the field never regresses.
        # DISTRIBUTED: gate on the GLOBAL corrected count + the GLOBAL bias verdict
        # (both already global) so EVERY rank accepts/rejects identically — a
        # rank-local ``n_corrected`` would let a rank owning no columns vacuously
        # accept a round its peers reject, diverging the campaign state (iter 88).
        accepted = not (
            accept_only_if_improved
            and _global_count(result.n_corrected, global_reduce) > 0
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
        # ABORT a DRY campaign: `patience` consecutive rounds that FLAGGED columns but
        # produced ZERO valid LES diagnoses globally ⇒ the spin-off LES is developing
        # no turbulence (too short / unforced), so every further multi-day round is
        # wasted. Checked BEFORE convergence so the stop_reason is the CORRECT cause
        # ("no_valid_diagnoses", not "converged"). COLLECTIVE-SAFE: the dry test is on
        # the GLOBAL corrected + valid counts (every rank sees the same totals → all
        # break or none; a rank-local count would diverge and deadlock — iter 88/98).
        if stop_on_no_valid_diagnoses and int(patience) >= 1:
            global_corrected = _global_count(result.n_corrected, global_reduce)
            global_valid = _global_count(result.n_diagnoses_valid, global_reduce)
            dry = global_corrected > 0 and global_valid == 0
            no_valid_streak = 0 if not dry else no_valid_streak + 1
            if no_valid_streak >= int(patience):
                stop_reason = "no_valid_diagnoses"
                break
        # Convergence early-stopping: stop once `patience` consecutive rounds make
        # no meaningful progress (rejected, or improving by < bias_tol) — the bias
        # has plateaued, so further expensive rounds are wasted.
        if bias_tol is not None:
            no_progress = (
                0 if _round_made_progress(accepted, result.bias, bias_tol)
                else no_progress + 1
            )
            if no_progress >= int(patience):
                stop_reason = "converged"
                break
    # ``base`` is the last ACCEPTED accumulated field (or the round-0 base when no
    # round was accepted / no iterations ran); reshape it to the grid uniformly.
    bg = jnp.asarray(base)
    final_field = (
        jnp.full(grid_shape, bg) if bg.ndim == 0 else bg.reshape(grid_shape)
    )
    return CampaignResult(
        final_config=config, iterations=tuple(iterations),
        final_field=final_field, accepted=tuple(accepted_flags),
        stop_reason=stop_reason,
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
    manifest_reducer: Callable[[list], list] | None = None,
    coordinate: Any = None,
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

    ``manifest_reducer`` (optional, ``list[ColumnRecord] -> list[ColumnRecord]``)
    post-processes the worst-column manifest; ``None`` (default) is a no-op.  Under
    distributed MPAS it is
    :func:`legoesm.training.distributed_manifest.gather_global_worst_columns`
    (closed over the rank's ``local_to_global`` map) — the cross-rank global top-k
    that turns per-rank owned rankings into the rank's share of the GLOBAL worst
    ``n_worst`` (iter 87).  Single-process runs leave it ``None`` (the local
    manifest already IS the global top-k).

    ``coordinate`` (optional) is the model's vertical coordinate object (``driver.sigma`` —
    a ``SigmaCoordinate`` or ``HybridSigmaPressureCoordinate``).  When supplied, each run's
    LAYER PRESSURES are computed from it (``coordinate.pressure_at_full/half(model.p_s)``) and
    passed to the compare, so the bias is mass-weighted by the TRUE per-column layer mass —
    correct for a HYBRID coordinate (the default; iter 337/338) and byte-identical for
    pure-sigma.  ``None`` keeps the original pure-sigma weighting (exact only at p_s = p_ref).
    """
    if "model" in compare_kwargs:
        raise ValueError(
            "compare_kwargs may not override 'model' — it is the per-config "
            "model state produced by run_amip_fn inside make_compare_fn."
        )
    if coordinate is not None and ("p_full" in compare_kwargs or "p_half" in compare_kwargs):
        raise ValueError(
            "make_compare_fn: pass coordinate= OR p_full/p_half (not both) — coordinate "
            "computes the per-run layer pressures from model.p_s itself."
        )

    def compare_fn(config: Any) -> CompareResult:
        model = run_amip_fn(config)
        # When the model's vertical COORDINATE is supplied, compute the per-run layer
        # pressures from it (``model.p_s``) so the compare weights the bias by the TRUE layer
        # mass — correct for a HYBRID coordinate (the default; iter 337/338) and byte-identical
        # for pure-sigma (``pressure_at_full = sigma·p_s``).  Without it the compare falls back
        # to pure-sigma pressures (the original behaviour, exact only at p_s = p_ref).
        p_overrides: dict[str, Any] = {}
        if coordinate is not None:
            p_overrides = {
                "p_full": coordinate.pressure_at_full(model.p_s),
                "p_half": coordinate.pressure_at_half(model.p_s),
            }
        comparison = compare_state_to_reference(
            model=model, reference=reference,
            sigma_full=sigma_full, sigma_half=sigma_half,
            lat_deg=lat_deg, lon_deg=lon_deg,
            time_index=time_index, n_worst=n_worst,
            valid_mask=valid_mask, **p_overrides, **compare_kwargs,
        )
        # DISTRIBUTED hook: reduce the rank-local (owned-only, iter-86 valid_mask)
        # manifest to this rank's owned subset of the GLOBAL n_worst worst columns
        # (cross-rank top-k, iter-87 distributed_manifest.gather_global_worst_columns)
        # so the campaign diagnoses each globally worst cell on exactly one rank.
        # None (single-process default) ⇒ the local manifest already IS the global.
        manifest = comparison.manifest
        if manifest_reducer is not None:
            manifest = list(manifest_reducer(list(manifest)))
        return CompareResult(
            combined_score=comparison.error_fields.combined_score,
            manifest=manifest,
            area_weights=area_weights,
            model_ctx=model,
            valid_mask=valid_mask,
            error_fields=comparison.error_fields,
            have_precip=comparison.have_precip,
        )

    return compare_fn


def _run_line_search(compare_fn, baseline, fractions, make_candidate, *,
                     global_reduce=None):
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
            global_reduce=global_reduce,
        )
        # FAIL-SAFE on a blown-up re-run: a non-finite STATE is NaN-masked to a
        # spurious ≈0 bias that reads as 'improved' (iter 392) — force NOT-improved so
        # it is neither chosen here NOR accepted by the monotonic gate (the k==0
        # fallback then carries this verdict, so the round is rejected, not shipped).
        if not model_state_is_finite(upd.model_ctx):
            imp = imp._replace(improved=jnp.asarray(False))
        candidate = (float(frac), field_state, cfg, upd, imp)
        if k == 0:
            fallback = candidate                   # largest configured step
        if bool(imp.improved):
            chosen = candidate                     # first (largest) improving step
            break
    return chosen if chosen is not None else fallback


def _diagnose_columns(records, diagnose_fn, model_ctx, les_budget, env_scales):
    """Diagnose each worst column, returning ``(per-column diagnoses, n_diagnosed,
    run_diagnoses)``.

    With ``les_budget = K < len(records)`` the worst columns are environment-
    clustered into ``K`` groups, the LES ``diagnose_fn`` runs ONLY on the ``K``
    representatives, and each representative's result is mapped to its cluster.
    ``diagnose_fn``'s return type is opaque (a single diagnosis OR a
    ``{method: diagnosis}`` dict) — the plumbing is identical, so this is shared by
    the single- and multi-coefficient iterations.

    ``run_diagnoses`` is the list of the ``n_diagnosed`` ACTUAL LES-run diagnoses
    (the ``K`` representatives when clustering, else one per column) — DISTINCT from
    the cluster-EXPANDED per-column ``diagnoses``.  Validity counts use it so the
    valid count is over LES RUNS (≤ ``n_diagnosed``), not the duplicated columns.
    """
    if les_budget is not None and int(les_budget) < len(records):
        if int(les_budget) <= 0:
            raise ValueError(f"les_budget must be > 0, got {les_budget}.")
        clusters = cluster_columns_by_environment(
            records, int(les_budget), env_scales=env_scales
        )
        rep = [
            diagnose_fn(records[ri], model_ctx)
            for ri in clusters.representative_indices
        ]
        return ([rep[label] for label in clusters.labels],
                len(clusters.representative_indices), rep)
    per_column = [diagnose_fn(rec, model_ctx) for rec in records]
    return per_column, len(records), per_column


def _env_grid_predictors(feedback_strategy, env_grid_fn, model_ctx):
    """``(grid_env, length_scales)`` for the env-generalization feedback strategy
    (``None, None`` for the static scatter); shared by single + multi."""
    if feedback_strategy == "environment":
        if env_grid_fn is None:
            raise ValueError(
                "feedback_strategy='environment' requires env_grid_fn(model_ctx) "
                "-> (grid_env, length_scales)."
            )
        return env_grid_fn(model_ctx)
    return None, None


def _raw_and_base_field(
    records, diagnoses, grid_shape, *, method, background, strategy,
    grid_env, length_scales, clip_to_bounds, config, promotion_key,
):
    """Assemble (+ optionally bounds-clamp) the diagnosed ``raw_field`` for ONE
    coefficient and the dtype-matched accumulated ``base_field`` — the per-
    coefficient piece shared by the single iteration and each multi spec.
    """
    raw_field = assemble_feedback_field(
        records, diagnoses, grid_shape,
        method=method, background=background,
        strategy=strategy, grid_env=grid_env, length_scales=length_scales,
    )
    if clip_to_bounds:
        # Clamp BEFORE the line search so every convex sub-step stays in range
        # (base is in-bounds by induction: prior accepted clamped field / default).
        raw_field = clip_field_to_promotable_bounds(config, promotion_key, raw_field)
    # base_field at raw_field's dtype: assemble writes the SAME background into the
    # untouched columns at that dtype, so raw − base is EXACTLY zero there (the
    # blend is the identity for any step) and full-step/partial share one dtype.
    bg = jnp.asarray(background, dtype=raw_field.dtype)
    base_field = (
        jnp.full(grid_shape, bg) if bg.ndim == 0 else bg.reshape(grid_shape)
    )
    return raw_field, base_field


def _round_made_progress(accepted: bool, bias, bias_tol: float) -> bool:
    """Whether a campaign round counts as PROGRESS for convergence early-stopping:
    it was accepted AND its area-weighted absolute bias reduction ≥ ``bias_tol``.
    A rejected round, or an accepted round whose improvement is below the
    tolerance, is "no progress" — ``patience`` consecutive ⇒ converged."""
    return accepted and float(bias.absolute_reduction) >= bias_tol


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


def _global_count(n_local: int, global_reduce) -> int:
    """Total selected columns across ranks (SUM of per-rank counts); ``== n_local``
    single-process (``global_reduce=None``).

    Drives the COLLECTIVE no-op gate so every rank takes the SAME branch (all
    no-op, or all proceed-and-re-run).  A rank-local ``len(records)`` gate would
    let a rank owning NONE of the globally-selected columns skip the second
    ``compare_fn`` (model re-run) while peers re-run, mismatching the model's MPI
    collectives → deadlock (Codex iter-87).  Owned subsets are disjoint, so the SUM
    of per-rank counts is the true global count.
    """
    if global_reduce is None:
        return int(n_local)
    return int(global_reduce(jnp.asarray(int(n_local), dtype=jnp.int32)))


def _maybe_per_variable_bias(baseline, updated, global_reduce):
    """Per-variable global-bias improvement (baseline→updated) for the round, or
    ``None`` when the ``compare_fn`` carried no per-variable ``error_fields`` (a mock
    that returns only the combined score).  Uses the SAME ``area_weights`` /
    ``valid_mask`` / ``global_reduce`` as the combined :class:`BiasImprovement` and
    the compare's ``have_precip`` (so precip is reported only when both states carried
    it, else ``NaN``).  A no-op round passes ``updated=baseline`` (all variables equal,
    not improved — mirroring the combined no-op bias)."""
    if getattr(baseline, "error_fields", None) is None or \
            getattr(updated, "error_fields", None) is None:
        return None
    from legoesm.training.bias_metrics import per_variable_bias_improvement
    return per_variable_bias_improvement(
        baseline.error_fields, updated.error_fields, baseline.area_weights,
        have_precip=bool(baseline.have_precip),
        valid_mask=baseline.valid_mask, global_reduce=global_reduce)


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
    kernel_background: float | None = None,
    global_reduce: Callable[[jax.Array], jax.Array] | None = None,
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

    **Env-kernel export** (``CorrectionResult.env_kernel``): for an
    ``feedback_strategy="environment"`` round whose ``promotion_key`` targets a
    CLUBB-lite coefficient, the result carries the RAW environment kernel
    (:class:`legoesm.training.deploy_correction.EnvKernel`) — the grid-AGNOSTIC
    env→coefficient regression — so a cheap low-res campaign deploys on an
    expensive high-res run by environmental similarity.  ``kernel_background`` is
    its scalar out-of-hull fallback (the production default); when ``background``
    is an accumulated per-column array it is REQUIRED (else a ``ValueError`` — no
    silent fallback).  ``env_kernel`` is ``None`` for static / non-CLUBB / no-op
    rounds.  SINGLE-coefficient only: the SIMULTANEOUS multi-coefficient path
    (:func:`run_multi_correction_iteration`) does not export a kernel.

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
    # Fail loud if the BASELINE (current-config) run DIVERGED — a non-finite model state
    # would otherwise flow into compare's score_columns (NaN-masked per level) and produce a
    # garbage bias the campaign cannot improve, wasting a multi-day HPC run. A diverged
    # CANDIDATE/line-search run is handled by the gate instead (its non-finite bias is never
    # < baseline → rejected, not fatal).
    if isinstance(baseline.model_ctx, ColumnState):
        assert_model_state_finite(baseline.model_ctx, name="baseline model run")
    records = list(baseline.manifest)

    if _global_count(len(records), global_reduce) == 0:
        # No flagged columns ANYWHERE → no-op correction (no second run, no config
        # edit) on EVERY rank in lockstep (the gate is GLOBAL, not rank-local).
        dtype = jnp.asarray(baseline.combined_score).dtype
        noop_bias = bias_improvement(
            baseline.combined_score, baseline.combined_score,
            baseline.area_weights, valid_mask=baseline.valid_mask,
            global_reduce=global_reduce,
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
            per_variable_bias=_maybe_per_variable_bias(baseline, baseline, global_reduce),
        )

    diagnoses, n_diagnosed, run_diagnoses = _diagnose_columns(
        records, diagnose_fn, baseline.model_ctx, les_budget, env_scales
    )
    # Count validity over the LES RUNS (representatives), so n_diagnoses_valid ≤
    # n_diagnosed even when clustering expands K runs across more columns.
    n_diagnoses_valid = count_valid_diagnoses(run_diagnoses, diagnosis_method)
    # Feedback field: scatter at the worst columns ("static"), or generalise the
    # diagnosed coefficients to ALL environmentally-similar columns ("environment").
    grid_env, length_scales = _env_grid_predictors(
        feedback_strategy, env_grid_fn, baseline.model_ctx
    )
    raw_field, base_field = _raw_and_base_field(
        records, diagnoses, grid_shape,
        method=diagnosis_method, background=background, strategy=feedback_strategy,
        grid_env=grid_env, length_scales=length_scales,
        clip_to_bounds=clip_to_bounds, config=baseline_config,
        promotion_key=promotion_key,
    )
    fractions = _validate_step_fractions(step_fractions)

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
        compare_fn, baseline, fractions, _make_candidate, global_reduce=global_reduce
    )
    worst_idx = jnp.asarray([int(r.flat_index) for r in records], dtype=jnp.int32)
    worst_change = worst_column_bias_change(
        baseline.combined_score, updated.combined_score, worst_idx
    )
    env_kernel = _maybe_build_env_kernel(
        feedback_strategy, promotion_key, records, diagnoses, diagnosis_method,
        length_scales, background, kernel_background,
    )
    return CorrectionResult(
        updated_config=updated_config,
        bias=improvement,
        worst_column_change=worst_change,
        feedback_field=field,
        n_corrected=len(records),
        n_diagnosed=n_diagnosed,
        n_diagnoses_valid=n_diagnoses_valid,
        step_fraction=frac,
        env_kernel=env_kernel,
        per_variable_bias=_maybe_per_variable_bias(baseline, updated, global_reduce),
    )


def _maybe_build_env_kernel(
    feedback_strategy, promotion_key, records, diagnoses, diagnosis_method,
    length_scales, background, kernel_background,
):
    """Build the exported RAW environment kernel for an "environment"-strategy
    CLUBB round, or ``None`` (static / non-CLUBB strategy → the cross-resolution
    env kernel does not apply).  The out-of-hull fallback MUST be a scalar (the
    production default): use ``kernel_background`` when given, else ``background``
    when it is still scalar; a non-scalar ``background`` (the accumulated per-column
    field of a multi-round campaign) with no ``kernel_background`` RAISES rather than
    silently picking an arbitrary fallback (``run_correction_campaign`` forwards its
    original scalar default).  The kernel captures the full-step, unclipped raw
    env→coefficient regression — deploy re-applies it and runs its own gate.
    """
    clubb_field = _clubb_field_for_promotion_key(promotion_key)
    if feedback_strategy != "environment" or clubb_field is None:
        return None
    from legoesm.training.deploy_correction import build_env_kernel
    from legoesm.training.feedback_assembly import reduce_column_diagnosis

    # A round whose LES diagnoses are ALL invalid (e.g. no turbulence developed in
    # any worst column) yields no transferable kernel.  The kernel is an OPTIONAL
    # deploy EXPORT, so skip it (return None) — the campaign must NOT crash (the
    # feedback assembly already keeps the background for invalid columns).  Without
    # this, build_env_kernel's strict "no VALID samples" raise would abort the run.
    if not any(
        bool(jnp.asarray(reduce_column_diagnosis(d, diagnosis_method)[1]).any())
        for d in diagnoses
    ):
        return None

    bg = jnp.asarray(background)
    if kernel_background is not None:
        kbg = float(kernel_background)
    elif bg.ndim == 0:
        kbg = float(bg)
    else:
        raise ValueError(
            "env-kernel export needs a SCALAR out-of-hull fallback: pass "
            "kernel_background (the coefficient's production default) when "
            "'background' is an accumulated per-column array — "
            "run_correction_campaign forwards its scalar default automatically."
        )
    return build_env_kernel(
        records, diagnoses, diagnosis_method,
        length_scales=length_scales, field=clubb_field, background=kbg,
    )


def run_multi_correction_iteration(
    baseline_config: Any,
    specs: Sequence[CorrectionSpec],
    *,
    compare_fn: Callable[[Any], CompareResult],
    diagnose_fn: Callable[[Any, Any], Any],
    grid_shape: tuple[int, ...],
    expected_ncol: int | None = None,
    les_budget: int | None = None,
    env_scales: Sequence[float] | None = None,
    feedback_strategy: str = "static",
    env_grid_fn: Callable[[Any], tuple] | None = None,
    step_fractions: Sequence[float] | None = None,
    clip_to_bounds: bool = False,
    sequential: bool = False,
    global_reduce: Callable[[jax.Array], jax.Array] | None = None,
) -> MultiCorrectionResult:
    """Correct SEVERAL coefficients SIMULTANEOUSLY from ONE LES run per column.

    Each :class:`CorrectionSpec` is corrected in the same round: the (multi)
    ``diagnose_fn(record, model_ctx)`` returns a ``{method: diagnosis}`` dict per
    worst column (one LES run — share with :func:`run_column_les_pipeline`'s
    ``methods=``), each spec assembles + bounds-clamps its own ``raw_field``, and
    the line search applies ALL specs' blends at the SAME fraction ``s`` (a single
    trust-region step), measuring the COMBINED bias ONCE.  The corrected config is
    ``apply(apply(base, key₁, f₁), key₂, f₂)`` — the specs target independent
    config slots, so order does not matter.

    **Two modes.**  ``sequential=False`` (default, *combined*): one fraction across
    all coefficients (the natural scalar-trust-region extension), the monotonic gate
    accepts the round iff the COMBINED bias improved.  Consequence: if one
    coefficient helps and another hurts, the whole step is rejected (the beneficial
    partial update is lost) — conservative and cheap.  ``sequential=True``
    (*staged*, block coordinate descent): the coefficients are corrected IN SPEC
    ORDER, each with its OWN line search + gate against the running state, so a
    coefficient is kept only if IT improves and the others are unaffected.  This
    handles COUPLED coefficients (e.g. C_eps sets the GCM wp2 that C_K then uses, so
    order C_eps→C_K→Pr_t lets wp2 converge first) and avoids the rejected-whole loss
    — at the cost of one line search per coefficient.  ``step_fractions_by_key``
    reports the accepted per-coefficient fraction (0.0 = that coefficient rejected).
    Reuses the shared
    :func:`_run_line_search`, :func:`_diagnose_columns`, :func:`_env_grid_predictors`
    and :func:`_raw_and_base_field`, so the numerics match the single path.
    """
    if not specs:
        raise ValueError("specs must be a non-empty sequence of CorrectionSpec.")
    keys = [s.promotion_key for s in specs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"specs must have distinct promotion_keys; got {keys}.")
    ncol = 1
    for d in grid_shape:
        ncol *= int(d)
    if expected_ncol is not None and int(expected_ncol) != ncol:
        raise ValueError(f"expected_ncol {expected_ncol} != prod(grid_shape) {ncol}.")

    baseline = compare_fn(baseline_config)
    # Fail loud if the BASELINE (current-config) run DIVERGED — a non-finite model state
    # would otherwise flow into compare's score_columns (NaN-masked per level) and produce a
    # garbage bias the campaign cannot improve, wasting a multi-day HPC run. A diverged
    # CANDIDATE/line-search run is handled by the gate instead (its non-finite bias is never
    # < baseline → rejected, not fatal).
    if isinstance(baseline.model_ctx, ColumnState):
        assert_model_state_finite(baseline.model_ctx, name="baseline model run")
    records = list(baseline.manifest)

    if _global_count(len(records), global_reduce) == 0:
        # GLOBAL no-op (no flagged column on ANY rank) — lockstep across ranks.
        dtype = jnp.asarray(baseline.combined_score).dtype
        noop_bias = bias_improvement(
            baseline.combined_score, baseline.combined_score,
            baseline.area_weights, valid_mask=baseline.valid_mask,
            global_reduce=global_reduce,
        )
        fields = {}
        for s in specs:
            bg = jnp.asarray(s.background)
            f = (jnp.full(grid_shape, bg.astype(dtype))
                 if bg.ndim == 0 else bg.reshape(grid_shape))
            if clip_to_bounds:
                f = clip_field_to_promotable_bounds(baseline_config, s.promotion_key, f)
            fields[s.promotion_key] = f
        return MultiCorrectionResult(
            updated_config=baseline_config, bias=noop_bias,
            worst_column_change=jnp.asarray(0.0, dtype=dtype),
            feedback_fields=fields, n_corrected=0, n_diagnosed=0, step_fraction=0.0,
            per_variable_bias=_maybe_per_variable_bias(baseline, baseline, global_reduce),
        )

    diagnoses, n_diagnosed, run_diagnoses = _diagnose_columns(
        records, diagnose_fn, baseline.model_ctx, les_budget, env_scales
    )
    # The MULTI diagnose_fn must return a {method: diagnosis} dict per column
    # covering every spec's method (one LES run → many diagnoses); fail clearly
    # rather than with a bare KeyError/TypeError deep in the per-spec extraction.
    required = {s.diagnosis_method for s in specs}
    for i, d in enumerate(diagnoses):
        if not isinstance(d, dict):
            raise TypeError(
                "multi-correction diagnose_fn must return a {method: diagnosis} "
                f"dict per column; got {type(d).__name__} at column {i}. (Use a "
                "diagnose_fn backed by run_column_les_pipeline(methods=...).)"
            )
        missing = required - d.keys()
        if missing:
            raise ValueError(
                f"diagnose_fn output at column {i} is missing the method(s) "
                f"{sorted(missing)} required by the specs (has {sorted(d.keys())})."
            )
    # Count validity over the LES RUNS, and ONLY for the spec-required methods (a
    # diagnose_fn may emit extra methods that no spec corrects — they must not mark a
    # column "valid").
    n_diagnoses_valid = count_valid_multi_diagnoses(run_diagnoses, required)
    grid_env, length_scales = _env_grid_predictors(
        feedback_strategy, env_grid_fn, baseline.model_ctx
    )
    # Per spec: extract this method's per-column diagnoses + build raw/base fields.
    per_spec = []
    for s in specs:
        method_diag = [d[s.diagnosis_method] for d in diagnoses]
        raw_i, base_i = _raw_and_base_field(
            records, method_diag, grid_shape,
            method=s.diagnosis_method, background=s.background,
            strategy=feedback_strategy, grid_env=grid_env, length_scales=length_scales,
            clip_to_bounds=clip_to_bounds, config=baseline_config,
            promotion_key=s.promotion_key,
        )
        per_spec.append((s, raw_i, base_i))
    fractions = _validate_step_fractions(step_fractions)

    def _blend(raw_i, base_i, frac):
        field = raw_i if frac == 1.0 else base_i + frac * (raw_i - base_i)
        return field

    if sequential:
        # Block coordinate descent: each coefficient gated against the running
        # state, in spec order (so a coupled coefficient sees the prior ones'
        # accepted effect — e.g. C_eps adjusts wp2 before C_K is line-searched).
        cfg = baseline_config
        running = baseline
        fields = {}
        fracs_by_key = {}
        for s, raw_i, base_i in per_spec:
            def _make_one(frac, _s=s, _raw=raw_i, _base=base_i, _cfg=cfg):
                field = _blend(_raw, _base, frac)
                if clip_to_bounds:
                    field = clip_field_to_promotable_bounds(
                        baseline_config, _s.promotion_key, field
                    )
                return apply_feedback_to_scheme(
                    _cfg, _s.promotion_key, field, expected_ncol=ncol
                ), field
            f, field, cand_cfg, upd, imp = _run_line_search(
                compare_fn, running, fractions, _make_one, global_reduce=global_reduce
            )
            if bool(imp.improved):
                cfg, running = cand_cfg, upd          # accept → advance running state
                fields[s.promotion_key] = field
                fracs_by_key[s.promotion_key] = float(f)
            else:
                fields[s.promotion_key] = base_i      # reject this coefficient only
                fracs_by_key[s.promotion_key] = 0.0
        updated_config, updated = cfg, running
        improvement = bias_improvement(
            baseline.combined_score, updated.combined_score,
            baseline.area_weights, valid_mask=baseline.valid_mask,
            global_reduce=global_reduce,
        )
        frac = max(fracs_by_key.values(), default=0.0)
        fracs_out: dict | None = fracs_by_key
    else:
        def _make_candidate(frac):
            cfg = baseline_config
            fields = {}
            for s, raw_i, base_i in per_spec:
                field = _blend(raw_i, base_i, frac)
                if clip_to_bounds:
                    field = clip_field_to_promotable_bounds(
                        baseline_config, s.promotion_key, field
                    )
                cfg = apply_feedback_to_scheme(
                    cfg, s.promotion_key, field, expected_ncol=ncol
                )
                fields[s.promotion_key] = field
            return cfg, fields

        frac, fields, updated_config, updated, improvement = _run_line_search(
            compare_fn, baseline, fractions, _make_candidate, global_reduce=global_reduce
        )
        fracs_out = None

    worst_idx = jnp.asarray([int(r.flat_index) for r in records], dtype=jnp.int32)
    worst_change = worst_column_bias_change(
        baseline.combined_score, updated.combined_score, worst_idx
    )
    return MultiCorrectionResult(
        updated_config=updated_config, bias=improvement,
        worst_column_change=worst_change, feedback_fields=fields,
        n_corrected=len(records), n_diagnosed=n_diagnosed,
        n_diagnoses_valid=n_diagnoses_valid, step_fraction=frac,
        step_fractions_by_key=fracs_out,
        per_variable_bias=_maybe_per_variable_bias(baseline, updated, global_reduce),
    )


def run_multi_correction_campaign(
    initial_config: Any,
    n_iterations: int,
    specs: Sequence[CorrectionSpec],
    *,
    compare_fn: Callable[[Any], CompareResult],
    diagnose_fn: Callable[[Any, Any], Any],
    grid_shape: tuple[int, ...],
    expected_ncol: int | None = None,
    les_budget: int | None = None,
    env_scales: Sequence[float] | None = None,
    feedback_strategy: str = "static",
    env_grid_fn: Callable[[Any], tuple] | None = None,
    accept_only_if_improved: bool = False,
    step_fractions: Sequence[float] | None = None,
    clip_to_bounds: bool = False,
    sequential: bool = False,
    bias_tol: float | None = None,
    patience: int = 2,
    stop_on_no_valid_diagnoses: bool = True,
    initial_fields: dict | None = None,
    start_round: int = 0,
    checkpoint_callback: Callable[[int, MultiCorrectionResult, dict], None] | None = None,
    global_reduce: Callable[[jax.Array], jax.Array] | None = None,
) -> MultiCampaignResult:
    """Run the SIMULTANEOUS multi-coefficient correction loop for ``n_iterations``.

    Mirrors :func:`run_correction_campaign` with a per-coefficient base DICT:
    ``specs`` give the initial backgrounds; each round corrects all coefficients
    together (one LES run, one fraction, one gate).  The monotonic gate
    (``accept_only_if_improved``) keeps the round only if the COMBINED bias fell —
    accept advances the config + ALL coefficient bases together; a rejected round
    reverts ALL bases atomically (no partial accumulation).  ``initial_fields``
    (``{promotion_key: field}``) RESUMES the accumulated per-coefficient state.
    """
    if int(n_iterations) < 0:
        raise ValueError(f"n_iterations must be >= 0, got {n_iterations}.")
    if bias_tol is not None and int(patience) < 1:
        raise ValueError(
            f"patience must be >= 1 when bias_tol is set, got {patience}.")
    config = initial_config
    bases = {s.promotion_key: s.background for s in specs}
    if initial_fields is not None:
        for key, field in initial_fields.items():
            if key not in bases:
                raise ValueError(
                    f"initial_fields key {key!r} is not a spec promotion_key {list(bases)}.")
            # Resume contract (same as the single path): initial_config must already
            # carry EVERY resumed field — the first round's compare reads the config.
            # Vacuous for n_iterations=0 (no round runs), so gate it the same way.
            if int(n_iterations) >= 1:
                _assert_resume_config_carries_field(initial_config, key, field)
            bases[key] = field
    iterations: list = []
    accepted_flags: list = []
    no_progress = 0
    no_valid_streak = 0
    stop_reason = "max_iterations"
    for i in range(int(n_iterations)):
        round_specs = [s._replace(background=bases[s.promotion_key]) for s in specs]
        result = run_multi_correction_iteration(
            config, round_specs,
            compare_fn=compare_fn, diagnose_fn=diagnose_fn, grid_shape=grid_shape,
            expected_ncol=expected_ncol, les_budget=les_budget, env_scales=env_scales,
            feedback_strategy=feedback_strategy, env_grid_fn=env_grid_fn,
            step_fractions=step_fractions, clip_to_bounds=clip_to_bounds,
            sequential=sequential, global_reduce=global_reduce,
        )
        # DISTRIBUTED: GLOBAL corrected count + GLOBAL bias verdict (see
        # run_correction_campaign) so every rank accepts/rejects identically.
        accepted = not (
            accept_only_if_improved
            and _global_count(result.n_corrected, global_reduce) > 0
            and not bool(result.bias.improved)
        )
        if accepted:
            config = result.updated_config
            bases = dict(result.feedback_fields)   # advance ALL coefficients together
        iterations.append(result)
        accepted_flags.append(accepted)
        if checkpoint_callback is not None:
            checkpoint_callback(int(start_round) + i, result, bases)
        # ABORT a DRY campaign (see run_correction_campaign): `patience` consecutive
        # rounds that flagged columns but produced ZERO valid LES diagnoses globally.
        # COLLECTIVE-SAFE (gated on the GLOBAL counts), checked BEFORE convergence.
        if stop_on_no_valid_diagnoses and int(patience) >= 1:
            global_corrected = _global_count(result.n_corrected, global_reduce)
            global_valid = _global_count(result.n_diagnoses_valid, global_reduce)
            dry = global_corrected > 0 and global_valid == 0
            no_valid_streak = 0 if not dry else no_valid_streak + 1
            if no_valid_streak >= int(patience):
                stop_reason = "no_valid_diagnoses"
                break
        if bias_tol is not None:   # convergence early-stopping (see run_correction_campaign)
            no_progress = (
                0 if _round_made_progress(accepted, result.bias, bias_tol)
                else no_progress + 1
            )
            if no_progress >= int(patience):
                stop_reason = "converged"
                break
    final_fields = {}
    for key, base in bases.items():
        bg = jnp.asarray(base)
        final_fields[key] = (
            jnp.full(grid_shape, bg) if bg.ndim == 0 else bg.reshape(grid_shape)
        )
    return MultiCampaignResult(
        final_config=config, iterations=tuple(iterations),
        final_fields=final_fields, accepted=tuple(accepted_flags),
        stop_reason=stop_reason,
    )
