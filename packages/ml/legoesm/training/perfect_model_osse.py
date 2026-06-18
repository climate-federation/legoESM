"""Perfect-model (identical-twin) OSSE for the LES-informed correction loop.

The done-criterion's empirical clause — "updating these parameters in the
AMIP/CMIP simulations IMPROVE the biases" — can only be settled for real on a
multi-day ERA5 HPC run.  This module is the cheaper, fully-controlled PRECURSOR
that an HPC user (or a CI nightly) runs FIRST: a perfect-model Observing-System
Simulation Experiment (OSSE).

The "truth" is the model itself run with a KNOWN turbulence coefficient
(``true_config``); its time mean is the pseudo-reanalysis.  A BIASED run
(``biased_config``, a different coefficient) is then driven through the SAME
correction loop (run -> compare to the pseudo-truth -> LES-diagnose worst columns
-> feed back -> re-run, under the monotonic gate).  Because the truth is known we
can ask two questions the real-ERA5 run cannot answer cheaply:

1. Did the loop LOWER the bias against the pseudo-truth?  (clause 5, controlled)
2. Did the corrected coefficient move TOWARD the known true value (parameter
   RECOVERY)?  If the loop cannot recover a known parameter in the perfect-model
   twin, it will not help against real ERA5 — a cheap go/no-go gate before HPC.

Everything heavy is INJECTED (``run_fn``, ``build_compare_fn``, ``diagnose_fn``)
so the harness reuses the real run->time-mean->compare->campaign machinery
(:mod:`legoesm.training.run_to_column_mean`, :mod:`legoesm.training.correction_loop`)
and stays unit-testable with synthetic runners.  The real-driver wiring lives in
``scripts/validate/run_perfect_model_osse.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from legoesm.training.correction_loop import run_correction_campaign

# A bias / parameter-error change smaller than this (relative to the initial
# magnitude) counts as "no change", not a real improvement — guards float noise.
_REL_TOL = 1e-6


class OSSEResult(NamedTuple):
    """Outcome of a perfect-model OSSE: bias trajectory + parameter recovery."""

    initial_bias: float
    final_bias: float
    bias_reduction: float          # initial_bias - final_bias (>= 0 under the gate)
    bias_reduced: bool             # final_bias strictly below initial_bias
    true_value: float              # the known true coefficient (scalar / mean)
    initial_value: float           # the biased start (scalar / mean)
    recovered_value: float         # mean of the final corrected per-column field
    initial_param_error: float     # |initial_value - true_value|
    final_param_error: float       # RMS(final_field - true_value)
    param_error_reduced: bool      # final_param_error strictly below initial
    n_rounds: int
    n_accepted: int
    summary: Any                   # the underlying CampaignSummary (bias detail)


class CoefRecovery(NamedTuple):
    """Per-coefficient parameter recovery (shared by the single + multi OSSE)."""

    field: str
    true_value: float
    initial_value: float
    recovered_value: float
    initial_param_error: float
    final_param_error: float
    param_error_reduced: bool


class OSSEVerdict(NamedTuple):
    status: str                    # recovered | bias_only | no_change | worsened
    message: str

    @property
    def ok(self) -> bool:
        """The perfect-model twin succeeded: bias fell AND the parameter recovered."""
        return self.status == "recovered"


def _recovery_metrics(field_name, true_config, biased_config, final_field) -> CoefRecovery:
    """Parameter recovery for one coefficient field, the SAME norm at both ends.

    RMS of the field's distance to the KNOWN scalar truth — for the (uniform)
    biased start and the corrected per-column field — so a non-uniform start or
    final field is compared fairly (for a uniform start this is just |bias-true|).
    """
    import jax.numpy as jnp

    true_value = float(jnp.mean(jnp.asarray(getattr(true_config, field_name))))
    biased_flat = jnp.asarray(getattr(biased_config, field_name)).reshape(-1)
    final_flat = jnp.asarray(final_field).reshape(-1)
    init_flat = jnp.broadcast_to(biased_flat, final_flat.shape)
    init_err = float(jnp.sqrt(jnp.mean((init_flat - true_value) ** 2)))
    final_err = float(jnp.sqrt(jnp.mean((final_flat - true_value) ** 2)))
    return CoefRecovery(
        field=field_name, true_value=true_value,
        initial_value=float(jnp.mean(biased_flat)),
        recovered_value=float(jnp.mean(final_flat)),
        initial_param_error=init_err, final_param_error=final_err,
        param_error_reduced=(init_err - final_err) > _REL_TOL * max(init_err, _REL_TOL),
    )


def _bias_outcome(summary):
    """``(initial_bias, final_bias, bias_reduced)`` from a CampaignSummary."""
    init_bias = float(summary.initial_bias)
    final_bias = float(summary.final_bias)
    bias_reduced = (init_bias - final_bias) > _REL_TOL * max(abs(init_bias), _REL_TOL)
    return init_bias, final_bias, bias_reduced


def run_perfect_model_osse(
    *,
    true_config: Any,
    biased_config: Any,
    coefficient_field: str,
    run_fn: Callable[[Any], Any],
    build_compare_fn: Callable[[Any], Callable[[Any], Any]],
    diagnose_fn: Callable[[Any, Any], Any],
    promotion_key: str,
    grid_shape: tuple[int, ...],
    diagnosis_method: str = "eddy_diffusivity",
    n_iterations: int = 3,
    accept_only_if_improved: bool = True,
    **campaign_kwargs: Any,
) -> OSSEResult:
    """Run a perfect-model OSSE and report bias reduction + parameter recovery.

    ``run_fn(config) -> state`` is the model run + time mean (the reference is
    ``run_fn(true_config)``); ``build_compare_fn(reference) -> compare_fn`` adapts
    the comparison to that pseudo-truth; ``diagnose_fn`` is the LES diagnosis.
    ``coefficient_field`` is the scheme-config field the loop corrects (``"C_K"`` /
    ``"Pr_t"`` / ``"C_eps"``); ``true_config``/``biased_config`` carry its known
    true value and the biased start.  The monotonic gate is ON by default (the
    accumulated field can never regress).  Extra ``campaign_kwargs`` forward to
    :func:`run_correction_campaign`.
    """
    import jax.numpy as jnp
    from legoesm.training.campaign_summary import summarize_campaign
    from legoesm.training.promotable_params import PROMOTABLE_FIELDS

    if n_iterations < 1:
        raise ValueError(
            f"run_perfect_model_osse needs n_iterations >= 1 (got {n_iterations}); "
            "zero rounds never compares the biased model to the pseudo-truth.")
    # The reported coefficient MUST be the one the loop actually corrects, or the
    # recovery metric would describe a different field than the campaign moved.
    promoted = PROMOTABLE_FIELDS.get(promotion_key)
    if promoted is None or promoted.field != coefficient_field:
        raise ValueError(
            f"coefficient_field {coefficient_field!r} does not match the field "
            f"corrected by promotion_key {promotion_key!r} "
            f"({getattr(promoted, 'field', None)!r}); the OSSE would report "
            "recovery for a different coefficient than the loop corrects.")

    # The pseudo-reanalysis: the model's OWN time mean under the known true config.
    reference = run_fn(true_config)
    compare_fn = build_compare_fn(reference)

    # Uncorrected columns hold the biased start (round-0 base) unless overridden.
    biased_flat = jnp.asarray(getattr(biased_config, coefficient_field)).reshape(-1)
    campaign_kwargs.setdefault("background", float(jnp.mean(biased_flat)))
    result = run_correction_campaign(
        biased_config, n_iterations,
        compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        promotion_key=promotion_key, grid_shape=grid_shape,
        diagnosis_method=diagnosis_method,
        accept_only_if_improved=accept_only_if_improved,
        **campaign_kwargs,
    )
    summary = summarize_campaign(result, promotion_key=promotion_key)
    rec = _recovery_metrics(
        coefficient_field, true_config, biased_config, result.final_field)
    init_bias, final_bias, bias_reduced = _bias_outcome(summary)

    return OSSEResult(
        initial_bias=init_bias, final_bias=final_bias,
        bias_reduction=init_bias - final_bias, bias_reduced=bias_reduced,
        true_value=rec.true_value, initial_value=rec.initial_value,
        recovered_value=rec.recovered_value,
        initial_param_error=rec.initial_param_error,
        final_param_error=rec.final_param_error,
        param_error_reduced=rec.param_error_reduced,
        n_rounds=len(result.iterations),
        n_accepted=int(sum(1 for a in result.accepted if a)),
        summary=summary,
    )


def osse_verdict(result: OSSEResult) -> OSSEVerdict:
    """Classify a perfect-model OSSE into a go/no-go verdict for the real campaign.

    * ``recovered`` — bias fell AND the corrected coefficient moved toward the
      known truth: the loop works in the controlled twin; proceed to real ERA5.
    * ``bias_only`` — bias fell but the parameter did NOT recover: the fit
      improved via compensating errors; the LES↔GCM closure transfer is suspect.
    * ``no_change`` — the gate rejected every round (the LES diagnosis never
      helped): the closure inversion does not recover the parameter in this setup.
    * ``worsened`` — final bias ABOVE initial: cannot happen with the monotonic
      gate ON (the harness default), so it signals either a disabled gate
      (``accept_only_if_improved=False``) or a bias-accounting bug; surfaced loudly.
    """
    if result.final_bias > result.initial_bias + _REL_TOL * max(
            abs(result.initial_bias), _REL_TOL):
        return OSSEVerdict(
            "worsened",
            f"final bias {result.final_bias:.4g} EXCEEDS initial "
            f"{result.initial_bias:.4g} — only possible with the monotonic gate "
            "disabled or a bias-accounting bug; investigate.")
    if result.bias_reduced and result.param_error_reduced:
        return OSSEVerdict(
            "recovered",
            f"bias {result.initial_bias:.4g} -> {result.final_bias:.4g} and "
            f"the coefficient moved toward the truth (param error "
            f"{result.initial_param_error:.4g} -> {result.final_param_error:.4g}, "
            f"true={result.true_value:.4g}). The loop works in the perfect-model "
            "twin; proceed to the real-ERA5 campaign.")
    if result.bias_reduced:
        return OSSEVerdict(
            "bias_only",
            f"bias fell ({result.initial_bias:.4g} -> {result.final_bias:.4g}) but "
            f"the parameter did NOT recover (error {result.initial_param_error:.4g} "
            f"-> {result.final_param_error:.4g}); suspect compensating errors / the "
            "LES->GCM closure transfer before trusting a real-ERA5 improvement.")
    return OSSEVerdict(
        "no_change",
        f"no accepted round reduced the bias (kept {result.n_accepted}/"
        f"{result.n_rounds}); the LES diagnosis did not recover the parameter in "
        "this perfect-model setup — fix that before spending HPC on real ERA5.")


class MultiOSSEResult(NamedTuple):
    """Outcome of a SIMULTANEOUS multi-coefficient perfect-model OSSE."""

    initial_bias: float
    final_bias: float
    bias_reduction: float
    bias_reduced: bool
    per_coefficient: dict          # {promotion_key: CoefRecovery}
    all_recovered: bool            # every coefficient's param error fell
    n_rounds: int
    n_accepted: int
    summary: Any


def run_multi_perfect_model_osse(
    *,
    true_config: Any,
    biased_config: Any,
    specs: Any,
    run_fn: Callable[[Any], Any],
    build_compare_fn: Callable[[Any], Callable[[Any], Any]],
    diagnose_fn: Callable[[Any, Any], Any],
    grid_shape: tuple[int, ...],
    n_iterations: int = 3,
    accept_only_if_improved: bool = True,
    sequential: bool = False,
    **campaign_kwargs: Any,
) -> MultiOSSEResult:
    """Perfect-model OSSE for the SIMULTANEOUS multi-coefficient correction.

    Like :func:`run_perfect_model_osse` but recovers SEVERAL known coefficients at
    once (each ``CorrectionSpec`` in ``specs``), exercising the coupled
    block-coordinate machinery (e.g. C_eps↔C_K): the pseudo-truth is
    ``run_fn(true_config)`` and the loop must lower the bias AND move EVERY
    coefficient toward its known value.  ``sequential`` selects the staged
    (block-coordinate) mode; the monotonic gate is ON by default.
    """
    from legoesm.training.campaign_summary import summarize_campaign
    from legoesm.training.correction_loop import run_multi_correction_campaign
    from legoesm.training.promotable_params import PROMOTABLE_FIELDS

    specs = list(specs)
    if n_iterations < 1:
        raise ValueError(
            f"run_multi_perfect_model_osse needs n_iterations >= 1 "
            f"(got {n_iterations}).")
    if not specs:
        raise ValueError(
            "run_multi_perfect_model_osse needs at least one CorrectionSpec.")
    keys = [s.promotion_key for s in specs]
    if len(set(keys)) != len(keys):
        # Else per_coefficient would silently collapse a duplicate's recovery.
        raise ValueError(
            f"duplicate promotion_key in specs ({keys}); each coefficient appears "
            "once.")
    for spec in specs:
        if PROMOTABLE_FIELDS.get(spec.promotion_key) is None:
            raise ValueError(
                f"unknown promotion_key {spec.promotion_key!r} in a CorrectionSpec; "
                f"choose from {tuple(PROMOTABLE_FIELDS)}.")

    reference = run_fn(true_config)
    compare_fn = build_compare_fn(reference)
    result = run_multi_correction_campaign(
        biased_config, n_iterations, specs,
        compare_fn=compare_fn, diagnose_fn=diagnose_fn, grid_shape=grid_shape,
        accept_only_if_improved=accept_only_if_improved, sequential=sequential,
        **campaign_kwargs,
    )
    summary = summarize_campaign(result)
    init_bias, final_bias, bias_reduced = _bias_outcome(summary)

    per_coefficient = {
        spec.promotion_key: _recovery_metrics(
            PROMOTABLE_FIELDS[spec.promotion_key].field,
            true_config, biased_config, result.final_fields[spec.promotion_key])
        for spec in specs
    }
    all_recovered = all(
        c.param_error_reduced for c in per_coefficient.values())
    return MultiOSSEResult(
        initial_bias=init_bias, final_bias=final_bias,
        bias_reduction=init_bias - final_bias, bias_reduced=bias_reduced,
        per_coefficient=per_coefficient, all_recovered=all_recovered,
        n_rounds=len(result.iterations),
        n_accepted=int(sum(1 for a in result.accepted if a)),
        summary=summary,
    )


def multi_osse_verdict(result: MultiOSSEResult) -> OSSEVerdict:
    """Classify a multi-coefficient OSSE; ``recovered`` requires ALL coefficients.

    Same partition as :func:`osse_verdict` but ``recovered`` demands the bias fell
    AND every coefficient moved toward its truth; ``bias_only`` reports how many of
    the N coefficients recovered.
    """
    n = len(result.per_coefficient)
    n_rec = sum(c.param_error_reduced for c in result.per_coefficient.values())
    if result.final_bias > result.initial_bias + _REL_TOL * max(
            abs(result.initial_bias), _REL_TOL):
        return OSSEVerdict(
            "worsened",
            f"final bias {result.final_bias:.4g} EXCEEDS initial "
            f"{result.initial_bias:.4g} — only possible with the monotonic gate "
            "disabled or a bias-accounting bug; investigate.")
    if result.bias_reduced and result.all_recovered:
        return OSSEVerdict(
            "recovered",
            f"bias {result.initial_bias:.4g} -> {result.final_bias:.4g} and all "
            f"{n} coefficients moved toward their truth. The simultaneous loop "
            "works in the perfect-model twin; proceed to the real-ERA5 campaign.")
    if result.bias_reduced:
        return OSSEVerdict(
            "bias_only",
            f"bias fell ({result.initial_bias:.4g} -> {result.final_bias:.4g}) but "
            f"only {n_rec}/{n} coefficients recovered; suspect compensating errors "
            "across the coupled coefficients before trusting a real-ERA5 gain.")
    return OSSEVerdict(
        "no_change",
        f"no accepted round reduced the bias (kept {result.n_accepted}/"
        f"{result.n_rounds}); the simultaneous LES diagnosis did not recover the "
        "coefficients in this perfect-model setup.")
