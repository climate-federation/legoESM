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


class OSSEVerdict(NamedTuple):
    status: str                    # recovered | bias_only | no_change | worsened
    message: str

    @property
    def ok(self) -> bool:
        """The perfect-model twin succeeded: bias fell AND the parameter recovered."""
        return self.status == "recovered"


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

    true_value = float(jnp.mean(jnp.asarray(getattr(true_config, coefficient_field))))

    # The pseudo-reanalysis: the model's OWN time mean under the known true config.
    reference = run_fn(true_config)
    compare_fn = build_compare_fn(reference)

    # Uncorrected columns hold the biased start (round-0 base) unless overridden.
    biased_flat = jnp.asarray(getattr(biased_config, coefficient_field)).reshape(-1)
    initial_value = float(jnp.mean(biased_flat))
    campaign_kwargs.setdefault("background", initial_value)
    result = run_correction_campaign(
        biased_config, n_iterations,
        compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        promotion_key=promotion_key, grid_shape=grid_shape,
        diagnosis_method=diagnosis_method,
        accept_only_if_improved=accept_only_if_improved,
        **campaign_kwargs,
    )
    summary = summarize_campaign(result, promotion_key=promotion_key)

    # Parameter recovery: SAME norm (RMS vs the known scalar truth) for the biased
    # start and the corrected field, so the comparison is fair even if the start /
    # final field is non-uniform.  For a uniform start this is just |bias - true|.
    final_flat = jnp.asarray(result.final_field).reshape(-1)
    init_flat = jnp.broadcast_to(biased_flat, final_flat.shape)
    initial_param_error = float(jnp.sqrt(jnp.mean((init_flat - true_value) ** 2)))
    final_param_error = float(jnp.sqrt(jnp.mean((final_flat - true_value) ** 2)))
    recovered_value = float(jnp.mean(final_flat))

    init_bias = float(summary.initial_bias)
    final_bias = float(summary.final_bias)
    scale = max(abs(init_bias), _REL_TOL)
    bias_reduced = (init_bias - final_bias) > _REL_TOL * scale
    err_scale = max(initial_param_error, _REL_TOL)
    param_error_reduced = (
        (initial_param_error - final_param_error) > _REL_TOL * err_scale)

    return OSSEResult(
        initial_bias=init_bias, final_bias=final_bias,
        bias_reduction=init_bias - final_bias, bias_reduced=bias_reduced,
        true_value=true_value, initial_value=initial_value,
        recovered_value=recovered_value,
        initial_param_error=initial_param_error,
        final_param_error=final_param_error,
        param_error_reduced=param_error_reduced,
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
