"""Condense a correction campaign into an actionable diagnostics summary.

After a (single- or multi-coefficient) correction campaign
(:func:`legoesm.training.correction_loop.run_correction_campaign` /
:func:`~legoesm.training.correction_loop.run_multi_correction_campaign`), an HPC
user needs to judge — without re-reading the full per-round trace — whether the
LES-informed corrections are actually lowering the bias, and whether the
diagnoses are being trusted.  This module reduces the result to:

* the bias trajectory (initial → final, absolute + fractional reduction),
* the acceptance rate + the convergence ``stop_reason``, and
* per-coefficient field statistics, INCLUDING how many columns are pinned at the
  registered ``__param_spec__`` bounds — a binding clamp signals the LES wants a
  coefficient MORE extreme than the calibratable range allows (the "monitor
  coefficient drift across iterations" the iter-46 review recommended,
  ``docs/COMPARE_REANALYSIS.md`` §9).

Pure host-side reduction (NumPy on the accumulated fields); no model run.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from legoesm.training.feedback import param_field_bounds
from legoesm.training.promotable_params import PROMOTABLE_FIELDS

# A field value within this tolerance of a registered bound counts as "at" it
# (the bounds clamp pins exactly, so the tolerance only absorbs float round-trip).
_BOUND_ATOL = 1.0e-9
_BOUND_RTOL = 1.0e-6


class CoefficientSummary(NamedTuple):
    """Field statistics for one corrected coefficient over the whole grid."""

    promotion_key: str
    n_columns: int              # grid columns (for the bound-pinned FRACTION)
    field_min: float
    field_max: float
    field_mean: float
    field_std: float            # 0 ⇒ a uniform field (no spatial correction)
    n_at_lower_bound: int       # columns pinned at the registered lower bound
    n_at_upper_bound: int       # columns pinned at the upper bound (clamp binding)
    bounds: tuple | None        # (lo, hi) or None if the field is not bound-spec'd

    @property
    def clamp_fraction(self) -> float:
        """Fraction of columns pinned at EITHER registered bound (0 if no bounds /
        no columns) — a high value means the LES wants a coefficient outside its
        calibratable range, i.e. the clamp is shaping the correction."""
        if not self.n_columns:
            return 0.0
        # min(1, …) guards the degenerate lo==hi case (a column counted at both).
        return min(1.0, (self.n_at_lower_bound + self.n_at_upper_bound) / self.n_columns)


class CampaignSummary(NamedTuple):
    """Actionable roll-up of a correction campaign."""

    n_rounds: int
    n_accepted: int
    acceptance_rate: float
    stop_reason: str
    initial_bias: float
    final_bias: float
    absolute_reduction: float       # initial_bias − final_bias (> 0 ⇒ improved)
    fractional_reduction: float     # absolute_reduction / initial_bias (0 if init 0)
    coefficients: tuple             # tuple[CoefficientSummary]

    def report(self) -> str:
        """A concise human-readable multi-line report."""
        pct = 100.0 * self.fractional_reduction
        lines = [
            f"Campaign: {self.n_rounds} rounds "
            f"({self.n_accepted} accepted, {100.0 * self.acceptance_rate:.0f}%), "
            f"{self.stop_reason}.",
            f"Bias: {self.initial_bias:.4g} -> {self.final_bias:.4g} "
            f"(reduced {self.absolute_reduction:.4g}, {pct:.1f}%).",
        ]
        for c in self.coefficients:
            line = (f"  {c.promotion_key}: mean {c.field_mean:.3g} "
                    f"[{c.field_min:.3g}, {c.field_max:.3g}] std {c.field_std:.3g}")
            if c.bounds is not None and (c.n_at_lower_bound or c.n_at_upper_bound):
                line += (f"; {c.n_at_lower_bound} at lo, {c.n_at_upper_bound} at hi"
                         " (clamp binding)")
            lines.append(line)
        return "\n".join(lines)


def _coefficient_summary(promotion_key: str, field, config: Any) -> CoefficientSummary:
    arr = np.asarray(field).reshape(-1)
    bounds = None
    promo = PROMOTABLE_FIELDS.get(promotion_key)
    if promo is not None:
        bounds = param_field_bounds(config, promo.field)
    n_lo = n_hi = 0
    if bounds is not None:
        lo, hi = bounds
        n_lo = int(np.sum(np.isclose(arr, lo, atol=_BOUND_ATOL, rtol=_BOUND_RTOL)))
        n_hi = int(np.sum(np.isclose(arr, hi, atol=_BOUND_ATOL, rtol=_BOUND_RTOL)))
    return CoefficientSummary(
        promotion_key=promotion_key, n_columns=int(arr.size),
        field_min=float(arr.min()), field_max=float(arr.max()),
        field_mean=float(arr.mean()), field_std=float(arr.std()),
        n_at_lower_bound=n_lo, n_at_upper_bound=n_hi, bounds=bounds,
    )


def summarize_campaign(result: Any, *, promotion_key: str | None = None) -> CampaignSummary:
    """Roll a campaign result up into a :class:`CampaignSummary`.

    Accepts a :class:`~legoesm.training.correction_loop.MultiCampaignResult` (the
    coefficients are read from ``final_fields``) or a single-coefficient
    :class:`~legoesm.training.correction_loop.CampaignResult` (pass the
    ``promotion_key`` so the single ``final_field`` can be named + bound-checked).
    The bias trajectory is ``iterations[0].bias.baseline_bias`` (the uncorrected
    bias) → the LAST ACCEPTED round's ``updated_bias`` (the achieved bias; the gate
    means rejected rounds did not change the state).
    """
    iters = result.iterations
    n_rounds = len(iters)
    accepted = result.accepted if result.accepted else tuple(True for _ in iters)
    n_accepted = int(sum(bool(a) for a in accepted))
    acceptance_rate = n_accepted / n_rounds if n_rounds else 0.0
    stop_reason = getattr(result, "stop_reason", "max_iterations")

    if n_rounds:
        initial_bias = float(iters[0].bias.baseline_bias)
        final_bias = initial_bias
        for it, a in zip(iters, accepted):
            if a:
                final_bias = float(it.bias.updated_bias)
    else:
        initial_bias = final_bias = 0.0
    absolute_reduction = initial_bias - final_bias
    fractional_reduction = (
        absolute_reduction / initial_bias if initial_bias != 0.0 else 0.0
    )

    config = result.final_config
    if hasattr(result, "final_fields"):          # MultiCampaignResult
        fields = result.final_fields
    else:                                        # single CampaignResult
        if promotion_key is None:
            raise ValueError(
                "summarize_campaign needs promotion_key for a single-coefficient "
                "CampaignResult (it carries no coefficient name).")
        fields = {promotion_key: result.final_field}
    coefficients = tuple(
        _coefficient_summary(key, f, config) for key, f in fields.items()
    )
    return CampaignSummary(
        n_rounds=n_rounds, n_accepted=n_accepted, acceptance_rate=acceptance_rate,
        stop_reason=stop_reason, initial_bias=initial_bias, final_bias=final_bias,
        absolute_reduction=absolute_reduction,
        fractional_reduction=fractional_reduction, coefficients=coefficients,
    )


# Health thresholds (campaign-control judgments, not physics): the fractional bias
# reduction below which a campaign is "stalled", and the bound-pinned fraction
# above which it is "clamp-limited".
_MIN_FRACTIONAL_REDUCTION = 0.02
_CLAMP_FRACTION_WARN = 0.2


class CampaignHealth(NamedTuple):
    """A one-look verdict on whether the corrections are working + why/why not."""

    status: str     # "improved" | "stalled" | "clamp_limited" | "no_rounds"
    message: str    # actionable one-line explanation

    @property
    def ok(self) -> bool:
        return self.status == "improved"


def campaign_health(
    summary: CampaignSummary,
    *,
    min_fractional_reduction: float = _MIN_FRACTIONAL_REDUCTION,
    clamp_fraction_warn: float = _CLAMP_FRACTION_WARN,
) -> CampaignHealth:
    """Classify a :class:`CampaignSummary` into an actionable verdict.

    * ``"improved"`` — the area-weighted bias fell by ≥ ``min_fractional_reduction``.
      A genuinely-improving run is reported "improved" (``.ok``) EVEN IF the clamp is
      binding — the clamp is then surfaced as a NOTE in the message (the in-range
      correction still helped, but the LES wanted more).
    * ``"clamp_limited"`` — the bias did NOT improve AND ≥ ``clamp_fraction_warn`` of
      columns are pinned at a coefficient's registered bounds: the clamp (not the
      diagnosis) is the likely cause — the LES wants a value OUTSIDE the calibratable
      range, so widen the bounds or check the diagnosis.
    * ``"stalled"`` — the bias did not improve and the clamp is not the cause, so the
      corrections are not improving the bias (check the LES↔GCM transfer or config).
    * ``"no_rounds"`` — the campaign ran no rounds.

    Improvement is judged FIRST: a strongly-improving run is never demoted to a
    warning status just because the clamp is binding (it is noted instead).  Pure
    host-side classification.
    """
    if summary.n_rounds == 0:
        return CampaignHealth("no_rounds", "Campaign ran no rounds.")
    worst_clamp, clamp_key = 0.0, None
    for c in summary.coefficients:
        if c.clamp_fraction > worst_clamp:
            worst_clamp, clamp_key = c.clamp_fraction, c.promotion_key
    acc = f"{summary.n_accepted}/{summary.n_rounds} rounds accepted"
    red = f"{summary.fractional_reduction:.0%}"
    clamp_binding = worst_clamp >= clamp_fraction_warn
    if summary.fractional_reduction >= min_fractional_reduction:
        note = (f" Note: {worst_clamp:.0%} of columns pinned at {clamp_key} bounds "
                "(the LES wants a value outside its calibratable range)."
                if clamp_binding else "")
        return CampaignHealth("improved", f"Bias reduced {red} ({acc}).{note}")
    if clamp_binding:
        return CampaignHealth(
            "clamp_limited",
            f"{worst_clamp:.0%} of columns pinned at {clamp_key} bounds — the LES "
            f"wants a coefficient outside its calibratable range; widen the bounds "
            f"or check the diagnosis. Bias reduced {red} ({acc}).")
    return CampaignHealth(
        "stalled",
        f"Bias reduced only {red} ({acc}) — the corrections are not improving "
        f"the bias; check the LES↔GCM transfer or the config.")
