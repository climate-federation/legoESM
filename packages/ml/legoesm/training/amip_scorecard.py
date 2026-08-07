"""Broad AMIP tuning objective: TOA radiation + near-surface temperature + precipitation.

This module turns the metrics table produced by
``scripts/plot/plot_amip_pattern_eval.py`` (the canonical AMIP scorer — the one
that produced every published ``bias`` / ``r_pattern`` number for this model)
into ONE scalar a parameter search can minimise, plus the per-term breakdown
that says *why* it moved.  It re-derives no metric: ``bias`` and
``pattern_corr`` are consumed exactly as that scorer computed them (area-weighted
with ``cos(lat)``, both fields centred on their own weighted means before the
correlation).  Keeping the arithmetic split this way is deliberate — one
implementation of the metric, one implementation of the objective, and the
objective is unit-testable without touching a single netCDF file.

Why not JAX
-----------
Nothing here is differentiable and nothing needs to be.  The objective consumes
monthly CMOR climatologies written to disk by a completed forward run, so the
only consumers are derivative-free searches (ETKI, sweeps).  Plain floats keep
the module importable without a JAX device and make the unit tests exact.
Contrast :mod:`legoesm.training.scm_rce_metrics`, which IS on an AD path and is
JAX for that reason.

The objective
-------------
Each field contributes a dimensionless score in "tolerances of error"::

    s_f = a * |bias_f| / bias_tol_f  +  (1 - a) * (1 - r_f) / PATTERN_TOLERANCE

with ``a = weights.bias_fraction``.  ``(1 - r)`` is 0 for a perfect pattern, 1
for an uncorrelated one and 2 for an anticorrelated one, so a field whose map is
*inverted* (the model's current ``rsut``) is penalised harder than one that is
merely unskilful — which is the behaviour we want, because an anticorrelated
field is a structural defect, not a tuning target.

The total is the weighted sum ``J = sum_f w_f * s_f``, readable directly as
"weighted mean number of acceptance tolerances of error".

Pattern correlation is BOTH a term and a constraint.  A tune that buys global-mean
bias by degrading spatial structure is a regression even when ``J`` falls, so
:func:`pattern_regressions` reports it independently of the scalar.  Call it on
every candidate; never rank on ``J`` alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

# --- Bias normalisation scales [field units] -------------------------------
# These are the ACCEPTANCE TOLERANCES already in use by the repo's global-mean
# gate, `scripts/validate/amip_skill_score.py::REFERENCE` (tas 2.0 K, pr
# 0.8 mm/day, rlut 8.0 W/m^2, rsut 8.0 W/m^2).  Reusing them rather than
# inventing new scales keeps ONE definition of "how far off is too far off" in
# the repo: a field at 1.0 tolerance here is exactly a field sitting on that
# gate's pass/fail boundary.  They are evaluation conventions, NOT physical
# constants, so they do not belong in `legoesm.constants`.
BIAS_TOLERANCES: dict[str, float] = {
    "rsut": 8.0,   # W/m^2, CERES-EBAF reflected SW
    "rlut": 8.0,   # W/m^2, CERES-EBAF OLR
    "tas": 2.0,    # K,      ERA5 2 m air temperature
    "pr": 0.8,     # mm/day, GPCP
}

# --- Pattern normalisation scale [dimensionless] ---------------------------
# One "tolerance" of pattern error is a 0.10 shortfall in centred pattern
# correlation.  Chosen so the two halves of a field score are commensurable:
# losing 0.10 of spatial correlation is treated as exactly as serious as being
# one bias tolerance (8 W/m^2 of TOA flux, 2 K of tas) off the global mean.
# Raising it makes the search care less about structure; lowering it more.
PATTERN_TOLERANCE = 0.10

# Default tolerated drop in centred pattern correlation vs a baseline before a
# candidate is called a structural regression.  0.02 is a deliberate near-zero:
# pattern skill is the quantity we are least willing to trade away, and the
# annual-mean correlation of a 365-day run is reproducible well inside this.
DEFAULT_MAX_PATTERN_DROP = 0.02

SCORED_FIELDS: tuple[str, ...] = ("rsut", "rlut", "tas", "pr")


class ScorecardWeights(NamedTuple):
    """The single knob controlling what the tuner cares about.

    Per-field weights must sum to 1 so the objective stays interpretable as a
    weighted mean number of tolerances.  Rationale for the defaults:

    * **TOA radiation carries half the objective** (``rsut`` 0.25 + ``rlut``
      0.25).  It is the largest error the model has, it is the most physically
      upstream (the TOA budget sets the energy the rest of the climate has to
      distribute), and it is the term the cloud work is aimed at.  ``rsut`` and
      ``rlut`` split it EVENLY rather than in proportion to their current
      errors: weighting by the present bias would fit the objective to today's
      baseline instead of to the target, and would silently re-weight itself
      every time a run improved.
    * **tas 0.25.**  The headline climate variable, and the one an observer
      checks first.  Its pattern correlation is already 0.98, so this weight
      mostly acts as a GUARD — it makes the search pay for wrecking a field
      that is currently good, rather than driving further improvement.
    * **pr 0.25.**  Equal to tas.  Precipitation is the hardest field and the
      noisiest, so it earns no more than an equal share; but it is the main
      independent check that a radiative tune has not been bought by breaking
      the hydrological cycle.

    ``bias_fraction`` splits each field's score between global-mean bias and
    spatial pattern.  The 0.5 default is the deliberate statement that the two
    failure modes are equally bad: a model with the right global mean and an
    inverted map is no more useful than one with the right map and a 60 W/m^2
    offset.  Set it to 1.0 to tune on magnitude alone (only defensible once
    every pattern correlation is already acceptable).
    """

    rsut: float = 0.25
    rlut: float = 0.25
    tas: float = 0.25
    pr: float = 0.25
    bias_fraction: float = 0.5

    def field_weight(self, field: str) -> float:
        """Weight for ``field``; raises on an unknown field rather than 0."""
        if field not in SCORED_FIELDS:
            raise ValueError(
                f"unknown scorecard field {field!r}; known: {SCORED_FIELDS}")
        return float(getattr(self, field))

    def validate(self) -> "ScorecardWeights":
        """Fail loudly on a mis-specified knob; returns self for chaining."""
        total = sum(self.field_weight(f) for f in SCORED_FIELDS)
        if abs(total - 1.0) > 1.0e-9:
            raise ValueError(
                f"scorecard field weights must sum to 1, got {total!r} "
                f"({ {f: self.field_weight(f) for f in SCORED_FIELDS} })")
        for f in SCORED_FIELDS:
            if self.field_weight(f) < 0.0:
                raise ValueError(f"weight for {f!r} is negative")
        if not 0.0 <= self.bias_fraction <= 1.0:
            raise ValueError(
                f"bias_fraction must be in [0, 1], got {self.bias_fraction!r}")
        return self


class FieldScore(NamedTuple):
    """One field's contribution, kept fully decomposed for reporting."""

    field: str
    bias: float             # field units, model - reference
    pattern_corr: float     # centred, dimensionless
    bias_term: float        # |bias| / bias tolerance
    pattern_term: float     # (1 - r) / PATTERN_TOLERANCE
    score: float            # blended field score
    weight: float
    contribution: float     # weight * score


class ScorecardResult(NamedTuple):
    """The scalar objective plus everything needed to explain it."""

    objective: float
    fields: tuple[FieldScore, ...]
    weights: ScorecardWeights

    def by_field(self) -> dict[str, FieldScore]:
        return {fs.field: fs for fs in self.fields}

    def format_table(self) -> str:
        """Human-readable breakdown; the contribution column is the ranking."""
        lines = [
            f"{'field':6s} {'bias':>10s} {'r_patt':>8s} "
            f"{'bias_t':>8s} {'patt_t':>8s} {'score':>8s} "
            f"{'weight':>7s} {'contrib':>9s}",
        ]
        for fs in self.fields:
            lines.append(
                f"{fs.field:6s} {fs.bias:+10.3f} {fs.pattern_corr:8.3f} "
                f"{fs.bias_term:8.3f} {fs.pattern_term:8.3f} {fs.score:8.3f} "
                f"{fs.weight:7.3f} {fs.contribution:9.3f}")
        lines.append(f"{'TOTAL':6s} {'':10s} {'':8s} {'':8s} {'':8s} "
                     f"{'':8s} {'':7s} {self.objective:9.3f}")
        return "\n".join(lines)


def bias_term(bias: float, tolerance: float) -> float:
    """``|bias| / tolerance`` — how many acceptance tolerances off the mean is."""
    if tolerance <= 0.0:
        raise ValueError(f"bias tolerance must be positive, got {tolerance!r}")
    return abs(float(bias)) / float(tolerance)


def pattern_term(pattern_corr: float,
                 tolerance: float = PATTERN_TOLERANCE) -> float:
    """``(1 - r) / tolerance``.

    Not clipped at ``r < 0``: an anticorrelated field SHOULD score worse than an
    uncorrelated one, because an inverted map is a structural defect a scalar
    knob cannot repair.  A perfect field scores 0, an uncorrelated one
    ``1 / tolerance``, a perfectly inverted one twice that.
    """
    if tolerance <= 0.0:
        raise ValueError(
            f"pattern tolerance must be positive, got {tolerance!r}")
    return (1.0 - float(pattern_corr)) / float(tolerance)


def field_score(field: str, bias: float, pattern_corr: float,
                weights: ScorecardWeights) -> FieldScore:
    """Blend one field's bias and pattern terms into its weighted contribution."""
    if field not in BIAS_TOLERANCES:
        raise ValueError(
            f"no bias tolerance for {field!r}; known: "
            f"{sorted(BIAS_TOLERANCES)}")
    a = float(weights.bias_fraction)
    b_t = bias_term(bias, BIAS_TOLERANCES[field])
    p_t = pattern_term(pattern_corr)
    score = a * b_t + (1.0 - a) * p_t
    w = weights.field_weight(field)
    return FieldScore(
        field=field, bias=float(bias), pattern_corr=float(pattern_corr),
        bias_term=b_t, pattern_term=p_t, score=score, weight=w,
        contribution=w * score)


def evaluate_scorecard(
    metrics: Mapping[str, Mapping[str, float]],
    weights: ScorecardWeights | None = None,
) -> ScorecardResult:
    """Score a pattern-eval metrics table.

    ``metrics`` maps a field name to at least ``{"bias": ..., "pattern_corr":
    ...}`` — i.e. exactly the per-field dict
    ``plot_amip_pattern_eval.pattern_stats`` returns, or the ``fields`` block of
    the JSON that script writes.  Extra fields in the table (``prw``, ``psl``,
    the ``zonal_*`` rows) are IGNORED, not silently folded in: this objective is
    defined on four fields and adding a fifth must be a deliberate edit here.

    A scored field missing from ``metrics`` raises — a run whose CMOR archive
    lacks ``rsut`` must not quietly score as if that term were perfect.
    """
    w = (weights or ScorecardWeights()).validate()
    scores = []
    for f in SCORED_FIELDS:
        if f not in metrics:
            raise ValueError(
                f"scorecard field {f!r} missing from metrics table; present: "
                f"{sorted(metrics)}")
        entry = metrics[f]
        for key in ("bias", "pattern_corr"):
            if key not in entry:
                raise ValueError(f"metrics[{f!r}] has no {key!r}")
            if not _is_finite(entry[key]):
                raise ValueError(
                    f"metrics[{f!r}][{key!r}] is not finite ({entry[key]!r}); "
                    "a non-finite metric means the run or the reference "
                    "regrid failed — scoring it would hide that")
        scores.append(field_score(f, entry["bias"], entry["pattern_corr"], w))
    return ScorecardResult(
        objective=sum(s.contribution for s in scores),
        fields=tuple(scores), weights=w)


class PatternRegression(NamedTuple):
    """One field whose spatial skill got worse."""

    field: str
    baseline_corr: float
    candidate_corr: float
    drop: float             # baseline - candidate, > 0 means degraded


def pattern_regressions(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    max_drop: float = DEFAULT_MAX_PATTERN_DROP,
    fields: tuple[str, ...] = SCORED_FIELDS,
) -> tuple[PatternRegression, ...]:
    """Fields whose centred pattern correlation fell by more than ``max_drop``.

    This is the constraint the scalar objective cannot express.  ``J`` can fall
    while spatial skill is being sold off — a global-mean bias is cheap to buy
    with a scalar knob and a pattern is not — so a candidate is only an
    improvement if ``J`` decreased AND this returns empty.  Evaluate both.
    """
    if max_drop < 0.0:
        raise ValueError(f"max_drop must be >= 0, got {max_drop!r}")
    out = []
    for f in fields:
        if f not in baseline or f not in candidate:
            raise ValueError(
                f"field {f!r} missing from baseline or candidate metrics")
        b = float(baseline[f]["pattern_corr"])
        c = float(candidate[f]["pattern_corr"])
        drop = b - c
        if drop > max_drop:
            out.append(PatternRegression(f, b, c, drop))
    return tuple(out)


class ScorecardComparison(NamedTuple):
    """Verdict for one candidate against a baseline."""

    baseline: ScorecardResult
    candidate: ScorecardResult
    delta_objective: float                      # candidate - baseline, < 0 good
    regressions: tuple[PatternRegression, ...]
    improved: bool                              # J fell AND no pattern loss

    def format_report(self) -> str:
        head = (f"objective {self.baseline.objective:.4f} -> "
                f"{self.candidate.objective:.4f} "
                f"(delta {self.delta_objective:+.4f})")
        if self.regressions:
            body = "\n".join(
                f"  PATTERN REGRESSION {r.field}: r {r.baseline_corr:.3f} -> "
                f"{r.candidate_corr:.3f} (drop {r.drop:+.3f})"
                for r in self.regressions)
            return f"{head}\nREJECTED — spatial skill degraded:\n{body}"
        verdict = "IMPROVED" if self.improved else "NOT IMPROVED"
        return f"{head}\n{verdict} — no pattern regression"


def compare_scorecards(
    baseline_metrics: Mapping[str, Mapping[str, float]],
    candidate_metrics: Mapping[str, Mapping[str, float]],
    weights: ScorecardWeights | None = None,
    max_pattern_drop: float = DEFAULT_MAX_PATTERN_DROP,
) -> ScorecardComparison:
    """Full candidate-vs-baseline verdict: scalar objective AND the constraint."""
    w = weights or ScorecardWeights()
    base = evaluate_scorecard(baseline_metrics, w)
    cand = evaluate_scorecard(candidate_metrics, w)
    regs = pattern_regressions(
        baseline_metrics, candidate_metrics, max_pattern_drop)
    delta = cand.objective - base.objective
    return ScorecardComparison(
        baseline=base, candidate=cand, delta_objective=delta,
        regressions=regs, improved=bool(delta < 0.0 and not regs))


def _is_finite(value: object) -> bool:
    """True for a real, finite number (rejects NaN/inf and non-numerics)."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))
