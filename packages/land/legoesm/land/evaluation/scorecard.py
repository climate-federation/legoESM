"""Aggregate per-variable metrics into a hierarchical, JSON-able scorecard.

Mirrors ILAMB's aggregation: each variable gets an overall unit-interval
score from a weighted mean of its component scores (bias / RMSE / Taylor),
variables roll up into groups, and groups roll up into one overall score for
the ``(model, reference)`` pair.  The default component weights follow
ILAMB (bias and RMSE weighted twice the distribution term), but they are
configurable per recipe.

The output is a plain nested ``dict`` (``write_scorecard_json`` dumps it),
deliberately close in spirit to the ``parity_summary.json`` that
``validate_clm_ml_canopy.py`` already emits, so downstream tooling and
collaborators read one stable shape.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from legoesm.land.evaluation import metrics as _m

# Default component-score weights for the overall variable score (ILAMB:
# bias and centralised-RMSE dominate, distribution/Taylor is secondary).
# Only score-type metrics (values already in [0, 1], higher-better) may be
# aggregated into an overall score.
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "bias_score": 2.0,
    "rmse_score": 2.0,
    "taylor_score": 1.0,
}


@dataclass
class VariableResult:
    """Metrics + overall score for one variable against one reference."""

    key: str
    label: str
    unit: str
    group: str
    n: int
    metrics: dict[str, float]
    score: float
    weight: float = 1.0

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "unit": self.unit,
            "group": self.group,
            "n": self.n,
            "weight": self.weight,
            "score": _jsonable(self.score),
            "metrics": {k: _jsonable(v) for k, v in self.metrics.items()},
        }


def _jsonable(x: float) -> float | None:
    """Map NaN/inf to ``None`` so the JSON is valid (no ``NaN`` literal)."""
    xf = float(x)
    return xf if np.isfinite(xf) else None


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    """Weighted mean over ``(value, weight)``, skipping non-finite values."""
    num = 0.0
    den = 0.0
    for val, w in pairs:
        if np.isfinite(val) and w > 0:
            num += val * w
            den += w
    return num / den if den > 0 else float("nan")


def score_variable(
    ref: np.ndarray,
    mod: np.ndarray,
    *,
    key: str,
    label: str = "",
    unit: str = "",
    group: str = "default",
    metric_names: list[str] | None = None,
    score_weights: dict[str, float] | None = None,
    weight: float = 1.0,
) -> VariableResult:
    """Compute ``metric_names`` for one variable and its overall score.

    The overall score is the ``score_weights``-weighted mean of the
    component *score* metrics that are both requested and present in
    ``score_weights``.  Requesting only error-type metrics (e.g. just
    ``rmse``) yields a ``nan`` overall score — intentional: an overall
    score needs at least one [0, 1] component.
    """
    metric_names = metric_names or [
        "bias", "rmse", "nrmse", "corr", "bias_score", "rmse_score",
        "taylor_score",
    ]
    weights = score_weights or DEFAULT_SCORE_WEIGHTS

    r, mvals = _m.finite_pair(ref, mod)
    n = int(r.size)
    computed: dict[str, float] = {}
    for name in metric_names:
        computed[name] = _m.get_metric(name)(ref, mod)

    # Overall score: weighted mean of requested score-type metrics.  If a
    # weighted score metric was not requested, compute it anyway so the
    # overall score is well-defined and reproducible.
    score_pairs: list[tuple[float, float]] = []
    for sname, sw in weights.items():
        val = computed.get(sname)
        if val is None:
            val = _m.get_metric(sname)(ref, mod)
            computed[sname] = val
        score_pairs.append((val, sw))
    overall = _weighted_mean(score_pairs)

    return VariableResult(
        key=key, label=label or key, unit=unit, group=group,
        n=n, metrics=computed, score=overall, weight=weight,
    )


@dataclass
class Scorecard:
    """A ``(model, reference)`` comparison over many variables."""

    case: str
    model: str
    reference: str
    variables: list[VariableResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def group_scores(self) -> dict[str, float]:
        """Weighted-mean score per variable group."""
        groups: dict[str, list[tuple[float, float]]] = {}
        for v in self.variables:
            groups.setdefault(v.group, []).append((v.score, v.weight))
        return {g: _weighted_mean(p) for g, p in groups.items()}

    def overall_score(self) -> float:
        """Unweighted mean of the group scores (each group counts once)."""
        gs = [s for s in self.group_scores().values() if np.isfinite(s)]
        return float(np.mean(gs)) if gs else float("nan")

    def as_dict(self) -> dict:
        return {
            "case": self.case,
            "model": self.model,
            "reference": self.reference,
            "overall_score": _jsonable(self.overall_score()),
            "group_scores": {
                g: _jsonable(s) for g, s in self.group_scores().items()
            },
            "variables": {v.key: v.as_dict() for v in self.variables},
            "metadata": self.metadata,
        }


def write_scorecard_json(
    cards: Scorecard | list[Scorecard], path: str | Path
) -> Path:
    """Write one or more scorecards to a JSON file; returns the path."""
    if isinstance(cards, Scorecard):
        cards = [cards]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scorecards": [c.as_dict() for c in cards],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    return path
