"""Unit tests for :mod:`legoesm.land.evaluation.scorecard`."""
from __future__ import annotations

import json

import numpy as np
import pytest

from legoesm.land.evaluation.scorecard import (
    Scorecard,
    VariableResult,
    score_variable,
    write_scorecard_json,
)


def test_score_variable_perfect_is_one():
    ref = np.linspace(0, 10, 200)
    mod = ref.copy()
    res = score_variable(
        ref, mod, key="flux:shflx", label="Sensible heat", unit="W m-2",
        group="flux",
    )
    assert res.n == 200
    assert res.score == pytest.approx(1.0, abs=1e-9)
    assert res.metrics["bias_score"] == pytest.approx(1.0)
    assert res.metrics["rmse_score"] == pytest.approx(1.0)


def test_overall_score_uses_default_components_even_for_error_only_display():
    # ILAMB behaviour: the overall score is a fixed functional over the
    # default bias/rmse/taylor score components, independent of which raw
    # metrics the recipe chose to display.  Requesting only 'rmse' for
    # display still yields a well-defined overall score.
    ref = np.linspace(0, 10, 50)
    mod = ref + 0.3
    res = score_variable(ref, mod, key="x", metric_names=["rmse"])
    assert "rmse" in res.metrics
    assert "bias_score" in res.metrics  # folded in for the overall score
    assert np.isfinite(res.score)
    assert 0.0 < res.score <= 1.0


def test_group_and_overall_aggregation():
    # Two groups, each with two variables of known score.
    def fake(key, group, score, weight=1.0):
        return VariableResult(
            key=key, label=key, unit="", group=group, n=10,
            metrics={}, score=score, weight=weight,
        )

    card = Scorecard(
        case="c", model="m", reference="r",
        variables=[
            fake("a", "flux", 1.0),
            fake("b", "flux", 0.0),
            fake("c", "fsun", 0.5),
            fake("d", "fsun", 0.5),
        ],
    )
    gs = card.group_scores()
    assert gs["flux"] == pytest.approx(0.5)
    assert gs["fsun"] == pytest.approx(0.5)
    # Overall = mean of group scores (each group once).
    assert card.overall_score() == pytest.approx(0.5)


def test_weighted_group_score():
    v1 = VariableResult("a", "a", "", "flux", 10, {}, 1.0, weight=3.0)
    v2 = VariableResult("b", "b", "", "flux", 10, {}, 0.0, weight=1.0)
    card = Scorecard("c", "m", "r", [v1, v2])
    # Weighted mean: (1*3 + 0*1)/4 = 0.75.
    assert card.group_scores()["flux"] == pytest.approx(0.75)


def test_nan_scores_are_skipped_not_poisoning():
    v1 = VariableResult("a", "a", "", "flux", 10, {}, float("nan"))
    v2 = VariableResult("b", "b", "", "flux", 10, {}, 0.8)
    card = Scorecard("c", "m", "r", [v1, v2])
    assert card.group_scores()["flux"] == pytest.approx(0.8)


def test_json_is_valid_and_nan_free(tmp_path):
    v = VariableResult(
        "a", "a", "W", "flux", 0, {"rmse": float("nan")}, float("nan")
    )
    card = Scorecard("c", "m", "r", [v])
    path = write_scorecard_json(card, tmp_path / "sc.json")
    text = path.read_text()
    assert "NaN" not in text  # nan mapped to null
    payload = json.loads(text)  # must parse
    sc = payload["scorecards"][0]
    assert sc["overall_score"] is None
    assert sc["variables"]["a"]["metrics"]["rmse"] is None
