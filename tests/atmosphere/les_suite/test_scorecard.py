"""Tests for the LES-suite scorecard assembly (Q2 ranking + Q3 spread)."""
from __future__ import annotations

import pytest
from legoesm.atmosphere.les_suite.scorecard import (
    ScorecardError,
    assemble_scorecard,
    coefficient_spreads,
    rank_closures_per_regime,
    render_markdown,
)


def _rec(case, scheme, regime, best_loss, default_loss=1.0, overrides=None):
    return {
        "case": case, "scheme": scheme, "regime": regime,
        "best_loss": best_loss, "default_loss": default_loss,
        "best_overrides": overrides or {},
    }


def test_ranking_orders_by_best_loss():
    recs = [
        _rec("c1", "louis", "dry_convective", 0.30),
        _rec("c1", "mynn25", "dry_convective", 0.10),
        _rec("c1", "ysu", "dry_convective", 0.20),
    ]
    ranking = rank_closures_per_regime(recs)
    assert len(ranking) == 1
    schemes = [row[0] for row in ranking[0].ranked]
    assert schemes == ["mynn25", "ysu", "louis"]  # ascending loss


def test_ranking_means_multiple_cases_per_regime():
    recs = [
        _rec("c1", "mynn25", "dry_convective", 0.10),
        _rec("c2", "mynn25", "dry_convective", 0.30),  # same regime+scheme
    ]
    ranking = rank_closures_per_regime(recs)
    # mean best_loss = 0.2
    assert ranking[0].ranked[0][1] == pytest.approx(0.2)


def test_ranking_separates_regimes():
    recs = [
        _rec("c1", "mynn25", "dry_convective", 0.10),
        _rec("c2", "mynn25", "dry_stable", 0.40),
    ]
    ranking = rank_closures_per_regime(recs)
    regimes = {r.regime for r in ranking}
    assert regimes == {"dry_convective", "dry_stable"}


def test_coefficient_spread_across_regimes():
    recs = [
        _rec("c1", "mynn25", "dry_convective", 0.1, overrides={"A1": 1.0}),
        _rec("c2", "mynn25", "dry_stable", 0.1, overrides={"A1": 1.6}),
    ]
    spreads = coefficient_spreads(recs)
    a1 = next(s for s in spreads if s.field == "A1")
    assert a1.n_regimes == 2
    assert a1.spread == pytest.approx(0.6)
    assert a1.per_regime["dry_convective"] == pytest.approx(1.0)


def test_coefficient_spread_picks_regime_best():
    # two records in the same regime: the lower-loss one's coeff is used
    recs = [
        _rec("c1", "mynn25", "dry_convective", 0.5, overrides={"A1": 2.0}),
        _rec("c1b", "mynn25", "dry_convective", 0.1, overrides={"A1": 1.0}),  # better
        _rec("c2", "mynn25", "dry_stable", 0.1, overrides={"A1": 1.0}),
    ]
    spreads = coefficient_spreads(recs)
    a1 = next(s for s in spreads if s.field == "A1")
    # dry_convective chooses A1=1.0 (loss 0.1) not 2.0 → spread 0
    assert a1.per_regime["dry_convective"] == pytest.approx(1.0)
    assert a1.spread == pytest.approx(0.0)


def test_single_regime_coeff_spread_zero():
    recs = [_rec("c1", "mynn25", "dry_convective", 0.1, overrides={"A1": 1.0})]
    spreads = coefficient_spreads(recs)
    assert spreads[0].spread == pytest.approx(0.0)
    assert spreads[0].n_regimes == 1


def test_assemble_and_render():
    recs = [
        _rec("c1", "mynn25", "dry_convective", 0.10, overrides={"A1": 1.0}),
        _rec("c2", "mynn25", "dry_stable", 0.20, overrides={"A1": 1.5}),
        _rec("c1", "louis", "dry_convective", 0.30, overrides={"Ri_crit": 0.25}),
    ]
    card = assemble_scorecard(recs)
    md = render_markdown(card)
    assert "Q2" in md and "Q3" in md
    assert "mynn25" in md and "louis" in md
    assert "dry_convective" in md


def test_missing_field_rejected():
    with pytest.raises(ScorecardError):
        assemble_scorecard([{"case": "c1", "scheme": "mynn25"}])  # no regime/best_loss


def test_empty_records_ok():
    card = assemble_scorecard([])
    assert card.rankings == ()
    assert card.spreads == ()
    assert "no tuned coefficients" in render_markdown(card)


def test_null_default_loss_falls_back_to_best():
    # a diverged default writes default_loss=null; ranking must stay finite
    recs = [_rec("c1", "mynn25", "dry_convective", 0.1, default_loss=None)]
    ranking = rank_closures_per_regime(recs)
    assert ranking[0].ranked[0][2] == pytest.approx(0.1)  # default←best fallback
