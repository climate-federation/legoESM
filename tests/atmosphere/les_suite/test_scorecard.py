"""Tests for the LES-suite scorecard assembly (Q2 ranking + Q3 spread)."""
from __future__ import annotations

import pytest
from legoesm.atmosphere.les_suite.scorecard import (
    ScorecardError,
    assemble_scorecard,
    coefficient_spreads,
    local_vs_nonlocal_skill,
    rank_closures_per_flux,
    rank_closures_per_regime,
    render_markdown,
)


def _rec(case, scheme, regime, best_loss, default_loss=1.0, overrides=None, q0=None):
    rec = {
        "case": case, "scheme": scheme, "regime": regime,
        "best_loss": best_loss, "default_loss": default_loss,
        "best_overrides": overrides or {},
    }
    if q0 is not None:
        rec["q0"] = q0
    return rec


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


def test_per_flux_groups_by_q0():
    # same regime+scheme at two distinct fluxes → two per-flux slices, one each
    recs = [
        _rec("cbl", "louis", "dry_convective", 0.10, q0=0.02),
        _rec("cbl", "mynn25", "dry_convective", 0.20, q0=0.02),
        _rec("cbl", "louis", "dry_convective", 0.40, q0=0.12),
        _rec("cbl", "mynn25", "dry_convective", 0.50, q0=0.12),
    ]
    per_flux = rank_closures_per_flux(recs)
    assert [rk.q0 for rk in per_flux] == [0.02, 0.12]  # sorted ascending flux
    lo, hi = per_flux
    assert [row[0] for row in lo.ranked] == ["louis", "mynn25"]  # 0.10 < 0.20
    assert lo.ranked[0][1] == pytest.approx(0.10)
    assert hi.ranked[0][1] == pytest.approx(0.40)


def test_dedup_same_flux_scheme_keeps_min():
    # a duplicate (flux, closure) — e.g. a stale tuned JSON — must NOT double-count;
    # the lower-loss record wins and the flux is a single slice.
    recs = [
        _rec("cbl", "louis", "dry_convective", 0.30, q0=0.06),  # stale, worse
        _rec("cbl", "louis", "dry_convective", 0.20, q0=0.06),  # replacement
    ]
    per_flux = rank_closures_per_flux(recs)
    assert len(per_flux) == 1
    assert len(per_flux[0].ranked) == 1
    assert per_flux[0].ranked[0][1] == pytest.approx(0.20)


def test_flux_mean_does_not_double_count_duplicate():
    # regime flux-mean of one closure over fluxes {0.02:0.10, 0.06:0.20} = 0.15,
    # even with a duplicate 0.06 record present (dedup → single 0.06 slice).
    recs = [
        _rec("cbl", "louis", "dry_convective", 0.10, q0=0.02),
        _rec("cbl", "louis", "dry_convective", 0.20, q0=0.06),
        _rec("cbl", "louis", "dry_convective", 0.99, q0=0.06),  # stale dup, ignored
    ]
    ranking = rank_closures_per_regime(recs)
    assert ranking[0].ranked[0][1] == pytest.approx(0.15)


def test_render_shows_per_flux_and_mean():
    recs = [
        _rec("cbl", "louis", "dry_convective", 0.10, q0=0.02),
        _rec("cbl", "louis", "dry_convective", 0.20, q0=0.12),
    ]
    md = render_markdown(assemble_scorecard(recs))
    assert "per surface-flux ranking" in md
    assert "Q0 = 0.02 K m/s" in md and "Q0 = 0.12 K m/s" in md
    assert "flux-mean ranking (mean over 2 flux points)" in md


def test_skill_local_vs_nonlocal_margin_and_winner():
    # at Q0=0.02: best local=louis 0.13, best nonlocal=holtslag 0.10 → nonlocal wins
    # (margin = 0.13 - 0.10 = +0.03 > 0). Other families are ignored.
    recs = [
        _rec("cbl", "louis", "dry_convective", 0.13, q0=0.02),        # local
        _rec("cbl", "smagorinsky", "dry_convective", 0.15, q0=0.02),  # local (worse)
        _rec("cbl", "holtslag_boville", "dry_convective", 0.10, q0=0.02),  # nonlocal
        _rec("cbl", "ysu", "dry_convective", 0.12, q0=0.02),          # nonlocal (worse)
        _rec("cbl", "mynn25", "dry_convective", 0.09, q0=0.02),       # 1.5-order: ignored
    ]
    skill = local_vs_nonlocal_skill(recs)
    assert len(skill) == 1
    s = skill[0]
    assert s.best_local == ("louis", pytest.approx(0.13))
    assert s.best_nonlocal == ("holtslag_boville", pytest.approx(0.10))
    assert s.margin == pytest.approx(0.03)
    assert s.nonlocal_wins is True


def test_skill_local_wins_negative_margin():
    recs = [
        _rec("cbl", "louis", "dry_convective", 0.10, q0=0.06),       # local better
        _rec("cbl", "holtslag_boville", "dry_convective", 0.20, q0=0.06),
    ]
    s = local_vs_nonlocal_skill(recs)[0]
    assert s.margin == pytest.approx(-0.10)
    assert s.nonlocal_wins is False


def test_skill_missing_family_gives_none():
    # only a local closure present → no crossing (nonlocal side absent)
    recs = [_rec("cbl", "louis", "dry_convective", 0.10, q0=0.06)]
    s = local_vs_nonlocal_skill(recs)[0]
    assert s.best_nonlocal is None
    assert s.margin is None and s.nonlocal_wins is None


def test_skill_rendered_in_scorecard():
    recs = [
        _rec("cbl", "louis", "dry_convective", 0.13, q0=0.02),
        _rec("cbl", "holtslag_boville", "dry_convective", 0.10, q0=0.02),
    ]
    md = render_markdown(assemble_scorecard(recs))
    assert "Q1b" in md and "nonlocal wins" in md
    assert "Q0=0.02" in md


def test_missing_field_rejected():
    with pytest.raises(ScorecardError):
        assemble_scorecard([{"case": "c1", "scheme": "mynn25"}])  # no regime/best_loss


def test_non_finite_best_loss_rejected():
    import math
    for bad in (math.nan, math.inf, None):
        with pytest.raises(ScorecardError):
            assemble_scorecard([_rec("c1", "mynn25", "dry_convective", bad)])


def test_typed_flux_key_no_label_collision():
    # a stamped Q0=0.06 record must NOT merge with a legacy record whose fallback
    # label happens to read "Q0=0.06" — typed keys keep them distinct slices.
    recs = [
        _rec("c1", "louis", "dry_convective", 0.10, q0=0.06),
        {"case": "c1", "scheme": "louis", "regime": "dry_convective",
         "best_loss": 0.20, "default_loss": 1.0, "best_overrides": {},
         "artifact": "Q0=0.06"},  # crafted collision-bait label, no q0
    ]
    per_flux = rank_closures_per_flux(recs)
    assert len(per_flux) == 2  # two distinct slices, not merged


def test_float32_and_exact_flux_merge():
    # the real mix: a tuner-stamped q0 read from a float32 artifact
    # (float32(0.06)=0.059999998…) MUST merge with an exact/legacy-parsed 0.06 —
    # otherwise stale + new records of the same flux split and corrupt the mean.
    import numpy as np
    recs = [
        _rec("c1", "louis", "dry_convective", 0.10, q0=float(np.float32(0.06))),
        _rec("c1", "mynn25", "dry_convective", 0.20, q0=0.06),  # exact
    ]
    per_flux = rank_closures_per_flux(recs)
    assert len(per_flux) == 1                     # one canonical 0.06 slice
    assert per_flux[0].q0 == pytest.approx(0.06)  # canonical value, not f32 noise
    assert per_flux[0].subcase == "Q0=0.06"       # label from the canonical flux
    assert {row[0] for row in per_flux[0].ranked} == {"louis", "mynn25"}


def test_flux_mean_denominator_flags_incomplete_matrix():
    # louis appears at both fluxes; mynn25 only at one → its mean denominator is 1,
    # flagged in the rendered flux-mean table so the mean is not read as over 2.
    recs = [
        _rec("c1", "louis", "dry_convective", 0.10, q0=0.02),
        _rec("c1", "louis", "dry_convective", 0.20, q0=0.12),
        _rec("c1", "mynn25", "dry_convective", 0.50, q0=0.02),  # missing at 0.12
    ]
    ranking = rank_closures_per_regime(recs)
    counts = ranking[0].counts
    assert counts["louis"] == 2 and counts["mynn25"] == 1
    md = render_markdown(assemble_scorecard(recs))
    assert "n fluxes" in md and "⚠" in md  # incomplete closure is flagged


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
