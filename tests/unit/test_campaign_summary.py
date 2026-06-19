"""Unit tests for the correction-campaign diagnostics summary."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
from legoesm.training.bias_metrics import BiasImprovement
from legoesm.training.campaign_summary import (
    CampaignSummary,
    summarize_campaign,
)
from legoesm.training.correction_loop import (
    CampaignResult,
    CorrectionResult,
    MultiCampaignResult,
    MultiCorrectionResult,
)


def _bias(base, upd):
    frac = (base - upd) / base if base != 0.0 else 0.0
    return BiasImprovement(
        baseline_bias=jnp.asarray(base), updated_bias=jnp.asarray(upd),
        absolute_reduction=jnp.asarray(base - upd),
        fractional_improvement=jnp.asarray(frac), improved=jnp.asarray(upd < base))


def _cres(base, upd, *, n_diagnosed=1, n_diagnoses_valid=1):
    return CorrectionResult(
        updated_config=None, bias=_bias(base, upd),
        worst_column_change=jnp.asarray(0.0), feedback_field=jnp.zeros((2, 2)),
        n_corrected=1, n_diagnosed=n_diagnosed, n_diagnoses_valid=n_diagnoses_valid)


def _mres(base, upd, *, n_diagnosed=1, n_diagnoses_valid=1):
    return MultiCorrectionResult(
        updated_config=None, bias=_bias(base, upd),
        worst_column_change=jnp.asarray(0.0), feedback_fields={}, n_corrected=1,
        n_diagnosed=n_diagnosed, n_diagnoses_valid=n_diagnoses_valid)


def test_summarize_single_campaign():
    field = jnp.array([0.1, 0.5, 1.2, 0.8])          # 1 at lo (0.1), 1 at hi (1.2)
    result = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=field),
        iterations=(_cres(1.0, 0.6), _cres(0.6, 0.3)),
        final_field=field.reshape(2, 2),
        accepted=(True, True), stop_reason="converged")
    s = summarize_campaign(result, promotion_key="clubb_lite_C_K")
    assert isinstance(s, CampaignSummary)
    assert s.n_rounds == 2 and s.n_accepted == 2
    assert s.acceptance_rate == pytest.approx(1.0)
    assert s.initial_bias == pytest.approx(1.0)
    assert s.final_bias == pytest.approx(0.3)        # last ACCEPTED updated_bias
    assert s.absolute_reduction == pytest.approx(0.7)
    assert s.fractional_reduction == pytest.approx(0.7)
    assert s.stop_reason == "converged"
    (c,) = s.coefficients
    assert c.promotion_key == "clubb_lite_C_K"
    assert c.bounds == (0.1, 1.2)
    assert c.n_at_lower_bound == 1 and c.n_at_upper_bound == 1
    assert "clamp binding" in s.report()
    assert "73.0%" not in s.report()                 # sanity: 70.0%
    assert "70.0%" in s.report()


def test_summarize_single_requires_promotion_key():
    result = CampaignResult(
        final_config=CLUBBLiteConfig(), iterations=(_cres(1.0, 0.5),),
        final_field=jnp.full((2, 2), 0.4), accepted=(True,))
    with pytest.raises(ValueError, match="needs promotion_key"):
        summarize_campaign(result)


def test_summarize_multi_campaign_bound_hitting():
    ck = jnp.array([[0.4, 0.5], [0.6, 1.2]])         # one column at the C_K upper bound
    prt = jnp.array([[0.7, 0.8], [0.9, 1.0]])
    result = MultiCampaignResult(
        final_config=CLUBBLiteConfig(C_K=ck.reshape(-1), Pr_t=prt.reshape(-1)),
        iterations=(_mres(2.0, 1.0), _mres(1.0, 1.0)),  # round 1 rejected (no change)
        final_fields={"clubb_lite_C_K": ck, "clubb_lite_Pr_t": prt},
        accepted=(True, False), stop_reason="max_iterations")
    s = summarize_campaign(result)
    assert s.n_accepted == 1 and s.acceptance_rate == pytest.approx(0.5)
    assert s.initial_bias == pytest.approx(2.0)
    assert s.final_bias == pytest.approx(1.0)        # last ACCEPTED (round 0), not the rejected
    by_key = {c.promotion_key: c for c in s.coefficients}
    assert by_key["clubb_lite_C_K"].bounds == (0.1, 1.2)
    assert by_key["clubb_lite_C_K"].n_at_upper_bound == 1
    assert by_key["clubb_lite_Pr_t"].n_at_upper_bound == 0
    assert by_key["clubb_lite_C_K"].field_std > 0.0  # spatially-varying correction


def test_summarize_no_rounds_safe():
    result = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=jnp.full((4,), 0.4)),
        iterations=(), final_field=jnp.full((2, 2), 0.4), accepted=())
    s = summarize_campaign(result, promotion_key="clubb_lite_C_K")
    assert s.n_rounds == 0 and s.fractional_reduction == 0.0
    assert s.coefficients[0].field_std == pytest.approx(0.0)   # uniform = no correction
    assert s.per_variable is None                              # no rounds ⇒ no per-variable


def _pv(t_rmse):
    from legoesm.training.bias_metrics import PerVariableBias
    return PerVariableBias(jnp.asarray(t_rmse), jnp.asarray(1.0e-3),
                           jnp.asarray(2.0), jnp.asarray(float("nan")))


def _cres_pv(base, upd, t_base, t_upd):
    from legoesm.training.bias_metrics import compare_per_variable_bias
    return _cres(base, upd)._replace(
        per_variable_bias=compare_per_variable_bias(_pv(t_base), _pv(t_upd)))


def test_summarize_per_variable_trajectory_round0_to_last_accepted():
    """The campaign per-variable bias is round-0 BASELINE → LAST-ACCEPTED updated (the
    SAME accepted-gate logic as the combined bias), and reported."""
    result = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=jnp.full((4,), 0.4)),
        iterations=(_cres_pv(1.0, 0.6, 4.0, 3.0), _cres_pv(0.6, 0.3, 3.0, 1.0)),
        final_field=jnp.full((2, 2), 0.4), accepted=(True, True), stop_reason="converged")
    s = summarize_campaign(result, promotion_key="clubb_lite_C_K")
    pv = s.per_variable
    assert pv is not None
    assert float(pv.baseline.global_T_rmse_K) == pytest.approx(4.0)   # round-0 baseline
    assert float(pv.updated.global_T_rmse_K) == pytest.approx(1.0)    # round-1 (accepted) updated
    assert bool(pv.T_improved)
    rep = s.report()
    assert "Per-variable RMSE" in rep and "T 4->1K" in rep


def test_summarize_per_variable_uses_last_accepted_not_rejected():
    """A REJECTED final round must NOT set the per-variable final (mirrors the combined
    final_bias): final per-variable = the last ACCEPTED round's updated."""
    result = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=jnp.full((4,), 0.4)),
        iterations=(_cres_pv(2.0, 1.0, 4.0, 2.0),       # accepted
                    _cres_pv(1.0, 1.0, 2.0, 9.9)),      # rejected (T would jump to 9.9)
        final_field=jnp.full((2, 2), 0.4), accepted=(True, False))
    s = summarize_campaign(result, promotion_key="clubb_lite_C_K")
    assert float(s.per_variable.updated.global_T_rmse_K) == pytest.approx(2.0)  # NOT 9.9


def test_summarize_per_variable_none_without_error_fields():
    """A campaign whose rounds carried no per_variable_bias ⇒ per_variable None."""
    result = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=jnp.full((4,), 0.4)),
        iterations=(_cres(1.0, 0.6),), final_field=jnp.full((2, 2), 0.4),
        accepted=(True,))
    s = summarize_campaign(result, promotion_key="clubb_lite_C_K")
    assert s.per_variable is None


from legoesm.training.campaign_summary import campaign_health  # noqa: E402


def _multi_result(ck_field, bias_pair, accepted, *, n_diagnosed=1, n_diagnoses_valid=1):
    prt = jnp.full(ck_field.shape, 0.8)
    iters = tuple(
        _mres(b, u, n_diagnosed=n_diagnosed, n_diagnoses_valid=n_diagnoses_valid)
        for (b, u) in bias_pair)
    return MultiCampaignResult(
        final_config=CLUBBLiteConfig(C_K=ck_field.reshape(-1), Pr_t=prt.reshape(-1)),
        iterations=iters,
        final_fields={"clubb_lite_C_K": ck_field, "clubb_lite_Pr_t": prt},
        accepted=accepted, stop_reason="max_iterations")


def test_campaign_health_improved():
    # In-bounds C_K, 60% bias reduction → "improved".
    ck = jnp.array([[0.4, 0.5], [0.6, 0.7]])
    s = summarize_campaign(_multi_result(ck, [(1.0, 0.4)], (True,)))
    h = campaign_health(s)
    assert h.status == "improved" and h.ok
    assert "60%" in h.message


def test_campaign_health_stalled():
    # In-bounds C_K, tiny bias reduction → "stalled".
    ck = jnp.array([[0.4, 0.5], [0.6, 0.7]])
    s = summarize_campaign(_multi_result(ck, [(1.0, 0.995)], (True,)))
    h = campaign_health(s)
    assert h.status == "stalled" and not h.ok
    assert "not improving" in h.message


def test_campaign_health_clamp_limited_when_not_improving():
    # Bias barely moved (1%) AND half the C_K columns pinned at the upper bound →
    # "clamp_limited" (the clamp is the likely cause of the non-improvement).
    ck = jnp.array([[1.2, 1.2], [0.6, 0.7]])      # 2/4 = 50% at hi bound
    s = summarize_campaign(_multi_result(ck, [(1.0, 0.99)], (True,)))
    h = campaign_health(s)
    assert h.status == "clamp_limited"
    assert "clubb_lite_C_K" in h.message and "outside its calibratable range" in h.message


def test_campaign_health_improved_takes_precedence_over_clamp():
    # A STRONGLY-improving run with heavy clamp-binding is still "improved" (ok),
    # with the clamp surfaced as a NOTE — not demoted to a warning status.
    ck = jnp.array([[1.2, 1.2], [0.6, 0.7]])      # 50% at hi bound
    s = summarize_campaign(_multi_result(ck, [(1.0, 0.3)], (True,)))   # 70% reduction
    h = campaign_health(s)
    assert h.status == "improved" and h.ok
    assert "Note:" in h.message and "calibratable range" in h.message


def test_summary_surfaces_diagnosis_validity_totals():
    """The summary sums n_diagnosed + n_diagnoses_valid across rounds, and report()
    states the valid fraction (iter 100)."""
    field = jnp.array([0.4, 0.5, 0.6, 0.7])
    result = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=field),
        iterations=(_cres(1.0, 0.6, n_diagnosed=5, n_diagnoses_valid=3),
                    _cres(0.6, 0.3, n_diagnosed=4, n_diagnoses_valid=2)),
        final_field=field.reshape(2, 2), accepted=(True, True))
    s = summarize_campaign(result, promotion_key="clubb_lite_C_K")
    assert s.n_diagnosed_total == 9 and s.n_diagnoses_valid_total == 5
    assert "5/9 valid" in s.report()


def test_campaign_health_no_valid_diagnoses():
    """A non-improving campaign where LES ran but EVERY diagnosis was rejected →
    'no_valid_diagnoses' (the root cause is the LES, not the correction; iter 100)."""
    ck = jnp.array([[0.4, 0.5], [0.6, 0.7]])         # in-bounds → no clamp
    s = summarize_campaign(_multi_result(
        ck, [(1.0, 0.995)], (True,), n_diagnosed=6, n_diagnoses_valid=0))
    h = campaign_health(s)
    assert h.status == "no_valid_diagnoses" and not h.ok
    assert "All 6 LES diagnoses were rejected" in h.message
    assert "lengthen or properly force" in h.message


def test_campaign_health_no_valid_diagnoses_precedes_clamp():
    """When NO diagnosis is valid, nothing is corrected, so 'no_valid_diagnoses' is
    reported even if the (background) field happens to sit at a bound — it is checked
    before 'clamp_limited' (iter 100 ordering)."""
    ck = jnp.array([[1.2, 1.2], [0.6, 0.7]])         # 50% at hi bound
    s = summarize_campaign(_multi_result(
        ck, [(1.0, 0.99)], (True,), n_diagnosed=3, n_diagnoses_valid=0))
    assert campaign_health(s).status == "no_valid_diagnoses"


def test_campaign_health_some_valid_is_not_no_valid_diagnoses():
    """≥1 valid diagnosis but still not improving (and clamp-bound) → 'clamp_limited',
    NOT 'no_valid_diagnoses' (the gate is n_diagnoses_valid_total == 0)."""
    ck = jnp.array([[1.2, 1.2], [0.6, 0.7]])
    s = summarize_campaign(_multi_result(
        ck, [(1.0, 0.99)], (True,), n_diagnosed=3, n_diagnoses_valid=1))
    assert campaign_health(s).status == "clamp_limited"


def test_campaign_health_no_rounds():
    s = summarize_campaign(
        CampaignResult(final_config=CLUBBLiteConfig(C_K=jnp.full((4,), 0.4)),
                       iterations=(), final_field=jnp.full((2, 2), 0.4), accepted=()),
        promotion_key="clubb_lite_C_K")
    h = campaign_health(s)
    assert h.status == "no_rounds"


def _pv_tw(t_rmse, wind_rmse):
    from legoesm.training.bias_metrics import PerVariableBias
    return PerVariableBias(jnp.asarray(t_rmse), jnp.asarray(1.0e-3),
                           jnp.asarray(wind_rmse), jnp.asarray(float("nan")))


def _cres_pv_tw(base, upd, pv_base, pv_upd):
    from legoesm.training.bias_metrics import compare_per_variable_bias
    return _cres(base, upd)._replace(
        per_variable_bias=compare_per_variable_bias(pv_base, pv_upd))


def _health_for(pv_base, pv_upd):
    result = CampaignResult(
        final_config=CLUBBLiteConfig(C_K=jnp.full((4,), 0.4)),
        iterations=(_cres_pv_tw(1.0, 0.6, pv_base, pv_upd),),  # combined 1.0->0.6 (40%)
        final_field=jnp.full((2, 2), 0.4), accepted=(True,))
    return campaign_health(summarize_campaign(result, promotion_key="clubb_lite_C_K"))


def test_health_flags_per_variable_tradeoff():
    """Combined bias improved (40%) but WIND RMSE worsened (2->5) while T improved
    (4->3): status stays 'improved' (.ok) — the combined target fell — but the message
    NOTES the per-variable trade-off naming wind (the metric-gaming case)."""
    h = _health_for(_pv_tw(4.0, 2.0), _pv_tw(3.0, 5.0))
    assert h.status == "improved" and h.ok          # combined improved ⇒ still ok
    assert "trade-off" in h.message and "wind" in h.message and "WORSENED" in h.message
    assert "T" not in h.message.split("WORSENED")[0].split("but")[1]  # T not flagged


def test_health_no_tradeoff_when_all_variables_improve():
    """Combined + every variable improved ⇒ NO trade-off note."""
    h = _health_for(_pv_tw(4.0, 5.0), _pv_tw(3.0, 2.0))   # T 4->3, wind 5->2, both down
    assert h.status == "improved" and "trade-off" not in h.message


def test_health_tradeoff_ignores_fp_noise():
    """A sub-threshold RMSE increase is NOT a trade-off (FP/noise-safe)."""
    h = _health_for(_pv_tw(4.0, 2.0), _pv_tw(3.0, 2.01))  # wind +0.5% < 5% default
    assert "trade-off" not in h.message


def test_health_tradeoff_default_threshold_5pct():
    """The default trade-off threshold is 5% (decoupled from the 2% improvement
    threshold): a 3% wind regression is NOT flagged, a 10% one IS — and the threshold
    is tunable via tradeoff_fraction_warn."""
    s3 = summarize_campaign(
        CampaignResult(final_config=CLUBBLiteConfig(C_K=jnp.full((4,), 0.4)),
                       iterations=(_cres_pv_tw(1.0, 0.6, _pv_tw(4.0, 2.0),
                                               _pv_tw(3.0, 2.06)),),   # wind +3%
                       final_field=jnp.full((2, 2), 0.4), accepted=(True,)),
        promotion_key="clubb_lite_C_K")
    assert "trade-off" not in campaign_health(s3).message            # 3% < 5% default
    assert "trade-off" in campaign_health(s3, tradeoff_fraction_warn=0.02).message  # tunable
    s10 = summarize_campaign(
        CampaignResult(final_config=CLUBBLiteConfig(C_K=jnp.full((4,), 0.4)),
                       iterations=(_cres_pv_tw(1.0, 0.6, _pv_tw(4.0, 2.0),
                                               _pv_tw(3.0, 2.2)),),   # wind +10%
                       final_field=jnp.full((2, 2), 0.4), accepted=(True,)),
        promotion_key="clubb_lite_C_K")
    assert "trade-off" in campaign_health(s10).message               # 10% >= 5% default


def test_per_variable_tradeoff_note_direct():
    """The helper: None -> ''; a >=threshold worsening -> note; equal/sub-threshold -> ''."""
    from legoesm.training.bias_metrics import compare_per_variable_bias
    from legoesm.training.campaign_summary import _per_variable_tradeoff_note

    assert _per_variable_tradeoff_note(None, 0.02) == ""
    worse = compare_per_variable_bias(_pv_tw(4.0, 2.0), _pv_tw(3.0, 5.0))
    assert "wind" in _per_variable_tradeoff_note(worse, 0.02)
    same = compare_per_variable_bias(_pv_tw(4.0, 2.0), _pv_tw(3.0, 2.0))   # wind unchanged
    assert _per_variable_tradeoff_note(same, 0.02) == ""


def test_coefficient_clamp_fraction():
    ck = jnp.array([0.1, 1.2, 0.5, 0.6])          # 1 lo + 1 hi of 4 = 0.5
    s = summarize_campaign(_multi_result(ck.reshape(2, 2), [(1.0, 0.5)], (True,)))
    ck_c = next(c for c in s.coefficients if c.promotion_key == "clubb_lite_C_K")
    assert ck_c.n_columns == 4
    assert ck_c.clamp_fraction == pytest.approx(0.5)
