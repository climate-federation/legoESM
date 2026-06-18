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


def _cres(base, upd):
    return CorrectionResult(
        updated_config=None, bias=_bias(base, upd),
        worst_column_change=jnp.asarray(0.0), feedback_field=jnp.zeros((2, 2)),
        n_corrected=1)


def _mres(base, upd):
    return MultiCorrectionResult(
        updated_config=None, bias=_bias(base, upd),
        worst_column_change=jnp.asarray(0.0), feedback_fields={}, n_corrected=1)


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
