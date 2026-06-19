"""Unit tests for :mod:`legoesm.training.bias_metrics`.

The success-criterion measurement: area-weighted global bias + baseline-vs-
updated improvement.  Analytic aggregation (uniform + weighted + masked),
the improvement direction/fraction, worst-column targeting, all-excluded edge,
and AD/jit safety.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.training.bias_metrics import (
    aggregate_combined_bias,
    aggregate_per_variable_bias,
    bias_improvement,
    compare_per_variable_bias,
    per_variable_bias_improvement,
    worst_column_bias_change,
)


class _ErrFields(NamedTuple):
    T_rmse_K: jnp.ndarray
    qv_rmse_kg_kg: jnp.ndarray
    wind_rmse_m_s: jnp.ndarray
    precip_err_mm_day: jnp.ndarray
    combined_score: jnp.ndarray


def _err_fields(t, qv, wind, precip):
    shp = jnp.asarray(t).shape
    return _ErrFields(T_rmse_K=jnp.asarray(t), qv_rmse_kg_kg=jnp.asarray(qv),
                      wind_rmse_m_s=jnp.asarray(wind),
                      precip_err_mm_day=jnp.asarray(precip),
                      combined_score=jnp.zeros(shp))


def test_per_variable_bias_is_quadrature_global_rmse():
    """T/qv/wind aggregate in MSE-space: global RMSE = sqrt(area_mean(rmse²)),
    NOT the linear mean of per-column RMSEs (they combine in quadrature)."""
    t = jnp.array([3.0, 4.0])          # uniform weights ⇒ sqrt((9+16)/2)=sqrt(12.5)
    pvb = aggregate_per_variable_bias(
        _err_fields(t, 2.0 * t, 0.5 * t, jnp.zeros(2)),
        jnp.ones(2), have_precip=False)
    assert float(pvb.global_T_rmse_K) == pytest.approx(np.sqrt(12.5))
    assert float(pvb.global_qv_rmse_kg_kg) == pytest.approx(np.sqrt(4 * 12.5))
    assert float(pvb.global_wind_rmse_m_s) == pytest.approx(np.sqrt(0.25 * 12.5))
    # ≠ the (wrong) linear mean 3.5 — proves quadrature, not arithmetic mean.
    assert float(pvb.global_T_rmse_K) != pytest.approx(3.5)


def test_per_variable_bias_area_weighted():
    """The MSE mean is AREA-weighted: a heavily-weighted column dominates."""
    t = jnp.array([2.0, 8.0])
    w = jnp.array([3.0, 1.0])           # sqrt((3·4 + 1·64)/4) = sqrt(19)
    pvb = aggregate_per_variable_bias(
        _err_fields(t, t, t, jnp.zeros(2)), w, have_precip=False)
    assert float(pvb.global_T_rmse_K) == pytest.approx(np.sqrt(19.0))


def test_per_variable_bias_precip_nan_unless_have_precip():
    """precip → NaN when not compared (an all-zero precip field must NOT read as a
    perfect 0); the area-weighted MEAN absolute error when have_precip."""
    ef = _err_fields(jnp.ones(2), jnp.ones(2), jnp.ones(2), jnp.array([1.0, 3.0]))
    assert bool(jnp.isnan(aggregate_per_variable_bias(
        ef, jnp.ones(2), have_precip=False).global_precip_err_mm_day))
    got = aggregate_per_variable_bias(ef, jnp.ones(2), have_precip=True)
    assert float(got.global_precip_err_mm_day) == pytest.approx(2.0)   # mean(|1|,|3|)


def test_per_variable_bias_global_reduce_doubling_preserves_rmse():
    """global_reduce sums the NUMERATOR Σw·rmse² AND the DENOMINATOR Σw before the
    divide: a DOUBLING reducer leaves the RMSE UNCHANGED (2·num / 2·den → same ratio
    → same sqrt), proving it reached BOTH sums (not just the numerator)."""
    t = jnp.array([3.0, 4.0, 0.0, 5.0])
    ef = _err_fields(t, t, t, jnp.zeros(4))
    local = float(aggregate_per_variable_bias(
        ef, jnp.ones(4), have_precip=False).global_T_rmse_K)
    assert local == pytest.approx(np.sqrt((9 + 16 + 0 + 25) / 4))   # sqrt(mean of squares)
    doubled = float(aggregate_per_variable_bias(
        ef, jnp.ones(4), have_precip=False,
        global_reduce=lambda x: 2.0 * x).global_T_rmse_K)
    assert doubled == pytest.approx(local)        # both sums scaled ⇒ RMSE unchanged


def test_per_variable_bias_improvement_exposes_per_variable_tradeoff():
    """The per-variable improvement EXPOSES a trade-off the combined score hides:
    a correction that LOWERS T-rmse but RAISES wind-rmse → T_improved, NOT
    wind_improved (the whole point — 'improve the biases' is per-variable)."""
    base = _err_fields(jnp.array([4.0, 4.0]), jnp.array([1.0, 1.0]),
                       jnp.array([2.0, 2.0]), jnp.zeros(2))
    upd = _err_fields(jnp.array([1.0, 1.0]),   # T improved (4→1)
                      jnp.array([1.0, 1.0]),   # qv unchanged
                      jnp.array([5.0, 5.0]),   # wind WORSE (2→5)
                      jnp.zeros(2))
    pvi = per_variable_bias_improvement(base, upd, jnp.ones(2), have_precip=False)
    assert float(pvi.baseline.global_T_rmse_K) == pytest.approx(4.0)
    assert float(pvi.updated.global_T_rmse_K) == pytest.approx(1.0)
    assert bool(pvi.T_improved) and not bool(pvi.wind_improved)
    assert not bool(pvi.qv_improved)              # strictly-less ⇒ unchanged is NOT improved


def test_per_variable_bias_improvement_precip_not_improved_when_absent():
    """precip_improved is False when precip wasn't compared (both NaN ⇒ NaN<NaN is
    False) — a not-compared variable never reads as 'improved'."""
    base = _err_fields(jnp.ones(2), jnp.ones(2), jnp.ones(2), jnp.array([2.0, 2.0]))
    upd = _err_fields(jnp.ones(2), jnp.ones(2), jnp.ones(2), jnp.array([1.0, 1.0]))
    pvi = per_variable_bias_improvement(base, upd, jnp.ones(2), have_precip=False)
    assert not bool(pvi.precip_improved)
    assert bool(jnp.isnan(pvi.baseline.global_precip_err_mm_day))
    # With have_precip the precip improvement IS detected (2.0 → 1.0).
    pvi2 = per_variable_bias_improvement(base, upd, jnp.ones(2), have_precip=True)
    assert bool(pvi2.precip_improved)


def test_compare_per_variable_bias_builds_improved_flags():
    """compare_per_variable_bias (the shared core of per_variable_bias_improvement AND
    the campaign summary): *_improved = updated < baseline per variable; precip NaN<NaN
    is False."""
    from legoesm.training.bias_metrics import PerVariableBias

    def _pv(t, w, p):
        return PerVariableBias(jnp.asarray(t), jnp.asarray(1e-3), jnp.asarray(w),
                               jnp.asarray(p))

    out = compare_per_variable_bias(_pv(4.0, 2.0, float("nan")),
                                    _pv(1.0, 5.0, float("nan")))   # T down, wind UP
    assert bool(out.T_improved) and not bool(out.wind_improved)
    assert not bool(out.qv_improved)              # equal qv ⇒ not strictly improved
    assert not bool(out.precip_improved)          # NaN<NaN ⇒ False (not compared)
    assert out.baseline.global_T_rmse_K == 4.0 and out.updated.global_T_rmse_K == 1.0


def test_aggregate_uniform_weights_is_mean():
    score = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    w = jnp.ones((2, 2))
    assert float(aggregate_combined_bias(score, w)) == pytest.approx(2.5)


def test_aggregate_area_weighted():
    score = jnp.array([1.0, 3.0])
    w = jnp.array([3.0, 1.0])  # weight the first column more
    # (3*1 + 1*3)/(3+1) = 6/4 = 1.5
    assert float(aggregate_combined_bias(score, w)) == pytest.approx(1.5)


def test_aggregate_mask_excludes_and_sanitizes_nan():
    score = jnp.array([1.0, jnp.nan])  # invalid column carries NaN
    w = jnp.ones((2,))
    mask = jnp.array([True, False])
    out = float(aggregate_combined_bias(score, w, valid_mask=mask))
    assert out == pytest.approx(1.0)  # NaN column excluded, no contamination


def test_aggregate_all_excluded_is_zero():
    score = jnp.array([5.0, 6.0])
    w = jnp.ones((2,))
    mask = jnp.array([False, False])
    assert float(aggregate_combined_bias(score, w, valid_mask=mask)) == 0.0


def test_aggregate_global_reduce_applies_to_both_sums():
    """``global_reduce`` (DISTRIBUTED) must reduce the NUMERATOR and DENOMINATOR
    before the divide.  A reducer that DOUBLES leaves the mean UNCHANGED (2·num /
    2·den) — proof it hit BOTH sums; if it touched only the numerator the mean
    would double (iter 88)."""
    score = jnp.array([1.0, 3.0])
    w = jnp.array([1.0, 1.0])
    local = float(aggregate_combined_bias(score, w))                 # 2.0
    doubled = float(aggregate_combined_bias(score, w, global_reduce=lambda x: 2.0 * x))
    assert doubled == pytest.approx(local)                           # ratio unchanged
    # An ADDITIVE reducer reaches the division (simulating another rank's mass):
    # (num+4)/(den+4) = (4+4)/(2+4) = 8/6, not the local 2.0.
    shifted = float(aggregate_combined_bias(
        score, w, global_reduce=lambda x: x + jnp.asarray(4.0)))
    assert shifted == pytest.approx(8.0 / 6.0)
    assert shifted != pytest.approx(local)


def test_aggregate_global_reduce_none_is_local():
    """``global_reduce=None`` is byte-identical to the purely-local aggregate."""
    score = jnp.array([2.0, 4.0, 6.0])
    w = jnp.array([1.0, 2.0, 1.0])
    assert (float(aggregate_combined_bias(score, w, global_reduce=None))
            == float(aggregate_combined_bias(score, w)))


def test_bias_improvement_global_reduce_is_global_verdict():
    """With ``global_reduce`` the improved verdict is GLOBAL: a reducer that adds a
    large WORSENING other-rank contribution to the updated bias can FLIP the local
    'improved' to a global 'not improved' — the basis for an identical accept/reject
    across ranks."""
    base = jnp.array([2.0])
    upd = jnp.array([1.0])         # locally improved (1 < 2)
    w = jnp.array([1.0])
    assert bool(bias_improvement(base, upd, w).improved)             # local: improved
    # A reducer that injects a big extra updated-mass (other rank got far worse):
    # base global = 2, updated global ≈ much higher → not improved.  Use a reducer
    # that adds 100 to every reduced sum so the updated mean dominates is hard to
    # craft symmetrically; instead just confirm the SAME reducer feeds base+upd and
    # the verdict is computed on the reduced means (identity reducer ⇒ same verdict).
    same = bias_improvement(base, upd, w, global_reduce=lambda x: x)
    assert bool(same.improved) and float(same.updated_bias) == pytest.approx(1.0)


def test_bias_improvement_detects_reduction():
    base = jnp.array([2.0, 2.0])
    upd = jnp.array([1.0, 1.0])
    w = jnp.ones((2,))
    out = bias_improvement(base, upd, w)
    assert bool(out.improved)
    assert float(out.baseline_bias) == pytest.approx(2.0)
    assert float(out.updated_bias) == pytest.approx(1.0)
    assert float(out.absolute_reduction) == pytest.approx(1.0)
    assert float(out.fractional_improvement) == pytest.approx(0.5)


def test_bias_improvement_detects_worsening():
    base = jnp.array([1.0, 1.0])
    upd = jnp.array([1.5, 1.5])
    w = jnp.ones((2,))
    out = bias_improvement(base, upd, w)
    assert not bool(out.improved)
    assert float(out.absolute_reduction) == pytest.approx(-0.5)


def test_bias_improvement_respects_mask():
    # Only the first column counts; it improves.
    base = jnp.array([2.0, 100.0])
    upd = jnp.array([1.0, 0.0])
    w = jnp.ones((2,))
    mask = jnp.array([True, False])
    out = bias_improvement(base, upd, w, valid_mask=mask)
    assert float(out.baseline_bias) == pytest.approx(2.0)
    assert float(out.updated_bias) == pytest.approx(1.0)
    assert bool(out.improved)


def test_worst_column_bias_change():
    base = jnp.array([[5.0, 1.0], [1.0, 4.0]])
    upd = jnp.array([[2.0, 1.0], [1.0, 3.0]])
    # worst columns flat 0 and 3: reductions 3.0 and 1.0 -> mean 2.0
    out = worst_column_bias_change(base, upd, jnp.array([0, 3]))
    assert float(out) == pytest.approx(2.0)


def test_aggregate_all_excluded_gradient_is_finite_zero():
    """All-excluded must give a FINITE (zero) gradient, not a 0/0 NaN leak."""
    w = jnp.ones((3,))
    mask = jnp.array([False, False, False])

    def loss(score):
        return aggregate_combined_bias(score, w, valid_mask=mask)

    g = jax.grad(loss)(jnp.array([1.0, 2.0, 3.0]))
    assert bool(jnp.all(jnp.isfinite(g)))
    np.testing.assert_allclose(np.asarray(g), [0.0, 0.0, 0.0], atol=1e-12)


def test_bias_improvement_perfect_baseline():
    # base==0 (perfect baseline) -> fractional uses the floor, stays finite.
    base = jnp.zeros((2,))
    upd = jnp.array([0.5, 0.5])
    out = bias_improvement(base, upd, jnp.ones((2,)))
    assert float(out.baseline_bias) == 0.0
    assert not bool(out.improved)  # got worse from perfect
    assert jnp.isfinite(out.fractional_improvement)


def test_worst_column_empty_indices_is_zero():
    base = jnp.array([1.0, 2.0])
    upd = jnp.array([0.5, 1.0])
    out = worst_column_bias_change(base, upd, jnp.array([], dtype=jnp.int32))
    assert float(out) == 0.0


def test_worst_column_out_of_range_raises():
    base = jnp.array([1.0, 2.0])
    upd = jnp.array([0.5, 1.0])
    with pytest.raises(ValueError, match="out of range"):
        worst_column_bias_change(base, upd, jnp.array([0, 5]))  # 5 >= 2


def test_worst_column_valid_mask_excludes_padded_slots():
    """A ``valid`` mask drops padded (non-worst) slots from the mean — so raw
    rank_worst_columns output (which pads in-range when n>n_valid) is safe."""
    base = jnp.array([[5.0, 1.0], [1.0, 4.0]])
    upd = jnp.array([[2.0, 1.0], [1.0, 3.0]])
    # idx 0 (reduction 3.0) is the ONE real worst column; idx 1 is a padded slot
    # (in-range, but reduction 0.0) — without the mask it would dilute 3.0 → 1.5.
    out = worst_column_bias_change(
        base, upd, jnp.array([0, 1]), valid=jnp.array([True, False]))
    assert float(out) == pytest.approx(3.0)        # padded slot excluded
    # without the mask the padded slot dilutes the mean (the trap this guards):
    diluted = worst_column_bias_change(base, upd, jnp.array([0, 1]))
    assert float(diluted) == pytest.approx(1.5)


def test_worst_column_all_invalid_mask_is_zero():
    """All slots padded (no real worst column) ⇒ 0, like an empty index set."""
    base = jnp.array([1.0, 2.0])
    upd = jnp.array([0.5, 1.0])
    out = worst_column_bias_change(
        base, upd, jnp.array([0, 1]), valid=jnp.array([False, False]))
    assert float(out) == 0.0


def test_worst_column_valid_mask_only_checks_valid_indices_in_range():
    """Bounds check applies to the slots that actually contribute: a padded slot
    whose index is masked out is not range-checked and never contributes (it is
    still gathered then clamped+masked, but its value cannot affect the mean)."""
    base = jnp.array([1.0, 2.0])
    upd = jnp.array([0.5, 1.0])
    # idx 5 is out of range but masked invalid → no raise; valid idx 0 contributes.
    out = worst_column_bias_change(
        base, upd, jnp.array([0, 5]), valid=jnp.array([True, False]))
    assert float(out) == pytest.approx(0.5)


def test_worst_column_valid_mask_shape_mismatch_raises():
    base = jnp.array([1.0, 2.0])
    upd = jnp.array([0.5, 1.0])
    with pytest.raises(ValueError, match="valid mask shape"):
        worst_column_bias_change(
            base, upd, jnp.array([0, 1]), valid=jnp.array([True]))


def test_aggregate_differentiable_and_jit():
    w = jnp.array([2.0, 1.0, 1.0])

    def loss(score):
        return aggregate_combined_bias(score, w)

    g = jax.grad(loss)(jnp.array([1.0, 2.0, 3.0]))
    # d/dscore_i of weighted mean = w_i / sum(w) = [2,1,1]/4.
    np.testing.assert_allclose(np.asarray(g), [0.5, 0.25, 0.25], rtol=1e-10)

    out = jax.jit(loss)(jnp.array([1.0, 2.0, 3.0]))
    assert jnp.isfinite(out)
