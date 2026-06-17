"""Unit tests for :mod:`legoesm.training.bias_metrics`.

The success-criterion measurement: area-weighted global bias + baseline-vs-
updated improvement.  Analytic aggregation (uniform + weighted + masked),
the improvement direction/fraction, worst-column targeting, all-excluded edge,
and AD/jit safety.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training.bias_metrics import (
    aggregate_combined_bias,
    bias_improvement,
    worst_column_bias_change,
)


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


def test_aggregate_differentiable_and_jit():
    w = jnp.array([2.0, 1.0, 1.0])

    def loss(score):
        return aggregate_combined_bias(score, w)

    g = jax.grad(loss)(jnp.array([1.0, 2.0, 3.0]))
    # d/dscore_i of weighted mean = w_i / sum(w) = [2,1,1]/4.
    np.testing.assert_allclose(np.asarray(g), [0.5, 0.25, 0.25], rtol=1e-10)

    out = jax.jit(loss)(jnp.array([1.0, 2.0, 3.0]))
    assert jnp.isfinite(out)
