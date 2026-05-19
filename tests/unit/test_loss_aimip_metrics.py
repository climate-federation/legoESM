"""Unit tests for the AIMIP scorecard helpers in ``legoesm.ml.loss``.

Addresses PR #270 slopbuster WARN: three scripts inlined the same
zonal-mean + latitude-weighted RMSE/bias formula.  After factoring the
helpers into ``ml/loss.py``, this test pins their behaviour.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.ml.loss import latitude_weighted_bias, latitude_weighted_rmse


@pytest.fixture
def uniform_weights():
    return jnp.ones((8,))


def test_rmse_zero_when_pred_equals_target(uniform_weights):
    pred = jnp.linspace(0.0, 1.0, 8)[:, None] * jnp.ones((1, 16))
    rmse = latitude_weighted_rmse(pred, pred, uniform_weights)
    assert float(rmse) == pytest.approx(0.0, abs=1e-12)


def test_bias_zero_when_pred_equals_target(uniform_weights):
    pred = jnp.linspace(0.0, 1.0, 8)[:, None] * jnp.ones((1, 16))
    bias = latitude_weighted_bias(pred, pred, uniform_weights)
    assert float(bias) == pytest.approx(0.0, abs=1e-12)


def test_rmse_constant_offset(uniform_weights):
    target = jnp.zeros((8, 16))
    pred = jnp.full_like(target, 2.0)
    rmse = latitude_weighted_rmse(pred, target, uniform_weights)
    assert float(rmse) == pytest.approx(2.0, abs=1e-10)


def test_bias_constant_offset(uniform_weights):
    target = jnp.zeros((8, 16))
    pred = jnp.full_like(target, 1.5)
    bias = latitude_weighted_bias(pred, target, uniform_weights)
    assert float(bias) == pytest.approx(1.5, abs=1e-10)


def test_lat_weighting_emphasises_chosen_row():
    target = jnp.zeros((4, 8))
    pred = jnp.zeros((4, 8))
    # Put all the error in row 1
    pred = pred.at[1].set(3.0)

    # Default uniform weights → mean square error = 9 * 1/4 → rmse = 1.5
    uniform = jnp.ones((4,))
    rmse_u = float(latitude_weighted_rmse(pred, target, uniform))
    assert rmse_u == pytest.approx(1.5, abs=1e-10)

    # Put all the weight on row 1 → rmse = 3.0
    weights = jnp.array([0.0, 1.0, 0.0, 0.0])
    rmse_w = float(latitude_weighted_rmse(pred, target, weights))
    assert rmse_w == pytest.approx(3.0, abs=1e-10)


def test_bias_handles_signed_errors(uniform_weights):
    target = jnp.zeros((8, 16))
    pred = jnp.zeros((8, 16))
    pred = pred.at[:4].set(1.0)
    pred = pred.at[4:].set(-1.0)
    bias = latitude_weighted_bias(pred, target, uniform_weights)
    assert float(bias) == pytest.approx(0.0, abs=1e-10)


# ----------------------------------------------------------------------
# Public-rename guards on trainable_params helpers
# ----------------------------------------------------------------------


def test_sigmoid_to_range_round_trip():
    """``sigmoid_to_range`` <-> ``range_to_sigmoid`` should be inverses."""
    from legoesm.training.trainable_params import (
        range_to_sigmoid,
        sigmoid_to_range,
    )

    lo, hi = 0.1, 2.5
    for val in (0.2, 1.0, 2.4):
        raw = range_to_sigmoid(val, lo, hi)
        back = float(sigmoid_to_range(jnp.array(raw), lo, hi))
        assert back == pytest.approx(val, abs=2e-3)


def test_private_aliases_removed():
    """The legacy `_sigmoid_to_range` / `_range_to_sigmoid` symbols should
    NOT be exported — only the public names."""
    import legoesm.training.trainable_params as tp

    assert hasattr(tp, "sigmoid_to_range")
    assert hasattr(tp, "range_to_sigmoid")
    assert not hasattr(tp, "_sigmoid_to_range")
    assert not hasattr(tp, "_range_to_sigmoid")
