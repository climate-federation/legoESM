"""Unit tests for shared core profile-comparison primitives."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from legoesm.core.profile_metrics import (
    layer_weights_from_heights,
    safe_sqrt,
    weighted_rmse,
    weighted_std,
)


def test_safe_sqrt_value():
    assert jnp.allclose(safe_sqrt(jnp.array(4.0)), 2.0)
    assert jnp.allclose(safe_sqrt(jnp.array(0.0)), 0.0)


def test_safe_sqrt_finite_grad_at_zero():
    g = jax.grad(lambda x: safe_sqrt(x))(0.0)
    assert jnp.isfinite(g)
    assert g == pytest.approx(0.0)


def test_weighted_rmse_perfect_fit_zero_and_finite_grad():
    w = jnp.full((5,), 0.2)
    rmse = weighted_rmse(jnp.zeros(5), w)
    assert rmse == pytest.approx(0.0)
    g = jax.grad(lambda d: weighted_rmse(d, w))(jnp.zeros(5))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_weighted_rmse_known_value():
    w = jnp.array([0.5, 0.5])
    diff = jnp.array([2.0, 0.0])
    # sqrt(0.5*4 + 0.5*0) = sqrt(2)
    assert weighted_rmse(diff, w) == pytest.approx(jnp.sqrt(2.0))


def test_weighted_std_uniform_is_zero():
    w = jnp.full((4,), 0.25)
    assert weighted_std(jnp.full((4,), 3.0), w) == pytest.approx(0.0)


def test_layer_weights_sum_to_one_uniform():
    z = jnp.linspace(0.0, 1000.0, 11)
    w = layer_weights_from_heights(z)
    assert w.shape == (11,)
    assert float(jnp.sum(w)) == pytest.approx(1.0)
    assert bool(jnp.all(w > 0))


def test_layer_weights_nonuniform_thicker_gets_more():
    # stretched grid: spacing grows with height → upper levels carry more weight
    z = jnp.array([0.0, 10.0, 30.0, 70.0, 150.0, 310.0])
    w = layer_weights_from_heights(z)
    assert float(jnp.sum(w)) == pytest.approx(1.0)
    assert float(w[-1]) > float(w[0])


def test_layer_weights_rejects_scalar():
    with pytest.raises(ValueError):
        layer_weights_from_heights(jnp.array([1.0]))
