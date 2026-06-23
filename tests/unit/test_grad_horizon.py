"""Tests for the chaos / long-window adjoint guardrails (grad_horizon.py)."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.training.grad_horizon import (  # noqa: E402
    check_grad_horizon,
    estimate_growth_rate,
    global_grad_norm,
    grad_norm_vs_horizon,
)


def test_global_grad_norm_matches_l2():
    g = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([0.0])}
    assert jnp.allclose(global_grad_norm(g), 5.0)


def test_global_grad_norm_empty():
    assert float(global_grad_norm({})) == 0.0


def test_grad_norm_grows_on_chaotic_map():
    # logistic map at r=3.9 is chaotic => |d x_n / d x_0| ~ exp(lyap * n)
    r = 3.9

    def loss_for_horizon(n, x):
        val = x
        for _ in range(n):
            val = r * val * (1.0 - val)
        return val ** 2

    x0 = jnp.asarray(0.4)
    norms = grad_norm_vs_horizon(loss_for_horizon, [1, 5, 10, 20], x0)
    assert norms[20] > norms[1]
    rate = estimate_growth_rate(norms)
    assert rate > 0.0  # positive Lyapunov exponent => exponential adjoint growth


def test_estimate_growth_rate_needs_two_points():
    import math
    assert math.isnan(estimate_growth_rate({5: 1.0}))
    assert math.isnan(estimate_growth_rate({1: 0.0, 2: 0.0}))  # no positive samples


def test_estimate_growth_rate_pure_exponential():
    # grad_norm = exp(0.3 n)  => recovered slope ~ 0.3
    norms = {n: float(jnp.exp(0.3 * n)) for n in (1, 2, 4, 8, 16)}
    assert abs(estimate_growth_rate(norms) - 0.3) < 1e-6


def test_check_grad_horizon_warn_and_raise():
    assert check_grad_horizon(1.0, 10, 5.0) is True
    assert check_grad_horizon(10.0, 10, 5.0) is False
    assert check_grad_horizon(jnp.inf, 10, 5.0) is False
    with pytest.raises(RuntimeError, match="predictability horizon"):
        check_grad_horizon(10.0, 10, 5.0, raise_on_exceed=True)
