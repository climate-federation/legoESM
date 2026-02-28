"""Unit tests for smooth approximations."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.smooth import (
    sigmoid_switch, smooth_max, smooth_min, smooth_clamp,
    smooth_relu, smooth_abs,
)


class TestSmoothApproximations:
    """Tests for differentiable smooth approximations."""

    def test_sigmoid_switch_limits(self):
        """Sigmoid should be ~0 for large negative, ~1 for large positive."""
        assert float(sigmoid_switch(jnp.array(-10.0))) < 0.01
        assert float(sigmoid_switch(jnp.array(10.0))) > 0.99

    def test_sigmoid_switch_midpoint(self):
        """Sigmoid at x=0 should be 0.5."""
        assert jnp.allclose(sigmoid_switch(jnp.array(0.0)), 0.5)

    def test_smooth_max(self):
        """smooth_max should approximate max(a, b)."""
        a = jnp.array([1.0, 5.0, 3.0])
        b = jnp.array([2.0, 3.0, 4.0])
        result = smooth_max(a, b)
        expected = jnp.maximum(a, b)
        assert jnp.allclose(result, expected, atol=0.05)

    def test_smooth_min(self):
        """smooth_min should approximate min(a, b)."""
        a = jnp.array([1.0, 5.0, 3.0])
        b = jnp.array([2.0, 3.0, 4.0])
        result = smooth_min(a, b)
        expected = jnp.minimum(a, b)
        assert jnp.allclose(result, expected, atol=0.05)

    def test_smooth_clamp(self):
        """smooth_clamp should keep values in [lo, hi]."""
        x = jnp.array([-5.0, 0.5, 10.0])
        result = smooth_clamp(x, 0.0, 1.0)
        assert float(result[0]) < 0.05   # Should be close to 0
        assert jnp.allclose(result[1], 0.5, atol=0.05)  # Unchanged
        assert float(result[2]) > 0.95   # Should be close to 1

    def test_smooth_relu(self):
        """smooth_relu should approximate max(x, 0)."""
        x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = smooth_relu(x)
        assert float(result[0]) < 0.01
        assert float(result[4]) > 1.99

    def test_smooth_abs(self):
        """smooth_abs should approximate |x|."""
        x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        result = smooth_abs(x)
        expected = jnp.abs(x)
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_all_differentiable(self):
        """All smooth functions should have finite gradients."""
        x = jnp.array(0.0)

        for fn_name, fn in [
            ("sigmoid", lambda x: sigmoid_switch(x)),
            ("smooth_max", lambda x: smooth_max(x, jnp.array(0.5))),
            ("smooth_min", lambda x: smooth_min(x, jnp.array(0.5))),
            ("smooth_clamp", lambda x: smooth_clamp(x, -1.0, 1.0)),
            ("smooth_relu", lambda x: smooth_relu(x)),
            ("smooth_abs", lambda x: smooth_abs(x)),
        ]:
            grad = jax.grad(fn)(x)
            assert jnp.isfinite(grad), f"{fn_name} has non-finite gradient at x=0"
