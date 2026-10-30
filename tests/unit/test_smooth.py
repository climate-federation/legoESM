"""Unit tests for smooth approximations."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.smooth import (
    pnorm_smooth_min, positive_part_c1, sigmoid_switch, smooth_max, smooth_min, smooth_clamp,
    smooth_relu, smooth_abs, smoothstep_cubic, smoothstep_quintic,
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

    def test_smoothstep_cubic(self):
        """Exactly 0 / 1 outside [0, 1], C1 at both ends, 1/2 at the middle."""
        t = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        assert jnp.array_equal(smoothstep_cubic(t), jnp.array([0.0, 0.0, 0.5, 1.0, 1.0]))
        d1 = jax.grad(smoothstep_cubic)
        for x in (0.0, 1.0):
            assert abs(float(d1(jnp.array(x)))) < 1e-12
        assert float(d1(jnp.array(0.5))) == pytest.approx(1.5)

    def test_smoothstep_quintic(self):
        """Exactly 0 / 1 outside [0, 1], C2 at both ends, 1/2 at the middle."""
        t = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        assert jnp.array_equal(smoothstep_quintic(t), jnp.array([0.0, 0.0, 0.5, 1.0, 1.0]))
        d1 = jax.grad(smoothstep_quintic)
        d2 = jax.grad(d1)
        for x in (0.0, 1.0):
            assert abs(float(d1(jnp.array(x)))) < 1e-12 and abs(float(d2(jnp.array(x)))) < 1e-12
        assert float(d1(jnp.array(0.5))) == pytest.approx(1.875)

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


def test_infinite_sharpness_is_the_exact_hard_limit():
    import math
    x = jnp.array([-2.0, -1e-30, 0.0, 1e-30, 3.0])
    y = jnp.array([1.0, 0.0, 0.0, -1.0, 5.0])
    inf = math.inf
    assert jnp.array_equal(sigmoid_switch(x, inf), (x >= 0).astype(x.dtype))
    assert jnp.array_equal(smooth_max(x, y, inf), jnp.maximum(x, y))
    assert jnp.array_equal(smooth_min(x, y, inf), jnp.minimum(x, y))
    assert jnp.array_equal(smooth_relu(x, inf), jnp.maximum(x, 0.0))
    assert jnp.array_equal(positive_part_c1(x, inf), jnp.maximum(x, 0.0))
    assert jnp.array_equal(smooth_clamp(x, -1.0, 1.0, inf), jnp.clip(x, -1.0, 1.0))
    import numpy as np
    assert jnp.array_equal(sigmoid_switch(x, np.float64(inf)), (x >= 0).astype(x.dtype))
    # a traced sharpness still takes the smooth path
    assert float(jax.jit(lambda s: sigmoid_switch(jnp.array(0.0), s))(10.0)) == 0.5


def test_pnorm_smooth_min():
    """Never above min(a, b); 2^(-1/p) of both where equal; zero stays zero."""
    a = jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 1e-3])
    b = jnp.array([2.0, 2.0, 1.0, 5.0, 0.0, 1e3])
    out = pnorm_smooth_min(a, b, 50.0, 1e-10)
    assert bool(jnp.all(out <= jnp.minimum(a, b)))
    assert jnp.allclose(out[1], 2.0 * 2.0 ** (-1.0 / 50.0))
    assert float(out[3]) == 0.0 and float(out[4]) == 0.0
    g = jax.grad(lambda x: pnorm_smooth_min(x, jnp.array(2.0), 50.0, 1e-10))(jnp.array(2.0))
    assert bool(jnp.isfinite(g)) and 0.0 < float(g) < 1.0


def test_positive_part_c1_is_zero_below_and_c1():
    """Exactly 0 for x <= 0 (softplus is not), exactly x at and above 1/s,
    never above max(x, 0), and value and slope continuous at both joins."""
    s = 4.0
    x = jnp.array([-1.0, 0.0, 0.1, 0.25, 0.5, 2.0])
    y = positive_part_c1(x, s)
    assert jnp.array_equal(y[:2], jnp.zeros(2))
    assert jnp.array_equal(y[3:], x[3:])
    assert jnp.allclose(y[2], 0.4 ** 2 * 1.6 / s)
    assert jnp.all(y <= jnp.maximum(x, 0.0))
    g = jax.vmap(jax.grad(lambda t: positive_part_c1(t, s)))
    for j in (0.0, 1.0 / s):
        lo, hi = g(jnp.array([j - 1e-7, j + 1e-7]))
        assert abs(float(hi - lo)) < 1e-5
