"""Unit tests for smooth approximations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.smooth import (
    abs_tanh,
    max_c1,
    max_rel,
    min_rel,
    step_rel,
    sign_tanh,
    zero_at_or_below,
    min_c1,
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


def test_pnorm_smooth_min_tie_curvature_and_zero_inputs():
    """#2241: exact second derivative at a tie (min/max-based AD zeroed it);
    exact min and finite gradients at zero inputs with eps = 0, at the
    default and at the hard limit."""
    if not jax.config.jax_enable_x64:
        pytest.skip("curvature checked in fp64 (JAX_ENABLE_X64=1)")
    from jax.test_util import check_grads
    p = 50.0
    f = lambda a, b: pnorm_smooth_min(a, b, p, 0.0)  # noqa: E731
    two = jnp.array(2.0)
    check_grads(lambda a: f(a, two), (two,), order=2, modes=("fwd", "rev"), eps=1e-4)
    check_grads(f, (two, two), order=2, modes=("fwd", "rev"), eps=1e-4)
    want = -(p + 1.0) * 2.0 ** (-1.0 / p) / (4.0 * 2.0)
    got = jax.grad(jax.grad(lambda a: f(a, two)))(two)
    np.testing.assert_allclose(float(got), want, rtol=1e-10)
    for a, b in ((0.0, 3.0), (3.0, 0.0), (0.0, 0.0)):
        a, b = jnp.array(a), jnp.array(b)
        for pp in (p, 1e12):
            assert float(pnorm_smooth_min(a, b, pp, 0.0)) == 0.0
            ga, gb = jax.grad(lambda x, y: pnorm_smooth_min(x, y, pp, 0.0), (0, 1))(a, b)
            assert bool(jnp.isfinite(ga)) and bool(jnp.isfinite(gb))
    np.testing.assert_allclose(float(pnorm_smooth_min(jnp.array(3.0), jnp.array(5.0), 1e12, 0.0)),
                               3.0, rtol=1e-12)


def test_pnorm_smooth_min_tiny_inputs_have_finite_gradients_in_fp32():
    """#2241 review: with eps = 0 a tiny hi (1e-20) gave a NaN d/dhi in fp32
    (lo/hi differentiated through hi^-2); the analytic JVP keeps it finite."""
    for a, b in ((0.0, 1e-20), (1e-30, 1e-20), (0.0, 1e-30), (1e-25, 1e-20)):
        a, b = jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)
        ga, gb = jax.grad(lambda x, y: pnorm_smooth_min(x, y, 50.0, 0.0), (0, 1))(a, b)
        assert bool(jnp.isfinite(ga)) and bool(jnp.isfinite(gb)), (a, b, ga, gb)
        f = lambda x, y: pnorm_smooth_min(x, y, 50.0, 0.0)  # noqa: E731
        for x, y in ((a, b), (b, a)):
            hess = jax.hessian(f, (0, 1))(x, y)
            assert all(bool(jnp.isfinite(h)) for row in hess for h in row), (x, y, hess)
            fwd = jax.jacfwd(jax.jacfwd(f, (0, 1)), (0, 1))(x, y)
            assert all(bool(jnp.isfinite(h)) for row in fwd for h in row), (x, y, fwd)
            rr = jax.jacrev(jax.jacrev(f, (0, 1)), (0, 1))(x, y)
            assert all(bool(jnp.isfinite(h)) for row in rr for h in row), (x, y, rr)
        if float(a) == 0.0:          # lo = 0: exactly the min's gradient
            ga, gb = jax.grad(f, (0, 1))(a, b)
            assert (float(ga), float(gb)) == (1.0, 0.0)
            gb2, ga2 = jax.grad(f, (0, 1))(b, a)
            assert (float(gb2), float(ga2)) == (0.0, 1.0)


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


def test_max_c1_min_c1_exact_outside_band_and_c1():
    """Bitwise the hard operand outside the band 0 < a - b < 1/s, C1 across both joins,
    never above max(a, b); min_c1 mirrors it; sharpness inf is jnp.maximum/minimum."""
    s = 4.0
    b = jnp.array(1.0)
    a = jnp.array([0.0, 1.0, 1.1, 1.25, 1.3, 3.0])
    y = max_c1(a, b, s)
    assert jnp.array_equal(y[:2], jnp.ones(2)) and jnp.array_equal(y[3:], a[3:])
    assert jnp.all(y <= jnp.maximum(a, b)) and jnp.all(y >= b)
    assert jnp.array_equal(min_c1(-a, -b, s), -y)
    assert jnp.array_equal(max_c1(a, b, jnp.inf), jnp.maximum(a, b))
    assert jnp.array_equal(min_c1(a, b, jnp.inf), jnp.minimum(a, b))
    g = jax.vmap(jax.grad(lambda t: max_c1(t, b, s)))
    for j in (1.0, 1.0 + 1.0 / s):
        lo, hi = g(jnp.array([j - 1e-7, j + 1e-7]))
        assert abs(float(hi - lo)) < 1e-5
def test_positive_part_c1_traced_inf_is_hard_and_finite():
    """A TRACED sharpness = inf gives max(x, 0) with finite gradients (no inf * 0)."""
    x = jnp.array([-1.0, 0.0, 0.3, 2.0])
    f = jax.jit(lambda x, s: positive_part_c1(x, s))
    assert jnp.array_equal(f(x, jnp.inf), jnp.maximum(x, 0.0))
    g = jax.jit(jax.grad(lambda x, s: jnp.sum(positive_part_c1(x, s))))(x, jnp.inf)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.array_equal(f(x, 4.0), positive_part_c1(x, 4.0))


def test_abs_sign_tanh_and_zero_at_or_below():
    """RULE 5 helpers (#2546): width 0 is the hard switch bitwise; the smooth
    forms stay within their stated bounds and are C2 at the switch."""
    x = jnp.linspace(-0.05, 0.05, 1001)
    assert jnp.array_equal(abs_tanh(x, 0.0), jnp.abs(x))
    assert jnp.array_equal(sign_tanh(x, 0.0), jnp.sign(x))
    w = 0.01
    a = abs_tanh(x, w)
    assert jnp.all(a <= jnp.abs(x)) and jnp.all(jnp.abs(x) - a <= 0.28 * w)
    assert jnp.all(jnp.abs(sign_tanh(x, w)) <= 1.0)
    tau = jnp.linspace(0.0, 3e-10, 301)
    assert jnp.array_equal(zero_at_or_below(tau, 1e-10, 0.0), jnp.where(tau <= 1e-10, 0.0, tau))
    r = zero_at_or_below(tau, 1e-10, 1e-10)
    assert jnp.all(r[tau <= 1e-10] == 0.0) and jnp.array_equal(r[tau >= 2e-10], tau[tau >= 2e-10])
    for f, x0 in ((lambda t: abs_tanh(t, w), 0.0), (lambda t: sign_tanh(t, w) * t * t, 0.0),
                  (lambda t: zero_at_or_below(t, 1.0, 1.0), 1.0)):
        for g in (jax.grad(f), jax.grad(jax.grad(f))):
            assert abs(float(g(x0 - 1e-7)) - float(g(x0 + 1e-7))) < 1e-3


def test_relative_switches_hard_limit_band_and_c1():
    """step_rel / max_rel / min_rel: width 0 is the hard form bitwise, the result is
    exact outside the relative band, and the derivative is continuous across it."""
    x = jnp.linspace(-3.0, 3.0, 61, dtype=jnp.float64)
    b = jnp.asarray(1.0, dtype=jnp.float64)
    assert jnp.array_equal(step_rel(x, 2.0, 0), (x > 0).astype(x.dtype))
    assert jnp.array_equal(max_rel(x, b, 0), jnp.maximum(x, b))
    assert jnp.array_equal(min_rel(x, b, 0), jnp.minimum(x, b))
    w = 0.1
    far = jnp.abs(x - b) >= w * (jnp.abs(x) + 1.0)
    assert jnp.allclose(max_rel(x, b, w)[far], jnp.maximum(x, b)[far], rtol=1e-15, atol=0)
    assert jnp.allclose(min_rel(x, b, w)[far], jnp.minimum(x, b)[far], rtol=1e-15, atol=0)
    s = step_rel(x, 2.0, w)
    assert jnp.all(s[x <= 0.0] == 0.0) and jnp.all(s[x >= 0.2] == 1.0)   # one-sided: ties as the hard form
    for f, x0 in ((lambda t: step_rel(t, 2.0, w), 0.1), (lambda t: max_rel(t, 1.0, w), 1.0)):
        g = jax.grad(f)
        assert abs(float(g(x0 - 1e-7)) - float(g(x0 + 1e-7))) < 1e-4
        assert float(g(x0)) > 0.0          # a derivative through the switch, unlike the hard form
    g = jax.grad(lambda t: step_rel(t, 2.0, w))
    assert float(g(0.0)) == 0.0 and abs(float(g(1e-7))) < 1e-4        # C1 at the one-sided join


def test_max_rel_gradient_is_finite_at_zero_operands():
    for w in (0.01, 0.1):
        g = jax.grad(lambda x: max_rel(x, 0.0, w))(0.0)
        assert np.isfinite(float(g))
        g2 = jax.grad(lambda x: min_rel(x * x, x, w))(0.0)
        assert np.isfinite(float(g2))


def test_relative_helpers_float32_gradient_at_tiny_operands():
    from legoesm.core.smooth import positive_part_rel
    x = jnp.float32(1e-20)
    for f in (lambda t: max_rel(t, jnp.float32(0.0), 0.01), lambda t: min_rel(t, t * t, 0.01),
              lambda t: step_rel(t, jnp.float32(0.0), 0.01), lambda t: positive_part_rel(t, t, 0.01)):
        for x0 in (x, jnp.float32(0.0)):
            assert np.isfinite(float(jax.grad(f)(x0)))


def test_step_rel_tie_at_zero_band_matches_the_hard_form():
    assert float(step_rel(jnp.float64(0.0), 0.0, 0.01)) == 0.0
    assert float(step_rel(jnp.float64(1.0), 0.0, 0.01)) == 1.0


def test_relative_helpers_float32_gradient_inside_a_tiny_band():
    a = jnp.float32(1e-18)
    for f in (lambda t: max_rel(t, a, 0.01), lambda t: step_rel(t - a, a, 0.01)):
        assert np.isfinite(float(jax.grad(f)(a * jnp.float32(1.002))))
