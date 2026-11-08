"""Derivative of the libm ``pow`` at bases that are not strictly positive."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core import transcendentals  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402


@pytest.fixture(autouse=True)
def _libm_policy():
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(saved)


def _grad(base, exponent):
    return float(jax.grad(lambda x: transcendentals.pow(x, exponent))(jnp.asarray(base)))


@pytest.mark.parametrize("base, exponent, expected", [
    (0.0, 1.0, 1.0),
    (-2.0, 2.0, -4.0),
    (-2.0, 3.0, 12.0),
    (0.0, 2.0, 0.0),
    (0.0, 0.5, 0.0),      # infinite slope: 0 is the stand-in
    (0.0, 0.0, 0.0),
    (-2.0, 0.5, 0.0),     # NaN forward value: 0 is the stand-in
    (2.0, 3.0, 12.0),
])
def test_first_derivative(base, exponent, expected):
    assert _grad(base, exponent) == pytest.approx(expected, rel=1e-14, abs=0.0)


def test_second_derivative_at_zero():
    second = jax.grad(jax.grad(lambda x: transcendentals.pow(x, 2.0)))
    assert float(second(jnp.asarray(0.0))) == pytest.approx(2.0, rel=1e-14)


def test_exponent_derivative_is_zero_at_base_zero():
    assert float(jax.grad(lambda p: transcendentals.pow(0.0, p))(jnp.asarray(2.0))) == 0.0


def test_mixed_vector_forward_reverse_and_jit():
    x = jnp.asarray([3.0, 0.0, -2.0])
    f = lambda v: jnp.sum(transcendentals.pow(v, 2.0))  # noqa: E731
    want = np.array([6.0, 0.0, -4.0])
    np.testing.assert_allclose(jax.grad(f)(x), want, rtol=1e-14)
    np.testing.assert_allclose(jax.jit(jax.grad(f))(x), want, rtol=1e-14)
    _, tangent = jax.jvp(f, (x,), (jnp.ones_like(x),))
    np.testing.assert_allclose(tangent, want.sum(), rtol=1e-14)
