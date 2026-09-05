"""Scalar-libm precision-policy regression tests."""

from __future__ import annotations

import ctypes

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from jax.test_util import check_grads
from legoesm.core.transcendentals import (
    cos,
    exp,
    log,
    log10,
    pow as policy_pow,
    sin,
    tanh,
)


def _libm_reference(name: str, values: np.ndarray) -> np.ndarray:
    libm = ctypes.CDLL("libm.so.6")
    function = getattr(libm, name)
    function.argtypes = (ctypes.c_double,)
    function.restype = ctypes.c_double
    source = np.asarray(values, dtype=np.float64)
    result = np.empty_like(source)
    for index, value in enumerate(source):
        result[index] = function(float(value))
    return result


def _libm_pow_reference(base: np.ndarray, exponent: np.ndarray) -> np.ndarray:
    libm = ctypes.CDLL("libm.so.6")
    function = libm.pow
    function.argtypes = (ctypes.c_double, ctypes.c_double)
    function.restype = ctypes.c_double
    left, right = np.broadcast_arrays(
        np.asarray(base, dtype=np.float64), np.asarray(exponent, dtype=np.float64)
    )
    result = np.empty_like(left)
    for index in range(left.size):
        result.flat[index] = function(float(left.flat[index]), float(right.flat[index]))
    return result


@pytest.fixture(autouse=True)
def _restore_policy():
    previous = get_policy()
    yield
    set_policy(previous)


@pytest.mark.parametrize(
    "name,function", [
        ("exp", exp), ("tanh", tanh), ("sin", sin), ("cos", cos),
        ("log", log), ("log10", log10),
    ]
)
def test_libm_policy_is_scalar_library_bit_exact_and_native_is_separate(
    name, function,
):
    # The non-power-of-two stride avoids repeatedly sampling easy reduction
    # points while keeping every EXP input in its ordinary finite range.
    values = np.linspace(-19.75, 19.25, 100_000, dtype=np.float64)
    values += (np.arange(values.size, dtype=np.float64)
               * np.float64(float.fromhex("0x1p-46")))
    if name in {"log", "log10"}:
        values = np.exp(values / 4.0)
    reference = _libm_reference(name, values)

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    jax.clear_caches()
    observed = np.asarray(jax.jit(function)(jnp.asarray(values)))
    assert np.array_equal(observed.view(np.uint64), reference.view(np.uint64))

    set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    # The selector is intentionally read at trace time; clear the first
    # executable so this is a genuinely independent native trace.
    jax.clear_caches()
    native = np.asarray(jax.jit(function)(jnp.asarray(values)))
    native_mismatches = np.count_nonzero(
        native.view(np.uint64) != reference.view(np.uint64)
    )
    if name in {"exp", "tanh", "log10"}:
        assert native_mismatches > 0
    else:
        # On this certification host XLA log/sine/cosine already agree with
        # scalar glibc.  The libm policy still routes each through its callback.
        assert native_mismatches == 0


@pytest.mark.parametrize("function", [exp, tanh, sin, cos, log, log10])
def test_libm_policy_jit_and_eager_are_bit_exact(function):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    values = jnp.asarray([0.125, 0.25, 1.0, 2.0, 4.5], dtype=jnp.float64)
    eager = np.asarray(function(values))
    compiled = np.asarray(jax.jit(function)(values))
    assert np.array_equal(eager.view(np.uint64), compiled.view(np.uint64))


@pytest.mark.parametrize("function", [exp, tanh, sin, cos, log, log10])
def test_libm_policy_custom_jvp_matches_finite_difference(function):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    point = jnp.asarray(0.375, dtype=jnp.float64)
    tangent = jnp.asarray(0.625, dtype=jnp.float64)
    _, derivative = jax.jvp(function, (point,), (tangent,))
    step = np.float64(2.0**-20)
    plus = float(function(point + step))
    minus = float(function(point - step))
    finite_difference = (plus - minus) / (2.0 * step) * float(tangent)
    assert float(derivative) == pytest.approx(finite_difference, rel=2.0e-10, abs=2.0e-12)

    reverse = jax.grad(lambda value: function(value))(point)
    assert np.isfinite(float(reverse))


def test_libm_log_log10_pow_have_pinned_system_bytes():
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    values = jnp.asarray([0.125, 0.5, 1.0, 2.0, 10.0, 123.456])
    log_bits = np.asarray(jax.jit(log)(values)).view(np.uint64)
    log10_bits = np.asarray(jax.jit(log10)(values)).view(np.uint64)
    np.testing.assert_array_equal(log_bits, np.asarray([
        0xC000A2B23F3BAB73, 0xBFE62E42FEFA39EF, 0x0000000000000000,
        0x3FE62E42FEFA39EF, 0x40026BB1BBB55516, 0x401343774F3E2362,
    ], dtype=np.uint64))
    np.testing.assert_array_equal(log10_bits, np.asarray([
        0xBFECE61CF8EF36FE, 0xBFD34413509F79FF, 0x0000000000000000,
        0x3FD34413509F79FF, 0x3FF0000000000000, 0x4000BB6ABFC968EF,
    ], dtype=np.uint64))
    np.testing.assert_array_equal(
        log_bits, _libm_reference("log", np.asarray(values)).view(np.uint64)
    )
    np.testing.assert_array_equal(
        log10_bits, _libm_reference("log10", np.asarray(values)).view(np.uint64)
    )

    base = jnp.asarray([0.125, 0.5, 2.0, 10.0, 1.25, 123.456])
    exponent = jnp.asarray([-3.5, 0.25, -4.0, np.pi, 3.75, 0.125])
    pow_bits = np.asarray(jax.jit(policy_pow)(base, exponent)).view(np.uint64)
    np.testing.assert_array_equal(pow_bits, np.asarray([
        0x4096A09E667F3BCD, 0x3FEAE89F995AD3AD, 0x3FB0000000000000,
        0x4095A5D2AB3E544A, 0x400278B542702B4C, 0x3FFD363B90355C72,
    ], dtype=np.uint64))

    np.testing.assert_array_equal(
        np.asarray(jax.jit(policy_pow)(base, exponent)).view(np.uint64),
        _libm_pow_reference(np.asarray(base), np.asarray(exponent)).view(np.uint64),
    )


@pytest.mark.parametrize("function", [log, log10])
def test_libm_new_unary_check_grads_order2(function):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    point = (jnp.asarray(1.375, dtype=jnp.float64),)
    check_grads(function, point, order=2, modes=("fwd", "rev"),
                atol=2.0e-5, rtol=2.0e-5)


def test_libm_pow_check_grads_order2_both_partials_and_broadcast_shape():
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    base = jnp.asarray([1.25, 2.5], dtype=jnp.float64)
    exponent = jnp.asarray(1.75, dtype=jnp.float64)
    observed = policy_pow(base, exponent)
    assert observed.shape == base.shape
    assert observed.dtype == base.dtype
    check_grads(policy_pow, (base, exponent), order=2, modes=("fwd", "rev"),
                atol=5.0e-5, rtol=5.0e-5)
