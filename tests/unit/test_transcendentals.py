"""Scalar-libm precision-policy regression tests."""

from __future__ import annotations

import ctypes

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.core.transcendentals import cos, exp, log, log10, sin, tanh


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


@pytest.fixture(autouse=True)
def _restore_policy():
    previous = get_policy()
    yield
    set_policy(previous)


@pytest.mark.parametrize(
    "name,function", [("exp", exp), ("tanh", tanh), ("sin", sin), ("cos", cos)]
)
def test_libm_policy_is_scalar_library_bit_exact_and_native_is_not(name, function):
    # The non-power-of-two stride avoids repeatedly sampling easy reduction
    # points while keeping every EXP input in its ordinary finite range.
    values = np.linspace(-19.75, 19.25, 100_000, dtype=np.float64)
    values += (np.arange(values.size, dtype=np.float64)
               * np.float64(float.fromhex("0x1p-46")))
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
    if name in {"exp", "tanh"}:
        assert native_mismatches > 0
    else:
        # On this certification host XLA sine/cosine already agree with scalar
        # glibc.  They still route through the poisoned callback in the NEMO
        # policy, which the GYRE discriminator tests behaviorally.
        assert native_mismatches == 0


@pytest.mark.parametrize("name,function", [("log", log), ("log10", log10)])
def test_libm_log_policy_is_scalar_library_bit_exact(name, function):
    values = np.geomspace(2.0**-40, 2.0**40, 100_000, dtype=np.float64)
    values += np.arange(values.size, dtype=np.float64) * np.float64(2.0**-60)
    reference = _libm_reference(name, values)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    observed = np.asarray(jax.jit(function)(jnp.asarray(values)))
    assert np.array_equal(observed.view(np.uint64), reference.view(np.uint64))


@pytest.mark.parametrize("function", [exp, tanh, sin, cos])
def test_libm_policy_jit_and_eager_are_bit_exact(function):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    values = jnp.asarray([-3.125, -0.25, 0.0, 0.75, 4.5], dtype=jnp.float64)
    eager = np.asarray(function(values))
    compiled = np.asarray(jax.jit(function)(values))
    assert np.array_equal(eager.view(np.uint64), compiled.view(np.uint64))


@pytest.mark.parametrize("function", [log, log10])
def test_libm_log_policy_jit_and_eager_are_bit_exact(function):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    values = jnp.asarray([0.125, 0.25, 0.5, 0.75, 4.5], dtype=jnp.float64)
    eager = np.asarray(function(values))
    compiled = np.asarray(jax.jit(function)(values))
    assert np.array_equal(eager.view(np.uint64), compiled.view(np.uint64))


@pytest.mark.parametrize("function", [exp, log, log10, tanh, sin, cos])
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
