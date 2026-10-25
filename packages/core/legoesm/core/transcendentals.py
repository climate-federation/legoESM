"""Precision-policy transcendental functions.

``native`` delegates to JAX/XLA.  ``libm`` calls the scalar ``exp``, ``log``,
``log10``, ``pow``, ``atan``, ``tanh``, ``sin``, and ``cos`` entry points from ``libm.so.6`` through
:func:`jax.pure_callback`.
That is the soname linked by the NEMO certification executables on the
campaign host (glibc 2.34).  The callback deliberately invokes the scalar C
function once per element: NumPy ufuncs may dispatch their own vector math
and are not an equivalent oracle.

The libm arm is CPU-only by construction.  Custom JVP rules preserve forward
and reverse autodiff using d(exp)=exp and d(tanh)=1-tanh**2, with the primal
transcendental evaluated through the same scalar library call.  The policy is
read while JAX traces the caller, so certification harnesses must set it
before creating/jitting the production step.
"""

from __future__ import annotations

import ctypes
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import get_policy

_LIBM_SONAME = "libm.so.6"
_LIBM = ctypes.CDLL(_LIBM_SONAME)


def _scalar_libm(name: str, values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"libm {name} requires float32/float64, got {source.dtype}")
    function = getattr(_LIBM, name)
    function.argtypes = (ctypes.c_double,)
    function.restype = ctypes.c_double
    result = np.empty_like(source)
    source_flat = source.reshape(-1)
    result_flat = result.reshape(-1)
    for index in range(source_flat.size):
        result_flat[index] = function(float(source_flat[index]))
    return result


def _callback(name: str, value: jax.Array) -> jax.Array:
    if jax.default_backend() != "cpu":
        raise RuntimeError(
            f"PrecisionPolicy.transcendentals='libm' is CPU-only; got "
            f"{jax.default_backend()!r}"
        )
    result = jax.ShapeDtypeStruct(value.shape, value.dtype)
    return jax.pure_callback(
        partial(_scalar_libm, name), result, value,
        vmap_method="broadcast_all",
    )


def _scalar_libm_binary(
    name: str, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left_source, right_source = np.broadcast_arrays(left, right)
    if left_source.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"libm {name} requires float32/float64")
    function = getattr(_LIBM, name)
    function.argtypes = (ctypes.c_double, ctypes.c_double)
    function.restype = ctypes.c_double
    result = np.empty_like(left_source)
    for index in range(result.size):
        result.flat[index] = function(
            float(left_source.flat[index]), float(right_source.flat[index])
        )
    return result


def _callback_binary(name: str, left: jax.Array, right: jax.Array) -> jax.Array:
    if jax.default_backend() != "cpu":
        raise RuntimeError("scalar libm policy is CPU-only")
    shape = jnp.broadcast_shapes(left.shape, right.shape)
    result = jax.ShapeDtypeStruct(shape, jnp.result_type(left, right))
    return jax.pure_callback(
        partial(_scalar_libm_binary, name), result, left, right,
        vmap_method="broadcast_all",
    )


@jax.custom_jvp
def _libm_exp(value: jax.Array) -> jax.Array:
    return _callback("exp", value)


@_libm_exp.defjvp
def _libm_exp_jvp(primals, tangents):
    (value,), (value_dot,) = primals, tangents
    result = _libm_exp(value)
    return result, result * value_dot


@jax.custom_jvp
def _libm_tanh(value: jax.Array) -> jax.Array:
    return _callback("tanh", value)


@_libm_tanh.defjvp
def _libm_tanh_jvp(primals, tangents):
    (value,), (value_dot,) = primals, tangents
    result = _libm_tanh(value)
    return result, (1.0 - result * result) * value_dot


@jax.custom_jvp
def _libm_sin(value: jax.Array) -> jax.Array:
    return _callback("sin", value)


@_libm_sin.defjvp
def _libm_sin_jvp(primals, tangents):
    (value,), (value_dot,) = primals, tangents
    result = _libm_sin(value)
    return result, _libm_cos(value) * value_dot


@jax.custom_jvp
def _libm_cos(value: jax.Array) -> jax.Array:
    return _callback("cos", value)


@_libm_cos.defjvp
def _libm_cos_jvp(primals, tangents):
    (value,), (value_dot,) = primals, tangents
    result = _libm_cos(value)
    return result, -_libm_sin(value) * value_dot


def exp(value) -> jax.Array:
    """Evaluate exponential under the active precision policy."""
    array = jnp.asarray(value)
    if get_policy().transcendentals == "native":
        return jnp.exp(array)
    return _libm_exp(array)


@jax.custom_jvp
def _libm_log(value: jax.Array) -> jax.Array:
    return _callback("log", value)


@_libm_log.defjvp
def _libm_log_jvp(primals, tangents):
    (value,), (value_dot,) = primals, tangents
    return _libm_log(value), value_dot / value


@jax.custom_jvp
def _libm_log10(value: jax.Array) -> jax.Array:
    return _callback("log10", value)


@_libm_log10.defjvp
def _libm_log10_jvp(primals, tangents):
    (value,), (value_dot,) = primals, tangents
    result = _libm_log10(value)
    return result, value_dot / (value * jnp.log(jnp.asarray(10.0, value.dtype)))


@jax.custom_jvp
def _libm_atan(value: jax.Array) -> jax.Array:
    return _callback("atan", value)


@_libm_atan.defjvp
def _libm_atan_jvp(primals, tangents):
    (value,), (value_dot,) = primals, tangents
    return _libm_atan(value), value_dot / (1.0 + value * value)


@jax.custom_jvp
def _libm_pow(left: jax.Array, right: jax.Array) -> jax.Array:
    return _callback_binary("pow", left, right)


@_libm_pow.defjvp
def _libm_pow_jvp(primals, tangents):
    left, right = primals
    left_dot, right_dot = tangents
    result = _libm_pow(left, right)
    tangent = result * (right_dot * _libm_log(left) + right * left_dot / left)
    return result, tangent


def log(value) -> jax.Array:
    """Evaluate natural logarithm under the active precision policy."""
    array = jnp.asarray(value)
    if get_policy().transcendentals == "native":
        return jnp.log(array)
    return _libm_log(array)


def log10(value) -> jax.Array:
    """Evaluate base-10 logarithm under the active precision policy."""
    array = jnp.asarray(value)
    if get_policy().transcendentals == "native":
        return jnp.log10(array)
    return _libm_log10(array)


def atan(value) -> jax.Array:
    """Evaluate arctangent under the active precision policy."""
    array = jnp.asarray(value)
    if get_policy().transcendentals == "native":
        return jnp.arctan(array)
    return _libm_atan(array)


def pow(left, right) -> jax.Array:
    """Evaluate binary power under the active precision policy."""
    left_array, right_array = jnp.broadcast_arrays(
        jnp.asarray(left), jnp.asarray(right)
    )
    if get_policy().transcendentals == "native":
        return jnp.power(left_array, right_array)
    return _libm_pow(left_array, right_array)


def tanh(value) -> jax.Array:
    """Evaluate hyperbolic tangent under the active precision policy."""
    array = jnp.asarray(value)
    if get_policy().transcendentals == "native":
        return jnp.tanh(array)
    return _libm_tanh(array)


def sin(value) -> jax.Array:
    """Evaluate sine under the active precision policy."""
    array = jnp.asarray(value)
    if get_policy().transcendentals == "native":
        return jnp.sin(array)
    return _libm_sin(array)


def cos(value) -> jax.Array:
    """Evaluate cosine under the active precision policy."""
    array = jnp.asarray(value)
    if get_policy().transcendentals == "native":
        return jnp.cos(array)
    return _libm_cos(array)


__all__ = ("atan", "cos", "exp", "log", "log10", "pow", "sin", "tanh")
