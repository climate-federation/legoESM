"""Precision-policy transcendental functions.

``native`` delegates to JAX/XLA.  ``libm`` calls the scalar ``exp``, ``log``,
``log10``, ``tanh``, ``sin``, and ``cos`` entry points from ``libm.so.6`` through
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

_LIBM_SONAMES = ("libm.so.6", "libm.dylib")
_LIBM: ctypes.CDLL | None = None
_LIBM_LOADED = False


def _load_libm() -> ctypes.CDLL:
    """Resolve the scalar libm backing the ``libm`` precision policy.

    ``libm.so.6`` is the glibc soname linked by the NEMO certification
    executables on the campaign host; ``libm.dylib`` is the same scalar
    library on macOS, where libm is folded into libSystem and no
    ``libm.so.6`` exists.  The load is deferred to the first ``libm``-policy
    evaluation so that importing this module — and everything downstream
    of it — succeeds on platforms without a ``libm.so.6``.
    """
    global _LIBM, _LIBM_LOADED
    if not _LIBM_LOADED:
        _LIBM_LOADED = True
        for soname in _LIBM_SONAMES:
            try:
                _LIBM = ctypes.CDLL(soname)
                break
            except OSError:
                continue
    if _LIBM is None:
        raise RuntimeError(
            "PrecisionPolicy.transcendentals='libm' requires a scalar libm "
            f"({', '.join(_LIBM_SONAMES)}); none could be loaded on this "
            "platform"
        )
    return _LIBM


def _scalar_libm(name: str, values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"libm {name} requires float32/float64, got {source.dtype}")
    function = getattr(_load_libm(), name)
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


__all__ = ("cos", "exp", "log", "log10", "sin", "tanh")
