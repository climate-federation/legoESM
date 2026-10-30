"""Precision-policy transcendental functions.

``native`` delegates to JAX/XLA.  ``libm`` calls the scalar ``exp``, ``log``,
``log10``, ``pow``, ``atan``, ``tanh``, ``sin``, and ``cos`` entry points from
``libm.so.6`` through :func:`jax.pure_callback`.
That is the soname linked by the NEMO certification executables on the
campaign host (glibc 2.34).  A small packaged C loop deliberately invokes the
scalar C function once per element, in element order: NumPy ufuncs may dispatch
their own vector math and are not an equivalent oracle.  The shared object is
compiled fail-closed with vectorization and builtin substitution disabled and
cached by C-source digest and GCC version.

The libm arm is CPU-only by construction.  Custom JVP rules preserve forward
and reverse autodiff using d(exp)=exp and d(tanh)=1-tanh**2, with the primal
transcendental evaluated through the same scalar library call.  The policy is
read while JAX traces the caller, so certification harnesses must set it
before creating/jitting the production step.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from functools import lru_cache, partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import get_policy

_LIBM_SONAME = "libm.so.6"
_LIBM: ctypes.CDLL | None = None
_C_SOURCE = Path(__file__).with_name("_scalar_libm.c")
_COMPILE_FLAGS = (
    "-O2",
    "-fno-builtin",
    "-fno-tree-vectorize",
    "-fPIC",
    "-shared",
)
_UNARY_NAMES = ("exp", "log", "log10", "atan", "tanh", "sin", "cos")
_FAST_SYMBOLS = {name: name for name in (*_UNARY_NAMES, "pow")}


def glibc_libm() -> ctypes.CDLL:
    """Return glibc's ``libm.so.6``, loaded on first use.

    Loading at import time made every ``legoesm`` import fail on non-glibc
    hosts (macOS), although only the ``libm`` certification arm needs it.
    """
    global _LIBM
    if _LIBM is None:
        try:
            _LIBM = ctypes.CDLL(_LIBM_SONAME)
        except OSError as exc:
            raise RuntimeError(
                f"the glibc scalar-libm oracle needs {_LIBM_SONAME!r}, which "
                "is not available on this platform"
            ) from exc
    return _LIBM


def _run_checked(command: list[str], purpose: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(f"scalar-libm {purpose} failed: {stderr.strip()}") from exc


@lru_cache(maxsize=1)
def _gcc_version() -> str:
    return _run_checked(
        ["gcc", "-dumpfullversion", "-dumpversion"], "GCC version query"
    ).stdout.strip()


@lru_cache(maxsize=1)
def _compiled_libm_path() -> Path:
    try:
        source = _C_SOURCE.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"scalar-libm C source is unavailable: {_C_SOURCE}") from exc
    source_digest = hashlib.sha256(source).hexdigest()
    version = _gcc_version()
    version_digest = hashlib.sha256(version.encode("utf-8")).hexdigest()
    cache_root = Path(
        os.environ.get(
            "LEGOESM_SCALAR_LIBM_CACHE",
            str(Path(tempfile.gettempdir()) / "legoesm-scalar-libm"),
        )
    )
    cache_dir = cache_root / f"{source_digest}-{version_digest}"
    shared_object = cache_dir / "scalar_libm.so"
    if shared_object.is_file():
        return shared_object
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="scalar_libm-", suffix=".so", dir=cache_dir, delete=False
        ) as tmp:
            temporary = Path(tmp.name)
        command = [
            "gcc",
            *_COMPILE_FLAGS,
            str(_C_SOURCE),
            "-o",
            str(temporary),
            "-Wl,-z,defs",
            "-lm",
        ]
        _run_checked(command, "build")
        os.replace(temporary, shared_object)
    except Exception:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        raise
    return shared_object


@lru_cache(maxsize=1)
def _compiled_libm() -> ctypes.CDLL:
    path = _compiled_libm_path()
    try:
        return ctypes.CDLL(str(path))
    except OSError as exc:
        raise RuntimeError(f"scalar-libm load failed for {path}: {exc}") from exc


def _dtype_pointer(dtype: np.dtype):
    return ctypes.POINTER(
        ctypes.c_float if dtype == np.dtype(np.float32) else ctypes.c_double
    )


def _fast_function(name: str, dtype: np.dtype, *, binary: bool):
    symbol = _FAST_SYMBOLS[name]
    suffix = "f32" if dtype == np.dtype(np.float32) else "f64"
    try:
        function = getattr(_compiled_libm(), f"legoesm_libm_{symbol}_{suffix}")
    except AttributeError as exc:
        raise RuntimeError(
            f"scalar-libm shared object lacks {name}/{suffix} entry point"
        ) from exc
    pointer = _dtype_pointer(dtype)
    function.argtypes = (
        (pointer, pointer, pointer, ctypes.c_size_t)
        if binary
        else (pointer, pointer, ctypes.c_size_t)
    )
    function.restype = None
    return function


def _scalar_libm(name: str, values: np.ndarray) -> np.ndarray:
    """Evaluate one unary libm entry point in the compiled element-order loop."""
    source = np.asarray(values)
    if source.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"libm {name} requires float32/float64, got {source.dtype}")
    if name not in _UNARY_NAMES:
        raise ValueError(f"unsupported unary libm function {name!r}")
    contiguous = np.array(source, dtype=source.dtype, order="C", copy=True)
    result = np.empty(contiguous.shape, dtype=contiguous.dtype)
    pointer = _dtype_pointer(contiguous.dtype)
    _fast_function(name, contiguous.dtype, binary=False)(
        contiguous.ctypes.data_as(pointer),
        result.ctypes.data_as(pointer),
        contiguous.size,
    )
    return result


def _scalar_libm_reference(name: str, values: np.ndarray) -> np.ndarray:
    """Slow Python/ctypes reference retained only for exactness tests."""
    source = np.asarray(values)
    if source.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"libm {name} requires float32/float64, got {source.dtype}")
    function = getattr(glibc_libm(), name)
    function.argtypes = (ctypes.c_double,)
    function.restype = ctypes.c_double
    result = np.empty_like(source)
    source_flat = source.reshape(-1)
    result_flat = result.reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
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
    """Evaluate one binary libm entry point in the compiled element-order loop."""
    left_source, right_source = np.broadcast_arrays(left, right)
    dtype = np.result_type(left_source.dtype, right_source.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"libm {name} requires float32/float64")
    if name != "pow":
        raise ValueError(f"unsupported binary libm function {name!r}")
    left_contiguous = np.array(left_source, dtype=dtype, order="C", copy=True)
    right_contiguous = np.array(right_source, dtype=dtype, order="C", copy=True)
    result = np.empty(left_contiguous.shape, dtype=dtype)
    pointer = _dtype_pointer(dtype)
    _fast_function(name, dtype, binary=True)(
        left_contiguous.ctypes.data_as(pointer),
        right_contiguous.ctypes.data_as(pointer),
        result.ctypes.data_as(pointer),
        result.size,
    )
    return result


def _scalar_libm_binary_reference(
    name: str, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Slow Python/ctypes binary reference retained only for exactness tests."""
    left_source, right_source = np.broadcast_arrays(left, right)
    dtype = np.result_type(left_source.dtype, right_source.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"libm {name} requires float32/float64")
    function = getattr(glibc_libm(), name)
    function.argtypes = (ctypes.c_double, ctypes.c_double)
    function.restype = ctypes.c_double
    result = np.empty(left_source.shape, dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
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
