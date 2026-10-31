"""Scalar-libm precision-policy regression tests."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core import transcendentals as tx
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.core.transcendentals import atan, cos, exp, log, log10, pow, sin, tanh

try:
    tx.glibc_libm()
    tx._glibc_version()
except RuntimeError:
    pytest.skip(
        "scalar-libm oracle needs glibc libm.so.6 (not on this platform)",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _restore_policy():
    previous = get_policy()
    yield
    set_policy(previous)
    tx._compiled_libm.cache_clear()
    tx._compiled_libm_path.cache_clear()
    tx._gcc_version.cache_clear()
    tx._glibc_version.cache_clear()


@pytest.mark.parametrize(
    "name,function",
    [("exp", exp), ("atan", atan), ("tanh", tanh), ("sin", sin), ("cos", cos)],
)
def test_libm_policy_is_scalar_library_bit_exact_and_native_is_not(name, function):
    # The non-power-of-two stride avoids repeatedly sampling easy reduction
    # points while keeping every EXP input in its ordinary finite range.
    values = np.linspace(-19.75, 19.25, 100_000, dtype=np.float64)
    values += (np.arange(values.size, dtype=np.float64)
               * np.float64(float.fromhex("0x1p-46")))
    reference = tx._scalar_libm_reference(name, values)

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
    elif name in {"sin", "cos"}:
        # On this certification host XLA sine/cosine already agree with scalar
        # glibc.  They still route through the poisoned callback in the NEMO
        # policy, which the GYRE discriminator tests behaviorally.
        assert native_mismatches == 0


@pytest.mark.parametrize("name,function", [("log", log), ("log10", log10)])
def test_libm_log_policy_is_scalar_library_bit_exact(name, function):
    values = np.geomspace(2.0**-40, 2.0**40, 100_000, dtype=np.float64)
    values += np.arange(values.size, dtype=np.float64) * np.float64(2.0**-60)
    reference = tx._scalar_libm_reference(name, values)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    observed = np.asarray(jax.jit(function)(jnp.asarray(values)))
    assert np.array_equal(observed.view(np.uint64), reference.view(np.uint64))


@pytest.mark.parametrize("function", [exp, atan, tanh, sin, cos])
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


@pytest.mark.parametrize("function", [exp, log, log10, atan, tanh, sin, cos])
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


def test_libm_pow_policy_preserves_broadcast_jit_and_custom_jvp() -> None:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    left = jnp.asarray([[0.5], [2.0], [4.0]], dtype=jnp.float64)
    right = jnp.asarray([[-2.0, 0.5, 3.0]], dtype=jnp.float64)
    eager = np.asarray(pow(left, right))
    compiled = np.asarray(jax.jit(pow)(left, right))
    reference = tx._scalar_libm_binary_reference("pow", np.asarray(left), np.asarray(right))
    _assert_bitwise_equal(eager, reference)
    _assert_bitwise_equal(compiled, reference)
    _, tangent = jax.jvp(
        pow,
        (jnp.asarray(2.0), jnp.asarray(3.0)),
        (jnp.asarray(0.25), jnp.asarray(0.5)),
    )
    reverse = jax.grad(lambda x: pow(x, jnp.asarray(3.0)))(jnp.asarray(2.0))
    assert np.isfinite(float(tangent))
    assert float(reverse) == pytest.approx(12.0)


@pytest.mark.parametrize(
    "left,right",
    [
        (jnp.asarray([2.0], dtype=jnp.float32), 3.0),
        (jnp.asarray([2.0], dtype=jnp.float32), 3),
        (jnp.asarray([2.0], dtype=jnp.float32), jnp.asarray([3], dtype=jnp.int64)),
        (jnp.asarray([2.0], dtype=jnp.float32), jnp.asarray([3.0], dtype=jnp.float64)),
        (jnp.asarray([2], dtype=jnp.int32), jnp.asarray([3.0], dtype=jnp.float64)),
    ],
)
def test_pow_libm_callback_dtype_matches_jax_result_type_and_native_lane(
    left, right
) -> None:
    expected_dtype = jnp.result_type(left, right)
    set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    native = pow(left, right)
    native_reference = jnp.power(*jnp.broadcast_arrays(jnp.asarray(left), jnp.asarray(right)))
    assert native.dtype == native_reference.dtype
    np.testing.assert_array_equal(np.asarray(native), np.asarray(native_reference))

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    observed = pow(left, right)
    assert observed.dtype == expected_dtype
    np.testing.assert_array_equal(np.asarray(observed), np.asarray(native_reference))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("base", [10.0, 10, jnp.asarray(10, dtype=jnp.int32)])
def test_pow_python_scalar_and_integer_base_support_grad_and_jvp(
    dtype, base
) -> None:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    exponent = jnp.asarray(2.0, dtype=dtype)
    tangent_in = jnp.asarray(0.25, dtype=dtype)
    derivative = jax.grad(lambda value: pow(base, value))(exponent)
    _, tangent = jax.jvp(lambda value: pow(base, value), (exponent,), (tangent_in,))
    expected = jnp.power(jnp.asarray(base, dtype=dtype), exponent) * jnp.log(
        jnp.asarray(base, dtype=dtype)
    )
    assert derivative.dtype == dtype
    assert tangent.dtype == dtype
    np.testing.assert_allclose(np.asarray(derivative), np.asarray(expected), rtol=2e-6)
    np.testing.assert_allclose(
        np.asarray(tangent), np.asarray(expected * tangent_in), rtol=2e-6
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("exponent", [3, jnp.asarray(3, dtype=jnp.int32)])
def test_pow_integer_exponent_supports_grad_and_jvp(dtype, exponent) -> None:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    base = jnp.asarray(2.0, dtype=dtype)
    tangent_in = jnp.asarray(0.25, dtype=dtype)
    derivative = jax.grad(lambda value: pow(value, exponent))(base)
    _, tangent = jax.jvp(lambda value: pow(value, exponent), (base,), (tangent_in,))
    expected = jnp.asarray(3.0, dtype=dtype) * base * base
    assert derivative.dtype == dtype
    assert tangent.dtype == dtype
    np.testing.assert_allclose(np.asarray(derivative), np.asarray(expected), rtol=2e-6)
    np.testing.assert_allclose(
        np.asarray(tangent), np.asarray(expected * tangent_in), rtol=2e-6
    )


def test_pow_integer_operands_promote_to_floating_callback_dtype() -> None:
    left = jnp.asarray([2], dtype=jnp.int32)
    set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    native = pow(left, 3)
    native_reference = jnp.power(left, 3)
    assert native.dtype == native_reference.dtype
    np.testing.assert_array_equal(np.asarray(native), np.asarray(native_reference))

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    observed = pow(left, 3)
    assert jnp.issubdtype(observed.dtype, jnp.floating)
    np.testing.assert_array_equal(np.asarray(observed), np.asarray([8.0]))


@pytest.mark.parametrize("function", [tx._scalar_libm_binary, tx._scalar_libm_binary_reference])
def test_direct_pow_callback_preserves_float32_with_python_scalar(function) -> None:
    observed = function("pow", np.asarray([2.0], dtype=np.float32), 3.0)
    assert observed.dtype == np.float32
    np.testing.assert_array_equal(observed, np.asarray([8.0], dtype=np.float32))


_MILLION = 1_000_000


def _large_unary_values(name: str, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    finfo = np.finfo(dtype)
    count = _MILLION - 9
    if name == "exp":
        ordinary = np.linspace(-800.0, 710.0, count, dtype=dtype)
    elif name in {"log", "log10"}:
        positive = np.exp(
            np.linspace(np.log(float(finfo.smallest_subnormal)), np.log(float(finfo.max)), count)
        )
        ordinary = positive.astype(dtype)
    else:
        ordinary = np.linspace(-1.0e6, 1.0e6, count, dtype=dtype)
    special = np.asarray(
        [
            0.0,
            -0.0,
            finfo.smallest_subnormal,
            -finfo.smallest_subnormal,
            finfo.tiny,
            finfo.max,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=dtype,
    )
    return np.concatenate((ordinary, special))


def _large_pow_values(dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.dtype(dtype)
    count = _MILLION - 18
    left = np.linspace(0.125, 32.0, count, dtype=dtype)
    right = np.linspace(-4.0, 4.0, count, dtype=dtype)
    special_left = np.asarray(
        [
            -2.0, -2.0, -2.0, 0.0, -0.0, 1.0, 1.0, np.inf, -np.inf,
            np.inf, np.nan, 2.0, np.nan, 0.0, -0.0, np.inf, -np.inf, 2.0,
        ],
        dtype=dtype,
    )
    special_right = np.asarray(
        [
            2.0, 3.0, 0.5, -1.0, -3.0, np.nan, np.inf, 0.0, 3.0,
            -1.0, 0.0, np.nan, 2.0, np.inf, -np.inf, np.nan, np.nan, -np.inf,
        ],
        dtype=dtype,
    )
    return np.concatenate((left, special_left)), np.concatenate((right, special_right))


def _assert_bitwise_equal(observed: np.ndarray, reference: np.ndarray) -> None:
    word = np.uint32 if observed.dtype == np.float32 else np.uint64
    assert observed.shape == reference.shape
    assert np.array_equal(observed.view(word), reference.view(word))


def test_python_reference_handles_non_contiguous_transposed_input() -> None:
    values = np.linspace(-4.0, 4.0, 60, dtype=np.float64).reshape(5, 12).T
    assert not values.flags.c_contiguous
    reference = tx._scalar_libm_reference("exp", values)
    observed = tx._scalar_libm("exp", values)
    _assert_bitwise_equal(reference, observed)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("name", ["exp", "log", "log10", "atan", "tanh", "sin", "cos"])
def test_compiled_unary_loop_matches_python_scalar_reference_on_one_million_values(
    name: str, dtype: np.dtype
) -> None:
    values = _large_unary_values(name, dtype)
    assert values.size >= _MILLION
    observed = tx._scalar_libm(name, values)
    reference = tx._scalar_libm_reference(name, values)
    _assert_bitwise_equal(observed, reference)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compiled_pow_loop_matches_python_scalar_reference_on_one_million_values(
    dtype: np.dtype,
) -> None:
    left, right = _large_pow_values(dtype)
    assert left.size == right.size >= _MILLION
    observed = tx._scalar_libm_binary("pow", left, right)
    reference = tx._scalar_libm_binary_reference("pow", left, right)
    _assert_bitwise_equal(observed, reference)


def _assert_disassembly_calls_scalar_symbol(disassembly: str, name: str) -> None:
    assert re.search(rf"callq?\s+[^\n]*<{name}@plt>", disassembly)
    assert not re.search(r"_ZGV|libmvec|v(?:exp|log|atan|tanh|sin|cos|pow)", disassembly)


def test_compiled_loops_call_only_the_named_scalar_libm_plt_symbols() -> None:
    shared_object = tx._compiled_libm_path()
    for name in (*tx._UNARY_NAMES, "pow"):
        for suffix in ("f32", "f64"):
            symbol = f"legoesm_libm_{name}_{suffix}"
            disassembly = subprocess.run(
                ["objdump", "-d", f"--disassemble={symbol}", str(shared_object)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            _assert_disassembly_calls_scalar_symbol(disassembly, name)

    planted = subprocess.run(
        [
            "objdump",
            "-d",
            "--disassemble=legoesm_libm_exp_f64",
            str(shared_object),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.replace("<exp@plt>", "<log@plt>")
    with pytest.raises(AssertionError):
        _assert_disassembly_calls_scalar_symbol(planted, "exp")


def test_fast_path_wrong_libm_function_plant_fails_bitwise_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.linspace(0.125, 4.0, 4096, dtype=np.float64)
    reference = tx._scalar_libm_reference("exp", values)
    monkeypatch.setitem(tx._FAST_SYMBOLS, "exp", "log")
    planted = tx._scalar_libm("exp", values)
    with pytest.raises(AssertionError):
        _assert_bitwise_equal(planted, reference)


def _clear_build_caches() -> None:
    tx._compiled_libm.cache_clear()
    tx._compiled_libm_path.cache_clear()
    tx._gcc_version.cache_clear()
    tx._glibc_version.cache_clear()


def test_compile_flags_participate_in_cache_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert {"-O2", "-fno-builtin", "-fno-tree-vectorize"} <= set(
        tx._COMPILE_FLAGS
    )
    assert "-ffast-math" not in tx._COMPILE_FLAGS
    monkeypatch.setenv("LEGOESM_SCALAR_LIBM_CACHE", str(tmp_path / "cache"))
    _clear_build_caches()
    first = tx._compiled_libm_path()
    monkeypatch.setattr(tx, "_COMPILE_FLAGS", (*tx._COMPILE_FLAGS, "-fwrapv"))
    tx._compiled_libm_path.cache_clear()
    second = tx._compiled_libm_path()
    assert first.parent != second.parent


def test_glibc_version_participates_in_cache_key_and_build_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LEGOESM_SCALAR_LIBM_CACHE", str(tmp_path / "cache"))
    _clear_build_caches()
    first = tx._compiled_libm_path()
    actual_version = tx._glibc_version()
    assert tx.libm_build_info()["glibc_version"] == actual_version
    monkeypatch.setattr(tx, "_glibc_version", lambda: actual_version + "-plant")
    tx._compiled_libm_path.cache_clear()
    second = tx._compiled_libm_path()
    assert first.parent != second.parent


def test_missing_compiler_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LEGOESM_SCALAR_LIBM_CACHE", str(tmp_path / "cache"))
    _clear_build_caches()

    def missing_compiler(*_args, **_kwargs):
        raise FileNotFoundError("planted missing gcc")

    monkeypatch.setattr(tx.subprocess, "run", missing_compiler)
    with pytest.raises(RuntimeError, match="GCC version query failed"):
        tx._compiled_libm_path()


def test_compile_error_removes_all_temporary_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "broken.c"
    source.write_text("this is not C\n", encoding="utf-8")
    cache = tmp_path / "cache"
    monkeypatch.setenv("LEGOESM_SCALAR_LIBM_CACHE", str(cache))
    monkeypatch.setattr(tx, "_C_SOURCE", source)
    _clear_build_caches()
    with pytest.raises(RuntimeError, match="scalar-libm build failed"):
        tx._compiled_libm_path()
    assert not list(cache.rglob("scalar_libm-*.so"))
    assert not list(cache.rglob("scalar_libm-*.sha256"))


def test_corrupt_shared_object_with_matching_sha_fails_to_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LEGOESM_SCALAR_LIBM_CACHE", str(tmp_path / "cache"))
    _clear_build_caches()
    shared_object = tx._compiled_libm_path()
    shared_object.write_bytes(b"not a shared object")
    checksum = hashlib.sha256(shared_object.read_bytes()).hexdigest()
    shared_object.with_name(shared_object.name + tx._SHA256_SUFFIX).write_text(
        checksum + "\n", encoding="ascii"
    )
    tx._compiled_libm.cache_clear()
    with pytest.raises(RuntimeError, match="scalar-libm load failed"):
        tx._compiled_libm()


def test_shared_object_sha_mismatch_fails_without_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LEGOESM_SCALAR_LIBM_CACHE", str(tmp_path / "cache"))
    _clear_build_caches()
    shared_object = tx._compiled_libm_path()
    with shared_object.open("ab") as stream:
        stream.write(b"planted corruption")
    tx._compiled_libm.cache_clear()
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        tx._compiled_libm()


def test_world_writable_cache_directory_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o777)
    monkeypatch.setenv("LEGOESM_SCALAR_LIBM_CACHE", str(cache))
    _clear_build_caches()
    with pytest.raises(RuntimeError, match="group/world-accessible"):
        tx._compiled_libm_path()


def test_foreign_owned_cache_directory_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    monkeypatch.setenv("LEGOESM_SCALAR_LIBM_CACHE", str(cache))
    monkeypatch.setattr(tx.os, "geteuid", lambda: cache.stat().st_uid + 1)
    _clear_build_caches()
    with pytest.raises(RuntimeError, match="not owned"):
        tx._compiled_libm_path()


def test_new_cache_directories_are_private(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv("LEGOESM_SCALAR_LIBM_CACHE", str(cache))
    _clear_build_caches()
    shared_object = tx._compiled_libm_path()
    assert os.stat(cache).st_mode & 0o777 == 0o700
    assert os.stat(shared_object.parent).st_mode & 0o777 == 0o700


def test_default_cache_root_uses_xdg_not_shared_tmp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("LEGOESM_SCALAR_LIBM_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert tx._cache_root() == tmp_path / "xdg" / "legoesm" / "scalar-libm"
