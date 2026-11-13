"""The run-wide numerics mode: ``nemo_exact``, ``nemo_fast``, or no scope (``None``)."""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core import transcendentals as tx
from legoesm.core.numerics_mode import (
    NUMERICS_MODES,
    bind_aot,
    current_numerics,
    numerics_scope,
    require_numerics,
)
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.core.source_rounding import nemo_source_round as sr
from legoesm.timestepping import tridiagonal

try:
    tx.glibc_libm()
except RuntimeError:
    pytest.skip("the libm policy needs glibc libm.so.6", allow_module_level=True)


@pytest.fixture
def use_policy():
    """Set the transcendentals policy; it is not in the jit key, so clear caches."""
    previous = get_policy()

    def use(transcendentals: str) -> None:
        set_policy(PrecisionPolicy.fp64(transcendentals=transcendentals))
        jax.clear_caches()

    yield use
    set_policy(previous)
    jax.clear_caches()


def _scope(mode):
    return numerics_scope(mode) if mode else contextlib.nullcontext()


def _guarded(x):
    return sr(sr(x * x) + tx.exp(sr(x)))


def _thomas(x):
    return tridiagonal.thomas_solve(
        -0.1 * x, 2.0 + x, -0.2 * x, x, "nemo_unnormalised")


_ENTRIES = {
    "exp": tx.exp, "log": tx.log, "log10": tx.log10, "atan": tx.atan,
    "tanh": tx.tanh, "sin": tx.sin, "cos": tx.cos,
    "pow": lambda x: tx.pow(x, 1.5 + x),
}


def _bits(array) -> np.ndarray:
    return np.atleast_1d(np.asarray(array)).view(np.uint64)


_CONSTRUCT_MODES: dict[str, set] = {}


def _spy(tag, value):
    _CONSTRUCT_MODES.setdefault(tag, set()).add(current_numerics())
    return sr(value)


def _cond_true(x):
    return _spy("cond", x * x)


def _cond_false(x):
    return _spy("cond", x + x)


def _scan_body(carry, x):
    carry = _spy("scan", carry * x + x)
    return carry, carry


@jax.custom_jvp
def _square(x):
    return _spy("custom_jvp", x * x)


@_square.defjvp
def _square_jvp(primals, tangents):
    (x,), (tangent,) = primals, tangents
    return _square(x), _spy("custom_jvp_rule", 2.0 * x) * tangent


def _constructs(x):
    """lax.cond, lax.scan, a custom_jvp and thomas_solve's custom_vjp, each with
    a guard; module-level bodies, so JAX's inner trace caches could hit."""
    y = jax.lax.cond(x[0] > 0, _cond_true, _cond_false, x)
    _, ys = jax.lax.scan(_scan_body, x[0], x)
    z = _square(y + ys)
    return jnp.sum(tridiagonal.thomas_solve(
        -0.1 * z, 2.0 + z, -0.2 * z, z, "nemo_unnormalised"))


def test_each_mode_traces_once_and_keeps_its_own_program(use_policy):
    use_policy("libm")
    traced_under = []

    @jax.jit
    def toy(x):
        traced_under.append(current_numerics())
        return _guarded(x)

    x = jnp.linspace(0.1, 0.9, 7)
    outputs = []
    for mode in ("nemo_exact", "nemo_fast", "nemo_exact", "nemo_fast"):
        with numerics_scope(mode):
            outputs.append(_bits(toy(x)))
    assert traced_under == ["nemo_exact", "nemo_fast"]
    assert not np.array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[2], outputs[0])
    np.testing.assert_array_equal(outputs[3], outputs[1])

    text = {}
    for mode in NUMERICS_MODES:
        with numerics_scope(mode):
            text[mode] = toy.lower(x).as_text()
    assert text["nemo_exact"] != text["nemo_fast"]
    assert "is_finite" in text["nemo_exact"] and "callback" in text["nemo_exact"]
    assert "is_finite" not in text["nemo_fast"] and "callback" not in text["nemo_fast"]


def test_two_user_contexts_are_independent_jit_key_slots():
    other = jax.make_user_context(None)
    traced_under = []

    @jax.jit
    def f(x):
        traced_under.append((current_numerics(), other.value))
        return x + 1.0

    calls = [("nemo_exact", 1), ("nemo_exact", 2), ("nemo_fast", 2), ("nemo_exact", 1)]
    for mode, value in calls:
        with numerics_scope(mode), other(value):
            assert (current_numerics(), other.value) == (mode, value)
            f(1.0)
    assert traced_under == calls[:3]


def test_one_shared_jit_retraces_cond_scan_and_custom_rules_per_mode(
        monkeypatch):
    monkeypatch.setattr(tridiagonal, "nemo_source_round",
                        lambda value: _spy("custom_vjp", value))
    _CONSTRUCT_MODES.clear()
    value, grad = jax.jit(_constructs), jax.jit(jax.grad(_constructs))
    x = jnp.linspace(0.1, 0.9, 6)
    outputs = []
    for mode in ("nemo_exact", "nemo_fast", "nemo_exact", "nemo_fast"):
        with numerics_scope(mode):
            outputs.append((_bits(value(x)), _bits(grad(x))))
    both = {"nemo_exact", "nemo_fast"}
    assert _CONSTRUCT_MODES == dict.fromkeys(
        ("cond", "scan", "custom_jvp", "custom_jvp_rule", "custom_vjp"), both)
    for later, earlier in ((2, 0), (3, 1)):
        for got, want in zip(outputs[later], outputs[earlier]):
            np.testing.assert_array_equal(got, want)
    for program in (value, grad):
        for mode in NUMERICS_MODES:
            with numerics_scope(mode):
                text = program.lower(x).as_text()
            assert ("is_finite" in text) is (mode == "nemo_exact")


def test_scope_rejects_unknown_names_and_a_different_nested_mode():
    for bad in ("fast", "exact", "NEMO_FAST", None):
        with pytest.raises(ValueError, match="numerics mode must be one of"):
            with numerics_scope(bad):
                pass
    with numerics_scope("nemo_exact"):
        with numerics_scope("nemo_exact"):
            assert current_numerics() == "nemo_exact"
        with pytest.raises(RuntimeError, match="inside 'nemo_exact'"):
            with numerics_scope("nemo_fast"):
                pass
        assert current_numerics() == "nemo_exact"
    assert current_numerics() is None


def test_require_numerics_passes_only_on_the_active_mode():
    require_numerics(None)
    with pytest.raises(RuntimeError):
        require_numerics("nemo_exact")
    with numerics_scope("nemo_fast"):
        require_numerics("nemo_fast")
        for other in (None, "nemo_exact"):
            with pytest.raises(RuntimeError):
                require_numerics(other)


@pytest.mark.parametrize("transcendentals", ["libm", "native"])
def test_unscoped_lowering_equals_nemo_exact_and_fast_overrides_the_policy(
        use_policy, transcendentals):
    """No scope must lower exactly as today.  Checked against ``nemo_exact`` in
    this process rather than a recorded digest, which any JAX upgrade would
    break; the one-time comparison with upstream main is in the commit."""
    use_policy(transcendentals)
    x = jnp.linspace(0.1, 0.9, 5)
    for name, fn in {"guarded": _guarded, "thomas": _thomas, **_ENTRIES}.items():
        text = {}
        for mode in (None, *NUMERICS_MODES):
            with _scope(mode):
                text[mode] = jax.jit(fn).lower(x).as_text()
        assert text[None] == text["nemo_exact"], name
        fast_differs = name in ("guarded", "thomas") or transcendentals == "libm"
        assert (text["nemo_fast"] != text["nemo_exact"]) is fast_differs, name


@pytest.mark.parametrize("value", [
    np.array([1.5, -0.0]), np.array([1.5], np.float32), np.float64(1.5), 1.5, 3,
    np.array([1, 2]), jnp.ones(2, jnp.float32), jnp.ones(2, jnp.int32),
    jnp.ones(2, bool),
], ids=["np_f64", "np_f32", "np_scalar", "py_float", "py_int", "np_int",
        "jax_f32", "jax_i32", "jax_bool"])
def test_fast_guard_returns_what_the_exact_guard_returns(value):
    result, shape = {}, {}
    for mode in NUMERICS_MODES:
        with numerics_scope(mode):
            result[mode], shape[mode] = sr(value), jax.eval_shape(sr, value)
    exact, fast = result["nemo_exact"], result["nemo_fast"]
    assert type(fast) is type(exact)
    assert (fast.dtype, fast.weak_type) == (exact.dtype, exact.weak_type)
    assert ((shape["nemo_fast"].dtype, shape["nemo_fast"].weak_type)
            == (shape["nemo_exact"].dtype, shape["nemo_exact"].weak_type))
    assert np.atleast_1d(fast).tobytes() == np.atleast_1d(exact).tobytes()


def test_fast_takes_the_native_path_for_every_entry_and_argument_kind(use_policy):
    floats, ints = jnp.linspace(0.1, 0.9, 5), jnp.arange(1, 6)
    cases = [
        (name, fn, args)
        for name, fn in _ENTRIES.items() if name != "pow"
        for args in ((floats,), (ints,), (0.5,), (2,))
    ] + [
        ("pow", tx.pow, args)
        for args in ((floats, 1.5 + floats), (ints, ints), (floats, 2),
                     (2.0, floats), (ints, 0.5), (2, 3))
    ]

    def run():
        return [(jax.jit(fn).lower(*args).as_text(), fn(*args))
                for _, fn, args in cases]

    use_policy("native")
    native = run()
    use_policy("libm")
    with numerics_scope("nemo_fast"):
        fast = run()
    for (name, _, args), (fast_text, fast_value), (native_text, native_value) in zip(
            cases, fast, native):
        label = (name, [type(a).__name__ for a in args])
        assert fast_text == native_text, label
        assert ((fast_value.dtype, fast_value.weak_type)
                == (native_value.dtype, native_value.weak_type)), label
        assert np.atleast_1d(fast_value).tobytes() == np.atleast_1d(
            native_value).tobytes(), label


def test_grad_is_exact_in_both_modes_and_fast_vmap_matches_single_calls(use_policy):
    use_policy("libm")
    x = jnp.linspace(0.1, 0.9, 7)
    for mode in NUMERICS_MODES:
        with numerics_scope(mode):
            grad = jax.jit(jax.grad(lambda v: jnp.sum(_guarded(v))))(x)
        np.testing.assert_allclose(grad, 2.0 * x + np.exp(x), rtol=1e-14)
    rows = jnp.stack([x, 2.0 * x])
    with numerics_scope("nemo_fast"):
        batched = jax.jit(jax.vmap(_guarded))(rows)
        single = jnp.stack([jax.jit(_guarded)(row) for row in rows])
    np.testing.assert_array_equal(_bits(batched), _bits(single))


def test_lazy_backward_is_traced_under_the_mode_current_when_pulled(monkeypatch):
    traced_under = []
    guard = tridiagonal.nemo_source_round

    def spy(value):
        traced_under.append(current_numerics())
        return guard(value)

    monkeypatch.setattr(tridiagonal, "nemo_source_round", spy)
    d = jnp.linspace(0.1, 0.9, 6)
    with numerics_scope("nemo_fast"):
        x, pull = jax.vjp(_thomas, d)
    forward = set(traced_under)
    traced_under.clear()
    pull(jnp.ones_like(x))
    # Accepted hole (numerics_mode docstring): the custom_vjp backward is
    # traced when the cotangent is pulled, under the mode current then.
    assert forward == {"nemo_fast"}
    assert set(traced_under) == {None}


def test_bind_aot_runs_only_under_the_mode_it_was_built_in():
    fn = jax.jit(lambda v: sr(sr(v * v) + v))
    x = jnp.linspace(0.1, 0.9, 5)
    with numerics_scope("nemo_exact"):
        compiled = bind_aot(fn.lower(x).compile())
        exported = bind_aot(jax.export.export(fn)(x))
        reference = _bits(compiled(x))
        np.testing.assert_array_equal(_bits(exported.call(x)), reference)
        exported.vjp()
    runs = (lambda: compiled(x), lambda: exported.call(x), exported.vjp)
    for mode in ("nemo_fast", None):
        with _scope(mode):
            for run in runs:
                with pytest.raises(RuntimeError, match="built for numerics mode"):
                    run()


def test_bind_aot_leaves_nothing_to_launder():
    """A traced or lowered object could be compiled later under another mode
    and stamped with that one; a bound program hands nothing on."""
    fn = jax.jit(lambda v: sr(v))
    x = jnp.ones(3)
    with numerics_scope("nemo_exact"):
        for unbindable in (fn, fn.trace(x), fn.lower(x)):
            with pytest.raises(TypeError):
                bind_aot(unbindable)
        compiled = bind_aot(fn.lower(x).compile())
        exported = bind_aot(jax.export.export(fn)(x))
    for bound in (compiled, exported):
        for name in ("lower", "compile", "serialize", "as_text", "mlir_module"):
            assert not hasattr(bound, name), name
