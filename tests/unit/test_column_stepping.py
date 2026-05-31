"""Unit tests for the shared single-column time-stepping kernels.

Exercises :mod:`legoesm.core.column_stepping` directly with a toy scalar
ODE and trivial ``apply`` / ``average`` operators (no atmosphere/ocean
state types), verifying the integrator numerics, sub-step time sampling,
the AB2 multistep weights, and the registry / arity validation that both
SCMs share.

Run with:
    .venv/bin/python -m pytest tests/unit/test_column_stepping.py -v
"""

from __future__ import annotations

import math
import warnings

import pytest

from legoesm.core.column_stepping import (
    ab2_effective_tendency,
    build_explicit_integrators,
    register_integrator,
)


# Trivial state algebra: state / tendency are plain Python floats.
def _apply(state, tend, dt):
    return state + dt * tend


def _average(*tends, weights):
    return sum(w * t for w, t in zip(weights, tends))


@pytest.fixture
def integrators():
    return build_explicit_integrators(_apply, _average)


# ---------------------------------------------------------------------------
# Registry contents and integrator order of accuracy
# ---------------------------------------------------------------------------


def test_registry_has_builtin_integrators(integrators):
    assert set(integrators) == {"forward_euler", "rk2", "rk4", "ab2"}


def test_euler_single_step(integrators):
    # dx/dt = -x ; one Euler step from x0=1 with dt=0.1 -> 1 - 0.1 = 0.9
    f = lambda s, aux, t: (-s, aux)
    new, aux = integrators["forward_euler"](1.0, None, f, 0.1, 0.0)
    assert new == pytest.approx(0.9)
    assert aux is None


def _integrate(step_fn, x0, dt, nsteps):
    f = lambda s, aux, t: (-s, aux)  # dx/dt = -x  -> x(t) = x0 e^{-t}
    x, aux = x0, None
    t = 0.0
    for _ in range(nsteps):
        x, aux = step_fn(x, aux, f, dt, t)
        t += dt
    return x


def test_integrator_order_of_accuracy(integrators):
    """RK4 error << RK2 error << Euler error for dx/dt=-x over [0, 1]."""
    x0, T, n = 1.0, 1.0, 20
    dt = T / n
    exact = math.exp(-T)
    err = {
        name: abs(_integrate(integrators[name], x0, dt, n) - exact)
        for name in ("forward_euler", "rk2", "rk4")
    }
    assert err["rk2"] < err["forward_euler"]
    assert err["rk4"] < err["rk2"]
    assert err["rk4"] < 1e-7  # 4th order on a smooth problem


def test_rk_convergence_rate(integrators):
    """Halving dt cuts RK2 error ~4x and RK4 error ~16x."""
    x0, T = 1.0, 1.0
    exact = math.exp(-T)

    def err(name, n):
        return abs(_integrate(integrators[name], x0, T / n, n) - exact)

    r2 = err("rk2", 10) / err("rk2", 20)
    r4 = err("rk4", 10) / err("rk4", 20)
    assert r2 == pytest.approx(4.0, rel=0.25)
    assert r4 == pytest.approx(16.0, rel=0.35)


# ---------------------------------------------------------------------------
# Sub-step time sampling (time-dependent RHS)
# ---------------------------------------------------------------------------


def test_stage_time_sampling(integrators):
    """dx/dt = t over one step: RK2/RK4 sample sub-step times exactly.

    Exact: x(dt) - x(0) = dt^2 / 2.  Forward Euler samples only t=0 and
    gives 0; RK2 and RK4 must hit the analytic answer.
    """
    f = lambda s, aux, t: (t, aux)
    dt = 0.5
    expected = 0.5 * dt * dt

    e_euler = integrators["forward_euler"](0.0, None, f, dt, 0.0)[0]
    e_rk2 = integrators["rk2"](0.0, None, f, dt, 0.0)[0]
    e_rk4 = integrators["rk4"](0.0, None, f, dt, 0.0)[0]
    assert e_euler == pytest.approx(0.0)
    assert e_rk2 == pytest.approx(expected)
    assert e_rk4 == pytest.approx(expected)


def test_aux_carry_threaded_from_stage_one(integrators):
    """The aux carry returned is the stage-1 value (single-valued)."""
    seen = []

    def f(s, aux, t):
        seen.append(aux)
        return -s, (aux or 0) + 1

    _, aux_out = integrators["rk4"](1.0, 0, f, 0.1, 0.0)
    # Stage 1 sees aux=0 and returns 1; later stages reuse that carry.
    assert aux_out == 1
    assert seen[0] == 0
    assert all(a == 1 for a in seen[1:])


# ---------------------------------------------------------------------------
# AB2 multistep weights
# ---------------------------------------------------------------------------


def test_ab2_effective_tendency_weights():
    # 1.5 * a - 0.5 * b
    assert ab2_effective_tendency(2.0, 1.0, _average) == pytest.approx(2.5)
    assert ab2_effective_tendency(1.0, 1.0, _average) == pytest.approx(1.0)


def test_ab2_first_entry_is_euler(integrators):
    f = lambda s, aux, t: (-s, aux)
    ab2 = integrators["ab2"](1.0, None, f, 0.1, 0.0)
    euler = integrators["forward_euler"](1.0, None, f, 0.1, 0.0)
    assert ab2[0] == pytest.approx(euler[0])


# ---------------------------------------------------------------------------
# register_integrator: arity validation and legacy 4-arg shim
# ---------------------------------------------------------------------------


def test_register_5arg(integrators):
    def custom(state, aux, f, dt, t):
        tend, aux_out = f(state, aux, t)
        return _apply(state, tend, 2.0 * dt), aux_out

    register_integrator(integrators, "double", custom)
    assert integrators["double"] is custom


def test_register_4arg_legacy_warns_and_tags(integrators):
    def legacy(state, aux, f, dt):  # old signature, no stage time
        tend, aux_out = f(state, aux)
        return _apply(state, tend, dt), aux_out

    with pytest.warns(DeprecationWarning):
        register_integrator(integrators, "legacy", legacy)
    wrapped = integrators["legacy"]
    assert getattr(wrapped, "_is_legacy_4arg", False) is True
    # The wrapper still produces a correct forward-Euler-like step.
    f = lambda s, aux, t: (-s, aux)
    out, _ = wrapped(1.0, None, f, 0.1, 0.0)
    assert out == pytest.approx(0.9)


def test_register_bad_arity_raises(integrators):
    with pytest.raises(TypeError):
        register_integrator(integrators, "bad3", lambda a, b, c: None)


def test_register_non_callable_raises(integrators):
    with pytest.raises(TypeError):
        register_integrator(integrators, "nope", 42)


def test_register_varargs_raises(integrators):
    def variadic(*args):
        return args

    with pytest.raises(TypeError):
        register_integrator(integrators, "varargs", variadic)


def test_register_required_kwonly_raises(integrators):
    def kwonly(state, aux, f, dt, *, t):
        return state, aux

    with pytest.raises(TypeError):
        register_integrator(integrators, "kwonly", kwonly)
