"""Unit tests for time integration schemes."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.timestepping.ssp_rk3 import ssp_rk3_step, integrate_scan
from legoesm.timestepping.ssp_rk34 import (
    ssp_rk34_step,
    integrate_scan as integrate_scan_rk34,
)
from legoesm.timestepping.ssp_rk54 import (
    ssp_rk54_step,
    integrate_scan as integrate_scan_rk54,
)

_IS_X64 = bool(jax.config.jax_enable_x64)


class TestSSPRK3:
    """Tests for the SSP-RK3 time integrator."""

    def test_constant_field_unchanged(self):
        """Zero tendency should leave state unchanged."""
        state = jnp.array([1.0, 2.0, 3.0])
        tendency_fn = lambda s: jnp.zeros_like(s)
        result = ssp_rk3_step(state, tendency_fn, dt=1.0)
        assert jnp.allclose(result, state)

    def test_linear_growth(self):
        """For dy/dt = 1, after dt=1: y should increase by ~1."""
        state = jnp.array([0.0])
        tendency_fn = lambda s: jnp.ones_like(s)
        result = ssp_rk3_step(state, tendency_fn, dt=1.0)
        assert jnp.allclose(result, jnp.array([1.0]))

    def test_exponential_growth(self):
        """For dy/dt = y, exact: y(t) = y0 * exp(t)."""
        y0 = jnp.array([1.0])
        tendency_fn = lambda y: y
        dt = 0.01
        n_steps = 100

        y = y0
        for _ in range(n_steps):
            y = ssp_rk3_step(y, tendency_fn, dt)

        exact = y0 * jnp.exp(dt * n_steps)
        assert jnp.allclose(y, exact, rtol=1e-4)

    def test_pytree_state(self):
        """SSP-RK3 should work with pytree states (dict, namedtuple, etc.)."""
        from collections import namedtuple
        State = namedtuple("State", ["x", "v"])

        state = State(x=jnp.array([0.0]), v=jnp.array([1.0]))

        def tendency(s):
            return State(x=s.v, v=-s.x)  # Simple harmonic oscillator

        result = ssp_rk3_step(state, tendency, dt=0.01)
        assert jnp.isfinite(result.x).all()
        assert jnp.isfinite(result.v).all()

    def test_differentiable(self):
        """jax.grad should work through SSP-RK3."""
        def loss(y0):
            tendency_fn = lambda y: -y  # Exponential decay
            y_final = ssp_rk3_step(y0, tendency_fn, dt=0.1)
            return jnp.sum(y_final ** 2)

        y0 = jnp.array([1.0, 2.0])
        grads = jax.grad(loss)(y0)
        assert jnp.all(jnp.isfinite(grads))

    def test_integrate_scan(self):
        """integrate_scan should produce the same result as a Python loop."""
        state = jnp.array([1.0])
        tendency_fn = lambda y: -0.1 * y  # Exponential decay
        dt = 0.1
        n_steps = 10

        # Scan method
        final_scan, traj_scan = integrate_scan(state, tendency_fn, n_steps, dt)

        # Loop method
        y = state
        for _ in range(n_steps):
            y = ssp_rk3_step(y, tendency_fn, dt)

        assert jnp.allclose(final_scan, y, atol=1e-6)


class TestSSPRK34:
    """Tests for the SSP-RK(4,3) time integrator."""

    def test_constant_field_unchanged(self):
        """Zero tendency should leave state unchanged."""
        state = jnp.array([1.0, 2.0, 3.0])
        tendency_fn = lambda s: jnp.zeros_like(s)
        result = ssp_rk34_step(state, tendency_fn, dt=1.0)
        assert jnp.allclose(result, state)

    def test_linear_growth(self):
        """For dy/dt = 1, after dt=1: y should increase by exactly 1."""
        state = jnp.array([0.0])
        tendency_fn = lambda s: jnp.ones_like(s)
        result = ssp_rk34_step(state, tendency_fn, dt=1.0)
        assert jnp.allclose(result, jnp.array([1.0]))

    def test_exponential_growth(self):
        """For dy/dt = y, exact: y(t) = y0 * exp(t)."""
        y0 = jnp.array([1.0])
        tendency_fn = lambda y: y
        dt = 0.01
        n_steps = 100

        y = y0
        for _ in range(n_steps):
            y = ssp_rk34_step(y, tendency_fn, dt)

        exact = y0 * jnp.exp(dt * n_steps)
        assert jnp.allclose(y, exact, rtol=5e-6)

    def test_higher_order_than_rk3(self):
        """RK34 should be more accurate than RK3 for dy/dt = y."""
        y0 = jnp.array([1.0])
        tendency_fn = lambda y: y
        dt = 0.1
        n_steps = 10

        y3 = y0
        for _ in range(n_steps):
            y3 = ssp_rk3_step(y3, tendency_fn, dt)

        y34 = y0
        for _ in range(n_steps):
            y34 = ssp_rk34_step(y34, tendency_fn, dt)

        exact = y0 * jnp.exp(dt * n_steps)
        err3 = jnp.abs(y3 - exact)
        err34 = jnp.abs(y34 - exact)
        assert jnp.all(err34 < err3)

    def test_integrate_scan(self):
        """integrate_scan should produce the same result as a Python loop."""
        state = jnp.array([1.0])
        tendency_fn = lambda y: -0.1 * y
        dt = 0.1
        n_steps = 10

        final_scan, _ = integrate_scan_rk34(state, tendency_fn, n_steps, dt)

        y = state
        for _ in range(n_steps):
            y = ssp_rk34_step(y, tendency_fn, dt)

        assert jnp.allclose(final_scan, y, atol=1e-6)


class TestSSPRK54:
    """Tests for the SSP-RK(5,4) time integrator."""

    def test_constant_field_unchanged(self):
        """Zero tendency should leave state unchanged."""
        state = jnp.array([1.0, 2.0, 3.0])
        tendency_fn = lambda s: jnp.zeros_like(s)
        result = ssp_rk54_step(state, tendency_fn, dt=1.0)
        assert jnp.allclose(result, state)

    def test_linear_growth(self):
        """For dy/dt = 1, after dt=1: y should increase by exactly 1."""
        state = jnp.array([0.0])
        tendency_fn = lambda s: jnp.ones_like(s)
        result = ssp_rk54_step(state, tendency_fn, dt=1.0)
        assert jnp.allclose(result, jnp.array([1.0]))

    def test_exponential_growth(self):
        """For dy/dt = y, exact: y(t) = y0 * exp(t). 4th order should beat RK3."""
        y0 = jnp.array([1.0])
        tendency_fn = lambda y: y
        dt = 0.01
        n_steps = 100

        y = y0
        for _ in range(n_steps):
            y = ssp_rk54_step(y, tendency_fn, dt)

        exact = y0 * jnp.exp(dt * n_steps)
        # x32 has expected roundoff accumulation over 100 steps.
        rtol = 1.0e-7 if _IS_X64 else 5.0e-6
        assert jnp.allclose(y, exact, rtol=rtol)

    def test_higher_order_than_rk3(self):
        """RK54 should be more accurate than RK3 for dy/dt = y."""
        y0 = jnp.array([1.0])
        tendency_fn = lambda y: y
        dt = 0.1
        n_steps = 10

        # RK3
        y3 = y0
        for _ in range(n_steps):
            y3 = ssp_rk3_step(y3, tendency_fn, dt)

        # RK54
        y54 = y0
        for _ in range(n_steps):
            y54 = ssp_rk54_step(y54, tendency_fn, dt)

        exact = y0 * jnp.exp(dt * n_steps)
        err3 = jnp.abs(y3 - exact)
        err54 = jnp.abs(y54 - exact)
        assert jnp.all(err54 < err3)

    def test_fourth_order_convergence(self):
        """Error should decrease as O(dt^4) when halving dt."""
        y0 = jnp.array([1.0])
        tendency_fn = lambda y: y
        T = 1.0  # Integrate to t=1

        errors = []
        # Choose dt-pair by precision so truncation error dominates roundoff.
        dts = [0.1, 0.05] if _IS_X64 else [0.25, 0.125]
        for dt in dts:
            n_steps = int(T / dt)
            y = y0
            for _ in range(n_steps):
                y = ssp_rk54_step(y, tendency_fn, dt)
            exact = y0 * jnp.exp(T)
            errors.append(float(jnp.abs(y - exact).sum()))

        # For 4th order: error(dt/2) / error(dt) ~ (1/2)^4 = 1/16
        ratio = errors[1] / errors[0]
        ratio_max = 0.1 if _IS_X64 else 0.2
        assert ratio < ratio_max

    def test_pytree_state(self):
        """SSP-RK54 should work with pytree states."""
        from collections import namedtuple
        State = namedtuple("State", ["x", "v"])

        state = State(x=jnp.array([0.0]), v=jnp.array([1.0]))

        def tendency(s):
            return State(x=s.v, v=-s.x)  # Simple harmonic oscillator

        result = ssp_rk54_step(state, tendency, dt=0.01)
        assert jnp.isfinite(result.x).all()
        assert jnp.isfinite(result.v).all()

    def test_differentiable(self):
        """jax.grad should work through SSP-RK54."""
        def loss(y0):
            tendency_fn = lambda y: -y  # Exponential decay
            y_final = ssp_rk54_step(y0, tendency_fn, dt=0.1)
            return jnp.sum(y_final ** 2)

        y0 = jnp.array([1.0, 2.0])
        grads = jax.grad(loss)(y0)
        assert jnp.all(jnp.isfinite(grads))

    def test_integrate_scan(self):
        """integrate_scan should produce the same result as a Python loop."""
        state = jnp.array([1.0])
        tendency_fn = lambda y: -0.1 * y  # Exponential decay
        dt = 0.1
        n_steps = 10

        # Scan method
        final_scan, traj_scan = integrate_scan_rk54(
            state, tendency_fn, n_steps, dt)

        # Loop method
        y = state
        for _ in range(n_steps):
            y = ssp_rk54_step(y, tendency_fn, dt)

        assert jnp.allclose(final_scan, y, atol=1e-6)

    def test_harmonic_oscillator_energy(self):
        """Energy should be nearly conserved for a harmonic oscillator."""
        from collections import namedtuple
        State = namedtuple("State", ["x", "v"])

        state = State(x=jnp.array([1.0]), v=jnp.array([0.0]))

        def tendency(s):
            return State(x=s.v, v=-s.x)

        E0 = float((state.x ** 2 + state.v ** 2).sum())

        s = state
        for _ in range(1000):
            s = ssp_rk54_step(s, tendency, dt=0.01)

        E_final = float((s.x ** 2 + s.v ** 2).sum())
        # x32 accumulates noticeably more roundoff over 1000 steps.
        tol = 1.0e-6 if _IS_X64 else 1.0e-4
        assert abs(E_final - E0) / E0 < tol
