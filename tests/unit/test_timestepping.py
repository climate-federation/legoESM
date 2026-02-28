"""Unit tests for time integration schemes."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.timestepping.ssp_rk3 import ssp_rk3_step, integrate_scan


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
