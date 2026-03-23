"""Tests for the 4D-Var cost function."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.da.control_vector import build_control_spec, state_to_control
from legoesm.da.background_error import DiagonalB
from legoesm.da.observation import DirectObsOperator, Observation
from legoesm.da.cost_function import build_cost_fn, build_cost_and_grad_fn


class _TrivialModel:
    """A trivial model that does nothing (identity step)."""
    def step(self, state, dt):
        return state


def _make_sw_state(shape=(4, 4)):
    return ShallowWaterState(
        h=Field(data=jnp.ones(shape) * 100.0, name="h", dims=(), units="m"),
        u=Field(data=jnp.zeros(shape), name="u", dims=(), units="m/s"),
        v=Field(data=jnp.zeros(shape), name="v", dims=(), units="m/s"),
        h_s=Field(data=jnp.zeros(shape), name="h_s", dims=(), units="m"),
    )


@pytest.fixture
def setup():
    """Common setup for cost function tests."""
    state = _make_sw_state()
    spec = build_control_spec(state)
    x_b = state_to_control(state, spec)
    sigma = jnp.ones(spec.total_size) * 10.0
    B = DiagonalB(sigma=sigma)
    model = _TrivialModel()
    return state, spec, x_b, B, model


class TestCostFunction:
    def test_J_at_background_no_obs(self, setup):
        """J(x_b) = 0 when no observations."""
        state, spec, x_b, B, model = setup
        cost_fn = build_cost_fn(
            model, x_b, observations=(), B=B,
            control_spec=spec, template_state=state,
            dt=1.0, n_steps=1,
        )
        J = cost_fn(x_b)
        assert jnp.allclose(J, 0.0, atol=1e-6)

    def test_J_non_negative(self, setup):
        """J(x) >= 0 for all x."""
        state, spec, x_b, B, model = setup
        idx = (jnp.array([0, 1]), jnp.array([0, 1]))
        op = DirectObsOperator("h", idx)
        obs = Observation(
            values=jnp.array([105.0, 95.0]),
            errors=jnp.array([5.0, 5.0]),
            time_index=0, operator=op,
        )
        cost_fn = build_cost_fn(
            model, x_b, observations=(obs,), B=B,
            control_spec=spec, template_state=state,
            dt=1.0, n_steps=1,
        )
        key = jax.random.PRNGKey(0)
        for _ in range(5):
            key, subkey = jax.random.split(key)
            x = x_b + jax.random.normal(subkey, x_b.shape) * 10.0
            assert cost_fn(x) >= 0.0

    def test_gradient_finite(self, setup):
        """Gradient should be finite."""
        state, spec, x_b, B, model = setup
        idx = (jnp.array([0]), jnp.array([0]))
        op = DirectObsOperator("h", idx)
        obs = Observation(
            values=jnp.array([110.0]),
            errors=jnp.array([5.0]),
            time_index=0, operator=op,
        )
        cost_and_grad = build_cost_and_grad_fn(
            model, x_b, observations=(obs,), B=B,
            control_spec=spec, template_state=state,
            dt=1.0, n_steps=1,
        )
        J, g = cost_and_grad(x_b)
        assert jnp.all(jnp.isfinite(g))
        assert g.shape == x_b.shape

    def test_gradient_correctness_finite_diff(self, setup):
        """AD gradient should match finite differences."""
        jax.config.update("jax_enable_x64", True)
        state, spec, x_b, B, model = setup
        # Use float64 for FD check
        x_b = x_b.astype(jnp.float64)
        sigma = jnp.ones(spec.total_size, dtype=jnp.float64) * 10.0
        B_64 = DiagonalB(sigma=sigma)

        # Recreate state in float64
        state64 = jax.tree.map(lambda f: f.replace(data=f.data.astype(jnp.float64))
                               if isinstance(f, Field) else f, state)

        idx = (jnp.array([0, 1]), jnp.array([0, 1]))
        op = DirectObsOperator("h", idx)
        obs = Observation(
            values=jnp.array([105.0, 95.0], dtype=jnp.float64),
            errors=jnp.array([5.0, 5.0], dtype=jnp.float64),
            time_index=0, operator=op,
        )
        cost_fn = build_cost_fn(
            model, x_b, observations=(obs,), B=B_64,
            control_spec=spec, template_state=state64,
            dt=1.0, n_steps=1,
        )
        grad_ad = jax.grad(cost_fn)(x_b)

        # Finite difference
        h = 1e-5
        grad_fd = jnp.zeros_like(x_b)
        # Check a subset of components (full FD is expensive)
        for i in range(min(10, x_b.shape[0])):
            e_i = jnp.zeros_like(x_b).at[i].set(1.0)
            fp = cost_fn(x_b + h * e_i)
            fm = cost_fn(x_b - h * e_i)
            grad_fd = grad_fd.at[i].set((fp - fm) / (2 * h))

        # Check first 10 components
        n_check = min(10, x_b.shape[0])
        rel_err = jnp.abs(grad_ad[:n_check] - grad_fd[:n_check]) / (
            jnp.maximum(jnp.abs(grad_ad[:n_check]), 1e-10)
        )
        assert jnp.all(rel_err < 1e-3), f"Max rel error: {jnp.max(rel_err)}"

    def test_jit_compiles(self, setup):
        """Cost function should JIT-compile."""
        state, spec, x_b, B, model = setup
        cost_fn = build_cost_fn(
            model, x_b, observations=(), B=B,
            control_spec=spec, template_state=state,
            dt=1.0, n_steps=1,
        )
        jit_cost = jax.jit(cost_fn)
        J = jit_cost(x_b)
        assert jnp.isfinite(J)
