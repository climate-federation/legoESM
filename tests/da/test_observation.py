"""Tests for observation operators."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.da.observation import (
    DirectObsOperator,
    InterpolatingObsOperator,
    CompositeObsOperator,
    generate_synthetic_obs,
    Observation,
)


def _make_sw_state_latlon(n_lat=8, n_lon=16, val=100.0):
    shape = (n_lat, n_lon)
    return ShallowWaterState(
        h=Field(data=jnp.ones(shape) * val, name="h", dims=(), units="m"),
        u=Field(data=jnp.ones(shape) * 5.0, name="u", dims=(), units="m/s"),
        v=Field(data=jnp.ones(shape) * -3.0, name="v", dims=(), units="m/s"),
        h_s=Field(data=jnp.zeros(shape), name="h_s", dims=(), units="m"),
    )


@pytest.fixture
def latlon_grid():
    from legoesm.grids import create_latlon_grid
    return create_latlon_grid(8, 16)


class TestDirectObsOperator:
    def test_exact_values(self):
        state = _make_sw_state_latlon(8, 16, val=42.0)
        # Observe h at specific indices
        idx = (jnp.array([0, 1, 2, 3]), jnp.array([0, 4, 8, 12]))
        op = DirectObsOperator("h", idx)
        result = op(state)
        assert jnp.allclose(result, 42.0)
        assert result.shape == (4,)

    def test_differentiable(self):
        state = _make_sw_state_latlon(8, 16)
        idx = (jnp.array([0, 1]), jnp.array([0, 1]))
        op = DirectObsOperator("h", idx)

        def f(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            return jnp.sum(op(s))

        grad = jax.grad(f)(state.h.data)
        assert jnp.all(jnp.isfinite(grad))


class TestInterpolatingObsOperator:
    def test_constant_field(self, latlon_grid):
        """Interpolation of constant field should return constant."""
        state = _make_sw_state_latlon(8, 16, val=99.0)
        obs_lat = jnp.array([0.0, 0.5, -0.5])
        obs_lon = jnp.array([0.0, 1.0, 3.0])
        op = InterpolatingObsOperator("h", obs_lat, obs_lon, grid=latlon_grid)
        result = op(state)
        assert jnp.allclose(result, 99.0, atol=1e-3)

    def test_differentiable(self, latlon_grid):
        state = _make_sw_state_latlon(8, 16)
        obs_lat = jnp.array([0.0, 0.3])
        obs_lon = jnp.array([0.0, 1.0])
        op = InterpolatingObsOperator("h", obs_lat, obs_lon, grid=latlon_grid)

        def f(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            return jnp.sum(op(s))

        grad = jax.grad(f)(state.h.data)
        assert jnp.all(jnp.isfinite(grad))

    def test_correct_shape(self, latlon_grid):
        state = _make_sw_state_latlon(8, 16)
        n_obs = 5
        obs_lat = jnp.linspace(-1.0, 1.0, n_obs)
        obs_lon = jnp.linspace(0.0, 2 * jnp.pi, n_obs)
        op = InterpolatingObsOperator("h", obs_lat, obs_lon, grid=latlon_grid)
        result = op(state)
        assert result.shape == (n_obs,)


class TestCompositeObsOperator:
    def test_concatenation(self):
        state = _make_sw_state_latlon(8, 16, val=10.0)
        idx1 = (jnp.array([0, 1]), jnp.array([0, 1]))
        idx2 = (jnp.array([2, 3, 4]), jnp.array([2, 3, 4]))
        op1 = DirectObsOperator("h", idx1)
        op2 = DirectObsOperator("u", idx2)
        composite = CompositeObsOperator((op1, op2))
        result = composite(state)
        assert result.shape == (5,)
        assert jnp.allclose(result[:2], 10.0)
        assert jnp.allclose(result[2:], 5.0)

    def test_differentiable(self):
        state = _make_sw_state_latlon(8, 16)
        idx1 = (jnp.array([0]), jnp.array([0]))
        idx2 = (jnp.array([1]), jnp.array([1]))
        op1 = DirectObsOperator("h", idx1)
        op2 = DirectObsOperator("u", idx2)
        composite = CompositeObsOperator((op1, op2))

        def f(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            return jnp.sum(composite(s))

        grad = jax.grad(f)(state.h.data)
        assert jnp.all(jnp.isfinite(grad))


class TestSyntheticObs:
    def test_shapes(self):
        state = _make_sw_state_latlon(8, 16)
        # Create a fake trajectory (2 time steps)
        trajectory = jax.tree.map(lambda a: jnp.stack([a, a * 1.1]), state)
        idx = (jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        op = DirectObsOperator("h", idx)
        obs = generate_synthetic_obs(
            trajectory,
            operators=(op,),
            time_indices=(0,),
            error_stds=(1.0,),
            key=jax.random.PRNGKey(0),
        )
        assert len(obs) == 1
        assert obs[0].values.shape == (3,)
        assert obs[0].errors.shape == (3,)
        assert obs[0].time_index == 0

    def test_noise_variance(self):
        """Noise should have expected variance."""
        state = _make_sw_state_latlon(8, 16, val=100.0)
        trajectory = jax.tree.map(lambda a: jnp.stack([a]), state)
        n_obs = 1000
        idx = (jnp.arange(n_obs) % 8, jnp.arange(n_obs) % 16)
        op = DirectObsOperator("h", idx)
        sigma = 2.0
        obs = generate_synthetic_obs(
            trajectory,
            operators=(op,),
            time_indices=(0,),
            error_stds=(sigma,),
            key=jax.random.PRNGKey(42),
        )
        noise = obs[0].values - 100.0  # truth is constant
        assert jnp.abs(jnp.std(noise) - sigma) < 0.3  # within ~15% of sigma
