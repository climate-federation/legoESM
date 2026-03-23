"""Category 11: Ensemble parallelism.

Tests vmap-based ensemble execution, statistics, and perturbation.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.parallel.ensemble import (
    stack_states,
    unstack_states,
    perturb_initial_conditions,
    make_ensemble_step,
    ensemble_integrate,
    ensemble_mean,
    ensemble_std,
    ensemble_spread,
)


# =========================================================================
# Helpers
# =========================================================================

from collections import namedtuple

_State = namedtuple("State", ["h", "u"])


def _make_state():
    """Create a simple NamedTuple state."""
    return _State(
        h=jnp.ones((6, 4, 4), dtype=jnp.float64) * 1000.0,
        u=jnp.ones((6, 4, 4), dtype=jnp.float64) * 10.0,
    )


def _simple_step(state, dt=1.0):
    """Simple dynamics: h -> h + 0.1*dt, u -> u * 0.99."""
    return state._replace(
        h=state.h + 0.1 * dt,
        u=state.u * 0.99,
    )


# =========================================================================
# 11a) Ensemble step — vmap correctness
# =========================================================================

class TestEnsembleStep:
    """make_ensemble_step should vmap correctly."""

    def test_ensemble_step_shape(self):
        """Output should have leading batch dimension."""
        state = _make_state()
        key = jax.random.PRNGKey(42)
        ensemble = perturb_initial_conditions(state, key, n_members=4, scale=0.01)

        step_fn = make_ensemble_step(_simple_step)
        result = step_fn(ensemble)

        assert result.h.shape == (4, 6, 4, 4)
        assert result.u.shape == (4, 6, 4, 4)

    def test_members_differ(self):
        """Different initial conditions should give different results."""
        state = _make_state()
        key = jax.random.PRNGKey(42)
        ensemble = perturb_initial_conditions(state, key, n_members=4, scale=0.01)

        step_fn = make_ensemble_step(_simple_step)
        result = step_fn(ensemble)

        # Members should differ (initial conditions were perturbed)
        h_std = float(jnp.std(result.h, axis=0).mean())
        assert h_std > 0


# =========================================================================
# 11b) Ensemble statistics
# =========================================================================

class TestEnsembleStatistics:
    """Ensemble statistics should match manual computation."""

    def test_ensemble_mean_matches_manual(self):
        state = _make_state()
        key = jax.random.PRNGKey(0)
        ensemble = perturb_initial_conditions(state, key, n_members=4, scale=0.01)

        mean = ensemble_mean(ensemble)
        manual_mean_h = jnp.mean(ensemble.h, axis=0)
        np.testing.assert_allclose(np.array(mean.h), np.array(manual_mean_h), atol=1e-10)

    def test_ensemble_std_matches_manual(self):
        state = _make_state()
        key = jax.random.PRNGKey(1)
        ensemble = perturb_initial_conditions(state, key, n_members=4, scale=0.01)

        std = ensemble_std(ensemble)
        manual_std_h = jnp.std(ensemble.h, axis=0)
        np.testing.assert_allclose(np.array(std.h), np.array(manual_std_h), atol=1e-10)

    def test_ensemble_spread_positive(self):
        state = _make_state()
        key = jax.random.PRNGKey(2)
        ensemble = perturb_initial_conditions(state, key, n_members=4, scale=0.01)

        spread = ensemble_spread(ensemble)
        assert isinstance(spread, dict)
        for name, val in spread.items():
            assert val >= 0, f"Spread for {name} is negative"


# =========================================================================
# 11d) Ensemble + scan integration
# =========================================================================

class TestEnsembleIntegrate:
    """ensemble_integrate should run multiple steps."""

    def test_integration_shape(self):
        state = _make_state()
        key = jax.random.PRNGKey(3)
        ensemble = perturb_initial_conditions(state, key, n_members=4, scale=0.01)

        result = ensemble_integrate(_simple_step, ensemble, n_steps=5, dt=1.0)

        assert result.h.shape == (4, 6, 4, 4)
        assert result.u.shape == (4, 6, 4, 4)

    def test_integration_changes_state(self):
        state = _make_state()
        key = jax.random.PRNGKey(4)
        ensemble = perturb_initial_conditions(state, key, n_members=4, scale=0.01)

        result = ensemble_integrate(_simple_step, ensemble, n_steps=10, dt=1.0)

        # h should increase (step adds 0.1*dt)
        assert float(jnp.mean(result.h)) > float(jnp.mean(ensemble.h))
        # u should decrease (step multiplies by 0.99)
        assert float(jnp.mean(result.u)) < float(jnp.mean(ensemble.u))


# =========================================================================
# 11e) Perturbation
# =========================================================================

class TestPerturbation:
    """perturb_initial_conditions should create diverse ensemble."""

    def test_output_shape(self):
        state = _make_state()
        key = jax.random.PRNGKey(5)
        ensemble = perturb_initial_conditions(state, key, n_members=4, scale=0.01)
        assert ensemble.h.shape == (4, 6, 4, 4)

    def test_perturbation_scale(self):
        """Perturbation should be O(scale)."""
        state = _make_state()
        key = jax.random.PRNGKey(6)
        scale = 0.01
        ensemble = perturb_initial_conditions(state, key, n_members=100, scale=scale)

        # Relative spread should be O(scale)
        h_mean = jnp.mean(ensemble.h, axis=0)
        h_std = jnp.std(ensemble.h, axis=0)
        relative_std = float(jnp.mean(h_std / h_mean))
        assert relative_std < scale * 5  # generous bound
        assert relative_std > scale * 0.1  # not too small

    def test_mean_close_to_base(self):
        """Ensemble mean should be close to base state."""
        state = _make_state()
        key = jax.random.PRNGKey(7)
        ensemble = perturb_initial_conditions(state, key, n_members=100, scale=0.01)
        mean = ensemble_mean(ensemble)
        np.testing.assert_allclose(
            np.array(mean.h), np.array(state.h), rtol=0.01
        )


# =========================================================================
# 11f) Stack/unstack roundtrip
# =========================================================================

class TestStackUnstack:
    """stack_states and unstack_states should be inverse operations."""

    def test_roundtrip(self):
        states = [_make_state() for _ in range(3)]
        # Modify each to be different
        states[1] = states[1]._replace(h=states[1].h * 2)
        states[2] = states[2]._replace(h=states[2].h * 3)

        batched = stack_states(states)
        assert batched.h.shape == (3, 6, 4, 4)

        recovered = unstack_states(batched)
        assert len(recovered) == 3
        np.testing.assert_allclose(np.array(recovered[0].h), np.array(states[0].h))
        np.testing.assert_allclose(np.array(recovered[1].h), np.array(states[1].h))
        np.testing.assert_allclose(np.array(recovered[2].h), np.array(states[2].h))

    def test_vmap_matches_loop(self):
        """vmap over ensemble should match Python loop."""
        state = _make_state()
        key = jax.random.PRNGKey(8)
        ensemble = perturb_initial_conditions(state, key, n_members=4, scale=0.01)

        # vmap
        vmapped = jax.vmap(_simple_step)(ensemble)

        # Loop
        members = unstack_states(ensemble)
        loop_results = [_simple_step(m) for m in members]
        loop_batched = stack_states(loop_results)

        np.testing.assert_allclose(
            np.array(vmapped.h), np.array(loop_batched.h), atol=1e-10
        )
