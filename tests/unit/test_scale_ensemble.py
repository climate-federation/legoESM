"""Category 11: Ensemble parallelism tests.

Tests vmap correctness, ensemble statistics, stack/unstack,
perturbation, integration, and sharding.
"""

from __future__ import annotations

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from legoesm.parallel.ensemble import (
    stack_states,
    unstack_states,
    perturb_initial_conditions,
    make_ensemble_step,
    make_ensemble_step_jit,
    ensemble_integrate,
    ensemble_mean,
    ensemble_std,
    ensemble_percentile,
    create_ensemble_mesh,
    shard_ensemble,
    gather_ensemble,
)


N_MEMBERS = 4


# ---------------------------------------------------------------------------
# stack / unstack
# ---------------------------------------------------------------------------

class TestStackUnstack:
    def test_stack_creates_leading_dim(self):
        states = [{"a": jnp.ones(3) * i} for i in range(N_MEMBERS)]
        batched = stack_states(states)
        assert batched["a"].shape == (N_MEMBERS, 3)

    def test_unstack_recovers_individuals(self):
        states = [{"a": jnp.ones(3) * i} for i in range(N_MEMBERS)]
        batched = stack_states(states)
        recovered = unstack_states(batched)
        assert len(recovered) == N_MEMBERS
        for i, s in enumerate(recovered):
            np.testing.assert_allclose(s["a"], float(i))

    def test_stack_unstack_roundtrip(self):
        states = [{"x": jnp.array([float(i)])} for i in range(3)]
        batched = stack_states(states)
        recovered = unstack_states(batched)
        for i, s in enumerate(recovered):
            np.testing.assert_allclose(s["x"], float(i))


# ---------------------------------------------------------------------------
# perturb_initial_conditions
# ---------------------------------------------------------------------------

class TestPerturbIC:
    def test_perturb_shape(self):
        state = {"T": jnp.ones((6, 4, 4))}
        key = jax.random.PRNGKey(42)
        batched = perturb_initial_conditions(state, key, n_members=N_MEMBERS)
        assert batched["T"].shape == (N_MEMBERS, 6, 4, 4)

    def test_perturb_members_differ(self):
        state = {"T": jnp.ones((6, 4, 4))}
        key = jax.random.PRNGKey(42)
        batched = perturb_initial_conditions(state, key, n_members=N_MEMBERS)
        # Different members should have different values
        diffs = batched["T"][0] - batched["T"][1]
        assert jnp.any(diffs != 0.0)

    def test_perturb_multiplicative(self):
        state = {"T": jnp.ones((3,)) * 300.0}
        key = jax.random.PRNGKey(0)
        batched = perturb_initial_conditions(
            state, key, n_members=N_MEMBERS, scale=0.01, multiplicative=True,
        )
        # Values should be close to 300 but not exactly 300
        assert jnp.all(jnp.abs(batched["T"] - 300.0) < 300.0 * 0.1)

    def test_perturb_additive(self):
        state = {"T": jnp.ones((3,)) * 300.0}
        key = jax.random.PRNGKey(0)
        batched = perturb_initial_conditions(
            state, key, n_members=N_MEMBERS, scale=1.0, multiplicative=False,
        )
        # Should be 300 + noise
        assert jnp.all(jnp.abs(batched["T"] - 300.0) < 10.0)


# ---------------------------------------------------------------------------
# make_ensemble_step (vmap wrapping)
# ---------------------------------------------------------------------------

class TestMakeEnsembleStep:
    def test_vmap_correctness(self):
        """vmapped step should apply the function to each member."""
        def step(state, dt):
            return {"T": state["T"] + dt}

        ensemble_step = make_ensemble_step(step, in_axes=(0, None))

        states = stack_states([{"T": jnp.ones(3) * i} for i in range(N_MEMBERS)])
        result = ensemble_step(states, 1.0)

        for i in range(N_MEMBERS):
            np.testing.assert_allclose(result["T"][i], float(i) + 1.0)

    def test_vmap_preserves_shape(self):
        def step(state, dt):
            return {"T": state["T"] * 2.0}

        ensemble_step = make_ensemble_step(step, in_axes=(0, None))
        states = {"T": jnp.ones((N_MEMBERS, 6, 4, 4))}
        result = ensemble_step(states, 1.0)
        assert result["T"].shape == (N_MEMBERS, 6, 4, 4)


# ---------------------------------------------------------------------------
# ensemble_integrate
# ---------------------------------------------------------------------------

class TestEnsembleIntegrate:
    def test_integration_n_steps(self):
        """Running 5 steps of +dt should add 5*dt to each member."""
        def step(state, dt):
            return {"T": state["T"] + dt}

        ensemble_step = make_ensemble_step(step, in_axes=(0, None))
        init = {"T": jnp.zeros((N_MEMBERS, 3))}
        final = ensemble_integrate(ensemble_step, init, n_steps=5, dt=2.0)
        np.testing.assert_allclose(final["T"], 10.0)

    def test_integration_with_trajectory(self):
        def step(state, dt):
            return {"T": state["T"] + dt}

        ensemble_step = make_ensemble_step(step, in_axes=(0, None))
        init = {"T": jnp.zeros((N_MEMBERS, 3))}
        final, traj = ensemble_integrate(
            ensemble_step, init, n_steps=3, dt=1.0, save_trajectory=True,
        )
        assert traj["T"].shape == (3, N_MEMBERS, 3)
        np.testing.assert_allclose(final["T"], 3.0)


# ---------------------------------------------------------------------------
# ensemble statistics
# ---------------------------------------------------------------------------

class TestEnsembleStatistics:
    def test_ensemble_mean(self):
        batched = {"T": jnp.array([[1.0, 2.0], [3.0, 4.0]])}
        mean = ensemble_mean(batched)
        np.testing.assert_allclose(mean["T"], [2.0, 3.0])

    def test_ensemble_std(self):
        batched = {"T": jnp.array([[1.0, 2.0], [3.0, 4.0]])}
        std = ensemble_std(batched)
        expected = jnp.std(batched["T"], axis=0)
        np.testing.assert_allclose(std["T"], expected)

    def test_ensemble_percentile(self):
        batched = {"T": jnp.arange(100.0).reshape(10, 10)}
        p50 = ensemble_percentile(batched, 50.0)
        assert p50["T"].shape == (10,)

    def test_mean_of_identical_is_original(self):
        """Mean of identical members should be the member itself."""
        single = jnp.ones((5,)) * 7.0
        batched = {"T": jnp.broadcast_to(single[None, :], (N_MEMBERS, 5))}
        mean = ensemble_mean(batched)
        np.testing.assert_allclose(mean["T"], 7.0)

    def test_std_of_identical_is_zero(self):
        single = jnp.ones((5,)) * 7.0
        batched = {"T": jnp.broadcast_to(single[None, :], (N_MEMBERS, 5)).copy()}
        std = ensemble_std(batched)
        np.testing.assert_allclose(std["T"], 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Ensemble mesh and sharding (single device)
# ---------------------------------------------------------------------------

class TestEnsembleMesh:
    def test_create_mesh(self):
        n_devices = len(jax.devices())
        # n_members must be divisible by n_devices
        n_members = n_devices * 2
        mesh = create_ensemble_mesh(n_members)
        assert mesh.axis_names == ('ensemble',)

    def test_create_mesh_indivisible_raises(self):
        n_devices = len(jax.devices())
        if n_devices == 1:
            # With 1 device, any n_members is divisible by 1 — nothing to test
            pytest.skip("Single device — all counts divisible by 1")
        # If >1 devices, use a count not divisible by n_devices
        bad_count = n_devices + 1
        if bad_count % n_devices == 0:
            bad_count = n_devices * 2 + 1  # guaranteed odd if n_devices > 1
        with pytest.raises(ValueError, match="divisible"):
            create_ensemble_mesh(bad_count)

    def test_shard_and_gather_roundtrip(self):
        n_devices = len(jax.devices())
        n_members = n_devices
        mesh = create_ensemble_mesh(n_members)
        state = {"T": jnp.arange(n_members * 3, dtype=jnp.float32).reshape(n_members, 3)}
        sharded = shard_ensemble(state, mesh)
        gathered = gather_ensemble(sharded)
        np.testing.assert_array_equal(gathered["T"], state["T"])


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestEnsembleJIT:
    def test_ensemble_step_jittable(self):
        def step(state, dt):
            return {"T": state["T"] + dt}

        ensemble_step = make_ensemble_step(step, in_axes=(0, None))
        jit_step = jax.jit(ensemble_step)
        states = {"T": jnp.ones((N_MEMBERS, 3))}
        result = jit_step(states, 1.0)
        np.testing.assert_allclose(result["T"], 2.0)

    def test_ensemble_differentiable(self):
        """Grad should work through vmapped ensemble step."""
        def step(state, dt):
            return {"T": state["T"] * dt}

        ensemble_step = make_ensemble_step(step, in_axes=(0, None))

        def loss(init_T):
            states = {"T": init_T}
            result = ensemble_step(states, 2.0)
            return jnp.sum(result["T"])

        init = jnp.ones((N_MEMBERS, 3))
        grad = jax.grad(loss)(init)
        # d/d(init_T) [sum(init_T * 2)] = 2.0 everywhere
        np.testing.assert_allclose(grad, 2.0)
