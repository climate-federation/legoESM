"""Tests for ensemble parallelization utilities.

Verifies:
- State stacking / unstacking round-trip
- Initial-condition perturbation (shape, diversity, static fields)
- Vmapped step produces correct shapes and results
- Ensemble scan integration (correctness, differentiability)
- Ensemble statistics (mean, std, spread)
- Multi-device sharding (if >1 device available)
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.parallel.ensemble import (
    stack_states,
    unstack_states,
    perturb_initial_conditions,
    perturb_parameters,
    make_ensemble_step,
    make_ensemble_step_jit,
    ensemble_integrate,
    ensemble_integrate_with_forcing,
    ensemble_mean,
    ensemble_std,
    ensemble_percentile,
    ensemble_spread,
    create_ensemble_mesh,
    shard_ensemble,
    gather_ensemble,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_field(shape, name="x", dtype=jnp.float64):
    """Create a simple Field for testing."""
    return Field(
        data=jnp.ones(shape, dtype=dtype),
        name=name,
        dims=("face", "x", "y") if len(shape) == 3 else ("face", "x", "y", "lev"),
        units="1",
    )


def _make_state(n=4, nlev=3, dtype=jnp.float64):
    """Create a minimal HydrostaticState for testing."""
    shape_3d = (6, n, n, nlev)
    shape_2d = (6, n, n)
    return HydrostaticState(
        u=_make_field(shape_3d, "u", dtype),
        v=_make_field(shape_3d, "v", dtype),
        T=Field(
            data=jnp.full(shape_3d, 250.0, dtype=dtype),
            name="T", dims=("face", "x", "y", "lev"), units="K",
        ),
        p_s=Field(
            data=jnp.full(shape_2d, 1e5, dtype=dtype),
            name="p_s", dims=("face", "x", "y"), units="Pa",
        ),
        phis=Field(
            data=jnp.zeros(shape_2d, dtype=dtype),
            name="phis", dims=("face", "x", "y"), units="m^2/s^2",
        ),
    )


# A trivial "model step" for testing: just scale by a factor.
def _trivial_step(state, dt):
    """state_new = state * (1 + 0.001*dt) for all fields."""
    factor = 1.0 + 0.001 * dt
    return jax.tree.map(lambda x: x * factor, state)


# A step that depends on per-member forcing.
def _forced_step(state, forcing, dt):
    """state_new = state + forcing * dt."""
    return jax.tree.map(lambda s, f: s + f * dt, state, forcing)


# ========================================================================
# State batching
# ========================================================================

class TestBatching:
    """Tests for stack_states / unstack_states."""

    def test_stack_unstack_roundtrip(self):
        """Stacking then unstacking recovers original states."""
        s1 = _make_state()
        s2 = jax.tree.map(lambda x: x * 2, s1)
        s3 = jax.tree.map(lambda x: x * 3, s1)

        batched = stack_states([s1, s2, s3])

        # Batched shape: leading dim = 3
        assert batched.u.data.shape[0] == 3
        assert batched.T.data.shape == (3, 6, 4, 4, 3)

        recovered = unstack_states(batched)
        assert len(recovered) == 3

        np.testing.assert_allclose(
            np.asarray(recovered[0].T.data),
            np.asarray(s1.T.data),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            np.asarray(recovered[2].u.data),
            np.asarray(s3.u.data),
            atol=1e-15,
        )

    def test_stack_preserves_metadata(self):
        """Field metadata survives stack/unstack."""
        s = _make_state()
        batched = stack_states([s, s])
        assert batched.T.name == "T"
        assert batched.T.units == "K"

    def test_unstack_count(self):
        """Correct number of states after unstack."""
        s = _make_state()
        batched = stack_states([s] * 5)
        assert len(unstack_states(batched)) == 5


# ========================================================================
# Perturbation
# ========================================================================

class TestPerturbation:
    """Tests for perturb_initial_conditions."""

    def test_output_shape(self):
        """Batched state has correct leading dimension."""
        s = _make_state()
        batched = perturb_initial_conditions(
            s, jax.random.PRNGKey(0), n_members=8,
        )
        assert batched.u.data.shape == (8, 6, 4, 4, 3)
        assert batched.p_s.data.shape == (8, 6, 4, 4)

    def test_members_differ(self):
        """Ensemble members have different values."""
        s = _make_state()
        batched = perturb_initial_conditions(
            s, jax.random.PRNGKey(42), n_members=4, scale=0.1,
        )
        T_data = np.asarray(batched.T.data)
        # Members 0 and 1 should differ
        assert not np.allclose(T_data[0], T_data[1])

    def test_phis_not_perturbed(self):
        """Surface geopotential is not perturbed by default."""
        s = _make_state()
        batched = perturb_initial_conditions(
            s, jax.random.PRNGKey(0), n_members=4,
        )
        # phis should be identical across members
        phis_data = np.asarray(batched.phis.data)
        np.testing.assert_allclose(phis_data[0], phis_data[1], atol=1e-15)
        np.testing.assert_allclose(phis_data[0], phis_data[3], atol=1e-15)

    def test_selective_perturbation(self):
        """Only specified fields are perturbed."""
        s = _make_state()
        batched = perturb_initial_conditions(
            s, jax.random.PRNGKey(0), n_members=4,
            fields=["T"], scale=0.1,
        )
        # T should differ across members
        T = np.asarray(batched.T.data)
        assert not np.allclose(T[0], T[1])

        # u should NOT differ (not in fields list)
        u = np.asarray(batched.u.data)
        np.testing.assert_allclose(u[0], u[1], atol=1e-15)

    def test_additive_perturbation(self):
        """Additive mode: x + scale * noise."""
        s = _make_state()
        batched = perturb_initial_conditions(
            s, jax.random.PRNGKey(1), n_members=4,
            scale=0.5, multiplicative=False,
        )
        T = np.asarray(batched.T.data)
        assert not np.allclose(T[0], T[1])

    def test_scale_zero_no_perturbation(self):
        """Scale=0 produces identical copies."""
        s = _make_state()
        batched = perturb_initial_conditions(
            s, jax.random.PRNGKey(0), n_members=4, scale=0.0,
        )
        T = np.asarray(batched.T.data)
        np.testing.assert_allclose(T[0], T[1], atol=1e-15)

    def test_different_keys_different_results(self):
        """Different PRNG keys produce different ensembles."""
        s = _make_state()
        b1 = perturb_initial_conditions(
            s, jax.random.PRNGKey(0), n_members=4, scale=0.1,
        )
        b2 = perturb_initial_conditions(
            s, jax.random.PRNGKey(99), n_members=4, scale=0.1,
        )
        assert not np.allclose(
            np.asarray(b1.T.data), np.asarray(b2.T.data),
        )

    def test_perturb_parameters(self):
        """perturb_parameters works the same as perturb_initial_conditions."""
        s = _make_state()
        batched = perturb_parameters(
            s, jax.random.PRNGKey(0), n_members=4,
            scale=0.05, fields=["T"],
        )
        assert batched.T.data.shape[0] == 4


# ========================================================================
# Ensemble step
# ========================================================================

class TestEnsembleStep:
    """Tests for vmapped step functions."""

    def test_vmap_step_shapes(self):
        """Vmapped step preserves batched shapes."""
        s = _make_state()
        batched = stack_states([s, s, s])

        ensemble_step = make_ensemble_step(_trivial_step, in_axes=(0, None))
        result = ensemble_step(batched, 1.0)

        assert result.u.data.shape == (3, 6, 4, 4, 3)
        assert result.p_s.data.shape == (3, 6, 4, 4)

    def test_vmap_step_values(self):
        """Vmapped step gives same result as manual per-member stepping."""
        s = _make_state()
        batched = stack_states([s, s])
        dt = 10.0

        ensemble_step = make_ensemble_step(_trivial_step, in_axes=(0, None))
        result = ensemble_step(batched, dt)

        expected = _trivial_step(s, dt)
        np.testing.assert_allclose(
            np.asarray(result.T.data[0]),
            np.asarray(expected.T.data),
            atol=1e-12,
        )

    def test_jit_step(self):
        """JIT-compiled vmapped step works correctly."""
        s = _make_state()
        batched = stack_states([s] * 4)

        ensemble_step = make_ensemble_step_jit(
            _trivial_step, in_axes=(0, None),
        )
        result = ensemble_step(batched, 5.0)
        assert result.u.data.shape == (4, 6, 4, 4, 3)

    def test_perturbed_members_diverge(self):
        """Perturbed members should produce different results after stepping."""
        s = _make_state()
        batched = perturb_initial_conditions(
            s, jax.random.PRNGKey(0), n_members=4, scale=0.01,
        )

        ensemble_step = make_ensemble_step(_trivial_step, in_axes=(0, None))
        result = ensemble_step(batched, 100.0)

        T = np.asarray(result.T.data)
        assert not np.allclose(T[0], T[1])


# ========================================================================
# Time integration
# ========================================================================

class TestEnsembleIntegrate:
    """Tests for ensemble_integrate (scan + vmap)."""

    def test_basic_integration(self):
        """Integration for n_steps produces correct result."""
        s = _make_state()
        batched = stack_states([s, s])

        ensemble_step = make_ensemble_step(_trivial_step, in_axes=(0, None))
        dt = 1.0
        n_steps = 10

        final = ensemble_integrate(ensemble_step, batched, n_steps, dt)

        # After n_steps, each value should be multiplied by (1.001)^10
        expected_factor = (1.0 + 0.001 * dt) ** n_steps
        np.testing.assert_allclose(
            np.asarray(final.u.data[0]),
            np.asarray(s.u.data) * expected_factor,
            rtol=1e-10,
        )

    def test_save_trajectory(self):
        """save_trajectory=True returns trajectory with time dimension."""
        s = _make_state()
        batched = stack_states([s] * 3)

        ensemble_step = make_ensemble_step(_trivial_step, in_axes=(0, None))
        n_steps = 5

        final, trajectory = ensemble_integrate(
            ensemble_step, batched, n_steps, dt=1.0,
            save_trajectory=True,
        )

        # Trajectory: (n_steps, n_members, ...)
        assert trajectory.T.data.shape == (5, 3, 6, 4, 4, 3)
        assert final.T.data.shape == (3, 6, 4, 4, 3)

    def test_checkpoint_integration(self):
        """Integration with checkpointing gives same result."""
        s = _make_state()
        batched = stack_states([s, s])
        ensemble_step = make_ensemble_step(_trivial_step, in_axes=(0, None))

        result_no_ckpt = ensemble_integrate(
            ensemble_step, batched, 10, dt=1.0, checkpoint=False,
        )
        result_ckpt = ensemble_integrate(
            ensemble_step, batched, 10, dt=1.0, checkpoint=True,
        )

        np.testing.assert_allclose(
            np.asarray(result_no_ckpt.T.data),
            np.asarray(result_ckpt.T.data),
            atol=1e-12,
        )

    def test_differentiable(self):
        """Gradient of ensemble integrate w.r.t. initial conditions."""
        s = _make_state(n=2, nlev=2)

        def loss_fn(T_init_data):
            state = s._replace(T=s.T.replace(data=T_init_data))
            batched = stack_states([state, state])
            ensemble_step = make_ensemble_step(
                _trivial_step, in_axes=(0, None),
            )
            final = ensemble_integrate(
                ensemble_step, batched, 5, dt=1.0,
            )
            return jnp.sum(final.T.data)

        grad = jax.grad(loss_fn)(s.T.data)
        assert grad.shape == s.T.data.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_with_forcing(self):
        """ensemble_integrate_with_forcing applies per-step forcing."""
        s = _make_state(n=2, nlev=2)
        n_members = 2
        batched = stack_states([s] * n_members)

        ensemble_step = make_ensemble_step(
            _forced_step, in_axes=(0, 0, None),
        )

        n_steps = 3
        # Forcing: constant 1.0 at each step, batched over members
        forcing = jax.tree.map(
            lambda x: jnp.broadcast_to(
                x[None, None],
                (n_steps, n_members) + x.shape,
            ),
            s,
        )

        final = ensemble_integrate_with_forcing(
            ensemble_step, batched, forcing, dt=1.0,
        )
        # After 3 steps: state + 3*forcing = 1+3=4 for u/v, etc.
        np.testing.assert_allclose(
            np.asarray(final.u.data[0, 0, 0, 0, 0]),
            4.0,
            atol=1e-12,
        )


# ========================================================================
# Statistics
# ========================================================================

class TestStatistics:
    """Tests for ensemble statistics."""

    def test_mean(self):
        """Ensemble mean computes correctly."""
        s = _make_state()
        s2 = jax.tree.map(lambda x: x * 3, s)
        batched = stack_states([s, s2])

        mean = ensemble_mean(batched)
        # mean(1, 3) = 2
        np.testing.assert_allclose(
            float(mean.u.data[0, 0, 0, 0]), 2.0, atol=1e-15,
        )

    def test_std(self):
        """Ensemble std computes correctly."""
        s1 = _make_state()
        s2 = jax.tree.map(lambda x: x * 3, s1)
        batched = stack_states([s1, s2])

        std = ensemble_std(batched)
        # std(1, 3) = 1.0
        np.testing.assert_allclose(
            float(std.u.data[0, 0, 0, 0]), 1.0, atol=1e-12,
        )

    def test_mean_removes_ensemble_dim(self):
        """Mean state has no ensemble dimension."""
        s = _make_state()
        batched = stack_states([s] * 5)
        mean = ensemble_mean(batched)
        assert mean.T.data.shape == s.T.data.shape

    def test_percentile(self):
        """Percentile computes correctly."""
        s = _make_state()
        states = [jax.tree.map(lambda x: x * i, s) for i in range(1, 11)]
        batched = stack_states(states)

        p50 = ensemble_percentile(batched, 50.0)
        # Median of 1..10 is 5.5
        np.testing.assert_allclose(
            float(p50.u.data[0, 0, 0, 0]), 5.5, atol=1e-12,
        )

    def test_spread(self):
        """ensemble_spread returns dict with correct keys."""
        s = _make_state()
        batched = perturb_initial_conditions(
            s, jax.random.PRNGKey(0), n_members=8, scale=0.1,
        )
        sp = ensemble_spread(batched)
        assert isinstance(sp, dict)
        assert "T" in sp
        assert "u" in sp
        assert sp["T"] > 0  # Non-zero spread
        assert sp["phis"] == 0.0  # phis not perturbed


# ========================================================================
# Multi-device sharding
# ========================================================================

class TestSharding:
    """Tests for multi-device ensemble sharding."""

    def test_single_device_sharding(self):
        """Sharding on a single device doesn't change values."""
        s = _make_state()
        n_members = 4
        batched = stack_states([s] * n_members)

        n_devices = len(jax.devices())
        # Adjust n_members to be divisible
        if n_members % n_devices != 0:
            n_members = n_devices
            batched = stack_states([s] * n_members)

        mesh = create_ensemble_mesh(n_members)
        sharded = shard_ensemble(batched, mesh)

        # Values should be preserved
        np.testing.assert_allclose(
            np.asarray(sharded.T.data),
            np.asarray(batched.T.data),
            atol=1e-15,
        )

    def test_gather_preserves_values(self):
        """Gather after shard recovers original values."""
        s = _make_state()
        n_devices = len(jax.devices())
        n_members = n_devices  # Ensures divisibility
        batched = stack_states([s] * n_members)

        mesh = create_ensemble_mesh(n_members)
        sharded = shard_ensemble(batched, mesh)
        gathered = gather_ensemble(sharded)

        np.testing.assert_allclose(
            np.asarray(gathered.T.data),
            np.asarray(batched.T.data),
            atol=1e-15,
        )

    def test_sharded_step(self):
        """Vmapped step works on sharded state."""
        s = _make_state()
        n_devices = len(jax.devices())
        n_members = n_devices
        batched = stack_states([s] * n_members)

        mesh = create_ensemble_mesh(n_members)
        sharded = shard_ensemble(batched, mesh)

        ensemble_step = make_ensemble_step(_trivial_step, in_axes=(0, None))
        result = ensemble_step(sharded, 1.0)

        expected = _trivial_step(s, 1.0)
        result_gathered = gather_ensemble(result)
        np.testing.assert_allclose(
            np.asarray(result_gathered.T.data[0]),
            np.asarray(expected.T.data),
            atol=1e-12,
        )

    def test_mesh_requires_divisible(self):
        """create_ensemble_mesh raises if n_members not divisible."""
        n_devices = len(jax.devices())
        if n_devices > 1:
            with pytest.raises(ValueError, match="divisible"):
                create_ensemble_mesh(n_devices + 1)


# ========================================================================
# Generic pytree (non-NamedTuple)
# ========================================================================

class TestGenericPytree:
    """Tests with dict-based state (not NamedTuple)."""

    def test_stack_unstack_dict(self):
        """stack/unstack works with dict state."""
        s1 = {"x": jnp.ones((3, 4)), "y": jnp.zeros((3,))}
        s2 = {"x": jnp.ones((3, 4)) * 2, "y": jnp.ones((3,))}

        batched = stack_states([s1, s2])
        assert batched["x"].shape == (2, 3, 4)

        recovered = unstack_states(batched)
        np.testing.assert_allclose(recovered[1]["x"], s2["x"], atol=1e-15)

    def test_perturb_generic_pytree(self):
        """Perturbation works on non-NamedTuple pytrees."""
        s = {"a": jnp.ones((3, 4)), "b": jnp.zeros((2,))}
        batched = perturb_initial_conditions(
            s, jax.random.PRNGKey(0), n_members=4, scale=0.1,
        )
        assert batched["a"].shape == (4, 3, 4)
        # Members should differ
        assert not np.allclose(
            np.asarray(batched["a"][0]),
            np.asarray(batched["a"][1]),
        )

    def test_integrate_dict_state(self):
        """Ensemble integrate works with dict state."""
        s = {"x": jnp.ones((3,), dtype=jnp.float64)}
        batched = stack_states([s] * 2)

        def step_fn(state, dt):
            return {"x": state["x"] * (1.0 + 0.1 * dt)}

        ensemble_step = make_ensemble_step(step_fn, in_axes=(0, None))
        final = ensemble_integrate(ensemble_step, batched, 5, dt=1.0)

        expected = (1.1) ** 5
        np.testing.assert_allclose(
            np.asarray(final["x"][0]), expected, rtol=1e-10,
        )
