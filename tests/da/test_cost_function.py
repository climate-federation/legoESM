"""Tests for the 4D-Var cost function."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.da.background_error import DiagonalB
from legoesm.da.control_vector import build_control_spec, state_to_control
from legoesm.da.cost_function import build_cost_and_grad_fn, build_cost_fn
from legoesm.da.observation import DirectObsOperator, Observation


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


class _DecayModel:
    """Non-trivial linear model so the trajectory genuinely evolves with time."""
    def step(self, state, dt):
        h = state.h
        return state._replace(h=h.replace(data=0.95 * h.data + 0.5))


class _FlipModel:
    """Sign-flipping model: with h0 > 0, h is negative after an ODD number of
    steps and positive after an EVEN number — so sqrt(h) is non-finite on the
    off-time intermediate states but finite at the (even-step) observation time."""
    def step(self, state, dt):
        h = state.h
        return state._replace(h=h.replace(data=-0.9 * h.data))


def _sqrt_h_operator(state):
    """Observation operator that is NaN (and NaN-gradient) wherever h < 0."""
    idx = (jnp.array([0, 1]), jnp.array([0, 1]))
    return jnp.sqrt(state.h.data[idx])


class TestCheckpointSchedules:
    """The 'none'/'uniform'/'binomial' adjoint-memory schedules must agree
    bit-for-bit on cost and gradient across a multi-time-observation window —
    only the memory footprint differs."""

    def _build_window(self, setup):
        jax.config.update("jax_enable_x64", True)
        state, spec, x_b, _B, _ = setup
        x_b = x_b.astype(jnp.float64)
        sigma = jnp.ones(spec.total_size, dtype=jnp.float64) * 10.0
        B64 = DiagonalB(sigma=sigma)
        state64 = jax.tree.map(
            lambda f: f.replace(data=f.data.astype(jnp.float64))
            if isinstance(f, Field) else f,
            state,
        )
        op = DirectObsOperator("h", (jnp.array([0, 1]), jnp.array([0, 1])))
        observations = tuple(
            Observation(
                values=jnp.array([105.0, 95.0], dtype=jnp.float64),
                errors=jnp.array([5.0, 5.0], dtype=jnp.float64),
                time_index=t, operator=op,
            )
            for t in (0, 2, 5)
        )
        return state64, spec, x_b, B64, _DecayModel(), observations

    @pytest.mark.parametrize("schedule", ["none", "uniform", "binomial"])
    def test_schedules_agree_on_cost_and_grad(self, setup, schedule):
        state64, spec, x_b, B64, model, observations = self._build_window(setup)
        n_steps = 6

        def cost_and_grad(sch):
            return build_cost_and_grad_fn(
                model, x_b, observations=observations, B=B64,
                control_spec=spec, template_state=state64,
                dt=1.0, n_steps=n_steps, checkpoint_schedule=sch,
            )

        J_ref, g_ref = cost_and_grad("uniform")(x_b)
        J, g = cost_and_grad(schedule)(x_b)
        assert jnp.allclose(J, J_ref, rtol=1e-10, atol=1e-10)
        assert jnp.allclose(g, g_ref, rtol=1e-8, atol=1e-10)

    def test_unknown_schedule_raises(self, setup):
        state, spec, x_b, B, model = setup
        with pytest.raises(ValueError, match="unknown checkpoint_schedule"):
            build_cost_fn(
                model, x_b, observations=(), B=B,
                control_spec=spec, template_state=state,
                dt=1.0, n_steps=1, checkpoint_schedule="bogus",
            )

    def test_schedules_agree_under_mixed_precision(self, setup):
        """float32 control/background + float64 observations: cost and gradient
        (and their dtypes) must be identical across all schedules (codex round-8)."""
        jax.config.update("jax_enable_x64", True)
        state, spec, x_b, _B, _ = setup
        x_b = x_b.astype(jnp.float32)
        sigma = jnp.ones(spec.total_size, dtype=jnp.float32) * 10.0
        B32 = DiagonalB(sigma=sigma)
        state32 = jax.tree.map(
            lambda f: f.replace(data=f.data.astype(jnp.float32))
            if isinstance(f, Field) else f,
            state,
        )
        model = _DecayModel()
        op = DirectObsOperator("h", (jnp.array([0, 1]), jnp.array([0, 1])))
        observations = tuple(
            Observation(
                values=jnp.array([105.0, 95.0], dtype=jnp.float64),
                errors=jnp.array([5.0, 5.0], dtype=jnp.float64),
                time_index=t, operator=op,
            )
            for t in (0, 2, 5)
        )
        n_steps = 6

        def cost_and_grad(sch):
            return build_cost_and_grad_fn(
                model, x_b, observations=observations, B=B32,
                control_spec=spec, template_state=state32,
                dt=1.0, n_steps=n_steps, checkpoint_schedule=sch,
            )

        J_u, g_u = cost_and_grad("uniform")(x_b)
        J_n, g_n = cost_and_grad("none")(x_b)
        J_b2, g_b2 = cost_and_grad("binomial")(x_b)
        # same dtype (control precision) across schedules
        assert J_u.dtype == J_n.dtype == J_b2.dtype == jnp.float32
        assert g_u.dtype == g_n.dtype == g_b2.dtype == jnp.float32
        # same value
        assert jnp.allclose(J_u, J_n) and jnp.allclose(J_u, J_b2)
        assert jnp.allclose(g_u, g_n) and jnp.allclose(g_u, g_b2)

    def test_binomial_matches_trajectory_for_negative_time_index(self, setup):
        """A final-state observation encoded as time_index=-1 must accumulate in
        the binomial in-loop path exactly as trajectory[-1] does (codex #1)."""
        state64, spec, x_b, B64, model, _ = self._build_window(setup)
        op = DirectObsOperator("h", (jnp.array([0, 1]), jnp.array([0, 1])))
        obs = (Observation(
            values=jnp.array([105.0, 95.0], dtype=jnp.float64),
            errors=jnp.array([5.0, 5.0], dtype=jnp.float64),
            time_index=-1, operator=op,
        ),)
        n_steps = 6

        def cost_and_grad(sch):
            return build_cost_and_grad_fn(
                model, x_b, observations=obs, B=B64,
                control_spec=spec, template_state=state64,
                dt=1.0, n_steps=n_steps, checkpoint_schedule=sch,
            )

        J_traj, g_traj = cost_and_grad("uniform")(x_b)
        J_bino, g_bino = cost_and_grad("binomial")(x_b)
        # the negative index must actually select something (non-zero misfit)
        assert float(J_bino) > 0.0
        assert jnp.allclose(J_bino, J_traj, rtol=1e-10, atol=1e-10)
        assert jnp.allclose(g_bino, g_traj, rtol=1e-8, atol=1e-10)

    def test_schedules_agree_with_unsorted_observation_times(self, setup):
        """Observation-order FP reduction is preserved in the binomial path:
        unsorted times with widely different magnitudes must still match the
        trajectory schedules (codex round-9, associativity)."""
        jax.config.update("jax_enable_x64", True)
        state, spec, x_b, _B, _ = setup
        x_b = x_b.astype(jnp.float32)
        sigma = jnp.ones(spec.total_size, dtype=jnp.float32) * 10.0
        B32 = DiagonalB(sigma=sigma)
        state32 = jax.tree.map(
            lambda f: f.replace(data=f.data.astype(jnp.float32))
            if isinstance(f, Field) else f,
            state,
        )
        model = _DecayModel()
        op = DirectObsOperator("h", (jnp.array([0, 1]), jnp.array([0, 1])))
        # deliberately UNSORTED times with very different misfit magnitudes
        triples = [(5, 1.0e3, 1.0), (0, 1.0, 0.1), (3, 1.0e-2, 5.0)]
        observations = tuple(
            Observation(
                values=jnp.array([v, v], dtype=jnp.float64),
                errors=jnp.array([e, e], dtype=jnp.float64),
                time_index=t, operator=op,
            )
            for (t, v, e) in triples
        )
        n_steps = 6

        def cost_and_grad(sch):
            return build_cost_and_grad_fn(
                model, x_b, observations=observations, B=B32,
                control_spec=spec, template_state=state32,
                dt=1.0, n_steps=n_steps, checkpoint_schedule=sch,
            )

        J_u, g_u = cost_and_grad("uniform")(x_b)
        J_b2, g_b2 = cost_and_grad("binomial")(x_b)
        assert J_u.dtype == J_b2.dtype == jnp.float32
        assert jnp.allclose(J_u, J_b2, rtol=1e-6, atol=1e-6)
        assert jnp.allclose(g_u, g_b2, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("schedule", ["none", "uniform", "binomial"])
    def test_unknown_storage_raises_for_every_schedule(self, setup, schedule):
        """storage typos must fail eagerly regardless of the schedule (codex #2)."""
        state, spec, x_b, B, model = setup
        with pytest.raises(ValueError, match="unknown storage"):
            build_cost_fn(
                model, x_b, observations=(), B=B,
                control_spec=spec, template_state=state,
                dt=1.0, n_steps=2, checkpoint_schedule=schedule, storage="bogus",
            )

    def test_host_storage_rejected_for_none_schedule(self, setup):
        """'none' + host is contradictory and must raise (codex round-4 consistency)."""
        state, spec, x_b, B, model = setup
        with pytest.raises(ValueError, match="requires checkpoint_schedule"):
            build_cost_fn(
                model, x_b, observations=(), B=B,
                control_spec=spec, template_state=state,
                dt=1.0, n_steps=2, checkpoint_schedule="none", storage="host",
            )

    def test_binomial_finite_with_offtime_nonfinite_operator(self, setup):
        """An operator that is NaN off its target step must NOT poison the
        binomial gradient — it is evaluated only at its own step via lax.cond
        (codex round-2 #1). Before the fix a jnp.where differentiated the dead
        branch and produced NaN gradients."""
        jax.config.update("jax_enable_x64", True)
        state, spec, x_b, _B, _ = setup
        x_b = x_b.astype(jnp.float64)
        sigma = jnp.ones(spec.total_size, dtype=jnp.float64) * 10.0
        B64 = DiagonalB(sigma=sigma)
        state64 = jax.tree.map(
            lambda f: f.replace(data=f.data.astype(jnp.float64))
            if isinstance(f, Field) else f,
            state,
        )
        model = _FlipModel()
        # h0=100: after 1 step -90 (sqrt NaN), after 2 steps +81 (sqrt OK).
        # time_index=1 selects trajectory[1] = state after 2 steps (positive).
        obs = (Observation(
            values=jnp.array([8.0, 8.0], dtype=jnp.float64),
            errors=jnp.array([1.0, 1.0], dtype=jnp.float64),
            time_index=1, operator=_sqrt_h_operator,
        ),)
        n_steps = 3

        def cost_and_grad(sch):
            return build_cost_and_grad_fn(
                model, x_b, observations=obs, B=B64,
                control_spec=spec, template_state=state64,
                dt=1.0, n_steps=n_steps, checkpoint_schedule=sch,
            )

        J_u, g_u = cost_and_grad("uniform")(x_b)
        J_b, g_b = cost_and_grad("binomial")(x_b)
        assert jnp.all(jnp.isfinite(g_b)), "binomial gradient poisoned by off-time operator"
        assert jnp.allclose(J_b, J_u, rtol=1e-10, atol=1e-10)
        assert jnp.allclose(g_b, g_u, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize("schedule", ["none", "uniform", "binomial"])
    @pytest.mark.parametrize("bad_ti", [6, 7, -7])  # n_steps=6 => valid [-6, 6)
    def test_out_of_window_time_index_raises(self, setup, schedule, bad_ti):
        """Out-of-window observation times fail eagerly for every schedule
        (codex round-2 #2): trajectory would clamp, binomial would silently
        drop — both unacceptable."""
        state, spec, x_b, B, model = setup
        op = DirectObsOperator("h", (jnp.array([0]), jnp.array([0])))
        obs = (Observation(
            values=jnp.array([1.0]), errors=jnp.array([1.0]),
            time_index=bad_ti, operator=op,
        ),)
        with pytest.raises(ValueError, match="out of range"):
            build_cost_fn(
                model, x_b, observations=obs, B=B,
                control_spec=spec, template_state=state,
                dt=1.0, n_steps=6, checkpoint_schedule=schedule,
            )

    @pytest.mark.parametrize("schedule", ["none", "uniform", "binomial"])
    def test_numpy_int_out_of_window_time_index_raises(self, setup, schedule):
        """A numpy integer (np.int64) out-of-window index must also be rejected,
        not just a Python int (codex round-7) — otherwise binomial silently drops it."""
        state, spec, x_b, B, model = setup
        op = DirectObsOperator("h", (jnp.array([0]), jnp.array([0])))
        obs = (Observation(
            values=jnp.array([1.0]), errors=jnp.array([1.0]),
            time_index=np.int64(6), operator=op,
        ),)
        with pytest.raises(ValueError, match="out of range"):
            build_cost_fn(
                model, x_b, observations=obs, B=B,
                control_spec=spec, template_state=state,
                dt=1.0, n_steps=6, checkpoint_schedule=schedule,
            )
