"""Tests for incremental 4D-Var."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.da.control_vector import build_control_spec, state_to_control
from legoesm.da.background_error import DiagonalB
from legoesm.da.observation import DirectObsOperator, Observation
from legoesm.da.incremental import incremental_4dvar, IncrementalConfig


class _IdentityModel:
    """Model with identity step (state unchanged)."""
    def step(self, state, dt):
        return state


def _make_sw_state(shape=(4, 4), h_val=100.0):
    return ShallowWaterState(
        h=Field(data=jnp.ones(shape) * h_val, name="h", dims=(), units="m"),
        u=Field(data=jnp.zeros(shape), name="u", dims=(), units="m/s"),
        v=Field(data=jnp.zeros(shape), name="v", dims=(), units="m/s"),
        h_s=Field(data=jnp.zeros(shape), name="h_s", dims=(), units="m"),
    )


class TestIncrementalSingleObs:
    def test_single_obs_shifts_toward_obs(self):
        """Analysis should shift toward a single observation."""
        bg_state = _make_sw_state(h_val=100.0)
        spec = build_control_spec(bg_state)
        sigma = jnp.ones(spec.total_size) * 10.0
        B = DiagonalB(sigma=sigma)
        model = _IdentityModel()

        # One observation: h at (0,0) = 120
        idx = (jnp.array([0]), jnp.array([0]))
        op = DirectObsOperator("h", idx)
        obs = Observation(
            values=jnp.array([120.0]),
            errors=jnp.array([5.0]),
            time_index=0, operator=op,
        )

        config = IncrementalConfig(
            n_outer=2, n_inner=30, inner_gtol=1e-6,
            inner_method="lbfgs", use_preconditioning=False,
        )

        analysis, diag = incremental_4dvar(
            model, bg_state, (obs,), B, spec,
            dt=1.0, n_steps=1, config=config,
        )

        # Analysis h at (0,0) should be between background (100) and obs (120)
        h_ana = float(analysis.h.data[0, 0])
        assert 100.0 < h_ana < 120.0, f"h_ana = {h_ana}"

    def test_cost_decreases(self):
        """Cost should decrease across outer iterations."""
        bg_state = _make_sw_state(h_val=100.0)
        spec = build_control_spec(bg_state)
        sigma = jnp.ones(spec.total_size) * 10.0
        B = DiagonalB(sigma=sigma)
        model = _IdentityModel()

        idx = (jnp.array([0, 1]), jnp.array([0, 1]))
        op = DirectObsOperator("h", idx)
        obs = Observation(
            values=jnp.array([110.0, 90.0]),
            errors=jnp.array([5.0, 5.0]),
            time_index=0, operator=op,
        )

        config = IncrementalConfig(
            n_outer=3, n_inner=20, inner_gtol=1e-6,
            inner_method="lbfgs", use_preconditioning=False,
        )

        _, diag = incremental_4dvar(
            model, bg_state, (obs,), B, spec,
            dt=1.0, n_steps=1, config=config,
        )

        # Cost should decrease
        for i in range(1, len(diag.cost_history)):
            assert diag.cost_history[i] <= diag.cost_history[i - 1] + 1e-6


class TestIncrementalTwinExperiment:
    def test_twin_experiment(self):
        """Twin experiment: analysis should be closer to truth than background."""
        shape = (4, 4)
        truth_state = _make_sw_state(shape, h_val=105.0)
        bg_state = _make_sw_state(shape, h_val=100.0)

        spec = build_control_spec(bg_state)
        sigma = jnp.ones(spec.total_size) * 10.0
        B = DiagonalB(sigma=sigma)
        model = _IdentityModel()

        # Observe all h grid points
        ii, jj = jnp.meshgrid(jnp.arange(4), jnp.arange(4), indexing="ij")
        idx = (ii.ravel(), jj.ravel())
        op = DirectObsOperator("h", idx)
        obs = Observation(
            values=truth_state.h.data[idx],
            errors=jnp.ones(16) * 2.0,
            time_index=0, operator=op,
        )

        config = IncrementalConfig(
            n_outer=2, n_inner=30, inner_gtol=1e-6,
            inner_method="lbfgs", use_preconditioning=False,
        )

        analysis, _ = incremental_4dvar(
            model, bg_state, (obs,), B, spec,
            dt=1.0, n_steps=1, config=config,
        )

        rmse_bg = float(jnp.sqrt(jnp.mean((bg_state.h.data - truth_state.h.data) ** 2)))
        rmse_ana = float(jnp.sqrt(jnp.mean((analysis.h.data - truth_state.h.data) ** 2)))
        assert rmse_ana < rmse_bg, f"RMSE bg={rmse_bg}, ana={rmse_ana}"


class TestIncrementalOuterLoop:
    def _setup(self):
        bg_state = _make_sw_state(h_val=100.0)
        spec = build_control_spec(bg_state)
        sigma = jnp.ones(spec.total_size) * 10.0
        B = DiagonalB(sigma=sigma)
        idx = (jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
        obs = Observation(
            values=jnp.array([120.0, 90.0, 104.0]),
            errors=jnp.array([1.0, 5.0, 20.0]),
            time_index=0, operator=DirectObsOperator("h", idx),
        )
        return bg_state, spec, B, obs, idx

    def test_preconditioned_outer_iterations_warm_start(self):
        """With one CG step per outer iteration, each outer iteration must
        continue from the current analysis (restarting from the background
        repeats the same solve, so the cost would stall after outer 1)."""
        bg_state, spec, B, obs, _ = self._setup()
        config = IncrementalConfig(
            n_outer=3, n_inner=1, inner_gtol=1e-10,
            inner_method="cg", use_preconditioning=True,
        )
        _, diag = incremental_4dvar(
            _IdentityModel(), bg_state, (obs,), B, spec,
            dt=1.0, n_steps=1, config=config,
        )
        c = diag.cost_history
        assert c[2] < c[1] - 1e-6, c

    def test_innovation_rms_is_obs_minus_analysis(self):
        bg_state, spec, B, obs, idx = self._setup()
        config = IncrementalConfig(
            n_outer=2, n_inner=30, inner_gtol=1e-8,
            inner_method="lbfgs", use_preconditioning=False,
        )
        analysis, diag = incremental_4dvar(
            _IdentityModel(), bg_state, (obs,), B, spec,
            dt=1.0, n_steps=1, config=config,
        )
        d = obs.values - analysis.h.data[idx]
        expected = float(jnp.sqrt(jnp.mean(d * d)))
        assert diag.innovation_rms[-1] == pytest.approx(expected, rel=1e-10)
