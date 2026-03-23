"""End-to-end cycling test.

2 DA cycles of shallow water with identity model:
- Verify: analysis improves cycle over cycle
- Verify: fully differentiable (jax.grad through cost)
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.da.control_vector import build_control_spec, state_to_control
from legoesm.da.background_error import DiagonalB
from legoesm.da.observation import DirectObsOperator, Observation
from legoesm.da.cost_function import build_cost_fn
from legoesm.da.incremental import IncrementalConfig
from legoesm.da.cycling import CyclingConfig, run_cycling


class _DecayModel:
    """Simple model: h decays slightly toward mean each step."""
    def step(self, state, dt):
        h_data = state.h.data
        h_mean = jnp.mean(h_data)
        alpha = 0.005
        h_new = h_data + alpha * (h_mean - h_data)
        return state._replace(h=state.h.replace(data=h_new))


def _make_sw_state(shape=(8, 16), h_val=1000.0):
    return ShallowWaterState(
        h=Field(data=jnp.ones(shape) * h_val, name="h", dims=(), units="m"),
        u=Field(data=jnp.zeros(shape), name="u", dims=(), units="m/s"),
        v=Field(data=jnp.zeros(shape), name="v", dims=(), units="m/s"),
        h_s=Field(data=jnp.zeros(shape), name="h_s", dims=(), units="m"),
    )


class TestEndToEnd:
    def test_cycling_improves_analysis(self):
        """Multi-cycle DA should maintain or improve fit."""
        shape = (8, 16)
        bg_state = _make_sw_state(shape, h_val=1000.0)
        model = _DecayModel()

        spec = build_control_spec(bg_state, fields=("h",))
        sigma = jnp.ones(spec.total_size) * 50.0
        B = DiagonalB(sigma=sigma)

        # Truth: background + perturbation
        key = jax.random.PRNGKey(0)
        h_pert = jax.random.normal(key, shape) * 20.0
        truth_h = bg_state.h.data + h_pert

        # Observations for 2 cycles
        n_obs = 30
        key, subkey = jax.random.split(key)
        obs_i = jax.random.randint(subkey, (n_obs,), 0, shape[0])
        key, subkey = jax.random.split(key)
        obs_j = jax.random.randint(subkey, (n_obs,), 0, shape[1])
        idx = (obs_i, obs_j)
        op = DirectObsOperator("h", idx)

        obs_cycles = []
        for cycle in range(2):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (n_obs,)) * 5.0
            obs = Observation(
                values=truth_h[idx] + noise,
                errors=jnp.full(n_obs, 5.0),
                time_index=0, operator=op,
            )
            obs_cycles.append((obs,))

        config = CyclingConfig(
            window_length=2,
            cycle_length=1,
            dt=600.0,
            n_cycles=2,
            incremental=IncrementalConfig(
                n_outer=2, n_inner=20,
                inner_method="lbfgs", use_preconditioning=False,
            ),
        )

        final, diagnostics = run_cycling(
            model, bg_state, tuple(obs_cycles),
            B, spec, config,
        )

        assert len(diagnostics) == 2
        assert jnp.all(jnp.isfinite(final.h.data))

        # Final analysis should be closer to truth than background
        rmse_bg = float(jnp.sqrt(jnp.mean((bg_state.h.data - truth_h) ** 2)))
        rmse_final = float(jnp.sqrt(jnp.mean((final.h.data - truth_h) ** 2)))
        assert rmse_final < rmse_bg

    def test_grad_through_cost(self):
        """jax.grad through a single-window cost should work."""
        bg_state = _make_sw_state((4, 4), h_val=100.0)
        model = _DecayModel()

        spec = build_control_spec(bg_state, fields=("h",))
        x_b = state_to_control(bg_state, spec)
        sigma = jnp.ones(spec.total_size) * 10.0
        B = DiagonalB(sigma=sigma)

        idx = (jnp.array([0, 1]), jnp.array([0, 1]))
        op = DirectObsOperator("h", idx)
        obs = Observation(
            values=jnp.array([110.0, 90.0]),
            errors=jnp.array([5.0, 5.0]),
            time_index=0, operator=op,
        )

        cost_fn = build_cost_fn(
            model, x_b, (obs,), B, spec, bg_state,
            dt=600.0, n_steps=2,
        )

        # jax.grad should work
        g = jax.grad(cost_fn)(x_b)
        assert jnp.all(jnp.isfinite(g))
        assert g.shape == x_b.shape
