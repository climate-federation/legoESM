"""Tests for DA cycling."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.da.control_vector import build_control_spec
from legoesm.da.background_error import DiagonalB
from legoesm.da.observation import DirectObsOperator, Observation
from legoesm.da.incremental import IncrementalConfig
from legoesm.da.cycling import CyclingConfig, run_cycling


class _IdentityModel:
    def step(self, state, dt):
        return state


def _make_sw_state(shape=(4, 4), h_val=100.0):
    return ShallowWaterState(
        h=Field(data=jnp.ones(shape) * h_val, name="h", dims=(), units="m"),
        u=Field(data=jnp.zeros(shape), name="u", dims=(), units="m/s"),
        v=Field(data=jnp.zeros(shape), name="v", dims=(), units="m/s"),
        h_s=Field(data=jnp.zeros(shape), name="h_s", dims=(), units="m"),
    )


class TestCycling:
    def test_two_cycles(self):
        """Two cycles should produce a valid result."""
        shape = (4, 4)
        bg_state = _make_sw_state(shape, h_val=100.0)
        spec = build_control_spec(bg_state)
        sigma = jnp.ones(spec.total_size) * 10.0
        B = DiagonalB(sigma=sigma)
        model = _IdentityModel()

        # Create observations for 2 cycles
        idx = (jnp.array([0, 1]), jnp.array([0, 1]))
        op = DirectObsOperator("h", idx)
        obs1 = Observation(
            values=jnp.array([110.0, 90.0]),
            errors=jnp.array([5.0, 5.0]),
            time_index=0, operator=op,
        )
        obs2 = Observation(
            values=jnp.array([112.0, 88.0]),
            errors=jnp.array([5.0, 5.0]),
            time_index=0, operator=op,
        )

        config = CyclingConfig(
            window_length=1,
            cycle_length=1,
            dt=1.0,
            n_cycles=2,
            incremental=IncrementalConfig(
                n_outer=1, n_inner=20,
                inner_method="lbfgs", use_preconditioning=False,
            ),
        )

        final, diagnostics = run_cycling(
            model, bg_state, ((obs1,), (obs2,)),
            B, spec, config,
        )

        assert len(diagnostics) == 2
        assert jnp.all(jnp.isfinite(final.h.data))
