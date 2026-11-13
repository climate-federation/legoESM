"""Contract tests for the ORCA2 rollout-gradient cost probe, on a toy model.

The ORCA2 card itself needs the recorded decks and a GPU; these gate the
probe's own logic: the damping-target stamp check, that every schedule gives
the eager chain's state and gradient, and that each step reads its own inputs.
"""
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))

from orca2_rollout_grad_cost import make_rollout, stack_inputs  # noqa: E402

from legoesm.core.field import Field  # noqa: E402


class Sponge(NamedTuple):
    T_ref: jax.Array
    nemo_kt: object = None


class State(NamedTuple):
    T: Field


class ToyModel:
    """Nonlinear step reading all three inputs; refuses a mismatched stamp the
    way ``LatLonCGridOceanModel.step`` does for a traced counter."""

    def step(self, s, dt, *, nemo_kt, freshwater, surface_forcing, sponge):
        if sponge.nemo_kt is not nemo_kt:
            raise ValueError("stale target")
        T = s.T.data
        T = T + dt * (jnp.sin(T) * freshwater + surface_forcing + 0.1 * (sponge.T_ref - T))
        return State(T=s.T.replace(data=T))


def _inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    return [{"freshwater": jnp.asarray(rng.normal(size=4)),
             "surface_forcing": jnp.asarray(rng.normal(size=4)),
             "sponge": Sponge(jnp.asarray(rng.normal(size=4)), nemo_kt=kt)}
            for kt in range(2, n + 2)]


def _eager(model, state1, inputs, dT):
    s = State(T=state1.T.replace(data=state1.T.data + dT))
    for kt, kw in enumerate(inputs, start=2):
        s = model.step(s, 0.1, nemo_kt=kt, **kw)
    return s


def test_stale_damping_target_is_refused():
    inputs = _inputs(3)
    inputs[1] = dict(inputs[1], sponge=inputs[0]["sponge"])
    with pytest.raises(ValueError, match="kt=2, expected 3"):
        stack_inputs(inputs)


@pytest.mark.parametrize("schedule", ["none", "uniform", "binomial"])
def test_every_schedule_matches_the_eager_chain(schedule):
    model, n = ToyModel(), 7
    inputs = _inputs(n)
    state1 = State(T=Field(jnp.linspace(0.0, 1.0, 4)))
    rollout = make_rollout(model, state1, 0.1, n, schedule, checkpoints=3)
    stacked = stack_inputs(inputs)
    dT = jnp.zeros(4)

    def loss_loop(d):
        return jnp.sum(rollout(d, stacked).T.data ** 2)

    def loss_eager(d):
        return jnp.sum(_eager(model, state1, inputs, d).T.data ** 2)

    np.testing.assert_allclose(jax.jit(loss_loop)(dT), loss_eager(dT), rtol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.grad(loss_loop))(dT), jax.grad(loss_eager)(dT),
                               rtol=1e-10)


def test_each_step_reads_its_own_inputs():
    model, n = ToyModel(), 5
    inputs = _inputs(n)
    state1 = State(T=Field(jnp.linspace(0.0, 1.0, 4)))
    rollout = make_rollout(model, state1, 0.1, n, "uniform")
    a = rollout(jnp.zeros(4), stack_inputs(inputs)).T.data
    swapped = list(inputs)
    swapped[1], swapped[3] = (dict(inputs[3], sponge=inputs[3]["sponge"]._replace(nemo_kt=3)),
                              dict(inputs[1], sponge=inputs[1]["sponge"]._replace(nemo_kt=5)))
    b = rollout(jnp.zeros(4), stack_inputs(swapped)).T.data
    assert not np.allclose(a, b)
    np.testing.assert_allclose(b, _eager(model, state1, swapped, 0.0).T.data, rtol=1e-12)
