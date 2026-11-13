"""Contract tests for the ORCA2 rollout-gradient cost probe, on a toy model.

The ORCA2 card itself needs the recorded decks and a GPU; these gate the
probe's own logic: the damping-target stamp check, that every schedule gives
the eager chain's state and gradient, that each step reads its own inputs and
step number, that a constant damping rate is held once, and that the
checkpointing schedules really shrink the compiled memory.
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

DT = 0.1


class Sponge(NamedTuple):
    gamma: object
    T_ref: jax.Array
    nemo_kt: object = None


class State(NamedTuple):
    T: Field


class ToyModel:
    """Nonlinear step reading every input and the step number; refuses a
    mismatched stamp the way ``LatLonCGridOceanModel.step`` does for a traced
    counter."""

    def step(self, s, dt, *, nemo_kt, freshwater, surface_forcing, sponge):
        if sponge.nemo_kt is not nemo_kt:
            raise ValueError("stale target")
        T = s.T.data
        T = T + dt * (jnp.sin(T) * freshwater + surface_forcing
                      + sponge.gamma * (sponge.T_ref - T) + 1e-2 * nemo_kt)
        return State(T=s.T.replace(data=T))


def _inputs(n, size=4, seed=0, constant_gamma=True):
    rng = np.random.default_rng(seed)
    g = jnp.asarray(rng.uniform(size=size))
    return [{"freshwater": jnp.asarray(rng.normal(size=size)),
             "surface_forcing": jnp.asarray(rng.normal(size=size)),
             "sponge": Sponge(g if constant_gamma else jnp.asarray(rng.uniform(size=size)),
                              jnp.asarray(rng.normal(size=size)), nemo_kt=kt)}
            for kt in range(2, n + 2)]


def _eager(model, state1, inputs, dT):
    s = State(T=state1.T.replace(data=state1.T.data + dT))
    for kt, kw in enumerate(inputs, start=2):
        s = model.step(s, DT, nemo_kt=kt, **kw)
    return s


def _state1(size=4):
    return State(T=Field(jnp.linspace(0.0, 1.0, size)))


def test_stale_damping_target_is_refused():
    inputs = _inputs(3)
    inputs[1] = dict(inputs[1], sponge=inputs[0]["sponge"])
    with pytest.raises(ValueError, match="kt=2, expected 3"):
        stack_inputs(inputs)


@pytest.mark.parametrize("constant_gamma", [True, False])
def test_constant_damping_rate_is_held_once(constant_gamma):
    stacked, gamma = stack_inputs(_inputs(5, constant_gamma=constant_gamma))
    if constant_gamma:
        assert gamma is not None and gamma.shape == (4,) and stacked["sponge"].gamma is None
    else:
        assert gamma is None and stacked["sponge"].gamma.shape == (5, 4)


@pytest.mark.parametrize("constant_gamma", [True, False])
@pytest.mark.parametrize("schedule", ["none", "uniform", "binomial"])
def test_every_schedule_matches_the_eager_chain(schedule, constant_gamma):
    model, n = ToyModel(), 7
    inputs = _inputs(n, constant_gamma=constant_gamma)
    state1 = _state1()
    rollout = make_rollout(model, DT, n, schedule, checkpoints=3)
    stacked, gamma = stack_inputs(inputs)

    def loss_loop(d):
        return jnp.sum(rollout(d, stacked, gamma, state1).T.data ** 2)

    def loss_eager(d):
        return jnp.sum(_eager(model, state1, inputs, d).T.data ** 2)

    dT = jnp.zeros(4)
    np.testing.assert_allclose(jax.jit(loss_loop)(dT), loss_eager(dT), rtol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.grad(loss_loop))(dT), jax.grad(loss_eager)(dT),
                               rtol=1e-10)


def test_each_step_reads_its_own_inputs():
    model, n = ToyModel(), 5
    inputs = _inputs(n, constant_gamma=False)
    state1 = _state1()
    rollout = make_rollout(model, DT, n, "uniform")
    a = rollout(jnp.zeros(4), *stack_inputs(inputs), state1).T.data
    swapped = list(inputs)
    swapped[1], swapped[3] = (dict(inputs[3], sponge=inputs[3]["sponge"]._replace(nemo_kt=3)),
                              dict(inputs[1], sponge=inputs[1]["sponge"]._replace(nemo_kt=5)))
    b = rollout(jnp.zeros(4), *stack_inputs(swapped), state1).T.data
    assert not np.allclose(a, b)
    np.testing.assert_allclose(b, _eager(model, state1, swapped, 0.0).T.data, rtol=1e-12)


def test_checkpointing_schedules_shrink_compiled_memory():
    model, n, size = ToyModel(), 32, 50_000
    inputs = _inputs(n, size=size)
    state1 = _state1(size)
    stacked, gamma = stack_inputs(inputs)
    temp = {}
    for schedule in ("none", "uniform", "binomial"):
        rollout = make_rollout(model, DT, n, schedule, checkpoints=4)
        vg = jax.jit(jax.grad(lambda d, *ops: jnp.sum(rollout(d, *ops).T.data ** 2)))
        mem = vg.lower(jnp.zeros(size), stacked, gamma, state1).compile().memory_analysis()
        if mem is None:
            pytest.skip("backend reports no memory analysis")
        temp[schedule] = mem.temp_size_in_bytes
    field = size * 8
    print({k: v / field for k, v in temp.items()})
    # measured on CPU, in units of one state: none ~65, uniform ~33, binomial ~8
    assert temp["none"] > 1.5 * temp["uniform"] > 0
    assert temp["uniform"] > 2 * temp["binomial"]
