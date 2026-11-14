"""Unit tests for the K0 rollout (no ORCA2 decks needed).

The real-card loop-vs-step-by-step measurement is
scripts/validate/ocean_fidelity/orca2_l4/orca2_k0_rollout_equivalence.py.
"""
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import legoesm.ocean.bulk_flux_omip as bulk
from legoesm.ocean.sponge import SpongeForcing
from legoesm.training.orca2_k0_rollout import (
    S2_CARRY,
    K0Carry,
    _controls_at,
    carry_from_result,
    make_k0_rollout,
    rollout_operands,
    split_controls,
)


class Clock(NamedTuple):
    field: object
    isecsbc: object
    stride: object


def _controls(n, kt0=3):
    mask = jnp.ones((2, 2))
    gamma = jnp.full((2, 2, 3), 1e-8)
    out = []
    for i in range(n):
        out.append({
            "s0f": Clock(field=jnp.full((2, 2), float(i)), isecsbc=(kt0 + i) * 10800, stride=2),
            "s2_base": {"mask": mask, "flag": False, "opt": None},
            "sponge": SpongeForcing(gamma=jnp.array(gamma), T_ref=jnp.full((2, 2, 3), float(i)),
                                    S_ref=jnp.zeros((2, 2, 3)), nemo_kt=kt0 + i),
        })
    return out


def test_leaves_are_classified_static_shared_or_stacked():
    split = split_controls(_controls(4), kt0=3)
    assert (split.kt0, split.n_steps) == (3, 4)
    ctrl = _controls_at(split.treedef, split.kinds, split.values, 2)
    assert ctrl["s0f"].stride == 2 and isinstance(ctrl["s0f"].stride, int)   # static stays Python
    assert ctrl["s2_base"]["flag"] is False and ctrl["s2_base"]["opt"] is None
    assert int(ctrl["s0f"].isecsbc) == 5 * 10800                             # varying int -> stacked
    np.testing.assert_array_equal(ctrl["s0f"].field, np.full((2, 2), 2.0))
    np.testing.assert_array_equal(ctrl["sponge"].T_ref, np.full((2, 2, 3), 2.0))
    assert ctrl["sponge"].nemo_kt is None                                    # re-stamped in the loop
    by_shape = {np.shape(v): k for k, v in zip(split.kinds, split.values) if k != "static"}
    assert by_shape[(2, 2)] == "shared"            # mask, equal every step: held once
    assert by_shape[(2, 2, 3)] == "shared"         # gamma: distinct objects, equal values
    assert by_shape[(4, 2, 2, 3)] == "stacked"     # T_ref varies


def test_stale_damping_target_is_refused():
    ctrls = _controls(3)
    ctrls[1] = dict(ctrls[1], sponge=ctrls[0]["sponge"])
    with pytest.raises(ValueError, match="step 4 has kt=3"):
        split_controls(ctrls, kt0=3)


def test_structure_change_is_refused():
    ctrls = _controls(2)
    ctrls[1] = dict(ctrls[1], s2_base={"mask": ctrls[1]["s2_base"]["mask"], "flag": False})
    with pytest.raises(ValueError, match="structure"):
        split_controls(ctrls, kt0=3)


def test_rollout_must_start_on_a_forcing_step_and_run_whole_pairs():
    with pytest.raises(ValueError, match="odd"):
        make_k0_rollout(None, None, split_controls(_controls(2, kt0=4), kt0=4))
    with pytest.raises(ValueError, match="whole pairs"):
        make_k0_rollout(None, None, split_controls(_controls(3), kt0=3))


# ----- the pair loop, with a fake K0 step that records how it is called -----

class S0(NamedTuple):
    prior: object
    shortwave_fraction_top: object
    forcing_stride: object


S2 = NamedTuple("S2", [(f, object) for f in S2_CARRY])


class Surf(NamedTuple):
    value: object


class Result(NamedTuple):
    surface: object
    ocean: object
    next_s0: object
    next_s2: object


def _fake_step(calls):
    def step(card, state, *, model, nemo_kt, s0f, ice_if, s0, s2, chl,
             geothermal_heat_flux_mwm2, prior_surface, sponge, forcing_update=None):
        if int(s0.forcing_stride) != 2:                  # as the real step: must stay Python
            raise ValueError("stride")
        if forcing_update is None:                       # eager: phase from the integer step
            forcing_update = (nemo_kt - 1) % 2 == 0
        calls.append((forcing_update, prior_surface is not None))
        carried = 0.0 if forcing_update else prior_surface.value
        new = state + nemo_kt * s0f["x"] + 0.1 * s0.prior + 0.01 * s2.prior_runoff_mass + carried
        return Result(Surf(value=0.5 * new),
                      new,
                      s0._replace(prior=s0.prior + s0f["x"],
                                  shortwave_fraction_top=0.9 * s0.shortwave_fraction_top),
                      s2._replace(**{f: getattr(s2, f) + nemo_kt for f in S2_CARRY}))
    return step


def _pair_controls(n, kt0):
    s0 = S0(prior=jnp.zeros(3), shortwave_fraction_top=jnp.ones(3), forcing_stride=2)
    s2 = S2(*(jnp.zeros(3) for _ in S2_CARRY))
    return [{"s0f": {"x": jnp.full(3, 1.0 + 0.25 * i)}, "ice_if": None, "s0_base": s0,
             "s2_base": s2, "chl": jnp.zeros(3), "geothermal_heat_flux_mwm2": jnp.zeros(3),
             "sponge": SpongeForcing(gamma=jnp.zeros(3), T_ref=jnp.zeros(3), S_ref=jnp.zeros(3),
                                     nemo_kt=kt0 + i)} for i in range(n)]


def _eager(step, ctrls, kt0, carry):
    prior = None
    for kt, ctrl in enumerate(ctrls, start=kt0):
        s0 = ctrl["s0_base"]._replace(prior=carry.s0_prior,
                                      shortwave_fraction_top=carry.s0_shortwave_fraction_top)
        s2 = ctrl["s2_base"]._replace(**dict(zip(S2_CARRY, carry.s2_prior)))
        res = step(None, carry.state, model=None, nemo_kt=kt, s0f=ctrl["s0f"], ice_if=None,
                   s0=s0, s2=s2, chl=None, geothermal_heat_flux_mwm2=None,
                   prior_surface=prior, sponge=ctrl["sponge"])
        carry, prior = carry_from_result(res), res.surface
    return carry


@pytest.mark.parametrize("schedule", ["none", "uniform", "binomial"])
def test_pair_loop_reproduces_the_step_by_step_hand_off(monkeypatch, schedule):
    kt0, n = 5, 6
    ctrls = _pair_controls(n, kt0)
    start = K0Carry(jnp.arange(3.0), jnp.ones(3), jnp.ones(3), tuple(jnp.zeros(3) for _ in S2_CARRY))
    ref = _eager(_fake_step([]), ctrls, kt0, start)

    calls = []
    monkeypatch.setattr(bulk, "nemo_orca2_k0_ocean_step", _fake_step(calls))
    split = split_controls(ctrls, kt0=kt0)
    out = jax.jit(make_k0_rollout(None, None, split, schedule=schedule, checkpoints=2))(
        start, rollout_operands(split))
    for a, b in zip(jax.tree_util.tree_leaves(out), jax.tree_util.tree_leaves(ref)):
        np.testing.assert_allclose(a, b, rtol=1e-14)
    # each pair: a forcing step without a previous surface, then a reuse step with one
    assert calls[:2] == [(True, False), (False, True)]


def test_rollout_refuses_controls_of_the_wrong_length(monkeypatch):
    monkeypatch.setattr(bulk, "nemo_orca2_k0_ocean_step", _fake_step([]))
    split = split_controls(_pair_controls(4, 3), kt0=3)
    short = split_controls(_pair_controls(2, 3), kt0=3)
    start = K0Carry(jnp.zeros(3), jnp.zeros(3), jnp.ones(3), tuple(jnp.zeros(3) for _ in S2_CARRY))
    with pytest.raises(ValueError, match="expected 4"):
        make_k0_rollout(None, None, split)(start, rollout_operands(short))


def test_traced_step_number_needs_a_static_phase_and_a_cleared_stamp(monkeypatch):
    monkeypatch.setattr(bulk, "_validate_k0_ocean_card", lambda card: None)
    s0 = SimpleNamespace(forcing_stride=2)
    clean = SpongeForcing(gamma=jnp.zeros(3), T_ref=jnp.zeros(3), S_ref=jnp.zeros(3))

    def call(kt, sponge, forcing_update):
        bulk.nemo_orca2_k0_ocean_step(None, None, s0f=None, ice_if=None, s0=s0, s2=None,
                                      chl=None, geothermal_heat_flux_mwm2=None,
                                      sponge=sponge, nemo_kt=kt, forcing_update=forcing_update)

    with pytest.raises(ValueError, match="static forcing_update"):
        jax.eval_shape(lambda k: call(k, clean, None), 3)
    with pytest.raises(ValueError, match="nemo_kt=None"):
        jax.eval_shape(lambda k: call(k, clean._replace(nemo_kt=3), True), 3)
    with pytest.raises(ValueError, match="disagrees"):
        call(3, clean._replace(nemo_kt=3), False)        # integer step 3 is a forcing step
