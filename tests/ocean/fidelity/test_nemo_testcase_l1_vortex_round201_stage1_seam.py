"""Fail-closed controls for round 201's stage-1 momentum seam.

Round 200 narrowed the VORTEX flux card's kt=2 velocity owner to two
statements inside RK3 stage 1 -- the flux-form advection trend
(``stprk3_stg.F90:316``) and the thickness-weighted explicit velocity update
(``stprk3_stg.F90:372-379``) -- and could not score either, because legoESM
had no stage-1 momentum seam at all.  Round 201 adds the stage-1 companions
of the stage-2 pair that already existed: two WRITE-only exposures and one
one-variable override.

Four things have to be true before any number they produce means anything:

* at their defaults they change NOTHING (they are private instruments, not
  schemes, and no card constructs them);
* the two exposures publish DIFFERENT frames from each other and from the
  stage-1 output, so a seam wired at the wrong boundary cannot pass;
* the override, set to the model's OWN stage-1 right-hand side, reproduces
  the production step BIT for BIT -- the known-answer control;
* set to something else it CHANGES the answer, so the control above cannot
  pass vacuously.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

CASE = "VORTEX-zco"


def _card_and_model(hooks=None):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    card = build_nemo_testcase_card(CASE)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    return card, model


@pytest.fixture(scope="module")
def fp64():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(old)


@pytest.fixture(scope="module")
def steps(fp64):
    """Every step this module needs, run once: each one is ~40 s of CPU."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    card, plain = _card_and_model()
    initial = card.recipe.initial_state
    out = {"card": card, "plain": plain.step(initial, dt=card.dt_s)}
    _, defaults = _card_and_model(_NEMOWSRK3TestHooks(
        expose_stage1_momentum_rhs=False,
        expose_stage1_raw_momentum=False,
        stage1_momentum_rhs_override=None))
    out["defaults"] = defaults.step(initial, dt=card.dt_s)
    _, rhs_model = _card_and_model(_NEMOWSRK3TestHooks(
        expose_stage1_momentum_rhs=True))
    out["rhs"] = rhs_model.step(initial, dt=card.dt_s)
    _, raw_model = _card_and_model(_NEMOWSRK3TestHooks(
        expose_stage1_raw_momentum=True))
    out["raw"] = raw_model.step(initial, dt=card.dt_s)
    _, stage_model = _card_and_model(_NEMOWSRK3TestHooks(
        expose_momentum_stage=1))
    out["stage1_out"] = stage_model.step(initial, dt=card.dt_s)
    own = (out["rhs"].u.data, out["rhs"].v.data)
    _, fed_back = _card_and_model(_NEMOWSRK3TestHooks(
        stage1_momentum_rhs_override=own))
    out["fed_back"] = fed_back.step(initial, dt=card.dt_s)
    _, bent_model = _card_and_model(_NEMOWSRK3TestHooks(
        stage1_momentum_rhs_override=(own[0] * 1.0000001, own[1])))
    out["bent"] = bent_model.step(initial, dt=card.dt_s)
    return out


def _leaves(state):
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(state)]


def _assert_identical(got, want):
    a, b = _leaves(got), _leaves(want)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_the_seam_at_its_defaults_changes_nothing(steps):
    _assert_identical(steps["defaults"], steps["plain"])


def test_the_rhs_exposure_publishes_a_frame_that_is_not_the_step_output(steps):
    # A hook that silently went inert would hand the walk the ordinary step
    # output and every row would still score a number.
    assert not np.array_equal(np.asarray(steps["rhs"].u.data),
                              np.asarray(steps["plain"].u.data))
    assert not np.array_equal(np.asarray(steps["rhs"].v.data),
                              np.asarray(steps["plain"].v.data))
    # Only the velocity slots are substituted; the rest of the step is the
    # production one, so the exposure cannot perturb a later stage.
    np.testing.assert_array_equal(np.asarray(steps["rhs"].T.data),
                                  np.asarray(steps["plain"].T.data))
    np.testing.assert_array_equal(np.asarray(steps["rhs"].eta.data),
                                  np.asarray(steps["plain"].eta.data))


def test_the_two_stage1_boundaries_are_different_frames(steps):
    # ``raw`` is Kaa after stprk3_stg.F90:372-379 and BEFORE the barotropic
    # replacement at :409-421; ``stage1_out`` is the same Kaa after it.  If
    # the new seam were wired at the old boundary these would be equal and
    # the walk would score one statement under the other's name.
    raw = np.asarray(steps["raw"].u.data)
    done = np.asarray(steps["stage1_out"].u.data)
    assert not np.array_equal(raw, done)
    # ... and it is the DEPTH MEAN that the replacement changes, so the two
    # frames must differ by one number per column, not by a general field.
    delta = raw - done
    departure = np.abs(delta - delta.mean(axis=-1, keepdims=True)).max()
    assert departure < 1.0e-12 * max(np.abs(delta).max(), 1.0e-30) + 1.0e-14
    # The RHS frame is a tendency, not a velocity: it cannot be either.
    assert not np.array_equal(np.asarray(steps["rhs"].u.data), raw)


def test_feeding_the_model_its_own_stage1_rhs_reproduces_the_step(steps):
    # The known-answer control: the override binds at the boundary it claims
    # and substitutes the SAME quantity the exposure published.
    _assert_identical(steps["fed_back"], steps["plain"])


def test_the_known_answer_control_is_not_vacuous(steps):
    # Bend one operand by 1e-7 relative: the step must move, or the control
    # above would pass for an override that binds nothing.
    assert not np.array_equal(np.asarray(steps["bent"].u.data),
                              np.asarray(steps["plain"].u.data))


@pytest.mark.parametrize("hooks_kwargs,match", [
    ({"expose_stage1_momentum_rhs": True,
      "expose_stage1_raw_momentum": True}, "share the returned u/v slots"),
    ({"expose_stage1_momentum_rhs": True,
      "expose_momentum_stage": 1}, "share the returned u/v slots"),
    ({"expose_stage1_raw_momentum": True,
      "expose_stage2_raw_momentum": True}, "share the returned u/v slots"),
    ({"expose_stage1_raw_momentum": True,
      "expose_momentum_operator": "hpg"}, "share the returned u/v slots"),
    ({"expose_stage1_momentum_rhs": 1}, "must be a bool"),
])
def test_the_construction_guard_refuses_an_ambiguous_exposure(
        fp64, hooks_kwargs, match):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    with pytest.raises(ValueError, match=match):
        _card_and_model(_NEMOWSRK3TestHooks(**hooks_kwargs))


def test_one_stage1_exposure_alone_is_accepted(fp64):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    # The guard above must not refuse the legal single selections, or the
    # walk could not run at all.
    _card_and_model(_NEMOWSRK3TestHooks(expose_stage1_momentum_rhs=True))
    _card_and_model(_NEMOWSRK3TestHooks(expose_stage1_raw_momentum=True))
