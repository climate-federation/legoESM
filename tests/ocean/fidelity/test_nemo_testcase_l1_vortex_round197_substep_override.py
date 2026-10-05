"""Fail-closed controls for round 197's per-substep barotropic override.

The override replaces one operand of NEMO's compiled sub-time-step loop --
the 2-D Coriolis trend (``dynspg_ts.f90:503``) or the surface pressure
gradient (``dynspg_ts.f90:498``) -- at the substep the scan is on.  Three
things have to be true before any number it produces means anything:

* unset, it changes NOTHING (it is a private instrument, not a scheme);
* set to the model's OWN per-substep trend, it reproduces the production
  step BIT for BIT -- which is the known-answer control, and the only test
  that can tell a correct per-substep index from one frame reused for the
  whole window;
* set to something else, it CHANGES the answer, so the control above cannot
  pass vacuously.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

CASE = "VORTEX_VEC-zco"


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


def _leaves(state):
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(state)]


def _assert_identical(got, want):
    a, b = _leaves(got), _leaves(want)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_unset_override_leaves_the_production_step_untouched(fp64):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    card, plain = _card_and_model()
    _, hooked = _card_and_model(_NEMOWSRK3TestHooks(
        barotropic_substep_coriolis_override=None,
        barotropic_substep_pgf_override=None))
    _assert_identical(hooked.step(card.recipe.initial_state, dt=card.dt_s),
                      plain.step(card.recipe.initial_state, dt=card.dt_s))


def test_substituting_the_models_own_trend_reproduces_the_step_bit_for_bit(fp64):
    """Known answer first, then the control that makes it non-vacuous."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    # The reference is the TRACED step, not the plain one: exposing the
    # substeps compiles the loop as a scan rather than a fori_loop, and the
    # two agree only to ~3e-16.  Every arm that uses this override reads the
    # trace, so the traced step is the right -- and the only honest --
    # comparison for it.
    card, traced = _card_and_model(
        _NEMOWSRK3TestHooks(expose_barotropic_substeps=True))
    result = jax.device_get(traced.step(card.recipe.initial_state,
                                        dt=card.dt_s))
    reference = result
    trace = result.substeps
    own = (jnp.asarray(trace["cor_u"]), jnp.asarray(trace["cor_v"]))
    assert own[0].shape[0] > 1, "a one-substep window cannot test the index"

    _, substituted = _card_and_model(_NEMOWSRK3TestHooks(
        expose_barotropic_substeps=True,
        barotropic_substep_coriolis_override=own))
    _assert_identical(
        substituted.step(card.recipe.initial_state, dt=card.dt_s), reference)

    # NON-VACUITY, two ways.  Zeroing the trend must move the answer, and so
    # must holding the FIRST substep's trend for the whole window -- the
    # exact defect a per-substep hook can hide.
    for broken in (
        (jnp.zeros_like(own[0]), jnp.zeros_like(own[1])),
        (jnp.broadcast_to(own[0][:1], own[0].shape),
         jnp.broadcast_to(own[1][:1], own[1].shape)),
        # ...and rolled by ONE substep, which is the defect a
        # per-substep hook hides most easily and which neither control
        # above can see.
        (jnp.roll(own[0], 1, axis=0), jnp.roll(own[1], 1, axis=0)),
    ):
        _, bent = _card_and_model(_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_substep_coriolis_override=broken))
        moved = bent.step(card.recipe.initial_state, dt=card.dt_s)
        assert any(
            not np.array_equal(x, y)
            for x, y in zip(_leaves(moved), _leaves(reference)))


def test_pressure_gradient_override_binds_at_its_own_boundary(fp64):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    card, traced = _card_and_model(
        _NEMOWSRK3TestHooks(expose_barotropic_substeps=True))
    reference = jax.device_get(traced.step(card.recipe.initial_state,
                                           dt=card.dt_s))
    own = (jnp.asarray(reference.substeps["pgf_u"]),
           jnp.asarray(reference.substeps["pgf_v"]))

    _, substituted = _card_and_model(_NEMOWSRK3TestHooks(
        expose_barotropic_substeps=True,
        barotropic_substep_pgf_override=own))
    _assert_identical(
        substituted.step(card.recipe.initial_state, dt=card.dt_s), reference)

    _, bent = _card_and_model(_NEMOWSRK3TestHooks(
        expose_barotropic_substeps=True,
        barotropic_substep_pgf_override=(jnp.zeros_like(own[0]),
                                         jnp.zeros_like(own[1]))))
    moved = bent.step(card.recipe.initial_state, dt=card.dt_s)
    assert any(
        not np.array_equal(x, y)
        for x, y in zip(_leaves(moved), _leaves(reference)))
