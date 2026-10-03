"""Fail-closed controls for round 205's ``dyn_adv_up3`` split.

Round 204 put the VORTEX flux card's whole stage-1 error inside the
flux-form momentum advection NEMO calls at ``stprk3_stg.f90:316``, and could
go no further: the record carries only the TOTAL (``adv`` minus ``base``),
so neither of the two halves ``dyn_adv_up3`` writes -- the horizontal UP3
flux divergence (``dynadv_up3.f90:174-215``) and the vertical block
(``:245-360``) -- could be read out of it.

Round 205 publishes legoESM's OWN two halves apart, and swaps the two
operands of ``stprk3_stg.f90:270`` one at a time.  Four things have to hold
before any of those frames means anything:

* at their defaults neither new hook changes anything;
* the two halves ADD BACK to the content round 204 removed, so the split is
  a partition of the same number and not of a different one;
* each causal arm is LIVE -- it moves the step -- because an arm that
  silently did nothing would report "no effect" and refute a true
  hypothesis;
* an unknown arm string RAISES, so a typo cannot select a different
  measurement in silence.
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
    out = {"plain": plain.step(initial, dt=card.dt_s)}
    for name, hooks in (
            ("defaults", _NEMOWSRK3TestHooks(
                expose_stage1_momentum_rhs_split="",
                momentum_transport_stage1_operand="")),
            ("completed", _NEMOWSRK3TestHooks(
                expose_stage1_momentum_rhs_split="completed")),
            ("pre_advection", _NEMOWSRK3TestHooks(
                expose_stage1_momentum_rhs_split="pre_advection")),
            ("hadv", _NEMOWSRK3TestHooks(
                expose_stage1_momentum_rhs_split="advection_horizontal")),
            ("vadv", _NEMOWSRK3TestHooks(
                expose_stage1_momentum_rhs_split="advection_vertical")),
            ("zub", _NEMOWSRK3TestHooks(
                expose_stage1_momentum_rhs_split="advection_zub_increment")),
            ("prognostic_mean", _NEMOWSRK3TestHooks(
                momentum_transport_stage1_operand="prognostic_mean")),
            ("qco_depth", _NEMOWSRK3TestHooks(
                momentum_transport_stage1_operand="qco_depth")),
    ):
        _, model = _card_and_model(hooks)
        out[name] = model.step(initial, dt=card.dt_s)
    return out


def _leaves(state):
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(state)]


def _assert_identical(got, want):
    a, b = _leaves(got), _leaves(want)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def _moved(got, want) -> float:
    a, b = _leaves(got), _leaves(want)
    return max(float(np.max(np.abs(x - y))) if x.shape == y.shape else np.inf
               for x, y in zip(a, b))


def test_both_new_hooks_at_their_defaults_change_nothing(steps):
    """SCOPE, said rather than implied: this covers the two fields' presence
    and the unconditionally-built qco column depth next to the production
    transport target -- that depth is computed on every WS-RK3 step now, and
    this test is what says it cannot reach the answer.  That the round moved
    no certified number is carried by the external registries (both VORTEX
    cards 0/50, the GYRE ladder and the 360-day year), not by this test.
    """
    _assert_identical(steps["defaults"], steps["plain"])


def test_the_two_halves_add_back_to_the_content_round_204_removed(steps):
    """The partition control.

    ``"completed"`` minus ``"pre_advection"`` IS the advection content round
    204 scored; ``"advection_horizontal"`` plus ``"advection_vertical"`` must
    be the same array.  Not asserted bit-exact: adding and subtracting the
    same numbers is not an identity in floating point.  The bar is the
    arithmetic floor of fields whose own size is ~1e-05.
    """
    for face in ("u", "v"):
        total = (np.asarray(getattr(steps["completed"], face).data)
                 - np.asarray(getattr(steps["pre_advection"], face).data))
        halves = (np.asarray(getattr(steps["hadv"], face).data)
                  + np.asarray(getattr(steps["vadv"], face).data))
        residual = float(np.max(np.abs(halves - total)))
        assert residual < 1.0e-18, (face, residual)
        # ...and the split is not trivial: both halves carry something.
        assert float(np.max(np.abs(
            np.asarray(getattr(steps["hadv"], face).data)))) > 1.0e-9
        assert float(np.max(np.abs(
            np.asarray(getattr(steps["vadv"], face).data)))) > 1.0e-12


def test_the_zub_increment_is_inside_the_horizontal_half(steps):
    """``stprk3_stg.f90:270``'s barotropic correction reaches the flux-form
    HORIZONTAL term, so the published cross-term must be no larger than the
    half that is claimed to contain it, and must not be zero."""
    for face in ("u", "v"):
        zub = np.abs(np.asarray(getattr(steps["zub"], face).data)).max()
        hadv = np.abs(np.asarray(getattr(steps["hadv"], face).data)).max()
        assert 0.0 < zub < hadv


@pytest.mark.parametrize("arm", ("prognostic_mean", "qco_depth"))
def test_each_causal_transport_arm_is_live(steps, arm):
    """NON-VACUITY, and the reason this test exists: round 205 REFUTES one of
    these two arms by measuring that it changes nothing at the scale in
    question.  That refutation is only worth anything if the arm ran.  An
    arm that silently fell through its guard would look identical."""
    moved = _moved(steps[arm], steps["plain"])
    assert moved > 0.0, arm
    assert np.isfinite(moved), arm
    for leaf in _leaves(steps[arm]):
        assert np.all(np.isfinite(leaf)), arm


def test_the_two_arms_do_different_things(steps):
    """They swap different operands of the same statement, so they must not
    be the same perturbation."""
    assert _moved(steps["qco_depth"], steps["prognostic_mean"]) > 0.0


@pytest.mark.parametrize("field,value", (
    ("expose_stage1_momentum_rhs_split", "advection"),
    ("expose_stage1_momentum_rhs_split", "qco_depth"),
    ("momentum_transport_stage1_operand", "advection_horizontal"),
    ("momentum_transport_stage1_operand", "min_rule"),
))
def test_an_unknown_arm_string_raises(fp64, field, value):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    card, model = _card_and_model(_NEMOWSRK3TestHooks(**{field: value}))
    with pytest.raises(ValueError, match=field):
        model.step(card.recipe.initial_state, dt=card.dt_s)
