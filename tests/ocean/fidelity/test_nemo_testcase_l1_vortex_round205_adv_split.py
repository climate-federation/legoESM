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


def test_the_qco_depth_arm_is_live_and_finite(steps):
    """It swaps the transport divisor, so it must move the step -- and it
    must not put a NaN on a dry column, where the qco depth is zero."""
    assert _moved(steps["qco_depth"], steps["plain"]) > 0.0
    for leaf in _leaves(steps["qco_depth"]):
        assert np.all(np.isfinite(leaf))


def test_the_prognostic_mean_arm_really_reads_uu_b(fp64):
    """NON-VACUITY FOR A REFUTATION.

    Round 205 refutes the ``prognostic_mean`` hypothesis by measuring that
    swapping that operand does not move the stage-1 advection trend at the
    scale in question.  A refutation is worthless if the arm never ran, and
    on THIS card's own initial state the arm is a no-op for a real reason:
    the prognostic ``uu_b`` and the depth mean re-reduced from the
    three-dimensional velocity are equal there (on the record-seeded state
    the walk scores they differ by 5.551115e-17, one unit in the last
    place, and the arm moves the trend by 1.355e-20).

    So liveness is proven by a CONTROL instead of by a number: perturb
    ``uu_b``/``vv_b`` and the arm's step must move AWAY from the production
    step, which it cannot do unless it reads them.  The unperturbed pair is
    the other half of the control -- there the two steps agree exactly.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    card, plain_model = _card_and_model()
    _, arm_model = _card_and_model(_NEMOWSRK3TestHooks(
        momentum_transport_stage1_operand="prognostic_mean"))
    clean = card.recipe.initial_state
    assert clean.uu_b is not None and clean.vv_b is not None
    bumped = clean._replace(
        uu_b=clean.uu_b.replace(
            data=clean.uu_b.data + 1.0e-4 * np.asarray(clean.u_mask.data)),
        vv_b=clean.vv_b.replace(
            data=clean.vv_b.data + 1.0e-4 * np.asarray(clean.v_mask.data)))
    # On the unperturbed state the two operands agree, so the arm is a
    # no-op -- that is the measured fact, not an inert hook.
    assert _moved(arm_model.step(clean, dt=card.dt_s),
                  plain_model.step(clean, dt=card.dt_s)) == 0.0
    # Perturb the operand the arm reads and it must separate.
    assert _moved(arm_model.step(bumped, dt=card.dt_s),
                  plain_model.step(bumped, dt=card.dt_s)) > 0.0
    # NOT covered here, said rather than implied: the arm's refusal when a
    # state carries no uu_b/vv_b pair.  Stripping that pair makes the
    # barotropic solver raise first on this card
    # (barotropic_latlon_cgrid.py, the carried NEMO depth mean), so the
    # refusal cannot be reached from a card and is left unexercised.


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

    # The guard is at CONSTRUCTION, not at step entry (round 205 review
    # finding): a typo must not survive to a card whose stage branch never
    # reads the hook.
    with pytest.raises(ValueError, match=field):
        _card_and_model(_NEMOWSRK3TestHooks(**{field: value}))
