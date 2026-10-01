"""Guards for the round-158 developed stage-2 advection split.

Under NEMO's vector-invariant arm ``dyn_adv`` is two compiled routines --
the kinetic-energy gradient and the explicit vertical advection -- accumulating
into one right-hand-side slot.  Round 158 scores them separately, which needs
two private seams on the model: the momentum-operator exposure has to publish
each half under its own name, and the stage-2 vertical-advection call has to
accept one substituted operand at a time.

Both guards are checked at CONSTRUCTION, so these run without a GYRE step.
The numbers the seams produce are in the round-158 receipt.
"""

from __future__ import annotations

import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

ADVECTION_HALVES = ("keg", "zad")


@pytest.fixture(scope="module")
def card():
    return build_nemo_testcase_card("GYRE-zco")


def _model(card, **hooks):
    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hooks))


@pytest.mark.parametrize("half", ADVECTION_HALVES)
def test_each_advection_half_is_exposable_by_name(card, half):
    """The two halves the bucket is made of are selectable on their own."""
    assert _model(card, expose_momentum_operator=half,
                  expose_momentum_operator_stage=2)


def test_an_unknown_operator_name_is_refused(card):
    """A typo must not score whatever the returned slots happen to hold."""
    with pytest.raises(ValueError, match="expose_momentum_operator"):
        _model(card, expose_momentum_operator="kegg",
               expose_momentum_operator_stage=2)


@pytest.mark.parametrize("stage", (2, 3))
@pytest.mark.parametrize("bad", [(None, None), (None,), "w", 3,
                                 (None, None, None, None)])
def test_a_stage_vertical_operand_override_that_is_not_a_triple_is_refused(
        card, stage, bad):
    """A pair or a bare array would substitute the WRONG dyn_zad operand."""
    name = f"stage{stage}_zad_operand_override"
    with pytest.raises(ValueError, match=name):
        _model(card, **{name: bad})


@pytest.mark.parametrize("stage", (2, 3))
def test_the_stage_vertical_operand_override_accepts_a_live_triple(card, stage):
    """All-``None`` is the identity: every operand stays the live one."""
    assert _model(card, **{
        f"stage{stage}_zad_operand_override": (None, None, None)})
