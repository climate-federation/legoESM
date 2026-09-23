"""Guards for the round-159 developed stage-2 continuity-solve arm.

NEMO solves continuity twice per stage on this deck: once on the raw stage
velocity for the momentum vertical advection, and once on the barotropically
corrected transports inside the tracer transport assembly.  legoESM's
production path takes the second form for both consumers, so round 159 needs a
private one-variable arm that selects the first at stage 2 with every other
operand held.

The guard is checked at CONSTRUCTION, so these run without a GYRE step.  The
numbers the arm produces are in the round-159 receipt.
"""

from __future__ import annotations

import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card


@pytest.fixture(scope="module")
def card():
    return build_nemo_testcase_card("GYRE-zco")


def _model(card, **hooks):
    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hooks))


@pytest.mark.parametrize("bad", [1, 0, "true", "", (True,), None])
def test_a_continuity_form_arm_that_is_not_a_bool_is_refused(card, bad):
    """Anything truthy would silently select NEMO's OTHER call form.

    The walk would then score one compiled program under another program's
    name, which is the failure this guard exists to make impossible.
    """
    with pytest.raises(ValueError, match="stage2_wzv_velocity_form"):
        _model(card, stage2_wzv_velocity_form=bad)


@pytest.mark.parametrize("good", [True, False])
def test_the_continuity_form_arm_accepts_either_call_form(card, good):
    """Both compiled forms are selectable; ``False`` is the production path."""
    assert _model(card, stage2_wzv_velocity_form=good)


def test_the_production_default_is_the_transport_form(card):
    """No card constructs the arm, so the default must stay the live path."""
    assert _NEMOWSRK3TestHooks().stage2_wzv_velocity_form is False
