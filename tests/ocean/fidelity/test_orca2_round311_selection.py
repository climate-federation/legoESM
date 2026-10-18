"""Round 311 (Decision 122): G8's stage-1 Krhs order, G1b's dyn_keg guard kept,
on the ORCA2 cards whose deck runs dyn_ldf (rung 0, OMT-3, OMT-4, rung 10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
    nemo_testcase_l4_orca2_round219_omt2_ladder_gate as omt2,
    nemo_testcase_l4_orca2_round221_omt3_ladder_gate as omt3,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round311_stage1_order_d122_gate import (  # noqa: E501
    GateError,
    selected_card,
)
from tests.ocean.fidelity.test_orca2_round310_selection import _step_jaxpr

DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
ON = ("rung10", "rung0", "omt3", "omt4")
ALL = ("rung10", "rung0", "omt0", "omt1", "omt2", "omt3", "omt4")


@pytest.fixture(scope="module")
def cards():
    if not DECK.is_dir():
        pytest.skip("ORCA2 input deck is unavailable")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    return {
        "rung10": build_orca2_zps_card(DECK),
        "rung0": rung0.build_rung0_card(DECK),
        "omt0": omt0.build_omt0_card(DECK),
        "omt1": omt1.build_omt1_card(DECK),
        "omt2": omt2.build_omt2_card(DECK),
        "omt3": omt3.build_omt3_card(DECK),
        "omt4": omt4.build_omt4_card(DECK),
    }


@pytest.mark.parametrize("rung", ALL)
def test_stage1_order_selection(cards, rung) -> None:
    cfg = cards[rung].recipe.model_config
    assert cfg.rk3_stage1_ldf_source_order is (rung in ON)
    if rung in ON:
        # the guard the order now keeps is G1b's, selected on every vector card
        assert cfg.rk3_vector_rhs_nemo_order is True


@pytest.mark.parametrize("rung", ("rung10", "rung0", "omt1", "omt3", "omt4"))
def test_shipped_arm_checks_the_selection(cards, rung) -> None:
    card = cards[rung]
    assert selected_card(card, arm="D122", rung=rung) is card
    flipped = card._replace(recipe=card.recipe._replace(
        model_config=card.recipe.model_config._replace(
            rk3_stage1_ldf_source_order=rung not in ON)))
    with pytest.raises(GateError):
        selected_card(flipped, arm="D122", rung=rung)


def test_stage1_order_is_consumed(cards) -> None:
    card = cards["rung0"]
    cfg = card.recipe.model_config
    assert _step_jaxpr(card, cfg._replace(rk3_stage1_ldf_source_order=False)) != (
        _step_jaxpr(card, cfg))


def test_retained_selector_plant_fires(cards) -> None:
    card = cards["rung0"]
    before = card._replace(recipe=card.recipe._replace(  # the round-310 card
        model_config=card.recipe.model_config._replace(
            rk3_stage1_ldf_source_order=False)))
    assert selected_card(before, arm="control", rung="rung0").recipe.model_config == (
        before.recipe.model_config)
    with pytest.raises(GateError, match="control arm moved the card"):
        selected_card(before, arm="control", rung="rung0", plant="retained-selector")
