"""Round 308 (Decision 122): the ORCA2 cards select G6's vector/wzv statements, not its fold set."""

from __future__ import annotations

from pathlib import Path

import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
    nemo_testcase_l4_orca2_round219_omt2_ladder_gate as omt2,
    nemo_testcase_l4_orca2_round221_omt3_ladder_gate as omt3,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round308_g6_d122_gate import (  # noqa: E501
    STATEMENTS,
    _get,
    selected_card,
)

DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
OFF = {**STATEMENTS["FOLD"]}


@pytest.fixture(scope="module")
def cards():
    if not DECK.is_dir():
        pytest.skip("ORCA2 input deck is unavailable")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    return {
        "rung10": build_orca2_zps_card(DECK),
        "rung0": rung0.build_rung0_card(DECK),
        "omt1": omt1.build_omt1_card(DECK),
        "omt2": omt2.build_omt2_card(DECK),
        "omt3": omt3.build_omt3_card(DECK),
        "omt4": omt4.build_omt4_card(DECK),
    }


@pytest.mark.parametrize("rung", ["rung10", "rung0", "omt1", "omt2", "omt3", "omt4"])
def test_card_selects_the_measured_set(cards, rung) -> None:
    cfg = cards[rung].recipe.model_config
    assert {k: _get(cfg, k) for k in STATEMENTS["SEL"]} == STATEMENTS["SEL"]


@pytest.mark.parametrize("rung", ["rung10", "rung0", "omt1", "omt2", "omt3", "omt4"])
def test_card_leaves_the_fold_set_off(cards, rung) -> None:
    cfg = cards[rung].recipe.model_config
    assert all(_get(cfg, k) is False for k in OFF), \
        {k: _get(cfg, k) for k in OFF}


def test_arm_moves_exactly_its_statements(cards) -> None:
    card = cards["rung0"]
    cfg = card.recipe.model_config
    ctrl = selected_card(card, arm="control", rung="rung0").recipe.model_config
    assert not {f for f in cfg._fields if getattr(cfg, f) is not getattr(ctrl, f)}
    moved = selected_card(card, arm="RAWV", rung="rung0").recipe.model_config
    diff = {f for f in cfg._fields if getattr(cfg, f) != getattr(moved, f)}
    assert diff == {"rk3_vector_raw_vmask"}
    fold = selected_card(card, arm="FOLD", rung="rung0").recipe.model_config
    assert all(_get(fold, k) == v for k, v in STATEMENTS["FOLD"].items())


def test_shipped_arm_refuses_a_card_without_the_selection(cards) -> None:
    card = cards["rung0"]
    stripped = card._replace(recipe=card.recipe._replace(
        model_config=card.recipe.model_config._replace(
            nemo_zad_written_association=False)))
    with pytest.raises(Exception, match="not round 308"):
        selected_card(stripped, arm="D122", rung="rung0")
