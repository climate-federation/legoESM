"""Round 307 (Decision 122/130): only the TKE card selects the V1 vertical-mixing statements."""

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
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round307_vmix_d122_gate import (  # noqa: E501
    V1_FLAGS,
    selected_card,
)

DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")


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


def _on(card):
    tke = card.recipe.model_config.physics.vertical_mixing.tke
    return {flag for flag in V1_FLAGS if getattr(tke, flag)}


def test_tke_card_selects_every_v1_statement(cards) -> None:
    vm = cards["rung10"].recipe.model_config.physics.vertical_mixing
    assert vm.scheme == "tke"
    assert _on(cards["rung10"]) == set(V1_FLAGS)
    # Round 312 (D135) added the nn_etau=1 penetration with V1's htau/EXP.
    assert vm.tke.etau_mode == "below_ml"
    assert vm.tke.tke_htau_evaluation == "nemo_scalar_libm"
    assert vm.tke.tke_etau_exponential_evaluation == "nemo_scalar_libm"


@pytest.mark.parametrize("rung", ["rung0", "omt0", "omt1", "omt2", "omt3", "omt4"])
def test_constant_mixing_cards_carry_no_v1_statement(cards, rung) -> None:
    assert cards[rung].recipe.model_config.physics.vertical_mixing.scheme != "tke"
    assert _on(cards[rung]) == set()


def test_arm_driver_refuses_a_shipped_card_without_the_selection(cards) -> None:
    card = selected_card(cards["rung10"], arm="control", rung="rung10")
    assert _on(card) == set()
    with pytest.raises(RuntimeError, match="not round 307's"):
        selected_card(card, arm="D122", rung="rung10")
    with pytest.raises(RuntimeError, match="not round 307's"):
        selected_card(cards["rung10"], arm="D122", rung="rung0")
