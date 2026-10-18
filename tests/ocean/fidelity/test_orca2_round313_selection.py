"""Round 313: V4's literal internal-wave mixing and V3's double diffusion on ORCA2."""

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
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round313_iwm_ddm_d122_gate import (  # noqa: E501
    BOTH,
    SHIPPED,
    carried,
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


def test_shipped_card_runs_orca2_iwm_and_ddm(cards):
    vm = cards["rung10"].recipe.model_config.physics.vertical_mixing
    assert carried(cards["rung10"]) == BOTH
    assert (vm.iwm.enabled, vm.iwm.mevar) == (True, False)  # namelist_cfg:396, :423
    assert (vm.ddm.rn_avts, vm.ddm.rn_hsbfr) == (1.0e-4, 1.6)  # namelist_cfg:394-395


@pytest.mark.parametrize("rung", ["rung0", "omt0", "omt1", "omt2", "omt3", "omt4"])
def test_constant_mixing_cards_keep_both_off(cards, rung):
    vm = cards[rung].recipe.model_config.physics.vertical_mixing
    assert (vm.iwm.enabled, vm.ddm.enabled) == (False, False)
    assert carried(cards[rung]) == SHIPPED


def test_arm_driver_control_strips_exactly_this_selection(cards):
    card = cards["rung10"]
    assert selected_card(card, arm="D122", rung="rung10") is card
    control = selected_card(card, arm="control", rung="rung10")
    assert carried(control) == SHIPPED
    assert carried(selected_card(control, arm="BOTH", rung="rung10")) == BOTH
    with pytest.raises(RuntimeError, match="not round 313's selection"):
        selected_card(control, arm="D122", rung="rung10")
