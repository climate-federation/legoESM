"""Decision 133: the ORCA2 cards select NEMO's linear implicit drag + G7's bottom statements."""

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

DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
LINEAR = {"bottom_drag_scheme": "nemo_linear", "bottom_drag_cd0": 1.0e-3,
          "bottom_drag_uc0": 0.4, "bottom_drag_r": 0.0,
          "bottom_drag_bbl_thickness": 0.0, "bottom_drag_bg_velocity": 0.0}


@pytest.fixture(scope="module")
def configs():
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


def _drag(cfg):
    return {name: getattr(cfg.bottom_drag, name) for name in LINEAR}


@pytest.mark.parametrize("rung", ["rung10", "rung0", "omt2", "omt3", "omt4"])
def test_linear_implicit_drag_with_bottom_statements(configs, rung) -> None:
    cfg = configs[rung].recipe.model_config
    assert _drag(cfg) == LINEAR
    assert cfg.zdf_drag_in_matrix and cfg.zdf_baroclinic_only
    assert cfg.barotropic_drag_substep
    assert cfg.nemo_bottom_drag_written_statements is True
    assert cfg.implicit_ice_ocean_top_drag is False


@pytest.mark.parametrize("rung", ["omt0", "omt1"])
def test_drag_off_rungs_stay_off(configs, rung) -> None:
    cfg = configs[rung].recipe.model_config
    assert cfg.bottom_drag.bottom_drag_scheme == "legacy"
    assert (cfg.bottom_drag.bottom_drag_r,
            cfg.bottom_drag.bottom_drag_bbl_thickness) == (0.0, 0.0)
    assert not cfg.zdf_drag_in_matrix and not cfg.barotropic_drag_substep
    assert not cfg.nemo_bottom_drag_written_statements
    assert not cfg.implicit_ice_ocean_top_drag


def test_card_validators_accept_the_selection(configs) -> None:
    rung0.validate_rung0_card(configs["rung0"])
    omt1.validate_omt1_card(configs["omt1"])
    omt2.validate_omt2_card(DECK, configs["omt2"])
    omt4.validate_omt4_card(DECK, configs["omt4"])
