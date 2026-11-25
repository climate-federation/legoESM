"""Round 312: D126 surface-only runoff and D135's V1 TKE forms on the ORCA2 cards."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
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
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round312_runoff_tke_d122_gate import (  # noqa: E501
    ETAU,
    LC,
    MXL,
    SHIPPED,
    TKE_FIELDS,
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


def _tke(card):
    tke = card.recipe.model_config.physics.vertical_mixing.tke
    return {f: getattr(tke, f) for f in TKE_FIELDS}


def test_shipped_card_runs_runoff_at_the_surface(cards):
    for card in cards.values():
        cfg = card.recipe.model_config
        assert cfg.runoff_depth_spread_m == 0.0
        assert cfg.runoff_depth_spread_map is None


def test_shipped_card_selects_v1_penetration_length_and_langmuir(cards):
    tke = cards["rung10"].recipe.model_config.physics.vertical_mixing.tke
    assert _tke(cards["rung10"]) == {**ETAU, **MXL, **LC}
    assert tke.nemo_mxlice == 0  # waits for the card's own hm_i


@pytest.mark.parametrize("rung", ["rung0", "omt0", "omt1", "omt2", "omt3", "omt4"])
def test_constant_mixing_cards_keep_the_previous_tke_forms(cards, rung):
    assert _tke(cards[rung]) == SHIPPED


def test_card_carries_native_gphit_and_dummy_bottom_thickness(cards):
    card = cards["rung10"]
    with netCDF4.Dataset(DECK / "ORCA_R2_zps_domcfg.nc") as ds:
        gphit = np.asarray(ds.variables["gphit"][:], dtype=np.float64)
        e3t_jpk = np.asarray(ds.variables["e3t_0"][30], dtype=np.float64)
    assert np.array_equal(np.asarray(card.recipe.grid.native_lat_T_deg), gphit)
    assert np.array_equal(np.asarray(card.recipe.z_coord.nemo_e3t_jpk), e3t_jpk)


def test_arm_driver_control_strips_exactly_this_selection(cards):
    assert selected_card(cards["rung10"], arm="D122", rung="rung10") is cards["rung10"]
    control = selected_card(cards["rung10"], arm="control", rung="rung10")
    assert _tke(control) == SHIPPED
    assert control.recipe.model_config.runoff_depth_spread_m == 150.0
    assert control.recipe.model_config._replace(
        runoff_depth_spread_m=0.0, physics=cards["rung10"].recipe.model_config.physics
    ) == cards["rung10"].recipe.model_config
    with pytest.raises(RuntimeError, match="not round 311's"):
        selected_card(cards["rung10"], arm="LC", rung="rung10")
    with pytest.raises(RuntimeError, match="not round 312's"):
        selected_card(cards["rung10"]._replace(recipe=cards["rung10"].recipe._replace(
            model_config=cards["rung10"].recipe.model_config._replace(
                runoff_depth_spread_m=150.0))), arm="D122", rung="rung10")
