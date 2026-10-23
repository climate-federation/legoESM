"""Round 325 (DECISION 151): rung 10 runs L3's eddy-induced velocity on lane
A's L9 route; rung 0 and OMT-0..4 do not; L9 refuses outside its exact
composition."""

from __future__ import annotations

from pathlib import Path

import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_orca2_zps_card,
    orca2_l2_isoneutral_gm_redi,
    orca2_l3_eiv_gm_redi,
    orca2_l8_krhs_selection,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
    nemo_testcase_l4_orca2_round219_omt2_ladder_gate as omt2,
    nemo_testcase_l4_orca2_round221_omt3_ladder_gate as omt3,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round325_l9_eiv_d122_gate import (  # noqa: E501
    ARMS,
    has_l3,
    selected_card,
    without_l3,
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


def _model(card, cfg):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    r = card.recipe
    return LatLonCGridOceanModel(r.grid, r.z_coord, cfg, iwm_forcing=r.iwm_forcing)


def test_rung10_runs_l3_on_l9_only(cards) -> None:
    cfg = cards["rung10"].recipe.model_config
    gm = cfg.gm_redi
    assert gm == orca2_l3_eiv_gm_redi() and cfg.rk3_stage3_l9_eiv_nemo_order
    # &namtra_eiv: nn_aei_ijk_t = 21 (Treguier), aei0 = 0.5*rn_Ue*rn_Le.
    assert gm.treguier.enabled and gm.treguier.aei0 == 0.5 * 0.03 * 200.0e3
    assert gm.gm_bolus_advection == "through_fct" and gm.eke is None
    assert all(getattr(cfg, k) == v for k, v in orca2_l8_krhs_selection().items())
    for rung in ("rung0", "omt0", "omt1", "omt2", "omt3", "omt4"):
        c = cards[rung].recipe.model_config
        assert not c.rk3_stage3_l9_eiv_nemo_order, rung
        assert c.gm_redi is None or not c.gm_redi.treguier.enabled, rung


def test_arms(cards) -> None:
    r10 = cards["rung10"]
    control = without_l3(r10).recipe.model_config
    assert control.gm_redi == orca2_l2_isoneutral_gm_redi()
    assert not control.rk3_stage3_l9_eiv_nemo_order and not has_l3(control)
    assert control == r10.recipe.model_config._replace(
        gm_redi=control.gm_redi, rk3_stage3_l9_eiv_nemo_order=False)
    for arm in ARMS:
        cfg = selected_card(r10, arm=arm, rung="rung10").recipe.model_config
        assert has_l3(cfg) is ("l3" in arm), arm
        assert cfg.implicit_ice_ocean_top_drag is arm.endswith("_drag"), arm
        for rung in ("rung0", "omt1", "omt2", "omt3", "omt4"):
            assert selected_card(cards[rung], arm=arm, rung=rung) is cards[rung]


def test_l9_refuses_outside_its_composition(cards) -> None:
    r10 = cards["rung10"]
    cfg = r10.recipe.model_config
    _model(r10, cfg)  # the shipped composition builds
    with pytest.raises(ValueError, match="the certified L2 selection, or L9"):
        _model(r10, cfg._replace(rk3_stage3_l9_eiv_nemo_order=False))
    with pytest.raises(ValueError, match="the certified L2 selection, or L9"):
        _model(r10, cfg._replace(gm_redi=orca2_l2_isoneutral_gm_redi()))
    with pytest.raises(ValueError, match="the certified L2 selection, or L9"):
        _model(r10, cfg._replace(gm_redi=cfg.gm_redi._replace(
            treguier_sqrt_evaluation="guarded_floor")))
