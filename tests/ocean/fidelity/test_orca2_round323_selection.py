"""Round 323 (Decision 122): rung 10 runs lane A L8's stage-3 Krhs route and
lane B L5's mixed-layer eddies; rung 0 and OMT-0..4 run neither; L3 + L8 is
refused (uncertified composition)."""

from __future__ import annotations

from pathlib import Path

import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_orca2_zps_card,
    orca2_l3_eiv_gm_redi,
    orca2_l5_mle,
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
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round323_l8_l3_l5_d122_gate import (  # noqa: E501
    PRE_SELECTION,
    selected_card,
    shipped_pieces,
    without_selection,
)

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round325_l9_eiv_d122_gate import (  # noqa: E501
    without_l3,
)

DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")


@pytest.fixture(scope="module")
def cards():
    if not DECK.is_dir():
        pytest.skip("ORCA2 input deck is unavailable")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    return {
        "rung10": without_l3(build_orca2_zps_card(DECK)),  # round 325: L3 reverted
        "rung0": rung0.build_rung0_card(DECK),
        "omt0": omt0.build_omt0_card(DECK),
        "omt1": omt1.build_omt1_card(DECK),
        "omt2": omt2.build_omt2_card(DECK),
        "omt3": omt3.build_omt3_card(DECK),
        "omt4": omt4.build_omt4_card(DECK),
    }


def test_rung10_runs_l8_and_l5_only(cards) -> None:
    r10 = cards["rung10"]
    cfg = r10.recipe.model_config
    assert shipped_pieces(cfg) == ("L8", "L5")
    assert cfg.physics.mle == orca2_l5_mle()
    assert r10.bbl_gamma_s == cfg.bbl_gamma_s == 10.0
    assert r10.recipe.tracer_damping.nemo_kt == 1
    for rung in ("rung0", "omt0", "omt1", "omt2", "omt3", "omt4"):
        c = cards[rung].recipe.model_config
        assert shipped_pieces(c) == (), rung
        assert c.physics.mle is None, rung
        assert cards[rung].recipe.tracer_damping is None, rung
        for name in orca2_l8_krhs_selection():
            if name.startswith("rk3_stage3") or name.startswith("bbl_nemo"):
                assert getattr(c, name) is False, (rung, name)


def test_control_rebuild_is_the_round322_card(cards) -> None:
    r10 = cards["rung10"]
    control = without_selection(r10)
    cfg = control.recipe.model_config
    assert shipped_pieces(cfg) == ()
    assert {k: getattr(cfg, k) for k in PRE_SELECTION} == PRE_SELECTION
    assert cfg.physics.mle is None and control.recipe.tracer_damping is None
    assert selected_card(r10, arm="D122", rung="rung10") is r10
    assert selected_card(r10, arm="control", rung="rung10") == control


def test_l3_with_l8_is_refused(cards) -> None:
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    r = cards["rung10"].recipe
    cfg = r.model_config._replace(gm_redi=orca2_l3_eiv_gm_redi())
    with pytest.raises(ValueError, match="the certified L2 selection, or L9"):
        LatLonCGridOceanModel(r.grid, r.z_coord, cfg, iwm_forcing=r.iwm_forcing)
