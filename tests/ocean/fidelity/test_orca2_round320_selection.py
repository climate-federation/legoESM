"""Round 320 (Decision 122): lane B L2's isoneutral block on rung 10 only.

Rung 0 (whose deck also runs it) refuses L2 without the stored stage-3
tracer thickness and keeps no isoneutral block; OMT-0..4 state it off.
Lane A L4's geothermal statement is imported but refuses with tracer
lateral diffusion, so no card selects it."""

from __future__ import annotations

from pathlib import Path

import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_orca2_zps_card,
    orca2_l2_isoneutral_gm_redi,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
    nemo_testcase_l4_orca2_round219_omt2_ladder_gate as omt2,
    nemo_testcase_l4_orca2_round221_omt3_ladder_gate as omt3,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round320_l2_l4_d122_gate import (  # noqa: E501
    GateError,
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


def test_rung10_runs_l2_and_no_card_runs_l4(cards) -> None:
    l2 = orca2_l2_isoneutral_gm_redi()
    assert l2.msc_stabilize and l2.kappa_redi_lat_scaling
    assert l2.kappa_redi_diffusive_velocity == 0.018
    assert l2.slope_prd_evaluation == "nemo_eos80_literal"
    assert cards["rung10"].recipe.model_config.gm_redi == l2
    for rung in ("rung0", "omt0", "omt1", "omt2", "omt3", "omt4"):
        assert cards[rung].recipe.model_config.gm_redi is None, rung
    for rung, card in cards.items():
        assert card.recipe.model_config.rk3_stage3_geothermal_nemo_written is False, rung


def test_geothermal_refuses_with_isoneutral_diffusion(cards) -> None:
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    r = cards["rung10"].recipe
    cfg = r.model_config._replace(rk3_stage3_geothermal_nemo_written=True)
    with pytest.raises(ValueError, match="tracer lateral diffusion"):
        LatLonCGridOceanModel(r.grid, r.z_coord, cfg,
                              iwm_forcing=getattr(r, "iwm_forcing", None))


def test_gate_refuses_a_card_that_does_not_match_the_arm(cards) -> None:
    # Arms run on the pre-selection card; the shipped card now carries L2.
    with pytest.raises(GateError, match="needs a shipped card"):
        selected_card(cards["rung10"], arm="control", rung="rung10")
    assert selected_card(cards["rung10"], arm="D122", rung="rung10") is cards["rung10"]
    unselected = cards["rung10"]._replace(recipe=cards["rung10"].recipe._replace(
        model_config=cards["rung10"].recipe.model_config._replace(gm_redi=None)))
    with pytest.raises(GateError, match="needs a shipped card"):
        selected_card(unselected, arm="D122", rung="rung10")
