"""Round 309 (Decision 122): OMT-0 keeps only the G6 statements np_LIN_dyn runs;
rung 10 reads EVD's trigger on the step-entry N2; double diffusion stays off."""

from __future__ import annotations

from pathlib import Path

import jax
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
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round309_omt0_v3_d122_gate import (  # noqa: E501
    OMT0_SHIPPED,
    R308_ON,
    GateError,
    selected_card,
)

DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
VECTOR = ["rung10", "rung0", "omt1", "omt2", "omt3", "omt4"]


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


def test_omt0_keeps_only_lin_dyn_statements(cards) -> None:
    cfg = cards["omt0"].recipe.model_config
    assert {k: getattr(cfg, k) for k in R308_ON} == OMT0_SHIPPED
    assert omt0.validate_omt0_card(cards["omt0"])["vector_only_statements_off"]


@pytest.mark.parametrize("rung", VECTOR)
def test_vector_cards_keep_round_308_set(cards, rung) -> None:
    cfg = cards[rung].recipe.model_config
    assert {k: getattr(cfg, k) for k in R308_ON} == R308_ON


@pytest.mark.parametrize("rung", ["rung10", "rung0", "omt0", "omt1", "omt2",
                                  "omt3", "omt4"])
def test_evd_trigger_and_ddm(cards, rung) -> None:
    ph = cards[rung].recipe.model_config.physics
    assert ph.convection.enhanced_diffusion.trigger_step_entry_bn2 is (
        rung == "rung10")
    # Round 313 selects V3's literal double diffusion on rung 10.
    assert ph.vertical_mixing.ddm.enabled is (rung == "rung10")
    assert ph.vertical_mixing.ddm.form == (
        "nemo_literal" if rung == "rung10" else "hybrid")


def test_omt0_step_traces(cards) -> None:
    # Round 308 shipped OMT-0 with the vector Krhs order and it refused at trace.
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    card = cards["omt0"]
    r = card.recipe
    freshwater, surface = omt1.rung0_ladder._zero_forcing((148, 180))
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)
    out = jax.eval_shape(lambda s: model.step(
        s, card.dt_s, freshwater=freshwater, surface_forcing=surface),
        r.initial_state)
    assert out.T.data.shape == r.initial_state.T.data.shape


@pytest.mark.parametrize("rung", ["rung10", "omt0"])
def test_retained_selector_plant_fires(cards, rung) -> None:
    with pytest.raises(GateError):
        selected_card(cards[rung], arm="control", rung=rung,
                      plant="retained-selector")


def test_shipped_arm_checks_the_selection(cards) -> None:
    moved = selected_card(cards["rung10"], arm="EVDSE", rung="rung10")
    assert selected_card(moved, arm="D122", rung="rung10") is moved
    ctrl = selected_card(cards["rung10"], arm="control", rung="rung10")
    with pytest.raises(GateError):
        selected_card(ctrl, arm="D122", rung="rung10")
