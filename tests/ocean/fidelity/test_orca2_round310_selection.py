"""Round 310 (Decision 122): G8's stored F area and stage-3 r3f on the ORCA2
cards whose deck runs dynldf_lev; its fold rows stay off.  Round 311 adds the
stage-1 order on the same cards (its selection test checks the shipped arm)."""

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
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round310_g8_d122_gate import (  # noqa: E501
    STATEMENTS,
    GateError,
    selected_card,
)

DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
LDF_ON = ("rung10", "rung0", "omt3", "omt4")
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
def test_g8_selection(cards, rung) -> None:
    cfg = cards[rung].recipe.model_config
    on = rung in LDF_ON
    assert (cfg.lateral_viscosity.A_h > 0.0) is on
    assert cfg.nemo_ldf_stored_f_area is on
    assert cfg.nemo_ldf_stage_r3f is on
    assert cfg.nemo_ldf_fold_rows is False
    assert cfg.rk3_stage1_ldf_source_order is on


def test_omt_validators_accept_the_selection(cards) -> None:
    assert omt0.validate_omt0_card(cards["omt0"])["ldf_only_statements_off"]
    omt3.validate_omt3_card(DECK, cards["omt3"])


def _step_jaxpr(card, cfg) -> str:
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    r = card.recipe
    freshwater, surface = omt1.rung0_ladder._zero_forcing((148, 180))
    model = LatLonCGridOceanModel(r.grid, r.z_coord, cfg)
    return str(jax.make_jaxpr(lambda s: model.step(
        s, card.dt_s, freshwater=freshwater, surface_forcing=surface))(
            r.initial_state))


def test_each_selected_statement_is_consumed(cards) -> None:
    card = cards["rung0"]
    cfg = card.recipe.model_config
    shipped = _step_jaxpr(card, cfg)
    for name in STATEMENTS["SEL"]:
        assert _step_jaxpr(card, cfg._replace(**{name: False})) != shipped, name


def test_retained_selector_plant_fires(cards) -> None:
    with pytest.raises(GateError):
        selected_card(cards["rung0"], arm="control", rung="rung0",
                      plant="retained-selector")
