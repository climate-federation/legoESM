"""Round 318 (Decision 122): the north-fold set on the ORCA2 cards.

G9's fold/transport statements, G6's raw pivot V mask, vector fold unit and
top-V exchange, and G8's fold rows.  FCT images only where FCT runs, fold
rows only where lateral diffusion runs; OMT-0 states the set off; G9's
stage-3 Krhs order stays off everywhere."""

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
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round318_fold_d122_gate import (  # noqa: E501
    D122_SELECTED,
    STATEMENTS,
    GateError,
    _comparison_admitting_later_first,
    _get,
    _set,
    selected_card,
)

DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
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
def test_fold_set_selection(cards, rung) -> None:
    cfg = cards[rung].recipe.model_config
    want = D122_SELECTED.get(rung, {})
    for key in STATEMENTS["ALL"]:
        assert bool(_get(cfg, key)) is (key in want), (rung, key)
    assert cfg.rk3_stage3_tracer_krhs_nemo_order is False


def test_omt_validators_accept_the_selection(cards) -> None:
    assert omt0.validate_omt0_card(cards["omt0"])["fold_set_off"]
    omt1.validate_omt1_card(cards["omt1"])
    omt3.validate_omt3_card(DECK, cards["omt3"])
    omt4.validate_omt4_card(DECK, cards["omt4"])


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


def test_each_landed_statement_is_consumed(cards) -> None:
    card = cards["rung0"]
    cfg = card.recipe.model_config
    shipped = _step_jaxpr(card, cfg)
    for name in D122_SELECTED["rung0"]:
        assert _step_jaxpr(card, _set(cfg, {name: False})) != shipped, name


def test_retained_selector_plant_fires(cards) -> None:
    with pytest.raises(GateError):
        selected_card(cards["rung0"], arm="control", rung="rung0",
                      plant="retained-selector")


def _doc(first):
    rows = [{"kt": 1, "checkpoint": "stage1", "field": f} for f in ("T", "S", "u")]
    return {"independent": {"rows": rows, "first_non_bit_checkpoint": first}}


def test_later_first_row_admitted_earlier_refused() -> None:
    seen = []
    compare = _comparison_admitting_later_first(
        lambda b, c: seen.append(c) or {})
    t, u = ({"kt": 1, "checkpoint": "stage1", "field": f} for f in ("T", "u"))
    assert compare(_doc(t), _doc(u))["first_non_bit_after"] == u
    assert seen[-1]["independent"]["first_non_bit_checkpoint"] == t
    with pytest.raises(GateError):
        compare(_doc(u), _doc(t))
