"""Round 318 (Decision 122): the north-fold set on the ORCA2 cards.

G9's fold/transport statements, G6's raw pivot V mask, vector fold unit and
top-V exchange, and G8's fold rows.  FCT images only where FCT runs, fold
rows only where lateral diffusion runs; OMT-0 states the set off; G9's
stage-3 Krhs order stays off everywhere."""

from __future__ import annotations

import json
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
    classify_later_first,
    _get,
    _set,
    selected_card,
)

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round323_l8_l3_l5_d122_gate import without_selection  # noqa: E501
DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
ALL = ("rung10", "rung0", "omt0", "omt1", "omt2", "omt3", "omt4")


@pytest.fixture(scope="module")
def cards():
    if not DECK.is_dir():
        pytest.skip("ORCA2 input deck is unavailable")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    return {
        # Round 322's card; round 323's L8 route is tested in its own file.
        "rung10": without_selection(build_orca2_zps_card(DECK)),
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


EVIDENCE = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
                "round318/arms2")
T1, S1, U1 = ({"kt": 1, "checkpoint": "stage1", "field": f} for f in "TSu")


def _classify(tmp_path, edit=None):
    """Classify the real OMT-1 control vs ALLN records, optionally planted."""
    paths = [EVIDENCE / f"omt1_independent_{arm}.json" for arm in ("control", "ALLN")]
    if not all(p.is_file() for p in paths):
        pytest.skip("round-318 OMT-1 evidence is unavailable")
    candidate = json.loads(paths[1].read_text(encoding="utf-8"))
    rows = {(r["kt"], r["checkpoint"], r["field"]): r
            for r in candidate["ladder"]["rows"]}
    if edit:
        edit(rows, candidate["ladder"])
    planted = tmp_path / "candidate.json"
    planted.write_text(json.dumps(candidate), encoding="utf-8")
    return classify_later_first(paths[0], planted, plant="none")


def _worsen(row):
    row.update(bit_identical=False, unequal=row["unequal"] + 1,
               rms=float(row["rms"]) + 1.0, max_abs=float(row["max_abs"]) + 1.0)


def test_omt1_later_first_row_admitted(tmp_path) -> None:
    out = _classify(tmp_path)
    assert (out["first_non_bit_before"], out["first_non_bit_after"]) == (T1, U1)
    assert out["decision96_eligible"] and not out["exact_row_losses"]
    assert out["first_over_bar"]["target"] == [1, "stage1", "u"]


def test_old_first_row_worsened_refused(tmp_path) -> None:
    # The marker still claims u; the rows say T is the first and got worse.
    with pytest.raises(GateError, match="marker"):
        _classify(tmp_path, lambda rows, _: _worsen(rows[(1, "stage1", "T")]))


def test_earlier_row_worsened_refused(tmp_path) -> None:
    # Entry T (non-exact by signed zeros, not a first-row candidate on OMT)
    # got worse before u: no exact row is lost, so only the new rule fires.
    with pytest.raises(GateError, match="got worse"):
        _classify(tmp_path, lambda rows, _: _worsen(rows[(1, "entry", "T")]))


def test_new_first_row_away_not_eligible(tmp_path) -> None:
    def edit(rows, ladder):
        _worsen(rows[(1, "stage1", "S")])
        ladder["first_non_bit_checkpoint"] = S1

    out = _classify(tmp_path, edit)
    assert out["first_over_bar"]["rms_direction"] == "away"
    assert not out["decision96_eligible"]


def test_earlier_first_row_refused(tmp_path) -> None:
    paths = [EVIDENCE / f"omt1_independent_{arm}.json" for arm in ("ALLN", "control")]
    if not all(p.is_file() for p in paths):
        pytest.skip("round-318 OMT-1 evidence is unavailable")
    with pytest.raises(GateError, match="earlier"):
        classify_later_first(*paths, plant="none")
