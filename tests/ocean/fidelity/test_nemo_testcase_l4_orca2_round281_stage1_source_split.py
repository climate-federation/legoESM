from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round281_stage1_source_split as gate,
)


def _score(unequal: int, maximum: float) -> dict[str, object]:
    return {
        "bit_exact": unequal == 0,
        "unequal": unequal,
        "count": 5400,
        "max_abs": maximum,
        "rms": maximum,
        "argmax": [gate.SOURCE_ROW, 0, 0],
    }


def _report() -> dict[str, object]:
    exact = _score(0, 0.0)
    debt = _score(1440, 0.016207369589460653)
    boundaries = {
        name: {"active": copy.deepcopy(exact), "source_row": copy.deepcopy(exact)}
        for name in gate.SOURCE_ORDER
    }
    boundaries["final_v"] = {
        "active": copy.deepcopy(debt), "source_row": copy.deepcopy(debt)}
    return {
        "claim_label": "independent_omt4",
        "admission": {"status": "PASS_R280_OMT4_STAGE1_ADMISSION"},
        "source_order": list(gate.SOURCE_ORDER),
        "source_row": gate.SOURCE_ROW,
        "program_points": {
            "pre_v": "after vector update, before barotropic correction",
            "cor_v": "after barotropic correction, before lbc_lnk",
            "final_v": "after completed-stage lbc_lnk in NEMO",
        },
        "boundaries": boundaries,
        "round279_source_row": copy.deepcopy(debt),
        "controls": {
            name: {"unequal": 1}
            for name in ("pre_v", "cor_v", "final_v", "one_bit")
        },
    }


def test_exact_pre_and_correction_select_the_post_stage_statement() -> None:
    result = gate.classify(_report())
    assert result["status"] == "HELD_FIRST_NONBIT_POST_STAGE_V_ASSOCIATION"
    assert result["prediction_ledger"] == {
        "R281-P1": "CONFIRMED",
        "R281-P2": "CONFIRMED",
        "R281-P3": "REFUTED",
        "R281-P4": "CONFIRMED",
        "R281-P5": "NOT_REACHED_POST_STAGE_OWNER",
    }


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant)
