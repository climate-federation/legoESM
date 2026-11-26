"""Known-answer controls for the round-289 top-V split."""

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round289_topv_split as gate,
)


def _row(unequal=0, maximum=0.0):
    return {"unequal": unequal, "max_abs": maximum, "rms": maximum,
            "bit_exact": unequal == 0, "count": 1, "argmax": [0]}


def _report():
    exact = _row()
    loud = _row(1, 1.0)
    return {
        "status": "PASS_R289_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1",
        "admission": {"status": "PASS_R280_OMT4_STAGE1_ADMISSION"},
        "passivity": {"T": True, "S": True, "u": True, "v": True,
                      "ssh": True},
        "mask_delta": {"unequal": 1319, "rows": [148]},
        "compiled_sources": {"association": "stprk3_stg.f90:812-820"},
        "rows": {
            "nemo_association_replay": copy.deepcopy(exact),
            "candidate_source_vs_nemo": copy.deepcopy(loud),
            "candidate_source_substitution_vs_nemo": copy.deepcopy(exact),
            "candidate_no_mask_vs_current_copy": copy.deepcopy(exact),
            "candidate_plus_sign_vs_nemo": copy.deepcopy(loud),
            "candidate_wrong_row_vs_nemo": copy.deepcopy(loud),
            "candidate_current_copy_vs_nemo": copy.deepcopy(loud),
            "candidate_right_half_only_vs_nemo": copy.deepcopy(loud),
        },
        "one_bit_control": copy.deepcopy(loud),
    }


def test_classifier_names_the_source_row_and_every_plant_fires():
    result = gate.classify(_report())
    assert result["status"] == "HELD_R289_SOURCE_ROW"
    assert result["prediction_ledger"]["R289-P3"] == "REFUTED_SOURCE_ROW_OWNS"
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(_report(), plant=plant)
