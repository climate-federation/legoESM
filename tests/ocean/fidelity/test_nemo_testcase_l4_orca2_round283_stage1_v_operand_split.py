from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round283_stage1_v_operand_split as gate,
)


def _report():
    row = {
        "bit_exact": True, "unequal": 0, "count": 1,
        "max_abs": 0.0, "rms": 0.0, "argmax": [0],
    }
    changed = {**row, "bit_exact": False, "unequal": 1, "max_abs": 1.0,
               "rms": 1.0}
    return {
        "format": "nemo-testcase-l4-orca2-round283-stage1-v-split-v1",
        "status": "PASS_R283_MEASUREMENT",
        "claim_label": "independent_omt4",
        "admission": {"status": "PASS_R280_OMT4_STAGE1_ADMISSION"},
        "source_order": list(gate.OPERAND_ORDER),
        "compiled_source": {"rDt": "rDt:136", "update": "update:477-480"},
        "passivity": {"T": True, "S": True, "u": True, "v": True, "eta": True},
        "self_replay": copy.deepcopy(row),
        "one_bit_control": copy.deepcopy(changed),
        "operand_rows": {
            "entry_v": copy.deepcopy(row),
            "rhs_v": copy.deepcopy(changed),
            "rDt": copy.deepcopy(row),
            "vmask": copy.deepcopy(row),
            "written_order": copy.deepcopy(row),
        },
        "prefix_rows": {
            "candidate": {**changed, "rms": 2.0},
            "entry_v": {**changed, "rms": 2.0},
            "rhs_v": {**changed, "rms": 1.0},
            "rDt": {**changed, "rms": 1.0},
            "vmask": copy.deepcopy(row),
        },
    }


def test_round283_classifies_rhs_as_first_valued_operand():
    result = gate.classify(_report())
    assert result["first_valued_operand"] == "rhs_v"
    assert result["prediction_ledger"]["R283-P3"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round283_every_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)
