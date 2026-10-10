from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round224_tracer_fold_walk as gate,
)


def _report():
    centred_row = {
        "literal_bit_exact": True,
        "wall_control_unequal": 3,
        "wall_control_maximum_absolute": 1.0,
    }
    return {
        "execution": "offline-pure-jit-cpu-fp64-libm",
        "record_scope": "admitted OMT-4 kt=1 completed states",
        "in_executable_observers": 0,
        "centred_stages": {
            stage: {tracer: copy.deepcopy(centred_row)
                    for tracer in ("T", "S")}
            for stage in ("stage1", "stage2")
        },
        "stage3": {
            "active_fold_support": 5,
            "first_moved_field": "first_v_raw",
            "off_fold_unequal": 0,
            "first_u_raw": {"bit_exact": True, "unequal": 0},
            "first_v_raw": {"bit_exact": False, "unequal": 5},
            "first_w_raw": {"bit_exact": True, "unequal": 0},
            "average_v_raw": {"bit_exact": False, "unequal": 5},
        },
        "statement_sufficiency": "UNMEASURED_WITH_SPEC",
    }


def test_clean_classification_names_first_statement():
    result = gate.classify(_report())
    assert result["status"] == "PASS_R224_FIRST_STATEMENT_DONOR_V_FOLD"
    assert result["predictions"]["R224-P4"] == "UNMEASURED_WITH_SPEC"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_planted_violation_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)
