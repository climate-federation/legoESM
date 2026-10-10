from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round225_nonosc_fold_walk as gate,
)


def _report():
    tracer = {
        "active_fold_support": 4,
        "nonnorth_member_unequal": 0,
        "north_member_unequal": 2,
        "off_fold_bound_unequal": 0,
        "zup": {"unequal": 2, "maximum_absolute": 1.0,
                 "literal_reproduced": True},
        "zdo": {"unequal": 1, "maximum_absolute": 1.0,
                 "literal_reproduced": True},
        "guards": {"guarded_off": 1, "guarded_result_finite": True,
                   "unguarded_nonfinite": 1},
        "v_coefficient": {"support": 3, "unequal": 2,
                          "maximum_absolute": 0.5,
                          "selected_pair_reproduced": True,
                          "off_fold_unequal": 0},
    }
    return {
        "execution": "offline-pure-jit-cpu-fp64-libm",
        "record_scope": "admitted OMT-4 kt=1 completed states",
        "in_executable_observers": 0,
        "source_order": ["zup", "zdo", "guards", "coef_v"],
        "tracers": {"T": copy.deepcopy(tracer), "S": copy.deepcopy(tracer)},
        "first_nonbit_statement": "zup",
        "statement_sufficiency": "UNMEASURED_WITH_SPEC",
    }


def test_round225_classifies_source_ordered_nonosc_fold():
    result = gate.classify(_report())
    assert result["status"] == "PASS_R225_FIRST_NONBIT_NONOSC_FOLD"
    assert result["predictions"]["R225-P1"] == "CONFIRMED"
    assert result["predictions"]["R225-P4"] == "UNMEASURED_WITH_SPEC"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round225_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)
