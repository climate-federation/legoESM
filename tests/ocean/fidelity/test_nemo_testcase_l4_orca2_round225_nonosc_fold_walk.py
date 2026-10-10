from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round225_nonosc_fold_walk as gate,
)


def _report():
    tracer = {
        "active_fold_support": 4,
        "nonnorth_member_unequal": 3,
        "nonnorth_differences_are_dry_sentinels": True,
        "sentinel_wet_bound_unequal": 0,
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
        "source_order": [
            "dry_bound_sentinel", "zup", "zdo", "guards", "coef_v"],
        "tracers": {"T": copy.deepcopy(tracer), "S": copy.deepcopy(tracer)},
        "first_nonbit_statement": "dry_bound_sentinel",
        "first_effective_statement": "zup",
        "statement_sufficiency": "REFUTED_IDENTICAL_KT8_REFUSAL",
        "sufficiency_evidence": {
            "byte_identical": True,
            "completed_steps": list(range(1, 8)),
            "refusal_step": 8,
        },
    }


def test_round225_classifies_source_ordered_nonosc_fold():
    result = gate.classify(_report())
    assert result["status"] == "PASS_R225_FIRST_NONBIT_NONOSC_FOLD"
    assert result["predictions"]["R225-P1"] == "REFUTED"
    assert result["predictions"]["R225-P4"] == "REFUTED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round225_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)
