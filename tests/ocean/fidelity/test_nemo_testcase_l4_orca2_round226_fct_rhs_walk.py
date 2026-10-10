from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round226_fct_rhs_walk as gate,
)


def _report():
    row = {
        "literal_vs_generic": {"unequal": 3, "rms": 2.0, "maximum_absolute": 3.0},
        "generic_vs_oracle": {"unequal": 4, "rms": 4.0, "maximum_absolute": 5.0},
        "literal_vs_oracle": {"unequal": 2, "rms": 1.0, "maximum_absolute": 2.0},
    }
    return {
        "execution": "offline-pure-jit-cpu-fp64-libm",
        "record_scope": "admitted OMT-4 kt=1 stage-3 state and rank-0 RKTR3 record",
        "in_executable_observers": 0,
        "record_alignment": {"matched_half": "south", "Kmm_unequal": 0},
        "tracers": {"T": copy.deepcopy(row), "S": copy.deepcopy(row)},
        "stage_consumer": "LIVE_WITH_SPEC",
        "statement_sufficiency": "UNMEASURED_WITH_SPEC",
    }


def test_round226_classifies_final_rhs_boundary():
    result = gate.classify(_report())
    assert result["status"] == "PASS_R226_FIRST_NONBIT_FINAL_FCT_RHS"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round226_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)
