from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round227_implicit_tracer_walk as gate,
)


def _report(first: str | None = "content_T") -> dict[str, object]:
    rows = {
        name: {
            "support": 4,
            "unequal": int(name == first),
            "maximum_absolute": float(name == first),
            "rms": float(name == first),
        }
        for name in gate.ROW_ORDER
    }
    return {
        "execution": "offline-replay-plus-passive-production-trace-cpu-fp64-libm",
        "in_executable_observers": 0,
        "observer_state_equal": {"T": True, "S": True, "eta": True},
        "record_alignment": {"Kmm_unequal": 0},
        "row_order": list(gate.ROW_ORDER),
        "candidate_vs_oracle": rows,
        "first_non_bit_statement": first,
        "oracle_replay_vs_recorded_Kaa": {"unequal": 0},
        "upstream_overlap": {
            "commit": "5e368e87ba",
            "literal_path_executes_upstream_rewrite": False,
            "first_non_bit_row_is_semantically_touched": (
                first in gate.UPSTREAM_ROWS),
        },
    }


def test_classify_keeps_source_order_and_path_scope() -> None:
    result = gate.classify(_report())
    assert result["status"] == "PASS_R227_IMPLICIT_TRACER_SOURCE_WALK"
    assert result["predictions"]["R227-P3"] == "REFUTED"
    assert result["predictions"]["R227-P4"] == "REFUTED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_fires(plant: str) -> None:
    report = _report("lower" if plant == "overlap" else "content_T")
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(report), plant=plant)
