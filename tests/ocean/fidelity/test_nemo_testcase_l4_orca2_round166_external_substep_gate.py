from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round166_external_substep_gate as gate,
)


def _report():
    return {
        "unobserved_terminal": {
            "completed_kt": 7, "kt8_stage3": False,
            "error": gate.EXPECTED_ERROR,
        },
        "observed_terminal": {
            "completed_kt": 7, "kt8_stage3": False,
            "error": gate.EXPECTED_ERROR,
        },
        "kt8_stages12_exposed": [True, True],
        "completed_checkpoint_passivity": [
            {"kt": kt, "bit_exact": True} for kt in range(1, 8)
        ],
        "trace_source_order": [name for name, _ in gate.SOURCE_ORDER],
        "kt8_first_nonfinite": {
            "substep": 8, "boundary": "after_ssh",
        },
        "kt8_record_candidates": [],
    }


def test_clean_report_passes():
    assert gate.classify(_report())["status"] == (
        "PASS_ROUND166_EXTERNAL_SUBSTEP_BOUNDARY")


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_all_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_first_nonfinite_is_substep_then_source_order():
    nrow = len(gate.SOURCE_ORDER)
    counts = [[0, 0] for _ in range(nrow)]
    flats = [[0, 0] for _ in range(nrow)]
    values = [[0.0, 0.0] for _ in range(nrow)]
    earlier_row = next(i for i, row in enumerate(gate.SOURCE_ORDER)
                       if row[0] == "transport_v")
    later_row = next(i for i, row in enumerate(gate.SOURCE_ORDER)
                     if row[0] == "after_ssh")
    counts[later_row][0] = 1
    flats[later_row][0] = 2
    values[later_row][0] = float("nan")
    counts[earlier_row][1] = 1
    frame = {
        "invalid_counts": counts,
        "first_flat_indices": flats,
        "first_invalid_values": values,
        "shapes": {trace: [2, 2] for _, trace in gate.SOURCE_ORDER},
    }
    first = gate._first_nonfinite(frame)
    assert first["substep"] == 1
    assert first["boundary"] == "after_ssh"
    assert first["index"] == [1, 0]


def test_trace_wrapper_returns_ordinary_solver_outputs():
    source = gate.Path(gate.__file__).read_text()
    assert "state_after, averages, trace = original" in source
    assert "return state_after, averages" in source
