from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round253_kt8_boundary_gate as gate,
)


def _report(label: str) -> dict[str, object]:
    first = {
        "substep": 3, "boundary": "eta_mid", "prior_boundary_valid": True,
        "invalid": 1, "first": {"index": [1, 49, 0]},
    }
    return {
        "status": "PASS_R253_MEASUREMENT",
        "label": label,
        "operand_route": "production-d114-resolved",
        "passivity": {"T": True, "S": True, "u": True, "v": True},
        "completed_kt": 7,
        "kt8_entry_guard": {"invalid": 0},
        "terminal_error": gate.EXPECTED_ERROR,
        "nemo_guard": {"checked": 40, "invalid_total": 0, "minimum": 9.0},
        "source_order": list(gate.ETA_BOUNDARIES),
        "first_invalid": first,
        "first_growth": {"kt": 7, "stage": 1, "field": "T", "growth": 30.0},
        "one_bit_control": {"unequal": 1, "max_abs": 2.0e-16},
        "compiled_source": {"midpoint": "dynspg_ts.f90:493"},
    }


def test_classifier_names_midpoint_writer_and_retains_refutation():
    result = gate.classify([_report(label) for label in gate.LABELS])
    assert result["status"] == "HELD_R253_FIRST_INVALID_WRITER_NAMED"
    assert result["first_invalid_writer"]["boundary"] == "eta_mid"
    assert result["first_invalid_writer"]["prediction"] == (
        "REFUTED_CONTINUITY_PREDICTION")


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_classifier_plants_fail_closed(plant):
    reports = copy.deepcopy([_report(label) for label in gate.LABELS])
    with pytest.raises(gate.GateError):
        gate.classify(reports, plant=plant)


def test_one_bit_control_is_non_vacuous():
    control = gate._one_bit_control()
    assert control["unequal"] == 1
    assert control["max_abs"] > 0.0
