from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round254_midpoint_operand_gate as midpoint,
    nemo_testcase_l4_orca2_round254_south_association_gate as south,
)


def _score(unequal: int = 0) -> dict[str, object]:
    return {"unequal": unequal, "max_abs": float(bool(unequal))}


def _south_report(label: str) -> dict[str, object]:
    return {
        "status": "PASS_R254_SOUTH_ASSOCIATION_MEASUREMENT",
        "label": label,
        "admission": "PASS_R253_OMT4_SPG8_ADMISSION",
        "operand_route": "production-d114-resolved",
        "source_order": [name for name, *_ in south.FIELDS],
        "south_changed_fields": [],
        "north_changed_cells": 1,
        "south_rows": {
            name: {"on_vs_off_row0": _score()} for name, *_ in south.FIELDS
        },
        "one_bit_control": _score(1),
    }


def _midpoint_report(label: str) -> dict[str, object]:
    rows = {name: _score(1) for name in midpoint.OPERANDS}
    for name in ("za1", "za2", "za3"):
        rows[name] = _score()
    return {
        "status": "PASS_R254_MIDPOINT_OPERAND_SPLIT",
        "label": label,
        "south_hypothesis": "REFUTED_ZERO_SOUTH_CHANGES",
        "operand_route": "production-d114-resolved",
        "source_order": list(midpoint.OPERANDS),
        "candidate_replay": _score(),
        "nemo_replay": _score(),
        "first_nonbit_operand": "sshn_e",
        "one_bit_control": _score(1),
        "operand_rows": rows,
        "candidate_output_vs_nemo": _score(1),
        "one_variable_substitutions": {
            name: _score(1) for name in midpoint.OPERANDS
        },
    }


def test_south_hypothesis_is_refuted_without_disabling_north_fold():
    result = south.classify([_south_report(label) for label in south.LABELS])
    assert result["status"] == "REFUTED_R254_SOUTH_ASSOCIATION_HYPOTHESIS"
    assert result["first_changed_fields"] == []


@pytest.mark.parametrize("plant", south.PLANTS[1:])
def test_south_classifier_plants_fail_closed(plant):
    reports = copy.deepcopy([_south_report(label) for label in south.LABELS])
    with pytest.raises(south.GateError):
        south.classify(reports, plant=plant)


def test_midpoint_classifier_names_current_ssh_history():
    result = midpoint.classify([
        _midpoint_report(label) for label in midpoint.LABELS])
    assert result["status"] == "HELD_R254_MIDPOINT_OPERAND_NAMED"
    assert result["first_nonbit_operand"] == "sshn_e"


@pytest.mark.parametrize("plant", midpoint.PLANTS[1:])
def test_midpoint_classifier_plants_fail_closed(plant):
    reports = copy.deepcopy([
        _midpoint_report(label) for label in midpoint.LABELS])
    with pytest.raises(midpoint.GateError):
        midpoint.classify(reports, plant=plant)
