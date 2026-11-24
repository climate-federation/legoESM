"""Controls for the round-139 FCT bound-input walk."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round139_bound_input_gate as gate,
)


def _report():
    members = []
    for name, index in zip(
            gate.STENCIL_NAMES, gate.STENCIL_CELLS, strict=True):
        nonfinite = name == "east"
        members.append({
            "name": name,
            "index": list(index),
            "value": "inf" if nonfinite else "1.0",
            "nonfinite": nonfinite,
        })
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "target_cell": list(gate.TARGET_CELL),
        "trace_field_order": list(gate.TRACE_FIELDS),
        "stencil_members": members,
        "first_nonfinite_member": "east",
        "selected_sources": {
            "member": "east",
            "index": list(gate.STENCIL_CELLS[2]),
            "wet": True,
            "pbef": "1.0",
            "paft": "inf",
            "zbup": "inf",
            "pbef_nonfinite": False,
            "paft_nonfinite": True,
            "zbup_nonfinite": True,
            "bound_reproduced": True,
        },
        "zup_link": {
            "stencil_max_nonfinite": True,
            "trace_zup_nonfinite": True,
            "beta_zup_target_equal": True,
        },
        "ordinary_fct_outputs_equal": {
            "horizontal": True, "vertical": True},
        "trace_payload_separate": True,
        "returned_first_nonfinite": {
            "field": "T", "index": [86, 159, 0], "value": "nan"},
        "ordinary_repeat_state_equal": {
            name: True for name in gate.state_gate.FIELDS},
        "observer_state_equal": {
            name: True for name in gate.state_gate.FIELDS},
    }


def test_valid_report_names_east_paft_owner():
    admitted = gate.classify(_report())
    assert admitted["status"] == "PASS_ROUND139_BOUND_INPUT_WALK"
    assert admitted["prediction_ledger"]["R139-P2"]["status"] == "CONFIRMED"
    assert admitted["prediction_ledger"]["R139-P3"]["status"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_planted_violation_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant=plant)
