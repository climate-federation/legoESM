"""Controls for the round-137 level-3 FCT/content walk."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round137_step36_fct_content_gate as gate,
)


def _report():
    groups = {}
    for group, fields in gate.TRACE_GROUPS[:-1]:
        details = {}
        for tracer in ("T", "S"):
            details[tracer] = {
                field: {
                    "target_nonfinite": int(
                        tracer == "T" and field in ("final_div", "rhs_final")),
                }
                for field in fields
            }
        target = {
            tracer: sum(row["target_nonfinite"]
                        for row in details[tracer].values())
            for tracer in ("T", "S")
        }
        support = {field: 10 for field in fields}
        groups[group] = {
            "fields": list(fields), "details": details,
            "support_count": support, "active_count": sum(support.values()),
            "nonfinite": dict(target), "nonfinite_total": sum(target.values()),
            "target_nonfinite": target,
            "target_nonfinite_total": sum(target.values()),
        }
    groups["caller_advection_content"] = {
        "fields": ["caller_content"],
        "details": {"passive_post_step_exposure": True},
        "support_count": {"caller_content": 20}, "active_count": 20,
        "nonfinite": {"T": 1, "S": 0}, "nonfinite_total": 1,
        "target_nonfinite": {"T": 1, "S": 0},
        "target_nonfinite_total": 1,
    }
    return {
        "claim_label": "independent", "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35, "target": list(gate.TARGET),
        "trace_field_order": list(gate.TRACE_FIELD_ORDER),
        "group_order": list(gate.GROUP_ORDER), "groups": groups,
        "first_target_nonfinite_field": "final_div",
        "ordinary_repeat_state_equal": {name: True for name in gate.FIELDS},
        "observer_state_equal": {name: True for name in gate.FIELDS},
        "returned_first_nonfinite": {
            "field": "T", "index": list(gate.RETURNED_TARGET), "value": "nan"},
        "side_output_type": "_NEMOWSFCTInputTrace",
        "side_output_field": "mass_flux_w",
        "stage3_advection_content": {"target_jik": list(gate.TARGET)},
        "round136_pre_zdf": {"first_nonfinite": list(gate.TARGET)},
    }


def test_a_valid_report_names_the_corrected_divergence():
    admitted = gate.classify(_report())
    assert admitted["status"] == "PASS_ROUND137_STEP36_FCT_CONTENT_WALK"
    assert admitted["prediction_ledger"]["R137-P2"] == {
        "status": "CONFIRMED",
        "predicted": "after rhs_after_up",
        "observed": "final_div",
    }


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_planted_violation_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant=plant)
