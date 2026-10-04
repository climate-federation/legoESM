"""Controls for the round-140 upstream-predictor walk."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round140_upstream_predictor_gate as gate,
)


def _report():
    rows = [{"field": name, "nonfinite": int(name == "average_u")}
            for name in gate.SOURCE_FIELDS]
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "target": list(gate.TARGET),
        "source_field_order": list(gate.SOURCE_FIELDS),
        "ordinary_repeat_state_equal": {name: True for name in gate.passive.FIELDS},
        "observer_state_equal": {"T": True, "S": True},
        "returned_first_nonfinite": {
            "field": "T", "index": list(gate.RETURNED_TARGET), "value": "nan"},
        "round139_link": {"cell": list(gate.TARGET),
                          "pbef_nonfinite": False, "paft_nonfinite": True,
                          "paft": "inf"},
        "target_rows": rows,
        "first_target_nonfinite_field": "average_u",
    }


def test_valid_report_names_averaged_flux():
    admitted = gate.classify(_report())
    assert admitted["status"] == "PASS_ROUND140_UPSTREAM_PREDICTOR_WALK"
    assert admitted["prediction_ledger"]["R140-P2"]["status"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_planted_violation_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant=plant)
