"""Controls for the round-136 passive tracer-ZDF walk."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round136_step36_zdf_gate as gate,
)


def _report():
    rows = {
        name: {
            "shape": [180, 182, 31],
            "nonfinite": 0,
            "first_nonfinite": None,
            "finite_max_abs": 1.0,
            "target_value": "0.0",
            "target_nonfinite": 0,
        }
        for name in gate.ROW_ORDER
    }
    rows["forward"].update(
        nonfinite=1, first_nonfinite=[86, 159, 0],
        target_value="nan", target_nonfinite=1)
    rows["solved_T"].update(
        nonfinite=1, first_nonfinite=[86, 159, 0],
        target_value="nan", target_nonfinite=1)
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "row_order": list(gate.ROW_ORDER),
        "rows": rows,
        "first_target_nonfinite_row": "forward",
        "returned_first_nonfinite": {
            "field": "T", "index": list(gate.TARGET), "value": "nan",
        },
        "ordinary_repeat_state_equal": {
            name: True for name in gate.prior.FIELDS
        },
        "observer_state_equal": {
            name: True for name in gate.prior.FIELDS
        },
    }


def test_a_valid_report_names_the_forward_recurrence():
    admitted = gate.classify(_report())
    assert admitted["status"] == "PASS_ROUND136_STEP36_ZDF_WALK"
    assert admitted["prediction_ledger"] == {
        "R136-P3": {"status": "CONFIRMED", "observed": "forward"},
        "R136-P4": {"status": "CONFIRMED", "observed": "forward"},
        "R136-P5": {
            "status": "CONFIRMED",
            "observed": "all ordinary state leaves bit-identical",
        },
    }


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_planted_violation_refuses(plant):
    report = copy.deepcopy(_report())
    with pytest.raises(gate.GateError):
        gate.classify(report, plant=plant)
