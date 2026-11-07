import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round81_month_stability_gate as gate,
)


def _failure_report():
    return {
        "claim_label": "independent",
        "arm": "landed",
        "protocol": {"steps_required": 240, "dt_s": 10800.0,
                     "initial_mode": "card_own_state"},
        "vertical_record_admission": "PASS",
        "steps_completed": 2,
        "failure": {"step": 3, "field": "raw_mesh_e3w_int",
                    "index": [10, 20, 0], "value": -1.0},
        "terminal": None,
    }


def test_failure_report_passes():
    assert gate.classify(_failure_report())["status"] == "PASS_ROUND81_ARM_FAILURE"


@pytest.mark.parametrize(
    "plant", ["arm-label", "protocol", "failure-step", "failure-cell", "early-complete"])
def test_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_failure_report()), plant)


def test_complete_requires_step_240_and_terminal():
    report = _failure_report()
    report["arm"] = "old-backgrounds"
    report["failure"] = None
    report["steps_completed"] = 240
    report["terminal"] = {"comparison": {}}
    assert gate.classify(report)["status"] == "PASS_ROUND81_ARM_COMPLETE"


@pytest.mark.parametrize(
    "plant", ["arm-label", "protocol", "failure-step", "failure-cell", "early-complete"])
def test_plants_fire_on_complete_report(plant):
    report = _failure_report()
    report["arm"] = "old-backgrounds"
    report["failure"] = None
    report["steps_completed"] = 240
    report["terminal"] = {"comparison": {}}
    with pytest.raises(gate.GateError):
        gate.classify(report, plant)
