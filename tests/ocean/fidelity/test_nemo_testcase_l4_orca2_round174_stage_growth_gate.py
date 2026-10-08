import copy
import json

import numpy as np

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round174_stage_growth_gate as gate,
)


def _row(value=0.0):
    return {
        "bit_exact": value == 0.0, "differing_cells": int(value != 0.0),
        "cells": 1, "candidate_nonfinite": 0, "oracle_nonfinite": 0,
        "max_abs": value, "argmax": [0], "candidate_max_abs": value,
        "oracle_max_abs": 0.0,
    }


def _report():
    boundaries = []
    for kt in gate.STEPS:
        for stage in gate.STAGES:
            if (kt, stage) == (8, 3):
                continue
            value = 1.0 if (kt, stage) == (1, 1) else 1.0
            boundaries.append({
                "kt": kt, "stage": stage,
                "rows": {field: _row(value) for field in gate.FIELDS},
            })
    growth = gate._growth_rows(boundaries)
    return {
        "claim_label": "independent", "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "admission": {"record_count": 80},
        "control": {"sha256": gate.CONTROL_SHA256},
        "field_order": list(gate.FIELDS), "floor": float(gate.FLOOR),
        "growth_threshold": float(gate.GROWTH),
        "passivity": {str(kt): {field: True for field in ("T", "S", "u", "v", "ssh")}
                      for kt in range(1, 8)},
        "boundaries": boundaries, "growth": growth,
        "first_growth": gate._first_growth(growth),
        "terminal": {"kt": 8, "stage": 3, "status": "REFUSED",
                     "error": gate.EXPECTED_TERMINAL},
        "one_ulp_control": {"bit_exact": False, "differing_cells": 1},
    }


def test_clean_report_selects_kt1_stage1():
    result = gate.classify(_report())
    assert result["first_growth"]["kt"] == 1
    assert result["first_growth"]["stage"] == 1
    assert result["prediction_ledger"]["R174-P3"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_all_plants_fire(plant):
    report = json.loads(json.dumps(_report(), sort_keys=True))
    with pytest.raises(gate.GateError):
        gate.classify(report, plant)


def test_sorted_json_report_reclassifies_cleanly():
    report = json.loads(json.dumps(_report(), sort_keys=True))
    assert gate.classify(report)["status"] == "PASS_ROUND174_STAGE_GROWTH_BOUNDARY"


def test_later_larger_growth_cannot_displace_first():
    report = _report()
    report["boundaries"][1]["rows"]["T"] = _row(1.0e30)
    report["growth"] = gate._growth_rows(report["boundaries"])
    report["first_growth"] = gate._first_growth(report["growth"])
    result = gate.classify(report)
    assert (result["first_growth"]["kt"], result["first_growth"]["stage"]) == (1, 1)


def test_one_ulp_control_is_mandatory():
    report = _report()
    report["one_ulp_control"]["differing_cells"] = 0
    with pytest.raises(gate.GateError, match="one-ULP"):
        gate.classify(report)


def test_measured_score_is_json_serializable():
    score = gate._score(np.array([[1.0]]), np.array([[0.0]]))
    assert json.loads(json.dumps(score))["argmax"] == [0, 0]


def test_nonfinite_completed_stage_is_reported_as_infinite_growth():
    report = _report()
    for boundary in report["boundaries"][-2:]:
        boundary["rows"]["T"]["candidate_nonfinite"] = 1
    report["growth"] = gate._growth_rows(report["boundaries"])
    report["first_growth"] = gate._first_growth(report["growth"])
    result = gate.classify(report)
    assert result["growth"][-1]["max_abs"] == float("inf")
    assert result["growth"][-1]["ratio"] == 1.0
