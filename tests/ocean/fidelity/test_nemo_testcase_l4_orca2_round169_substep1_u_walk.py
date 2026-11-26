from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round169_substep1_u_walk as gate,
)


REPORT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round169/substep1_u_walk.json"
)


def _report():
    if not REPORT.exists():
        pytest.skip("round-169 measurement report not present")
    return json.loads(REPORT.read_text())


def test_clean_report_passes_and_disposes_predictions():
    result = gate.classify(_report())
    assert result["status"] == "PASS_ROUND169_SUBSTEP1_U_WALK"
    assert set(result["prediction_dispositions"]) == {
        "R169-P1", "R169-P2", "R169-P3", "R169-P4", "R169-P5",
    }


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_all_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant)


def test_first_nonbit_is_ordered_not_minimum_magnitude():
    rows = [
        {"name": "first", "bit_exact": False, "finite_absolute_max": 2.0},
        {"name": "second", "bit_exact": False, "finite_absolute_max": 1.0},
    ]
    assert gate._first_nonbit(rows)["name"] == "first"


def test_failed_prediction_is_retained():
    report = copy.deepcopy(_report())
    report["first_nonbit"] = {"name": "pressure_u", "bit_exact": False}
    result = gate.classify(report)
    assert result["prediction_dispositions"]["R169-P1"] == "REFUTED"
