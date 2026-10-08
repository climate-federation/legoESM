from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round170_slow_producer_walk as gate,
)


REPORT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round170/slow_producer_walk_final.json"
)


def _report():
    if not REPORT.exists():
        pytest.skip("round-170 measurement report not present")
    return json.loads(REPORT.read_text())


def test_clean_report_passes_and_retains_refutation():
    result = gate.classify(_report())
    assert result["status"] == "PASS_ROUND170_SLOW_PRODUCER_WALK"
    assert result["prediction_dispositions"] == {
        "R170-P1": "CONFIRMED",
        "R170-P2": "CONFIRMED",
        "R170-P3": "REFUTED",
        "R170-P4": "CONFIRMED",
        "R170-P5": "CONFIRMED",
    }


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_all_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant)


def test_first_nonbit_is_rhs_u_and_u_arm_closes():
    report = gate.classify(_report())
    assert report["first_nonbit_operand"]["name"] == "rhs_u"
    assert report["rhs_override_depth"]["u"]["bit_exact"]
    assert report["rhs_override_depth"]["v"]["differing_cells"] == 68


def test_record_identity_is_not_a_candidate_source_row():
    report = gate.classify(_report())
    assert "final_u" not in report["source_order"]
    assert report["recorded_post_drag_wind_final_identity"] == {
        "u": True, "v": True,
    }


def test_failed_prediction_remains_failed_when_reclassified():
    report = copy.deepcopy(_report())
    assert gate.classify(report)["prediction_dispositions"]["R170-P3"] == "REFUTED"
