import copy
import os
import subprocess
import sys

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round165_vertical_boundary_gate as gate,
)


def _report():
    return {
        "unobserved_control": {
            "kt7_stage3_completed": True,
            "kt8_stages12_exposed": True,
            "kt8_stage3_completed": False,
            "error": True,
        },
        "observed_terminal": {
            "kt8_stages12_exposed": True,
            "kt8_stage3_completed": False,
            "error": gate.EXPECTED_ERROR,
        },
        "first_invalid": {
            "matching_boundaries": ["stage2"],
            "source_match_status": "MATCHED_BOUNDARY",
        },
        "boundaries": {"entry": {"all_finite_positive": True}},
    }


def test_clean_report_passes():
    assert gate.classify(_report())["status"] == (
        "PASS_ROUND165_VERTICAL_BOUNDARY")


def test_falsified_source_prediction_is_retained():
    report = _report()
    report["first_invalid"] = {
        "matching_boundaries": [],
        "source_match_status": "UNMATCHED_REQUIRES_NEXT_WALK",
    }
    assert gate.classify(report)["status"] == (
        "PASS_ROUND165_VERTICAL_BOUNDARY")


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_stage_observer_is_host_side_before_final_model():
    source = gate.Path(gate.ladder.__file__).read_text()
    host_observer = source.index("stage_observer(kt, state, stage_states)")
    final_call = source.index("state_after = jax.device_get(final_model.step(")
    assert host_observer < final_call


def test_gate_imports_under_campaign_pythonpath():
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join((
        "packages/core", "packages/ocean", "packages/atmosphere",
        "packages/coupler", "packages/ice", "packages/land", "packages/ml",
        "packages/tools", "src",
    ))
    result = subprocess.run(
        [sys.executable, str(gate.Path(gate.__file__)), "--help"],
        cwd=gate.REPO_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
