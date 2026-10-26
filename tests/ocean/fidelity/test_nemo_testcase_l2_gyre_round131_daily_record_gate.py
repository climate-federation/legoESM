"""Direct tests for the Round-131 daily NEMO-record admission gate."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


GATE = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
        / "ocean_fidelity" / "testcases"
        / "nemo_testcase_l2_gyre_round131_daily_record_gate.py")
DAILY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners/nemo_seed0")
MONTHLY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/nemo_seed0")
EVIDENCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round131")
RECEIPT = (Path(__file__).resolve().parents[3] / "docs" / "ocean"
           / "fidelity" / "testcases"
           / "nemo_testcases_l2_gyre_round131_daily_nudging_record_receipt.md")


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("round131_record_gate", GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_virtual_record_passes(gate):
    report = gate.validate_inventory(gate.REQUIRED_STEPS)
    assert report["required_count"] == 360
    assert report["observed_count"] == 360
    assert report["missing_steps"] == []
    gate.validate_required_variables(gate.REQUIRED_VARIABLES)


@pytest.mark.parametrize(
    "plant", ["missing-boundary", "required-variable", "monthly-overlap-ulp"])
def test_plants_exit_nonzero_on_complete_virtual_record(plant):
    result = subprocess.run(
        [sys.executable, str(GATE), "--self-check", "--plant", plant],
        capture_output=True, text=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert f"STATUS PLANT-FIRED: {plant}" in result.stdout
    assert f"REFUSE Round-131 {plant}" in result.stderr


def test_missing_boundary_and_variable_are_fail_closed(gate):
    with pytest.raises(gate.GateError, match="missing 1 daily boundaries"):
        gate.validate_inventory(gate.REQUIRED_STEPS[:-1])
    with pytest.raises(gate.GateError, match="dissl"):
        gate.validate_required_variables(gate.REQUIRED_VARIABLES[:-1])
    without_live_velocity = tuple(
        name for name in gate.REQUIRED_VARIABLES if name != "uu_n")
    with pytest.raises(gate.GateError, match="uu_n"):
        gate.validate_required_variables(without_live_velocity)


def test_schema_includes_both_live_barotropic_velocity_slots(gate):
    assert len(gate.REQUIRED_VARIABLES) == 18
    assert {"uu_n", "vv_n"} <= set(gate.REQUIRED_VARIABLES)


@pytest.mark.skipif(not DAILY_ROOT.is_dir() or not MONTHLY_ROOT.is_dir(),
                    reason="Round-131 NEMO roots are not on this machine")
def test_named_record_stops_after_day_30_and_overlap_is_exact(gate):
    report = gate.audit_record(DAILY_ROOT, MONTHLY_ROOT)
    assert report["status"] == "STOPPED_FOR_RECORD"
    assert report["inventory"]["observed_count"] == 30
    assert report["inventory"]["last_observed_step"] == 180
    assert report["inventory"]["missing_count"] == 330
    assert report["inventory"]["missing_days"][0] == 31
    assert report["inventory"]["missing_days"][-1] == 360
    assert report["namelist"]["nn_itend"] == 180
    assert report["namelist"]["nn_stock"] == 6
    assert report["schema"]["checked_file_count"] == 30
    assert report["schema"]["errors"] == []
    overlap = report["monthly_overlap"]
    assert overlap["compared_count"] == 1
    assert overlap["comparisons"]["180"]["bit_identical"] is True


def test_audit_json_shape_is_serializable(gate):
    report = {
        "format": "nemo-testcase-l2-gyre-round131-daily-record-audit-v1",
        "required": list(gate.REQUIRED_STEPS),
        "variables": list(gate.REQUIRED_VARIABLES),
    }
    assert json.loads(json.dumps(report)) == report


@pytest.mark.skipif(not (EVIDENCE / "daily_record_audit.json").is_file(),
                    reason="Round-131 audit is not on this machine")
def test_receipt_headlines_are_bound_to_the_committed_audit():
    report = json.loads((EVIDENCE / "daily_record_audit.json").read_text())
    text = RECEIPT.read_text()
    assert report["status"] == "STOPPED_FOR_RECORD"
    assert report["tool"]["worktree_clean"] is True
    assert report["inventory"]["observed_count"] == 30
    assert report["inventory"]["missing_count"] == 330
    assert report["inventory"]["last_observed_step"] == 180
    assert report["namelist"]["nn_itend"] == 180
    assert report["schema"]["errors"] == []
    comparison = report["monthly_overlap"]["comparisons"]["180"]
    assert comparison["bit_identical"] is True
    assert comparison["daily_file_sha256"] == comparison["monthly_file_sha256"]
    for value in ("30", "330", "180", comparison["daily_file_sha256"],
                  report["tool"]["commit"]):
        assert value in text
