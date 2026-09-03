"""Controls for the partial SI3 rung-3.3 gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_GATE_PATH = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_rung33_gate.py"
)
_SPEC = importlib.util.spec_from_file_location("nemo_si3_phase2_rung33_gate", _GATE_PATH)
assert _SPEC and _SPEC.loader
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)
_AGRID_PATH = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_rung33_agrid_guard.py"
)
_AGRID_SPEC = importlib.util.spec_from_file_location("rung33_agrid_guard", _AGRID_PATH)
assert _AGRID_SPEC and _AGRID_SPEC.loader
agrid_guard = importlib.util.module_from_spec(_AGRID_SPEC)
_AGRID_SPEC.loader.exec_module(agrid_guard)


def test_partial_gate_has_exact_registry_and_loud_debt_boundary():
    report, code = gate.run_gate()
    assert code == 1
    assert report["status"] == "UNMEASURED"
    assert report["numeric_status"] == "DEBT"
    assert len(report["rows"]) == 68
    names = {row["name"] for row in report["debt"]}
    assert "trajectory.post_step_00000001.stress1_i" in names
    assert "trajectory.post_step_00000001.stress2_i" in names
    assert "trajectory.post_step_00000001.stress12_i" in names
    assert report["binding_control"]["status"] == "RED_AS_REQUIRED"
    assert report["unmeasured"]


def test_geometry_plant_goes_red_and_registry_omission_is_fatal():
    report, _ = gate.run_gate(plant_geometry=True)
    row = next(item for item in report["rows"] if item["name"] == "geometry.e1t")
    assert row["status"] == "DEBT"
    with pytest.raises(gate.GateError, match="tripwire registry mismatch"):
        gate._validate_registry(report["rows"][:-1])


def test_existing_a_grid_solver_is_byte_identical_to_preregister_boundary():
    report = agrid_guard.run_guard()
    assert report["byte_identical"] is True
    assert report["before_sha256"] == report["after_sha256"]
