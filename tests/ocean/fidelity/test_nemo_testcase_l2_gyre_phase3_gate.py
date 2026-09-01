"""Non-vacuity and fail-closed tests for the lane-2 GYRE Phase 3 gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

PATH = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_phase3_gate.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_phase3_gate", PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_score_exact_and_planted_violation():
    values = np.array([1.0, 2.0], dtype=np.float64)
    mask = np.ones(2, dtype=bool)
    assert gate.score("exact", values, values.copy(), mask)["status"] == "AT-BAR"
    assert gate.score("plant", values, values.copy(), mask, plant=True)["status"] == "DEBT"


def test_kt1_at_rest_controls_are_uninformative_but_debt_stays_red():
    for field in ("u", "v", "ssh"):
        row = gate._mark_kt1_uninformative({"status": "AT-BAR"}, field, 1)
        assert row["status"] == "UNINFORMATIVE"
        assert "at-rest" in row["reason"]
        assert gate._mark_kt1_uninformative({"status": "DEBT"}, field, 1)["status"] == "DEBT"


def test_one_variable_manifest_control_fires():
    arms = {"omit_stage_barotropic_correction": {"changed_operands": ["one"]}}
    assert gate.validate_one_variable_arms(arms)[0]["status"] == "VERIFIED"
    assert gate.validate_one_variable_arms(arms, plant=True)[0]["status"] == "DEBT"


def test_gate_reuses_registry_and_lane1_growth_instrument():
    source = PATH.read_text()
    assert "time_level_for_dump" in source
    assert 'with_name("nemo_testcase_phase3_trajectory_gate.py")' in source
    assert '"continue_after_first": True' in source
    assert '"UNMEASURED_AFTER_REGISTERED_ARMS"' in source


def test_full_rk3_inventory_and_scaling_precede_owner_labels():
    source = PATH.read_text()
    for token in ("read_stage", "read_transport", "read_rhs", "read_bt"):
        assert token in source
    assert '"scaling_check_before_owner_label": True' in source
    assert '"CONFIRMED_OWNER"' in source
    assert '"PLAUSIBLE_CONTRIBUTOR_NOT_OWNER"' in source
    assert '"REFUTED_AS_PRIMARY_OWNER"' in source
