"""Direct non-vacuity tests for the phase-3 first-divergence gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


GATE_PATH = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_phase3_trajectory_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_phase3_trajectory_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_score_exact_control_is_at_bar():
    values = np.array([1.0, 2.0], dtype=np.float64)
    row = gate.score("control", values, values.copy(), np.ones(2, bool))
    assert row["exact"] is True
    assert row["status"] == "AT-BAR"


def test_planted_wet_state_violation_turns_row_red():
    values = np.array([1.0, 2.0], dtype=np.float64)
    row = gate.score(
        "control", values, values.copy(), np.ones(2, bool), plant=True)
    assert row["exact"] is False
    assert row["status"] == "DEBT"
    assert row["normalized_max_abs"] > gate.BAR


def test_empty_structural_face_is_loud_unmeasured_but_nonzero_still_red():
    zeros = np.zeros(2, dtype=np.float64)
    empty = np.zeros(2, dtype=bool)
    row = gate.score(
        "v", zeros, zeros, empty, allow_empty_no_active_face=True)
    assert row["status"] == "UNMEASURED"
    nonzero = gate.score(
        "v", zeros, np.array([1.0, 0.0]), empty,
        allow_empty_no_active_face=True)
    assert nonzero["status"] == "DEBT"
