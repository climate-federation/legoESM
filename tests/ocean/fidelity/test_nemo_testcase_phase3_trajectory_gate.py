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


def test_report_selector_inventory_includes_eos_depth_and_resolved_substeps():
    """The trajectory receipt must expose both post-review card corrections."""
    # Full ``run`` coverage lives in the science invocation because constructing
    # the card is the expensive part.  Keep this direct source-inventory guard
    # independent so deleting either provenance field fails on CPU-only CI.
    source = GATE_PATH.read_text()
    assert '"eos_depth": cfg.eos_depth' in source
    assert '"n_barotropic_substeps": cfg.barotropic.n_barotropic_substeps' in source
    assert '"tracer_rk3_transport_time_levels"' in source
    assert '"tracer_fct_low_order_predictor"' in source
    assert '"rk3_ws_momentum_transport_reconcile"' in source


def test_continue_after_first_is_explicit_and_preserves_first_debt():
    source = GATE_PATH.read_text()
    assert "continue_after_first=False" in source
    assert "if first_over_bar is None:" in source
    assert "if not continue_after_first:" in source


def test_uniform_and_zero_rows_are_not_allowed_to_imply_corroboration():
    mask = np.ones(3, dtype=bool)
    salt = gate.mark_uninformative(
        {"status": "AT-BAR"}, "S", 2, np.full(3, 35.0), mask)
    ssh = gate.mark_uninformative(
        {"status": "AT-BAR"}, "ssh", 2, np.zeros(3), mask)
    assert salt["status"] == "UNINFORMATIVE"
    assert "n_unique=1" in salt["reason"]
    assert ssh["status"] == "UNINFORMATIVE"
    assert "identically zero" in ssh["reason"]


def test_uninformative_classifier_never_hides_debt():
    row = gate.mark_uninformative(
        {"status": "DEBT"}, "S", 2, np.full(2, 35.0), np.ones(2, bool))
    assert row["status"] == "DEBT"
