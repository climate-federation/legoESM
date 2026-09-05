"""Direct non-vacuity tests for the phase-3 WS stage sweep gate."""

import importlib.util
from pathlib import Path

import numpy as np


PATH = Path("scripts/validate/ocean_fidelity/testcases/nemo_testcase_phase3_stage_sweep_gate.py")
SPEC = importlib.util.spec_from_file_location("nemo_stage_sweep_gate", PATH)
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GATE)


def test_stage_score_planted_control_goes_red():
    oracle = np.zeros((2, 3), dtype=np.float64)
    mask = np.ones_like(oracle, dtype=bool)
    clean = GATE.score("clean", oracle, oracle.copy(), mask)
    planted = GATE.score("plant", oracle, oracle.copy(), mask, plant=True)
    assert clean["status"] == "AT-BAR"
    assert clean["unequal"] == 0
    assert clean["max_row_scale_ulp_error"] == 0.0
    assert planted["status"] == "DEBT"
    assert planted["absolute_max"] == 1.0
    assert planted["unequal"] == 1
    assert planted["row_scale_ulp"] == np.spacing(1.0)


def test_owner_label_requires_scale_and_bar_clearance():
    faithful = {"absolute_max": 4.0, "normalized_max_abs": 4.0}
    control = {"absolute_max": 3.0, "normalized_max_abs": 3.0}
    plausible = GATE.classify_arm(faithful, control, {"absolute_max": 1.0}, improving=True)
    assert plausible["classification"] == "PLAUSIBLE_CONTRIBUTOR_NOT_OWNER"
    control["absolute_max"] = 0.0
    control["normalized_max_abs"] = 0.0
    confirmed = GATE.classify_arm(faithful, control, {"absolute_max": 4.0}, improving=True)
    assert confirmed["classification"] == "CONFIRMED"
