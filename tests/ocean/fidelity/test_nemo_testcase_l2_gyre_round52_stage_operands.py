"""Controls for the Round-52 live stage-operand gate."""

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l2_gyre_round52_stage_operands as gate  # noqa: E402


def test_cell_plant_changes_one_exact_active_value():
    values = np.arange(6.0).reshape(2, 3)
    mask = np.ones_like(values, dtype=bool)
    clean = gate._row("clean", values, values, mask)
    planted = gate._row("planted", values, values, mask, plant=True)
    assert clean["n_unequal"] == 0
    assert planted["n_unequal"] == 1
    assert planted["absolute_max"] == 1.0


def test_frozen_exit_thresholds_are_round51_live_values():
    assert gate.OLD_EXIT_MAX == {
        "u": 2.7478404751243857e-12,
        "v": 3.305560306813421e-12,
    }
