"""Controls for the round-56 OVERFLOW UP3 boundary gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l1_overflow_round56_up3_boundary_gate as gate  # noqa: E402


def _case() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # Stored records are Fortran (x,y,z); the helper maps them to (y,x,z).
    shape = (4, 3, 2)
    before = np.zeros(shape, dtype=np.float64, order="F")
    after = before.copy(order="F")
    after[1, 1, 0] = 1.0
    stage = {
        "after_vor_u": before,
        "after_adv_u": after,
        "after_vor_v": before.copy(order="F"),
        "after_adv_v": before.copy(order="F"),
    }
    masks = {
        "u": np.ones((3, 4, 2), dtype=bool),
        "v": np.zeros((3, 4, 2), dtype=bool),
    }
    return stage, masks


def test_boundary_counts_real_active_u_change_and_empty_v_domain():
    stage, masks = _case()
    u_row, v_row = gate._boundary_rows(stage, masks, plant=False)
    assert u_row["n_unequal"] == 1
    assert u_row["baseline_n_unequal"] == 1
    assert v_row["status"] == "UNMEASURED_NO_ACTIVE_FACE"


def test_endpoint_plant_adds_exactly_one_active_u_refusal():
    stage, masks = _case()
    u_row, _ = gate._boundary_rows(stage, masks, plant=True)
    assert u_row["baseline_n_unequal"] == 1
    assert u_row["n_unequal"] == 2
    assert u_row["plant_at"] is not None

