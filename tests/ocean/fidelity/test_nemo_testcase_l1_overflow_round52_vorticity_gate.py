"""Unit controls for the round-52 OVERFLOW vorticity gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l1_overflow_round52_vorticity_gate as gate  # noqa: E402


def _record(after_hpg: np.ndarray, after_vor: np.ndarray) -> dict:
    return {
        2: {
            "after_hpg_u": after_hpg.transpose(1, 0, 2),
            "after_vor_u": after_vor.transpose(1, 0, 2),
            "after_hpg_v": np.zeros((3, 2, 2), dtype=np.float64),
            "after_vor_v": np.zeros((3, 2, 2), dtype=np.float64),
        }
    }


def test_zero_addend_replay_is_exact_and_v_domain_is_explicitly_unmeasured():
    values = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
    masks = {
        "u": np.ones_like(values, dtype=bool),
        "v": np.zeros_like(values, dtype=bool),
    }
    rows = gate._vorticity_rows(_record(values, values), masks, plant=False)
    assert rows[0]["status"] == "BIT_EXACT" and rows[0]["n_unequal"] == 0
    assert rows[1]["status"] == "UNMEASURED_NO_ACTIVE_FACE"


def test_active_u_one_ulp_plant_fires_exactly_once():
    values = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
    masks = {
        "u": np.ones_like(values, dtype=bool),
        "v": np.zeros_like(values, dtype=bool),
    }
    rows = gate._vorticity_rows(_record(values, values), masks, plant=True)
    assert rows[0]["status"] == "NON_BIT"
    assert rows[0]["n_unequal"] == 1


def test_plant_adds_one_refusal_when_baseline_has_signed_zero_debt():
    before = np.zeros((2, 3, 2), dtype=np.float64)
    after = before.copy()
    after[0, 0, 0] = -0.0
    masks = {
        "u": np.ones_like(before, dtype=bool),
        "v": np.zeros_like(before, dtype=bool),
    }
    clean = gate._vorticity_rows(_record(before, after), masks, plant=False)[0]
    planted = gate._vorticity_rows(_record(before, after), masks, plant=True)[0]
    assert clean["n_unequal"] == 1
    assert clean["signed_zero_unequal"] == 1
    assert planted["n_unequal"] == clean["n_unequal"] + 1


def test_nonzero_vorticity_delta_refuses_exact_bar():
    before = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
    after = before.copy()
    after[0, 0, 0] = np.nextafter(after[0, 0, 0], np.inf)
    masks = {
        "u": np.ones_like(before, dtype=bool),
        "v": np.zeros_like(before, dtype=bool),
    }
    rows = gate._vorticity_rows(_record(before, after), masks, plant=False)
    assert rows[0]["status"] == "NON_BIT"
    assert rows[0]["n_unequal"] == 1
