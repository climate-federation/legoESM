"""Fail-closed unit controls for the round-194 ZAD operand gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_round194_zad_operands.py"
SPEC = importlib.util.spec_from_file_location("round194_zad_operands", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_row_detects_one_ulp_in_an_active_cell():
    reference = np.ones((2, 3, 4), dtype=np.float64)
    candidate = reference.copy()
    candidate[1, 1, 2] = np.nextafter(candidate[1, 1, 2], np.inf)
    row = gate._row("plant", reference, candidate, np.ones_like(reference, bool))
    assert row["cells_unequal"] == 1
    assert row["max_abs"] > 0.0
    assert not row["bit_exact"]


def test_row_ignores_an_inactive_cell_but_not_an_active_one():
    reference = np.zeros((2, 2), dtype=np.float64)
    candidate = reference.copy()
    candidate[0, 0] = 1.0
    mask = np.array([[False, True], [True, True]])
    assert gate._row("masked", reference, candidate, mask)["bit_exact"]
    candidate[1, 1] = 1.0
    assert not gate._row("active", reference, candidate, mask)["bit_exact"]


def test_operand_order_is_compiled_call_order():
    assert gate.OPERAND_ORDER == ("velocity_u", "velocity_v", "w", "h_u", "h_v")
