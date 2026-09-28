"""Unit controls for the Round-86 ZAD operand walk."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases"
    / "nemo_testcase_l2_gyre_round86_zad_operands.py"
)
SPEC = importlib.util.spec_from_file_location("round86_zad_operands", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ROUND86 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUND86)


def test_owned_injection_preserves_halos_and_replaces_payload():
    full = np.zeros((6, 7, 3), dtype=np.float64)
    owned = np.ones((2, 3, 2), dtype=np.float64)
    changed = ROUND86._inject_owned(full, owned, nlev=2)
    assert np.array_equal(changed[2:-2, 2:-2, :2], owned)
    assert np.count_nonzero(changed) == owned.size


def test_operand_order_selects_first_nonbit():
    rows = {name: {"bit_exact": True} for name in ROUND86.OPERAND_ORDER}
    rows["r3u"]["bit_exact"] = False
    rows["thickness_u"]["bit_exact"] = False
    assert ROUND86._first_nonbit(rows) == "r3u"


def test_one_ulp_control_moves_maximum_residual():
    candidate = np.asarray([1.0, 2.0], dtype=np.float64)
    reference = np.asarray([1.0, 1.5], dtype=np.float64)
    changed, at = ROUND86._away_one_ulp(
        reference, candidate, np.ones(2, dtype=bool))
    assert at == (1,)
    assert abs(candidate[1] - changed[1]) > abs(candidate[1] - reference[1])
