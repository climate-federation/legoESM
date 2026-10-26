from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_round73_stage1_transport.py"
)
SPEC = importlib.util.spec_from_file_location("round73_stage1_transport", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def test_comparison_is_bitwise_and_source_order_stops_at_un_adv() -> None:
    active = np.ones((2, 2), dtype=bool)
    oracle = np.ones((2, 2), dtype=np.float64)
    candidate = oracle.copy()
    candidate[0, 0] = np.nextafter(candidate[0, 0], np.inf)
    row = GATE.comparison(candidate, oracle, active)
    assert row["differing_cells"] == 1
    assert GATE.classify_first_u({"un_adv": row}) == "un_adv"


def test_null_un_adv_plant_removes_required_first_boundary() -> None:
    active = np.ones((2, 2), dtype=bool)
    live = np.full((2, 2), 3.0, dtype=np.float64)
    row = GATE.comparison(live, live.copy(), active)
    assert row["bit_exact"]
    assert GATE.classify_first_u({"un_adv": row}) is None


def test_comparison_distinguishes_signed_zero() -> None:
    active = np.ones((1,), dtype=bool)
    row = GATE.comparison(
        np.array([0.0]), np.array([-0.0]), active)
    assert not row["bit_exact"]
