from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round78_uamid_walk.py"
)
SPEC = importlib.util.spec_from_file_location("round78_uamid_walk", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _row(exact: bool, cells: int = 0) -> dict:
    return {
        "bit_exact": exact,
        "differing_cells": cells,
        "wet_cells": 4,
        "absolute_max": float(not exact),
        "reference_max_abs": 1.0,
    }


def test_source_order_stops_at_first_midpoint_input() -> None:
    exact = _row(True)
    rows = [
        {
            "substep": 1,
            "coefficient_1": exact,
            "coefficient_2": exact,
            "coefficient_3": exact,
            "un_e": _row(False, 2),
            "ub_e": _row(False, 3),
            "ubb_e": _row(False, 4),
            "ua_e": _row(False, 4),
        }
    ]
    first = GATE.first_live_boundary(rows)
    assert first["substep"] == 1
    assert first["boundary"] == "un_e"
    assert first["differing_cells"] == 2


def test_null_un_e_plant_moves_boundary() -> None:
    exact = _row(True)
    rows = [
        {
            "substep": 1,
            "coefficient_1": exact,
            "coefficient_2": exact,
            "coefficient_3": exact,
            "un_e": exact,
            "ub_e": exact,
            "ubb_e": exact,
            "ua_e": _row(False, 4),
        }
    ]
    first = GATE.first_live_boundary(rows)
    assert first["boundary"] == "ua_e"


def test_comparison_distinguishes_signed_zero() -> None:
    row = GATE.comparison(
        np.array([0.0], dtype=np.float64),
        np.array([-0.0], dtype=np.float64),
        np.array([True]),
    )
    assert not row["bit_exact"]
    assert row["differing_cells"] == 1


def test_array_digest_distinguishes_one_ulp() -> None:
    left = np.array([1.0], dtype=np.float64)
    right = np.nextafter(left, np.inf)
    assert GATE._array_digest(left)["sha256"] != GATE._array_digest(right)["sha256"]
