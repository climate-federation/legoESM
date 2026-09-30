from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_round76_advmean_walk.py"
)
SPEC = importlib.util.spec_from_file_location("round76_advmean_walk", SCRIPT)
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


def test_source_order_stops_at_metric_transport() -> None:
    exact = _row(True)
    rows = [
        {
            "substep": 1,
            "sum_entry": exact,
            "weight": exact,
            "metric_transport": _row(False, 4),
            "r1_e2u": _row(False, 1),
            "sum_exit": _row(False, 4),
        }
    ]
    first = GATE.first_live_u_boundary(exact, rows)
    assert first["substep"] == 1
    assert first["boundary"] == "metric_transport"
    assert first["differing_cells"] == 4


def test_null_metric_transport_plant_moves_boundary() -> None:
    exact = _row(True)
    rows = [
        {
            "substep": 1,
            "sum_entry": exact,
            "weight": exact,
            "metric_transport": exact,
            "r1_e2u": exact,
            "sum_exit": _row(False, 4),
        }
    ]
    first = GATE.first_live_u_boundary(exact, rows)
    assert first["boundary"] == "sum_exit"


def test_comparison_distinguishes_signed_zero() -> None:
    row = GATE.comparison(
        np.array([0.0], dtype=np.float64),
        np.array([-0.0], dtype=np.float64),
        np.array([True]),
    )
    assert not row["bit_exact"]
    assert row["differing_cells"] == 1


def test_owned_face_restoration_round_trips_trace_mapping() -> None:
    owned_u = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    owned_v = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    assert np.array_equal(GATE._native(GATE._full_u(owned_u), "u"), owned_u)
    assert np.array_equal(GATE._native(GATE._full_v(owned_v), "v"), owned_v)
