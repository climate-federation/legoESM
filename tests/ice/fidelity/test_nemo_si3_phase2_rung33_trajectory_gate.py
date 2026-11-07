"""Direct controls for the full SI3 rung-3.3 trajectory/restart gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_GATE_PATH = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_si3_phase2_rung33_trajectory_gate.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "nemo_si3_phase2_rung33_trajectory_gate_direct", _GATE_PATH
)
assert _SPEC and _SPEC.loader
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


def test_detailed_score_reports_absolute_relative_and_bar_status() -> None:
    oracle = np.asarray([0.024, -0.012], dtype=np.float64)
    exact = gate._detailed_score("velocity.exact", oracle, oracle.copy())
    assert exact["status"] == "AT-BAR"
    assert exact["relative_max_abs"] == 0.0

    candidate = oracle.copy()
    candidate[0] += 2.4e-15
    debt = gate._detailed_score("velocity.plant", oracle, candidate)
    assert debt["status"] == "DEBT"
    assert debt["normalized_max_abs"] < debt["relative_max_abs"]
    assert debt["relative_max_abs"] == pytest.approx(
        debt["max_abs"] / debt["oracle_max_abs"]
    )


@pytest.mark.parametrize(
    ("oracle", "candidate", "message"),
    (
        (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            "shape mismatch",
        ),
        (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float32),
            "not fp64",
        ),
        (
            np.asarray([np.inf], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            "non-finite input",
        ),
    ),
)
def test_detailed_score_fails_closed(oracle, candidate, message: str) -> None:
    with pytest.raises(gate.TrajectoryGateError, match=message):
        gate._detailed_score("control.invalid", oracle, candidate)


def test_shape_census_counts_extrema_and_negative_cells() -> None:
    field = np.zeros((3, 3), dtype=np.float64)
    field[1, 1] = 2.0
    field[0, 0] = -1.0
    census = gate._shape_census(field)
    assert census == {
        "maximum": 2.0,
        "minimum": -1.0,
        "negative_cells": 1,
        "strict_four_neighbor_local_maxima": 1,
        "strict_four_neighbor_local_minima": 1,
    }


def test_main_writes_exact_run_gate_payload_and_uses_derived_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "gate.json"
    expected = {"status": "DEBT", "rows": [{"status": "DEBT"}]}
    monkeypatch.setattr(gate, "run_gate", lambda _root: (expected, 1))
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_GATE_PATH), "--root", str(tmp_path), "--artifact", str(artifact)],
    )
    assert gate.main() == 1
    assert json.loads(artifact.read_text()) == expected
