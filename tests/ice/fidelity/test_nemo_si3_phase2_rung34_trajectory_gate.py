"""Direct controls for the SI3 rung-3.4 trajectory and regime gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_GATE_PATH = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_si3_phase2_rung34_trajectory_gate.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "nemo_si3_phase2_rung34_trajectory_gate_direct", _GATE_PATH
)
assert _SPEC and _SPEC.loader
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)
_CASE_DT_S = 30.0  # tests/ICE_RHEO/EXPREF/namelist_cfg:35


def test_closing_row_replays_si3_evp_equations_and_binds() -> None:
    divergence = np.asarray([[-1.0e-12, 2.0e-12]], dtype=np.float64)
    deformation = np.asarray([[3.0e-12, 6.0e-12]], dtype=np.float64)
    row = gate._closing_row(3, divergence, deformation, _CASE_DT_S)
    assert row["max_abs_delta_s_inv"] == 6.0e-12
    assert row["max_closing_net_s_inv"] == 1.5e-12
    assert row["max_opening_s_inv"] == 3.0e-12
    assert row["max_closing_net_area_per_step"] == 4.5e-11
    assert row["source_significant"] is False

    planted = gate._closing_row(3, divergence, deformation * 10.0, _CASE_DT_S)
    assert planted["source_significant"] is True


def test_oracle_zero_fields_are_uninformative_not_at_bar() -> None:
    zero = np.zeros((2, 2), dtype=np.float64)
    row = gate.gate._score(
        "kt1.oa_i", zero, zero.copy(), uninformative_zero=True
    )
    assert row["status"] == "UNINFORMATIVE"

    with pytest.raises(gate.gate.Rung34GateError, match="not oracle-zero"):
        gate.gate._score(
            "kt1.oa_i",
            np.asarray([[1.0]], dtype=np.float64),
            np.asarray([[1.0]], dtype=np.float64),
            uninformative_zero=True,
        )


@pytest.mark.parametrize(
    ("divergence", "deformation", "message"),
    (
        (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            "shape mismatch",
        ),
        (
            np.asarray([[np.inf]], dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            "non-finite",
        ),
    ),
)
def test_closing_row_fails_closed(divergence, deformation, message: str) -> None:
    with pytest.raises(gate.Rung34TrajectoryError, match=message):
        gate._closing_row(1, divergence, deformation, _CASE_DT_S)
