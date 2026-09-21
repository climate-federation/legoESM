"""Direct controls for the rung-3.4 written-order moment replay."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_GATE_PATH = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_si3_phase2_rung34_moment_replay.py"
)
_SPEC = importlib.util.spec_from_file_location("nemo_si3_rung34_moment_replay_test", _GATE_PATH)
assert _SPEC and _SPEC.loader
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)
_SHAPE = (6, 6, len(gate.recipe.ICE_RHEO_TRACERS))
_PLANT_VALUE = 1.0e-12


def _stages() -> dict[str, tuple[np.ndarray, tuple[np.ndarray, ...]]]:
    content = np.zeros(_SHAPE, dtype=np.float64)
    moments = tuple(np.zeros(_SHAPE, dtype=np.float64) for _ in range(5))
    return {"after_y": (content.copy(), moments), "after_x": (content.copy(), moments)}


def test_first_operand_scan_binds_to_y_stage_and_nemo_order() -> None:
    baseline = _stages()
    oracle_velocity = _stages()
    changed = list(oracle_velocity["after_y"][1])
    changed[2] = changed[2].copy()
    changed[2][2, 2, 0] = _PLANT_VALUE
    oracle_velocity["after_y"] = (oracle_velocity["after_y"][0], tuple(changed))

    first = gate._first_arm_difference(baseline, oracle_velocity)
    assert first is not None
    assert first["stage"] == "after_y"
    assert first["tracer"] == "v_i"
    assert first["moment"] == "sxx"


def test_arm_summary_and_row_bar_are_derived() -> None:
    oracle = np.asarray([[0.5]], dtype=np.float64)
    at_bar = gate._moment_row("sxa", oracle, oracle + np.finfo(np.float64).eps)
    debt = gate._moment_row("sxxe_l01", oracle, oracle + _PLANT_VALUE)
    summary = gate._arm_summary([at_bar, debt])
    assert at_bar["status"] == "AT-BAR"
    assert debt["status"] == "DEBT"
    assert summary["status"] == "DEBT"
    assert summary["debt_count"] == 1
    assert summary["owner"]["name"] == "sxxe_l01"
    assert summary["byte_exact_count"] == 0
    assert summary["non_bit_exact_count"] == 2


def test_arm_summary_records_zero_over_n_for_byte_exact_row() -> None:
    oracle = np.asarray([[0.5, -0.25]], dtype=np.float64)
    row = gate._moment_row("sxa", oracle, oracle.copy())
    summary = gate._arm_summary([row])
    assert row["bitwise_nonzero_over_n"] == "0 / 2"
    assert summary["byte_exact_count"] == 1
    assert summary["non_bit_exact_count"] == 0


def test_plant_exit_requires_a_jit_over_two_ulp_row() -> None:
    report = {
        "exit_code": 0,
        "plant": {
            "jit_over_two_ulp_count": 4,
            "binding_row": {
                "name": gate.PLANT_BINDING_ROW,
                "status": "DEBT",
                "max_ulp": 3,
            },
            "exit_code": 1,
        },
    }
    assert gate._selected_exit_code(report, plant=False) == 0
    assert gate._selected_exit_code(report, plant=True) == 1
    report["plant"]["binding_row"]["max_ulp"] = 0
    with pytest.raises(gate.MomentReplayError, match=gate.PLANT_BINDING_ROW):
        gate._selected_exit_code(report, plant=True)


def test_named_plant_binding_row_is_stable() -> None:
    assert gate.PLANT_BINDING_ROW == "plant_delta.syye_l01"
