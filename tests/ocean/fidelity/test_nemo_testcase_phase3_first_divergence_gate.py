"""Direct fail-closed tests for the phase-3 stage diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


GATE_PATH = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_phase3_first_divergence_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_phase3_first_divergence_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _stage(stage, kaa):
    return {"kt": 1, "stage": stage, "Kaa": kaa}


def _transport(stage, kmm):
    return {"kt": 1, "stage": stage, "Kmm": kmm}


def test_time_level_registry_accepts_only_the_instrumented_ladder():
    stages = {stage: _stage(stage, level) for stage, level in ((1, 3), (2, 2), (3, 3))}
    transports = {
        stage: _transport(stage, level)
        for stage, level in ((1, 1), (2, 3), (3, 2))
    }
    rows = gate.validate_registry(stages, transports)
    assert [row["status"] for row in rows] == ["VERIFIED"] * 3


def test_planted_time_level_violation_turns_registry_red():
    stages = {stage: _stage(stage, level) for stage, level in ((1, 3), (2, 2), (3, 3))}
    transports = {
        stage: _transport(stage, level)
        for stage, level in ((1, 1), (2, 3), (3, 2))
    }
    rows = gate.validate_registry(stages, transports, plant=True)
    assert rows[0]["status"] == "DEBT"


def test_planted_rhs_violation_turns_numerical_row_red():
    values = np.array([1.0, 2.0], dtype=np.float64)
    mask = np.ones(2, dtype=bool)
    assert gate.score("rhs", values, values.copy(), mask)["status"] == "AT-BAR"
    planted = gate.score("rhs", values, values.copy(), mask, plant=True)
    assert planted["status"] == "DEBT"
    assert planted["normalized_max_abs"] > gate.BAR


def test_thickness_weighted_mean_does_not_silently_become_level_mean():
    values = np.array([[0.0, 2.0]], dtype=np.float64)
    stretched = np.array([[1.0, 3.0]], dtype=np.float64)
    got = gate.thickness_weighted_mean(values, stretched)
    np.testing.assert_array_equal(got, np.array([1.5]))
    assert float(got[0]) != float(np.mean(values, axis=-1)[0])


def test_level_mean_assumption_holds_only_for_uniform_thickness():
    values = np.array([[0.0, 2.0]], dtype=np.float64)
    uniform = np.ones_like(values)
    got = gate.thickness_weighted_mean(values, uniform)
    np.testing.assert_array_equal(got, np.mean(values, axis=-1))


def test_parsers_reject_bad_magic(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"NOT_A_STAGE_____" + b"\0" * 64)
    try:
        gate.read_stage(path)
    except (gate.GateError, UnicodeError):
        pass
    else:
        raise AssertionError("bad stage magic passed")


def test_bt_frame_parser_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad-frame.bin"
    path.write_bytes(b"NOT_A_BT_FRAME__" + b"\0" * 64)
    try:
        gate.read_bt_frames(path)
    except (gate.GateError, UnicodeError):
        pass
    else:
        raise AssertionError("bad barotropic-frame magic passed")


def test_frame_rows_name_both_staggering_and_reduction():
    source = GATE_PATH.read_text()
    assert "instantaneous_prognostic_u" in source
    assert "time_mean_u_transport" in source
    assert source.count('"staggering_and_reduction"') >= 3
