"""Direct controls for the ORCA2 round-19 U-history gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
GATE = (REPO / "scripts/validate/ocean_fidelity/orca2_l4"
        / "nemo_testcase_l4_orca2_round19_u_history_gate.py")


def _gate():
    spec = importlib.util.spec_from_file_location("_r19_u_history_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_record(path: Path, gate, rank: int) -> None:
    full = np.arange(gate.JPI * gate.JPJ, dtype=np.float64)
    interior = np.arange(
        gate.INTERIOR_NX * gate.INTERIOR_NY, dtype=np.float64)
    with path.open("wb") as stream:
        stream.write(gate.MAGIC)
        np.asarray(
            [1, 1, gate.NROWS, rank, gate.JPI, gate.JPJ, 64],
            dtype=np.int32,
        ).tofile(stream)
        for substep in range(1, gate.NROWS + 1):
            np.asarray([substep], dtype=np.int32).tofile(stream)
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64).tofile(stream)
            for offset in range(len(gate.FULL_FIELDS_BEFORE)):
                (full + substep + offset).tofile(stream)
            np.asarray([2.0], dtype=np.float64).tofile(stream)
            for offset in range(len(gate.FULL_FIELDS_AFTER)):
                (full + 10 + substep + offset).tofile(stream)
            (interior + 20 + substep).tofile(stream)
            for offset in range(len(gate.FULL_FIELDS_TAIL)):
                (full + 30 + substep + offset).tofile(stream)


def test_mixed_extent_reader_preserves_layout_and_rank(tmp_path):
    gate = _gate()
    path = tmp_path / "rank0.bin"
    _write_record(path, gate, rank=0)

    record = gate.read_u_history(path, expected_rank=0)

    assert path.stat().st_size == gate.EXPECTED_BYTES == 2_042_100
    assert record["fields"]["un_e"].shape == (2, 152, 94)
    assert record["fields"]["zu_frc"].shape == (2, 148, 90)
    assert record["fields"]["un_e"][0, 1, 2] == 1 + gate.JPI + 2


def test_reader_rejects_swapped_rank_header(tmp_path):
    gate = _gate()
    path = tmp_path / "rank0.bin"
    _write_record(path, gate, rank=0)

    with pytest.raises(gate.GateError, match="header mismatch"):
        gate.read_u_history(path, expected_rank=1)


def test_midpoint_replay_keeps_compiled_association():
    gate = _gate()
    coefficients = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    got = gate.replay_midpoint(
        coefficients,
        np.asarray([1.0e16]), np.asarray([-1.0e16]), np.asarray([1.0]))
    assert got.view(np.uint64).item() == np.float64(1.0).view(np.uint64).item()


def test_vector_replay_and_first_boundary_are_non_vacuous():
    gate = _gate()
    got = gate.replay_vector(
        np.asarray([1.0]), 2.0, np.asarray([3.0]), np.asarray([4.0]),
        np.asarray([5.0]), np.asarray([1.0]))
    assert got.item() == 25.0
    exact = {"bit_exact": True, "differing_cells": 0}
    moved = {"bit_exact": False, "differing_cells": 1}
    row = {name: exact for name in gate.SOURCE_ORDER}
    row = {"substep": 1, **row, "ua_exit": moved}
    first = gate.first_non_bit([row])
    assert first["boundary"] == "ua_exit"
