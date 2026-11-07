"""Direct controls for the ORCA2 round-18 ranked-halo gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
GATE = (REPO / "scripts/validate/ocean_fidelity/orca2_l4"
        / "nemo_testcase_l4_orca2_round18_halo_operand_gate.py")


def _gate():
    spec = importlib.util.spec_from_file_location("_r18_halo_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_record(path: Path, gate, rank: int) -> None:
    with path.open("wb") as stream:
        stream.write(gate.MAGIC)
        np.asarray(
            [1, 1, gate.NROWS, rank, gate.JPI, gate.JPJ, 64],
            dtype=np.int32,
        ).tofile(stream)
        for substep in range(1, gate.NROWS + 1):
            np.asarray([substep], dtype=np.int32).tofile(stream)
            for field in range(len(gate.FIELD_NAMES)):
                values = np.arange(
                    gate.JPI * gate.JPJ, dtype=np.float64
                ).reshape((gate.JPJ, gate.JPI)) + substep + field
                values.tofile(stream)


def test_reader_preserves_fortran_local_layout_and_rank(tmp_path):
    gate = _gate()
    path = tmp_path / "rank0.bin"
    _write_record(path, gate, rank=0)

    record = gate.read_halo(path, expected_rank=0)

    assert record["rank"] == 0
    assert record["substeps"] == [1, 2]
    assert record["fields"]["zhU"].shape == (2, gate.JPJ, gate.JPI)
    assert record["fields"]["zhU"][0, 1, 2] == 1 + gate.JPI + 2


def test_reader_rejects_swapped_rank_header(tmp_path):
    gate = _gate()
    path = tmp_path / "rank0.bin"
    _write_record(path, gate, rank=0)

    with pytest.raises(gate.GateError, match="header mismatch"):
        gate.read_halo(path, expected_rank=1)


def test_rank0_window_keeps_left_halo_and_all_owned_right_faces():
    gate = _gate()
    local = np.arange(
        gate.NROWS * gate.JPJ * gate.JPI
    ).reshape((gate.NROWS, gate.JPJ, gate.JPI))

    window = gate._rank0_u_window(local)

    assert window.shape == (2, 148, 91)
    assert np.array_equal(window, local[:, 2:150, 1:92])


def test_face_support_marks_both_operands():
    gate = _gate()
    t_support = np.zeros((2, 3), dtype=bool)
    t_support[1, 0] = True

    support = gate._face_support(t_support)

    assert np.array_equal(
        support,
        np.array([[False, False, False, False],
                  [True, True, False, False]]),
    )
