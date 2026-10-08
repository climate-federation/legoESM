"""Direct controls for the ORCA2 round-20 exchange gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
GATE = (REPO / "scripts/validate/ocean_fidelity/orca2_l4"
        / "nemo_testcase_l4_orca2_round20_exchange_gate.py")


def _gate():
    spec = importlib.util.spec_from_file_location("_r20_exchange_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_record(path: Path, gate, rank: int) -> None:
    full = np.arange(gate.JPI * gate.JPJ, dtype=np.float64)
    with path.open("wb") as stream:
        stream.write(gate.MAGIC)
        np.asarray(
            [1, 1, gate.NROWS, rank, gate.JPI, gate.JPJ, 64],
            dtype=np.int32,
        ).tofile(stream)
        for substep in range(1, gate.NROWS + 1):
            np.asarray([substep], dtype=np.int32).tofile(stream)
            (full + substep).tofile(stream)


def test_reader_preserves_fortran_layout_and_rank(tmp_path):
    gate = _gate()
    path = tmp_path / "rank0.bin"
    _write_record(path, gate, rank=0)
    record = gate.read_preexchange(path, expected_rank=0)
    assert path.stat().st_size == gate.EXPECTED_BYTES == 228_660
    assert record["u"].shape == (2, 152, 94)
    assert record["u"][0, 1, 2] == 1 + gate.JPI + 2


def test_reader_rejects_swapped_rank_header(tmp_path):
    gate = _gate()
    path = tmp_path / "rank0.bin"
    _write_record(path, gate, rank=0)
    with pytest.raises(gate.GateError, match="header mismatch"):
        gate.read_preexchange(path, expected_rank=1)


def test_vector_replay_keeps_compiled_association():
    gate = _gate()
    got = gate.replay_vector(
        np.asarray([1.0]), 2.0, np.asarray([3.0]), np.asarray([4.0]),
        np.asarray([5.0]), np.asarray([1.0]))
    assert got.item() == 25.0


def test_compare_is_bitwise_and_one_ulp_fires():
    gate = _gate()
    value = np.asarray([1.0], dtype=np.float64)
    moved = np.nextafter(value, np.inf)
    assert gate._compare(value, value)["bit_exact"]
    assert not gate._compare(value, moved)["bit_exact"]
