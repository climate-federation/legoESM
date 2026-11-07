"""Controls for the round-111 passive FCT-writer acquisition gate."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round111_fct_writer_gate as gate,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_card(root: Path, commit: str = "synthetic-commit") -> None:
    record = root / gate.RECORD
    payload = bytearray(gate.MAGIC.ljust(16).encode("ascii"))
    payload.extend(struct.pack("=14i", *gate.HEADER))
    for name, rank, shape, origin in gate.EXPECTED_ROWS:
        payload.extend(name.ljust(16).encode("ascii"))
        payload.extend(struct.pack("=7i", rank, *shape, *origin))
        values = np.zeros(shape, dtype="=f8", order="F")
        if name == "p2dt":
            values.fill(14400.0)
        payload.extend(values.tobytes(order="F"))
    record.write_bytes(payload)
    (root / f"{gate.RECORD}.stamp").write_text(
        f"{_digest(record)} {commit} {gate.RECORD}\n", encoding="utf-8")
    (root / "producer_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (root / "nemo").write_bytes(b"synthetic-nemo-binary")
    (root / "binary.sha256").write_text(
        f"{_digest(root / 'nemo')}  nemo.exe\n", encoding="utf-8")


def test_round111_record_layout_and_truncation(tmp_path: Path) -> None:
    _write_card(tmp_path)
    parsed = gate.read_record(tmp_path / gate.RECORD)
    assert len(parsed["fields"]) == 34
    assert parsed["fields"]["average_u_T"]["shape"] == (33, 23, 30)
    with pytest.raises(gate.GateError, match="truncated"):
        gate.read_record(tmp_path / gate.RECORD, truncate=True)


@pytest.mark.parametrize("plant", ("stamp", "truncation", "rhs-entry-ulp"))
def test_round111_gate_plants_exit_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], plant: str,
) -> None:
    commit = "synthetic-commit"
    _write_card(tmp_path, commit)
    monkeypatch.setattr(gate, "worktree_stamp", lambda: {"clean": True})
    output = tmp_path / "report.json"
    assert gate.main([
        "--root", str(tmp_path), "--expect-commit", commit,
        "--output", str(output), "--plant", plant,
    ]) == 1
    assert "STATUS PLANT-FIRED" in capsys.readouterr().err


def test_round111_gate_accepts_complete_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commit = "synthetic-commit"
    _write_card(tmp_path, commit)
    monkeypatch.setattr(gate, "worktree_stamp", lambda: {"clean": True})
    output = tmp_path / "report.json"
    assert gate.main([
        "--root", str(tmp_path), "--expect-commit", commit,
        "--output", str(output),
    ]) == 0
    assert "STATUS PASS" in capsys.readouterr().out
    assert output.is_file()
