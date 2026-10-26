"""Controls for the Round-117 direct pre-loop acquisition gate."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round117_preloop_gate as gate,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full2_bytes(parsed: np.ndarray) -> bytes:
    return np.asarray(parsed.T, dtype="=f8", order="F").tobytes(order="F")


def _owned2_bytes(parsed: np.ndarray) -> bytes:
    return np.asarray(parsed.T, dtype="=f8", order="F").tobytes(order="F")


def _full3_bytes(parsed: np.ndarray) -> bytes:
    source = parsed.transpose(1, 0, 2)
    return np.asarray(source, dtype="=f8", order="F").tobytes(order="F")


def _write_records(root: Path, commit: str) -> tuple[np.ndarray, np.ndarray]:
    owned_shape = (gate.OWNED_Y, gate.OWNED_X)
    full_shape = (gate.JPJ, gate.JPI)
    full3_shape = (gate.JPJ, gate.JPI, gate.JPK)
    incoming_u = np.arange(np.prod(owned_shape), dtype=np.float64).reshape(
        owned_shape) * 1.0e-12
    incoming_v = -incoming_u
    cor_u = np.full(full_shape, 3.0e-13, dtype=np.float64)
    cor_v = np.full(full_shape, -7.0e-13, dtype=np.float64)
    mask = np.ones(full_shape, dtype=np.float64)
    final_u = incoming_u - cor_u[2:-2, 2:-2] * mask[2:-2, 2:-2]
    final_v = incoming_v - cor_v[2:-2, 2:-2] * mask[2:-2, 2:-2]

    slow = bytearray(gate.SLOW_MAGIC.ljust(16).encode("ascii"))
    slow.extend(struct.pack("=8i", *gate.SLOW_HEADER))
    slow.extend(struct.pack("=7i", *gate.SLOW_SIZES))
    zero3 = np.zeros(full3_shape, dtype=np.float64)
    for _name in ("e3u", "krhs_u", "umask", "e3v", "krhs_v", "vmask"):
        slow.extend(_full3_bytes(zero3))
    zero_owned = np.zeros(owned_shape, dtype=np.float64)
    for _name in ("depth_u", "depth_v"):
        slow.extend(_owned2_bytes(zero_owned))
    zero_full = np.zeros(full_shape, dtype=np.float64)
    for _name in ("r1_hu0", "r1_hv0"):
        slow.extend(_full2_bytes(zero_full))
    for _name in ("post_drag_u", "post_drag_v"):
        slow.extend(_owned2_bytes(zero_owned))
    for _name in ("cd_u", "cd_v"):
        slow.extend(_full2_bytes(zero_full))
    slow.extend(struct.pack("=d", 1.0))
    for _name in ("utau", "vtau", "r1_hu", "r1_hv"):
        slow.extend(_full2_bytes(zero_full))
    slow.extend(_owned2_bytes(incoming_u))
    slow.extend(_owned2_bytes(incoming_v))
    slow_path = root / gate.SLOW_RECORD
    slow_path.write_bytes(slow)

    preloop = bytearray(gate.PRELOOP_MAGIC.ljust(16).encode("ascii"))
    preloop.extend(struct.pack("=8i", *gate.PRELOOP_HEADER))
    values = {
        "u_kmm": zero_full, "v_kmm": zero_full,
        "incoming_u": incoming_u, "incoming_v": incoming_v,
        "u_mask": mask, "v_mask": mask,
        "cor_u": cor_u, "cor_v": cor_v,
        "final_u": final_u, "final_v": final_v,
    }
    for name, layout in gate.PRELOOP_FIELD_LAYOUT:
        value = values.get(
            name, zero_full if layout == "full" else zero_owned)
        preloop.extend(
            _full2_bytes(value) if layout == "full" else _owned2_bytes(value))
    preloop_path = root / gate.PRELOOP_RECORD
    preloop_path.write_bytes(preloop)

    for record in (slow_path, preloop_path):
        record.with_name(record.name + ".stamp").write_text(
            f"{_digest(record)} {commit} {record.name}\n", encoding="utf-8")
    (root / "producer_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (root / "nemo").write_bytes(b"synthetic-nemo")
    (root / "binary.sha256").write_text(
        f"{_digest(root / 'nemo')}  nemo.exe\n", encoding="utf-8")
    return final_u, final_v


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, commit: str,
    final_u: np.ndarray, final_v: np.ndarray,
) -> None:
    monkeypatch.setattr(
        gate, "worktree_stamp",
        lambda: {"clean": True, "commit": commit},
    )

    def reference(_path: Path, *, expected_kt: int) -> dict:
        assert expected_kt == 2
        return {
            "slow_u": np.stack((np.array(final_u, copy=True),)),
            "slow_v": np.stack((np.array(final_v, copy=True),)),
        }

    monkeypatch.setattr(gate.round81, "read_record", reference)


def test_round117_exact_mixed_layout_and_sizes(tmp_path: Path) -> None:
    final_u, final_v = _write_records(tmp_path, "a" * 40)
    slow = gate.read_slow_record(tmp_path / gate.SLOW_RECORD)
    preloop = gate.read_preloop_record(tmp_path / gate.PRELOOP_RECORD)
    assert slow["bytes"] == gate.SLOW_EXPECTED_SIZE == 1_486_548
    assert preloop["bytes"] == gate.PRELOOP_EXPECTED_SIZE == 112_560
    np.testing.assert_array_equal(slow["fields"]["post_wind_u"],
                                  preloop["fields"]["incoming_u"])
    np.testing.assert_array_equal(preloop["fields"]["final_u"], final_u)
    np.testing.assert_array_equal(preloop["fields"]["final_v"], final_v)


def test_round117_gate_accepts_closed_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commit = "b" * 40
    final_u, final_v = _write_records(tmp_path, commit)
    _install_fakes(monkeypatch, commit, final_u, final_v)
    output = tmp_path / "report.json"
    assert gate.main([
        "--root", str(tmp_path), "--reference-root", str(tmp_path),
        "--expect-commit", commit, "--output", str(output),
    ]) == 0
    assert "STATUS PASS" in capsys.readouterr().out


def test_round117_gate_separates_tool_and_record_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record_commit = "d" * 40
    tool_commit = "e" * 40
    final_u, final_v = _write_records(tmp_path, record_commit)
    _install_fakes(monkeypatch, tool_commit, final_u, final_v)
    assert gate.main([
        "--root", str(tmp_path), "--reference-root", str(tmp_path),
        "--expect-commit", tool_commit,
        "--expect-record-commit", record_commit,
        "--output", str(tmp_path / "separate-commits.json"),
    ]) == 0
    assert "STATUS PASS" in capsys.readouterr().out


@pytest.mark.parametrize(
    "plant", (
        "stamp", "truncation", "header", "layout", "input-ulp",
        "reference-ulp",
    ),
)
def test_round117_plants_exit_nonzero_and_name_firing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], plant: str,
) -> None:
    commit = "c" * 40
    final_u, final_v = _write_records(tmp_path, commit)
    _install_fakes(monkeypatch, commit, final_u, final_v)
    assert gate.main([
        "--root", str(tmp_path), "--reference-root", str(tmp_path),
        "--expect-commit", commit, "--output", str(tmp_path / "plant.json"),
        "--plant", plant,
    ]) == 1
    assert f"STATUS PLANT-FIRED: {plant}" in capsys.readouterr().err
