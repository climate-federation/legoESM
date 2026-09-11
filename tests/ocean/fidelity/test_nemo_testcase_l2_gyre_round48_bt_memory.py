"""Hermetic fail-closed checks for the round-48 memory acquisition package."""

from __future__ import annotations

import importlib.util
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[3]
TESTCASES = ROOT / "scripts/validate/ocean_fidelity/testcases"
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_gyre_round48_bt_memory_gate",
    TESTCASES / "nemo_testcase_l2_gyre_round48_bt_memory_gate.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(TESTCASES))
SPEC.loader.exec_module(gate)
import nemo_testcase_l2_gyre_round21_admission as admission


def _write(path: Path, *, kt: int, phase: int) -> Path:
    y, x = gate.DIMS[1], gate.DIMS[0]
    base = np.arange(y * x, dtype=np.float64).reshape(y, x)
    values = {name: base + i for i, name in enumerate(gate.HISTORIES)}
    if phase == 1:
        values.update({
            "un_e": base + 20, "vn_e": base + 21, "sshn_e": base + 22,
            "un_adv": base + 23, "vn_adv": base + 24,
            "ubar_Kmm": base + 30, "vbar_Kmm": base + 31, "ssh_Kmm": base + 32,
            "ubar_Kaa": base + 20, "vbar_Kaa": base + 21, "ssh_Kaa": base + 22,
        })
    else:
        values.update({
            "un_e": base + 30, "vn_e": base + 31, "sshn_e": base + 32,
            "un_adv": np.zeros_like(base), "vn_adv": np.zeros_like(base),
            "ubar_Kmm": base + 30, "vbar_Kmm": base + 31, "ssh_Kmm": base + 32,
            "ubar_Kaa": np.zeros_like(base), "vbar_Kaa": np.zeros_like(base),
            "ssh_Kaa": np.zeros_like(base),
        })
    with path.open("wb") as stream:
        stream.write(gate.MAGIC.ljust(16).encode())
        stream.write(struct.pack(
            "=13i", 1, kt, phase, 1, 1, 2, *gate.DIMS, 3, 34, 3, 24, 64
        ))
        for name in sorted(gate.REQUIRED):
            stream.write(name.ljust(16).encode())
            stream.write(struct.pack("=4i", 2, *gate.DIMS, 1))
            stream.write(values[name].T.ravel(order="F").tobytes())
    return path


def _root(tmp_path: Path) -> Path:
    _write(tmp_path / "oracle_bt_memory_kt00000001_end.bin", kt=1, phase=1)
    _write(tmp_path / "oracle_bt_memory_kt00000002_start.bin", kt=2, phase=2)
    return tmp_path


def test_reader_and_boundary_gate(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(gate, "worktree_stamp", lambda: {
        "commit": "a" * 40, "clean": True})
    report = gate.run(root, expect_commit="a" * 40, plant=None)
    assert report["status"] == "PASS"
    assert all(report["history_boundary_bit_identical"].values())


@pytest.mark.parametrize("plant", ["boundary", "seed", "reset", "stamp"])
def test_semantic_plants_are_red(tmp_path, monkeypatch, plant):
    root = _root(tmp_path)
    monkeypatch.setattr(gate, "worktree_stamp", lambda: {
        "commit": "a" * 40, "clean": True})
    with pytest.raises(Exception):
        gate.run(root, expect_commit="a" * 40, plant=plant)


def test_header_and_truncation_plants_are_red(tmp_path):
    path = _write(tmp_path / "record.bin", kt=1, phase=1)
    with pytest.raises(Exception, match="wrong kt/phase"):
        gate.read_record(path, plant="header")
    with pytest.raises(Exception, match="short payload"):
        gate.read_record(path, plant="truncation")


def test_writer_is_write_only_and_run_has_every_plant():
    package = TESTCASES / "nemo_testcase_l2_gyre_round48_bt_memory"
    writer = (package / "l2_r48_bt_memory.F90").read_text()
    patch = (package / "dynspg_ts_round48.patch").read_text()
    run = (package / "run.sh").read_text()
    assert "ACTION='WRITE'" in writer and "STATUS='REPLACE'" in writer
    assert writer.count("INTENT(in)") >= 8
    assert patch.count("+      CALL r48_bt_memory") == 2
    assert not any(line.startswith("-") for line in patch.splitlines()[2:])
    model_names = "pubb_e|pub_e|pvbb_e|pvb_e|psshbb_e|psshb_e|pun_e|pvn_e|psshn_e"
    assert not re.search(rf"\b(?:{model_names})\s*(?:\([^\n]*\))?\s*=", writer)
    for plant in ("header", "truncation", "boundary", "seed", "reset", "stamp"):
        assert f'"{plant}"' in (TESTCASES / "nemo_testcase_l2_gyre_round48_bt_memory_gate.py").read_text()
        assert plant in run
    assert "--plant-consumed" in run


def test_round46_stage_stream_is_registered_with_its_named_reader():
    assert admission.SELF_DESCRIBING["NEMO_L2_R46STG1"] == 16
    assert "NEMO_L2_R46STG1" in admission.SOURCES
    assert "NEMO_L2_R46STG1" in admission.PARSERS
