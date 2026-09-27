import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (Path(__file__).parents[3] / "scripts" / "validate" /
          "ocean_fidelity" / "testcases" /
          "nemo_testcase_l2_gyre_round186_qsr_walk.py")
SPEC = importlib.util.spec_from_file_location("round186_qsr", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _root(tmp_path):
    root = tmp_path / "record"
    root.mkdir()
    values = np.zeros(MOD.VALUE_COUNT, dtype="=f8")
    raw = MOD.MAGIC + struct.pack(f"={MOD.HEADER_INTS}i", *MOD.HEADER) + values.tobytes()
    record = root / "oracle_qsr_walk_kt00001080.bin"
    record.write_bytes(raw)
    digest = MOD.sha256(record)
    (root / "producer_commit.txt").write_text("abc\n")
    (root / "qsr_record.stamp").write_text(f"{digest} abc {record.name}\n")
    return root


def test_round186_qsr_record_admits_exact_replay(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(MOD.subprocess, "check_output", lambda *a, **k: "tip\n")
    assert MOD.admit(root, "abc")["calibration"]["cells_unequal"] == 0


def test_round186_qsr_record_ulp_plant_fires(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(MOD.subprocess, "check_output", lambda *a, **k: "tip\n")
    with pytest.raises(MOD.GateError, match="STATUS PLANT-FIRED"):
        MOD.admit(root, "abc", "actual-increment-ulp")
