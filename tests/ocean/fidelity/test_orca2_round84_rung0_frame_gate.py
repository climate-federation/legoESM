"""Controls for the round-84 ORCA2 rung-0 frame gate."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
GATE_PATH = ROOT / (
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round84_rung0_frame_gate.py"
)
SPEC = importlib.util.spec_from_file_location("orca2_r84_frame_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _write_frame(path: Path) -> None:
    nx, ny, nz = 3, 4, 2
    with path.open("wb") as handle:
        handle.write(f"{gate.MAGIC:<16}".encode("ascii"))
        handle.write(gate.HEADER.pack(
            1, 1, 0, 1, 0, nx, ny, nz, 2, 1, 1, 1, 1, nx, ny, 64, 5,
        ))
        for name in gate.FIELDS:
            handle.write(f"{name:<16}".encode("ascii"))
            if name == "ssh":
                dims = (2, nx, ny, 1)
            else:
                dims = (3, nx, ny, nz)
            handle.write(gate.FIELD_HEADER.pack(*dims))
            count = dims[1] * dims[2] * dims[3]
            handle.write(np.arange(count, dtype="=f8").tobytes())


def test_preflight_is_additions_only_and_applies() -> None:
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["removed_source_lines"] == 0
    assert report["frame_calls"] == 4


def test_self_describing_frame_and_plants(tmp_path: Path) -> None:
    path = tmp_path / "frame.bin"
    _write_frame(path)
    record = gate.read_frame(path)
    assert tuple(record["fields"]) == gate.FIELDS
    assert record["header"]["shape"] == [3, 4, 2]
    for plant in ("header", "field-name", "truncation", "nonfinite"):
        with pytest.raises(gate.GateError):
            gate.read_frame(path, plant)
