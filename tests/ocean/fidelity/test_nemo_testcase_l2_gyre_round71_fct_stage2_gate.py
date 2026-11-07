from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round71_fct_stage2_gate.py"
)
SPEC = importlib.util.spec_from_file_location("round71_stage2_gate", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _record(
    path: Path, stage: int, *, kt: int = 2, bad_magic: bool = False,
) -> None:
    nx, ny, nz = GATE.DIMS
    n3, n2 = nx * ny * nz, nx * ny
    levels = GATE.LEVELS.get(kt, GATE.LEVELS[2])[stage]
    values = np.arange(15 * n3 + 3 * n2, dtype=np.float64)
    with path.open("wb") as handle:
        magic = "BAD_MAGIC" if bad_magic else "NEMO_L2_RKTRA_1"
        handle.write(magic.ljust(16).encode())
        handle.write(struct.pack("=11i", 1, kt, stage, *levels, nx, ny, nz, 64))
        values.tofile(handle)


def test_reader_accepts_exact_stage2_schema(tmp_path: Path) -> None:
    path = tmp_path / "record.bin"
    _record(path, 2)
    row = GATE.read_record(path, 2)
    assert row["header"] == (1, 2, 2, 3, 1, 2, 2, 36, 26, 31, 64)
    assert row["fields"]["Kmm_T"].shape == (32, 22, 31)
    assert row["fields"]["r3t_Kbb"].shape == (32, 22)
    assert row["fields"]["r3t_Kaa"].shape == (32, 22)


def test_reader_rejects_wrong_magic_and_truncation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.bin"
    _record(bad, 2, bad_magic=True)
    with pytest.raises(SystemExit, match="bad magic"):
        GATE.read_record(bad, 2)
    good = tmp_path / "good.bin"
    _record(good, 2)
    with pytest.raises(SystemExit, match="bad payload size"):
        GATE.read_record(good, 2, truncate=True)


def test_reader_accepts_registered_kt1_schema(tmp_path: Path) -> None:
    path = tmp_path / "record.bin"
    _record(path, 1, kt=1)
    row = GATE.read_record(path, 1, expected_kt=1)
    assert row["header"][:3] == (1, 1, 1)


def test_reader_rejects_unregistered_kt(tmp_path: Path) -> None:
    path = tmp_path / "record.bin"
    _record(path, 1, kt=3)
    with pytest.raises(SystemExit, match="unsupported kt"):
        GATE.read_record(path, 1, expected_kt=3)
