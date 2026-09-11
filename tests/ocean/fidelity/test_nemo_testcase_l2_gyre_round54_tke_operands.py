from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).parents[3] / "scripts" / "validate" / "ocean_fidelity"
          / "testcases" / "nemo_testcase_l2_gyre_round54_tke_operands.py")
SPEC = importlib.util.spec_from_file_location("round54_tke_operands", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record(path: Path) -> None:
    parts = [MODULE.MAGIC, struct.pack("=13i", 1, 2, 1, 2, 32, 22, 31, 30,
                                      1, 32, 1, 22, 64)]
    shape = (32, 22, 31)
    for name in MODULE.FIELDS:
        parts.append(name.ljust(16).encode("ascii"))
        if name == "rn_Dt":
            parts.extend((struct.pack("=4i", 0, 1, 1, 1),
                          struct.pack("=d", 14400.0)))
        else:
            parts.append(struct.pack("=4i", 3, *shape))
            parts.append(np.zeros(shape, dtype="=f8").tobytes(order="F"))
    path.write_bytes(b"".join(parts))


def test_reader_accepts_complete_schema(tmp_path):
    path = tmp_path / "record.bin"
    _record(path)
    result = MODULE.read_record(path)
    assert result["header"]["kt"] == 2
    assert tuple(result["arrays"]["avt_output"].shape) == (32, 22, 31)


@pytest.mark.parametrize("plant", ["header", "truncation", "nan"])
def test_reader_plants_fail(tmp_path, plant):
    path = tmp_path / "record.bin"
    _record(path)
    with pytest.raises(MODULE.GateError):
        MODULE.read_record(path, plant=plant)
