import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest


PATH = (Path(__file__).resolve().parents[3] / "scripts" / "validate" /
        "ocean_fidelity" / "testcases" /
        "nemo_testcase_l2_gyre_round146_rhs_family_gate.py")
SPEC = importlib.util.spec_from_file_location("round146_family_gate", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _payload() -> bytes:
    header = MODULE.MAGIC.encode("ascii").ljust(16, b" ")
    header += struct.pack("=8i", 1, 1081, 1, 3, MODULE.NX, MODULE.NY,
                          MODULE.NZ, 64)
    header += struct.pack(f"={len(MODULE.FIELDS)}i",
                          *((MODULE.COUNT,) * len(MODULE.FIELDS)))
    values = np.zeros(MODULE.COUNT, dtype=np.float64).tobytes()
    return header + values * len(MODULE.FIELDS)


def test_round146_reader_closes_the_field_census():
    record = MODULE.read_record_bytes(_payload())
    assert tuple(record["fields"]) == MODULE.FIELDS
    assert len(_payload()) == MODULE.EXPECTED_SIZE


def test_round146_reader_rejects_shifted_header():
    payload = bytearray(_payload())
    payload[16:20] = struct.pack("=i", 2)
    with pytest.raises(MODULE.GateError, match="bad header"):
        MODULE.read_record_bytes(bytes(payload))


def test_round146_reader_rejects_truncation():
    with pytest.raises(MODULE.GateError, match="record is"):
        MODULE.read_record_bytes(_payload()[:-1])
