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


def _round140_payload(rhs_u_value: float = 0.0) -> bytes:
    count = MODULE.COUNT
    interior = (MODULE.NX - 4) * (MODULE.NY - 4)
    header = b"NEMO_L2_R140RHS".ljust(16, b" ") + struct.pack(
        "=8i", 3, 1081, 1, 3, MODULE.NX, MODULE.NY, MODULE.NZ, 64)
    header += struct.pack("=7i", *((count,) * 6 + (interior,)))
    zero3 = np.zeros(count, dtype=np.float64)
    rhs_u = np.full(count, rhs_u_value, dtype=np.float64)
    umask = np.zeros(count, dtype=np.float64)
    umask[0] = 1.0
    chunks = [zero3, rhs_u, umask, zero3, zero3, zero3]
    chunks += [np.zeros(interior, dtype=np.float64) for _ in range(2)]
    chunks += [np.zeros(MODULE.NX * MODULE.NY, dtype=np.float64)
               for _ in range(2)]
    chunks += [np.zeros(interior, dtype=np.float64) for _ in range(2)]
    chunks += [np.zeros(MODULE.NX * MODULE.NY, dtype=np.float64)
               for _ in range(2)]
    chunks += [np.zeros(1, dtype=np.float64)]
    chunks += [np.zeros(MODULE.NX * MODULE.NY, dtype=np.float64)
               for _ in range(4)]
    chunks += [np.zeros(interior, dtype=np.float64) for _ in range(2)]
    return header + b"".join(chunk.tobytes() for chunk in chunks)


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


def test_round140_full_reader_closes_every_field():
    record = MODULE.read_round140_bytes(_round140_payload())
    assert tuple(record["fields"]) == MODULE.ROUND140_FIELDS
    assert len(_round140_payload()) == MODULE.ROUND140_EXPECTED_SIZE


def test_parent_comparison_rejects_owned_rhs_change():
    baseline = MODULE.read_round140_bytes(_round140_payload())
    candidate = MODULE.read_round140_bytes(_round140_payload())
    candidate["fields"]["rhs_u"][0, 0, 0] = 1.0
    result = MODULE._parent_comparison(candidate, baseline)
    assert not result["passive"]
    assert result["rows"]["rhs_u"]["owned"]["differing_cells"] == 1


def test_parent_comparison_registers_excluded_rhs_change():
    baseline = MODULE.read_round140_bytes(_round140_payload())
    candidate = MODULE.read_round140_bytes(_round140_payload())
    candidate["fields"]["rhs_u"][0, 0, 1] = 1.0
    result = MODULE._parent_comparison(candidate, baseline)
    assert result["passive"]
    assert result["rows"]["rhs_u"]["owned"]["differing_cells"] == 0
    assert result["rows"]["rhs_u"]["excluded"]["differing_cells"] == 1


def test_parent_comparison_rejects_interior_workspace_change():
    baseline = MODULE.read_round140_bytes(_round140_payload())
    candidate = MODULE.read_round140_bytes(_round140_payload())
    candidate["fields"]["cd_u"][2, 2] = 1.0
    result = MODULE._parent_comparison(candidate, baseline)
    assert not result["passive"]
    assert result["rows"]["cd_u"]["owned"]["differing_cells"] == 1


def test_parent_comparison_registers_workspace_halo_change():
    baseline = MODULE.read_round140_bytes(_round140_payload())
    candidate = MODULE.read_round140_bytes(_round140_payload())
    candidate["fields"]["cd_u"][0, 0] = 1.0
    result = MODULE._parent_comparison(candidate, baseline)
    assert result["passive"]
    assert result["rows"]["cd_u"]["owned"]["differing_cells"] == 0
    assert result["rows"]["cd_u"]["excluded"]["differing_cells"] == 1
