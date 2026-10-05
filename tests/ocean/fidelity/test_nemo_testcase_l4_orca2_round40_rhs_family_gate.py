"""Binding tests for the ORCA2 round-40 acquisition reader."""

import struct

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round40_rhs_family_gate as gate,
)


def _payload(rank: int = 0) -> bytes:
    header = gate.MAGIC.encode("ascii").ljust(16, b" ")
    header += struct.pack(
        "=9i", 1, 1, 1, 3, rank, gate.NX, gate.NY, gate.NZ, 64
    )
    header += struct.pack(
        f"={len(gate.FIELDS)}i", *((gate.COUNT,) * len(gate.FIELDS))
    )
    values = np.zeros(gate.COUNT, dtype=np.float64).tobytes()
    return header + values * len(gate.FIELDS)


def test_round40_reader_closes_the_registered_field_census():
    record = gate.read_record_bytes(_payload(), 0)
    assert tuple(record["fields"]) == gate.FIELDS
    assert len(_payload()) == gate.EXPECTED_SIZE == 35_434_332


def test_round40_reader_rejects_swapped_rank():
    with pytest.raises(gate.GateError, match="bad header"):
        gate.read_record_bytes(_payload(rank=0), 1)


def test_round40_reader_rejects_truncation():
    with pytest.raises(gate.GateError, match="record is"):
        gate.read_record_bytes(_payload()[:-1], 0)


def test_round40_identity_is_bitwise_and_one_ulp_sensitive():
    left = np.zeros((2,), dtype=np.float64)
    right = left.copy()
    right[0] = np.nextafter(right[0], np.inf)
    assert gate._identity(left, left)["bit_exact"]
    row = gate._identity(left, right)
    assert not row["bit_exact"]
    assert row["differing_cells"] == 1
