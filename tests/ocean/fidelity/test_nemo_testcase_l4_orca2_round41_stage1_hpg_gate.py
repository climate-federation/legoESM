"""Binding tests for the ORCA2 round-41 stage-1 HPG reader."""

import struct

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round41_stage1_hpg_gate as gate,
)


def _payload(rank: int = 0) -> bytes:
    header = gate.MAGIC.encode("ascii").ljust(16, b" ")
    header += struct.pack(
        "=9i", 1, 1, 1, 3, rank, gate.NX, gate.NY, gate.NZ, 64)
    field3 = np.zeros(gate.N3, dtype=np.float64).tobytes()
    field2 = np.zeros(gate.N2, dtype=np.float64).tobytes()
    return (header + field3 * len(gate.FIELDS_3D)
            + field2 * len(gate.FIELDS_2D))


def test_reader_closes_registered_field_census():
    record = gate.read_record_bytes(_payload(), 0)
    assert tuple(record["fields"]) == gate.FIELDS_3D + gate.FIELDS_2D
    assert len(_payload()) == gate.EXPECTED_SIZE == 32_119_476


def test_reader_rejects_swapped_rank_and_truncation():
    with pytest.raises(gate.GateError, match="bad header"):
        gate.read_record_bytes(_payload(0), 1)
    with pytest.raises(gate.GateError, match="record is"):
        gate.read_record_bytes(_payload()[:-1], 0)


def test_reader_rejects_nonfinite_values():
    payload = bytearray(_payload())
    payload[52:60] = np.asarray([np.nan], dtype=np.float64).tobytes()
    with pytest.raises(gate.GateError, match="non-finite"):
        gate.read_record_bytes(bytes(payload), 0)
