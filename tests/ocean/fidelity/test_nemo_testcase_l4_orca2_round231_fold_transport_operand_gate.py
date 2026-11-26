from __future__ import annotations

import struct

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round231_fold_transport_operand_gate as gate,
)


def _write(path, rank: int) -> None:
    nx, ny, nz = 94, 152, 31
    with path.open("wb") as handle:
        handle.write(b"NEMO_L4_R229FLD1")
        handle.write(struct.pack(
            "=15i", 1, 1, 1, rank, nx, ny, nz,
            1 + 90 * rank, 1, 3, 3, 92, 150, 64, 5,
        ))
        for field_index, name in enumerate(gate.FIELDS):
            handle.write(name.encode("ascii").ljust(16, b" "))
            handle.write(struct.pack("=4i", 3, nx, ny, nz))
            value = float(10 * rank + field_index)
            values = np.full((nx, ny, nz), value, np.float64)
            if name == "e3t_Kmm":
                values.fill(1.0)
            values.ravel(order="F").tofile(handle)


def test_rank_records_assemble_owned_slabs_and_north_halo(tmp_path) -> None:
    for rank in (0, 1):
        _write(tmp_path / f"oracle_r229_fold_rank{rank:04d}_kt00000001_s1.bin", rank)
    record = gate.assemble_record(tmp_path)
    assert record["T_Kmm"].shape == (148, 180, 31)
    assert np.all(record["T_Kmm"][:, :90] == 1.0)
    assert np.all(record["T_Kmm"][:, 90:] == 11.0)
    assert np.all(record["T_Kmm_north_halo"][:90] == 1.0)
    assert np.all(record["T_Kmm_north_halo"][90:] == 11.0)


def test_self_describing_reader_refuses_trailing_payload(tmp_path) -> None:
    path = tmp_path / "record.bin"
    _write(path, 0)
    path.write_bytes(path.read_bytes() + b"extra")
    try:
        gate._read(path)
    except gate.GateError as error:
        assert "trailing payload" in str(error)
    else:
        raise AssertionError("trailing payload plant stayed green")
