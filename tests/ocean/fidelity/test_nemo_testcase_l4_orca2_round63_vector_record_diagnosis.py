"""Controls for the round-63 incomplete vector-record diagnosis."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_round63_vector_record_diagnosis.py"
)
SPEC = importlib.util.spec_from_file_location("round63_vector_record", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _write_incomplete(path: Path) -> None:
    nx, ny, nz = 2, 3, 2
    header = (1, 1, 2, 3, 2, 0, 1, 1, 1, 2, 1, 3, nx, ny, nz, 1, 64, 0, 0, 0)
    payload3 = b"\0" * (8 * nx * ny * nz)
    payload2 = b"\0" * (8 * nx * ny)
    with path.open("wb") as handle:
        handle.write(gate.MAGIC.ljust(16).encode("ascii"))
        handle.write(struct.pack("=20i", *header))
        for name, rank in gate.PREFIX:
            handle.write(name.ljust(16).encode("ascii"))
            handle.write(struct.pack("=4i", rank, nx, ny, nz))
            handle.write(payload3)
        for name in ("e3u_Kmm", "e3v_Kmm"):
            handle.write(name.ljust(16).encode("ascii"))
            handle.write(struct.pack("=4i", 3, nx, ny, nz))
        for name, rank in gate.SUFFIX:
            handle.write(name.ljust(16).encode("ascii"))
            handle.write(struct.pack("=4i", rank, nx, ny, nz if rank == 3 else 1))
            handle.write(payload3 if rank == 3 else payload2)


def test_diagnosis_recovers_fields_after_both_missing_payloads(tmp_path):
    path = tmp_path / "record.bin"
    _write_incomplete(path)
    report = gate.diagnose_record(path, expected_dims=(2, 3, 2, 1))
    assert report["missing_payloads"] == ["e3u_Kmm", "e3v_Kmm"]
    assert report["later_fields_recovered"] == len(gate.SUFFIX)
    assert report["physical_eof_parsed"] is True


@pytest.mark.parametrize("plant", ["offset", "signature"])
def test_diagnosis_plants_fire(tmp_path, plant):
    path = tmp_path / "record.bin"
    _write_incomplete(path)
    with pytest.raises(gate.GateError):
        gate.diagnose_record(path, plant=plant, expected_dims=(2, 3, 2, 1))
