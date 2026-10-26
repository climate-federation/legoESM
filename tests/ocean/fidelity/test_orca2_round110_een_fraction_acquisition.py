import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round110_een_fraction_acquisition import (
    check_record as gate,
)


def _record_bytes(*, corrupt_name: bool = False) -> bytes:
    raw = bytearray(gate.MAGIC.ljust(16, b" "))
    raw.extend(gate.HEADER.pack(1, 1, 0, 4, 4, 3, 1, 1, 2, 2, 3, 3, 64,
                                len(gate.FIELDS)))
    for index, name in enumerate(gate.FIELDS):
        encoded = (("bad" if corrupt_name and index == 0 else name).encode("ascii")
                   .ljust(16, b" "))
        raw.extend(encoded)
        n3 = 1 if name == "mbku" else 3
        raw.extend(gate.GROUP.pack(2 if name == "mbku" else 3, 2, 2, n3))
        raw.extend(np.zeros((2, 2, n3), dtype=np.float64, order="F")
                   .tobytes(order="F"))
    return bytes(raw)


def test_self_describing_parser_reads_declared_field_order(tmp_path: Path):
    path = tmp_path / "record.bin"
    path.write_bytes(_record_bytes())
    row = gate.read_record(path)
    assert tuple(row["groups"]) == gate.FIELDS
    assert row["owned"] == (2, 2, 3, 3)


def test_self_describing_parser_rejects_field_registry_plant(tmp_path: Path):
    path = tmp_path / "record.bin"
    path.write_bytes(_record_bytes(corrupt_name=True))
    with pytest.raises(gate.Refusal, match="field registry/order moved"):
        gate.read_record(path)


def test_fraction_record_has_all_three_operand_groups():
    for label in ("west", "center", "south"):
        assert (f"{label}_ff", f"{label}_e3f0", f"{label}_r3f",
                f"{label}_mask", f"{label}_denom", f"frac_{label}") == tuple(
                    name for name in gate.FIELDS if name.startswith(label)
                    or name == f"frac_{label}")
