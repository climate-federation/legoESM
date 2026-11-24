import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round118_een_v_recurrence_acquisition import (
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
        n3 = 1 if name == "mbkv" else 3
        raw.extend(gate.GROUP.pack(2 if name == "mbkv" else 3, 2, 2, n3))
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


def test_record_covers_all_four_v_recurrences():
    assert gate.LABELS == ("nw", "ne", "sw", "se")
    for label in gate.LABELS:
        assert tuple(name for name in gate.FIELDS if name.endswith(f"_{label}")) == (
            f"zpvo_{label}", f"e3v_{label}", f"e3u_{label}", f"mask_{label}",
            f"term_{label}", f"before_{label}", f"after_{label}",
        )


def test_product_and_recurrence_bit_controls_detect_signed_zero():
    positive = np.array([0.0], dtype=np.float64)
    negative = np.array([-0.0], dtype=np.float64)
    assert not gate.bit_equal(positive, negative)
    assert gate.bit_equal(positive, positive.copy())


def test_every_admission_claim_has_a_registered_plant():
    assert gate.PLANTS == (
        "none", "header", "field-name", "field-dims", "truncation",
        "missing-field", "duplicate-rank", "bottom", "product", "recurrence",
        "inherited", "association", "restart-byte",
    )
