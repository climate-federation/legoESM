from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round120_een_v_fraction_acquisition import (
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


def test_record_covers_both_northern_paths_and_three_fractions():
    assert gate.PATHS == ("ne", "nw")
    assert gate.COMPONENTS == ("1", "2", "3")
    assert gate.OPERANDS == ("ff", "e3f0", "r3f", "mask", "denom", "frac")
    for path in gate.PATHS:
        for component in gate.COMPONENTS:
            for operand in gate.OPERANDS:
                assert f"{path}_{component}_{operand}" in gate.FIELDS
        assert gate.FIELDS.index(f"{path}_partial") < gate.FIELDS.index(f"{path}_sum")
        assert gate.FIELDS.index(f"{path}_sum") < gate.FIELDS.index(f"{path}_nemo_sum")


def test_bit_control_detects_signed_zero():
    positive = np.array([0.0], dtype=np.float64)
    negative = np.array([-0.0], dtype=np.float64)
    assert not gate.bit_equal(positive, negative)
    assert gate.bit_equal(positive, positive.copy())


def test_every_admission_claim_has_a_registered_plant():
    assert gate.PLANTS == (
        "none", "header", "field-name", "field-dims", "truncation",
        "missing-field", "duplicate-rank", "bottom", "denominator",
        "quotient", "sum", "inherited-final", "inherited-stream",
        "restart-byte",
    )
