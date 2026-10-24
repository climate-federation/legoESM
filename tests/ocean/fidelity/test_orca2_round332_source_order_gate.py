from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round332_source_order_gate as gate,
)


def _field(name: str, value: np.ndarray) -> bytes:
    name_bytes = name.encode("ascii").ljust(24)
    metadata = struct.pack("=7i", 3, *value.shape, 3, 3, 1)
    return name_bytes + metadata + struct.pack("=Q", 0) + np.asarray(
        value, dtype="=f8", order="F").tobytes(order="F")


def _record(path: Path) -> None:
    shape = (2, 2, 2)
    before = np.ones(shape, dtype="=f8", order="F")
    added = np.full(shape, 2.0, dtype="=f8", order="F")
    before[0, 0, 0] = -0.0
    added[0, 0, 0] = 0.0
    after = before + added
    lengths = np.ones(shape, dtype="=f8", order="F")
    header = gate.MAGIC.encode("ascii").ljust(16) + struct.pack(
        "=12i", 1, 1, 1, 3, 0, *shape, 3, 3, 64, len(gate.FIELDS))
    payloads = (before, added, after, lengths, lengths)
    path.write_bytes(header + b"".join(
        _field(name, value) for name, value in zip(gate.FIELDS, payloads, strict=True)))


def test_reader_and_source_order_semantics(tmp_path: Path) -> None:
    path = tmp_path / gate.record_name(0)
    _record(path)
    record = gate.read_record(path)
    gate._verify_semantics(record)
    before = record["fields"]["stage1_v_before"]
    after = record["fields"]["stage1_v_after"]
    assert before[0, 0, 0] == after[0, 0, 0]
    assert before.view(np.uint64)[0, 0, 0] != after.view(np.uint64)[0, 0, 0]


@pytest.mark.parametrize("plant", ["header", "field-order", "truncation"])
def test_reader_plants_refuse(tmp_path: Path, plant: str) -> None:
    path = tmp_path / gate.record_name(0)
    _record(path)
    with pytest.raises(gate.GateError):
        gate.read_record(path, plant=plant)


@pytest.mark.parametrize("plant", ["vor", "mxl"])
def test_semantic_plants_refuse(tmp_path: Path, plant: str) -> None:
    path = tmp_path / gate.record_name(0)
    _record(path)
    record = gate.read_record(path)
    gate._plant(record, plant)
    with pytest.raises(gate.GateError):
        gate._verify_semantics(record)


def test_observer_patches_are_additions_only_and_target_owning_statements() -> None:
    root = Path(gate.__file__).with_name(
        "nemo_testcase_l4_orca2_round332_source_order_acquisition")
    dynvor = (root / "dynvor_round332.patch").read_text()
    tke = (root / "zdftke_round332.patch").read_text()
    for patch in (dynvor, tke):
        assert not any(line.startswith("-") and not line.startswith("---")
                       for line in patch.splitlines())
    assert "r332_vor_before_cell(ji,jj,jk,pv_rhs(ji,jj,jk),zva)" in dynvor
    assert "r332_vor_after_cell(ji,jj,jk,pv_rhs(ji,jj,jk))" in dynvor
    assert "r332_tke_row(Kbb,Kmm,jj,zmxlm" in tke


def test_launcher_imports_reviewed_acquisition_driver() -> None:
    root = Path(gate.__file__).with_name(
        "nemo_testcase_l4_orca2_round332_source_order_acquisition")
    launcher = (root / "run.sh").read_text()
    assert "R328_ACQ_TARGET=tke332" in launcher
    assert "round328_dynzdf_acquisition/run.sh" in launcher
