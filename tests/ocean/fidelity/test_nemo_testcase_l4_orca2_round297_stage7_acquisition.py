"""Known-answer controls for the round-297 kt=7 stage record admission."""

import struct
from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round175_stage1_acquisition import (
    check_record as schema,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round297_stage7_acquisition import (
    check_record as gate,
)


def _record(rank):
    return {
        "rank": rank,
        "levels": [1, 1, 3, 3],
        "shape": [94, 152, 31],
        "origin": [1 if rank == 0 else 91, 1],
        "owned": [3, 3, 92, 150],
        "fields": {},
        "bytes": 0,
        "sha256": "0" * 64,
    }


def _families(root: Path, baseline: Path):
    for index in range(80):
        name = f"oracle_r84_frame_{index:02d}.bin"
        (root / name).write_bytes(bytes([index % 251]))
        (baseline / name).write_bytes(bytes([index % 251]))
    for rank in (0, 1):
        name = f"ORCA2_00000010_restart_{rank:04d}.nc"
        (root / name).write_bytes(bytes([rank + 1]))
        (baseline / name).write_bytes(bytes([rank + 1]))


def test_schema_accepts_explicit_step_and_magic(tmp_path, monkeypatch):
    monkeypatch.setattr(schema, "NAMES", ())
    monkeypatch.setattr(schema, "NAMES_2D", ())
    magic = gate.MAGIC
    header = struct.pack(
        "=19i", 1, 7, 1, 1, 1, 3, 3, 0, 94, 152, 31,
        1, 1, 3, 3, 92, 150, 64, 0,
    )
    path = tmp_path / "record.bin"
    path.write_bytes(magic + header)
    assert schema.read_record(
        path, expected_kt=7, expected_magic=magic)["rank"] == 0
    with pytest.raises(schema.Refusal, match="version/step/stage"):
        schema.read_record(path, expected_magic=magic)


def test_admission_and_all_plants_bind(tmp_path, monkeypatch):
    root, baseline = tmp_path / "root", tmp_path / "baseline"
    root.mkdir()
    baseline.mkdir()
    _families(root, baseline)

    seen = []

    def fake_read(path, plant="none", *, expected_kt=1,
                  expected_magic=schema.MAGIC):
        seen.append((plant, expected_kt, expected_magic))
        if plant != "none":
            raise schema.Refusal(f"{plant} fired")
        rank = 0 if "rank0000" in path.name else 1
        return _record(rank)

    monkeypatch.setattr(schema, "read_record", fake_read)
    result = gate.run(root, baseline)
    assert result["status"] == "PASS_R297_OMT4_KT7_STAGE1_ADMISSION"
    assert all(row[1:] == (7, gate.MAGIC) for row in seen)

    for plant in gate.PLANTS[1:]:
        with pytest.raises((gate.Refusal, schema.Refusal)):
            gate.run(root, baseline, plant)


def test_writer_patch_selects_only_kt7():
    patch = Path(gate.__file__).with_name("l4_r175_to_r297_stage7.patch")
    text = patch.read_text(encoding="utf-8")
    assert "target_kt = 7, target_stage = 1, expected_fields = 23" in text
    assert "NEMO_L4_R297S7" in text
    assert "oracle_r297_stage1_rank" in text
