"""Known-answer controls for the round-299 kt=7 W record admission."""

from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round299_w_acquisition import (
    check_record as gate,
)


def _record(rank):
    return {
        "rank": rank, "levels": [1, 1, 3, 3], "shape": [94, 152, 31],
        "origin": [1 if rank == 0 else 91, 1],
        "owned": [3, 3, 92, 150], "fields": {}, "bytes": 0,
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


def test_admission_and_all_plants_bind(tmp_path, monkeypatch):
    root, baseline = tmp_path / "root", tmp_path / "baseline"
    root.mkdir()
    baseline.mkdir()
    _families(root, baseline)

    def fake_read(path, plant="none"):
        if plant != "none":
            raise gate.Refusal(f"{plant} fired")
        return _record(0 if "rank0000" in path.name else 1)

    monkeypatch.setattr(gate, "read_record", fake_read)
    result = gate.run(root, baseline)
    assert result["status"] == "PASS_R299_OMT4_KT7_W_ADMISSION"
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.Refusal):
            gate.run(root, baseline, plant)


def test_schema_is_self_describing_and_source_complete():
    assert len(gate.NAMES) == 31
    assert gate.NAMES[10:21] == (
        "trp_u", "trp_v", "trp_w", "w_r1area", "w_area", "w_e3t0",
        "w_r3bb", "w_r3mm", "w_r3aa", "w_tmask", "w_ww",
    )
    source = Path(gate.__file__).read_text(encoding="utf-8")
    assert "struct.unpack_from(\"=4i\"" in source
    assert "count = n1 * n2 * n3" in source


def test_patch_is_additions_only_at_the_stage_program():
    patch = Path(gate.__file__).with_name("l4_r297_to_r299_w.patch")
    text = patch.read_text(encoding="utf-8")
    stage = text.split("--- stprk3_stg.F90", 1)[1]
    assert not any(line.startswith("-") for line in stage.splitlines()[2:])
    assert "expected_fields = 31" in text
    assert "NEMO_L4_R299W7" in text
    assert "'w_ww', ww" in text
