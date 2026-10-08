"""Controls for the round-172 self-describing HPG acquisition."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
ACQ = (
    ROOT
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_round172_hpg8_acquisition"
)


def _module():
    path = ACQ / "check_record.py"
    spec = importlib.util.spec_from_file_location("round172_hpg8_record", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_record(path: Path) -> None:
    gate = _module()
    nx, ny, nz = 2, 3, 2
    chunks = [gate.MAGIC, struct.pack(
        "=16i", 1, 8, 3, 1, 0, nx, ny, nz, 1, 1, 1, 1, nx, ny, 64,
        len(gate.NAMES),
    )]
    for index, name in enumerate(gate.NAMES):
        ndim, n3 = (3, nz) if name in gate.THREE_D else (2, 1)
        chunks.append(name.encode("ascii").ljust(16, b" "))
        chunks.append(struct.pack("=4i", ndim, nx, ny, n3))
        values = np.full((nx, ny, n3), index + 0.25, dtype="=f8", order="F")
        chunks.append(values.tobytes(order="F"))
    path.write_bytes(b"".join(chunks))


def test_self_describing_record_round_trips(tmp_path: Path) -> None:
    gate = _module()
    record = tmp_path / "record.bin"
    _write_record(record)
    parsed = gate.read_record(record)
    assert parsed["levels"] == [3, 1]
    assert parsed["shape"] == [2, 3, 2]
    assert tuple(parsed["fields"]) == gate.NAMES


@pytest.mark.parametrize(
    "plant", ["header", "field-name", "field-dims", "truncation", "swapped-rank"],
)
def test_record_plants_fire(tmp_path: Path, plant: str) -> None:
    gate = _module()
    record = tmp_path / "record.bin"
    _write_record(record)
    if plant == "swapped-rank":
        assert gate.read_record(record, plant)["rank"] == 1
    else:
        with pytest.raises(gate.Refusal):
            gate.read_record(record, plant)


def test_writer_patch_is_additions_only_and_ranked() -> None:
    text = (ACQ / "dynhpg_round172.patch").read_text(encoding="utf-8")
    removed = [line for line in text.splitlines() if line.startswith("-") and not line.startswith("---")]
    assert removed == []
    assert "oracle_r172_hpg8_rank" in text
    assert "narea - 1" in text
    assert "STORAGE_SIZE(1._wp), 11" in text


def test_launcher_is_fail_closed_and_content_pinned() -> None:
    text = (ACQ / "run.sh").read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in text
    assert "producer_content.sha256" in text
    assert "ORCA2_OMIP_L4_R172HPG8" in text
    assert "/usr/bin/time" not in text
    assert "--fuzz=0" in text
