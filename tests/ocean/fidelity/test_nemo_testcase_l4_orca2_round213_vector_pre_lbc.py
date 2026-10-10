from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round213_vector_pre_lbc_acquisition import (
    check_record as gate,
)


def _write_record(path: Path, rank: int) -> None:
    nx, ny = 94, 152
    nimpp = 1 if rank == 0 else 91
    header = (1, 1, 1, 1, 3, rank, nx, ny, nimpp, 1,
              3, 3, 92, 150, 64, len(gate.NAMES))
    raw = bytearray(gate.MAGIC)
    raw.extend(struct.pack("=16i", *header))
    for index, name in enumerate(gate.NAMES):
        shape = (90, 148) if name == "zv_frc" else (nx, ny)
        values = np.full(shape, index + rank / 10, dtype="=f8", order="F")
        raw.extend(name.encode("ascii").ljust(16, b" "))
        raw.extend(struct.pack("=4i", 2, *shape, 1))
        raw.extend(values.tobytes(order="F"))
    path.write_bytes(raw)


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    root, baseline = tmp_path / "root", tmp_path / "baseline"
    root.mkdir()
    baseline.mkdir()
    for rank in (0, 1):
        _write_record(
            root / f"oracle_r213_vector_rank{rank:04d}_kt00000001_jn001.bin",
            rank,
        )
    for index in range(64):
        name = f"oracle_r84_frame_{index:03d}.bin"
        payload = f"frame-{index}".encode()
        (baseline / name).write_bytes(payload)
        (root / name).write_bytes(payload)
    for rank in (0, 1):
        name = f"ORCA2_00000008_restart_{rank:04d}.nc"
        payload = f"restart-{rank}".encode()
        (baseline / name).write_bytes(payload)
        (root / name).write_bytes(payload)
    return root, baseline


def test_round213_record_admits_headers_coverage_and_passivity(tmp_path: Path) -> None:
    root, baseline = _tree(tmp_path)
    result = gate.run(root, baseline)
    assert result["status"] == "PASS_R213_OMT1_VECTOR_PRE_LBC_ADMISSION"
    assert result["rank_coverage"] == "exactly-once"
    assert [row["levels"] for row in result["records"]] == [[1, 3], [1, 3]]
    assert len(result["existing_frame_comparisons"]) == 64
    assert len(result["terminal_restart_comparisons"]) == 2


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round213_record_plants_fire(tmp_path: Path, plant: str) -> None:
    root, baseline = _tree(tmp_path)
    with pytest.raises(gate.Refusal):
        gate.run(root, baseline, plant)
