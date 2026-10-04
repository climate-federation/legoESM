from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round144_initial_growth_acquisition import (
    check_record as gate,
)


def _write_record(path: Path, step: int, rank: int) -> None:
    owned_x = 90
    owned_y = 148
    origin_x = 1 if rank == 0 else 91
    header = (1, step, rank, 94, 152, 31, origin_x, 1, 3, 3,
              92, 150, 64, len(gate.FIELDS))
    chunks = [gate.MAGIC.encode().ljust(16), gate.HEADER.pack(*header)]
    for index, name in enumerate(gate.FIELDS):
        ndim = 2 if name in gate.FIELDS_2D else 3
        n3 = 1 if ndim == 2 else 31
        chunks.extend((name.encode().ljust(16), gate.GROUP.pack(
            rank, ndim, owned_x, owned_y, n3)))
        values = np.full((owned_x, owned_y, n3), step + rank + index / 100, dtype="=f8")
        if ndim == 2:
            values = values[:, :, 0]
        chunks.append(values.tobytes(order="F"))
    path.write_bytes(b"".join(chunks))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    record = tmp_path / "record"
    calibration = tmp_path / "calibration"
    baseline = tmp_path / "baseline"
    for root in (record, calibration, baseline):
        root.mkdir()
    for step in gate.STEPS:
        for rank in (0, 1):
            _write_record(record / f"oracle_r144_growth_rank{rank:04d}_kt{step:08d}.bin", step, rank)
    (record / "ORCA2_00000010_restart_0000.nc").write_bytes(b"terminal")
    for step in range(1, 11):
        for rank in (0, 1):
            name = f"ORCA2_{step:08d}_restart_{rank:04d}.nc"
            payload = f"restart-{step}-{rank}".encode()
            (baseline / name).write_bytes(payload)
            (calibration / name).write_bytes(payload)
    return record, calibration, baseline


def test_rank_complete_growth_record_passes(tmp_path: Path) -> None:
    record, calibration, baseline = _fixture(tmp_path)
    report = gate.run(record, calibration, baseline, "none")
    assert report["status"] == "PASS_R144_INITIAL_GROWTH_RECORD"
    assert len(report["records"]) == 20
    assert len(report["calibration_restart_comparisons"]) == 20


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_growth_record_plant_fires(tmp_path: Path, plant: str) -> None:
    record, calibration, baseline = _fixture(tmp_path)
    with pytest.raises(gate.Refusal):
        gate.run(record, calibration, baseline, plant)


def test_writer_and_patch_are_additions_only_and_rank_complete() -> None:
    root = Path("scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round144_initial_growth_acquisition")
    writer = (root / "l4_r144_growth.F90").read_text()
    patch = (root / "growth_round144.patch").read_text()
    assert "STATUS='NEW'" in writer
    assert "mpprank" in writer
    assert "kt >= 1 .AND. kt <= 10" in writer
    assert "WRITE(record_unit) 1, kt, mpprank" in writer
    assert all(not line.startswith("-") or line.startswith("---") for line in patch.splitlines())
