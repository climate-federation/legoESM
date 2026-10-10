from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round229_fold_transport_acquisition import (
    check_record as gate,
)


def _write(path, rank: int) -> None:
    nx, ny, nz = 94, 152, 31
    with path.open("wb") as handle:
        handle.write(b"NEMO_L4_R229FLD1")
        handle.write(struct.pack(
            "=15i", 1, 1, 1, rank, nx, ny, nz,
            1 + 90 * rank, 1, 3, 3, 92, 150, 64, 5))
        for name in ("zFv_after_trp", "T_Kmm", "S_Kmm", "e3t_Kmm", "tmask"):
            handle.write(name.encode("ascii").ljust(16, b" "))
            handle.write(struct.pack("=4i", 3, nx, ny, nz))
            value = 1.0 if name == "e3t_Kmm" else 0.0
            np.full(nx * ny * nz, value, np.float64).tofile(handle)


def test_rank_complete_record_and_plants(tmp_path) -> None:
    for rank in (0, 1):
        _write(tmp_path / f"oracle_r229_fold_rank{rank:04d}_kt00000001_s1.bin", rank)
    assert gate.validate(tmp_path, "none")["status"] == "PASS_R229_FOLD_RECORD"
    for plant in ("rank", "field-name", "truncation"):
        with pytest.raises(gate.GateError):
            gate.validate(tmp_path, plant)


def test_live_thickness_writer_avoids_whole_array_e3t_macro() -> None:
    root = Path(__file__).parents[3]
    acquisition = root / (
        "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round229_fold_transport_acquisition")
    patch = (acquisition / "stprk3_stg_round229.patch").read_text()
    writer = (acquisition / "l4_r229_fold.F90").read_text()
    assert "e3t(:,:,:,Kmm)" not in patch
    assert "e3t_3d(:,:,:)" in patch
    assert "r3t(:,:,Kmm)" in patch
    assert "DO jk = 1, SIZE(e3t_ref,3)" in writer
    assert "r3t_live(:,:) * tmask_live(:,:,jk)" in writer
