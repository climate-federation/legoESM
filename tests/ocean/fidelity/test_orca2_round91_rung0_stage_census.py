"""Controls for the round-91 ORCA2 rung-0 stage census."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / (
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round91_rung0_stage_census.py"
)
SPEC = importlib.util.spec_from_file_location("orca2_r91_stage_census", SCRIPT)
assert SPEC and SPEC.loader
census = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(census)


def test_load_payloads_reuses_validated_offsets(tmp_path: Path) -> None:
    path = tmp_path / "frame.bin"
    nx, ny, nz = 3, 4, 2
    with path.open("wb") as handle:
        handle.write(f"{census.frame_gate.MAGIC:<16}".encode("ascii"))
        handle.write(census.frame_gate.HEADER.pack(
            1, 1, 0, 1, 0, nx, ny, nz, 2, 1, 1, 1, 1, nx, ny, 64, 5,
        ))
        for name in census.frame_gate.FIELDS:
            handle.write(f"{name:<16}".encode("ascii"))
            dims = (2, nx, ny, 1) if name == "ssh" else (3, nx, ny, nz)
            handle.write(census.frame_gate.FIELD_HEADER.pack(*dims))
            count = dims[1] * dims[2] * dims[3]
            handle.write(np.arange(count, dtype="=f8").tobytes())
    header, arrays = census.load_payloads(path)
    assert header["stage"] == 0
    assert tuple(arrays) == census.frame_gate.FIELDS
    assert arrays["T"].shape == (nx * ny * nz,)
    assert arrays["ssh"].shape == (nx * ny,)


def test_require_refuses() -> None:
    with pytest.raises(census.CensusError):
        census.require(False, "planted refusal")
