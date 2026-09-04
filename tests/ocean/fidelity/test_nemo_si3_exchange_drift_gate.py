"""Tests and planted violations for the SI3 exchange drift gate."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

GATE_PATH = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/nemo_si3_exchange_drift_gate.py"
SPEC = importlib.util.spec_from_file_location("nemo_si3_exchange_drift_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _write(path: Path, *, active_plant: bool = False,
           inactive_plant: bool = False, active_rcdu_plant: bool = False) -> None:
    layout, record_bytes = gate._layout(5, 5, 1)
    assert record_bytes == 2120
    with path.open("wb") as stream:
        for step in range(1, 4):
            stream.write(gate.MAGIC)
            stream.write(struct.pack("=6i", 1, step, 5, 5, 1, 64))
            values = np.zeros((record_bytes - gate.HEADER_BYTES) // 8, np.float64)
            if active_plant and step == 2:
                values[0] = 1.0
            if inactive_plant and step == 2:
                # utau_ice begins after ten reduced values; index zero is halo.
                values[10] = 1.0
            if active_rcdu_plant and step == 2:
                # rCdU_ice begins at value 72; A2D(1)'s center is index four.
                values[76] = 1.0
            values.tofile(stream)


def test_schema_is_writer_complete() -> None:
    assert len(gate.FIELD_SPECS) == 36
    assert gate._layout(5, 5, 1)[1] == 2120


def test_identical_stream_passes(tmp_path: Path) -> None:
    reference = tmp_path / "reference.bin"
    candidate = tmp_path / "candidate.bin"
    _write(reference)
    _write(candidate)
    assert gate.require_identical(reference, candidate)["classification"] == "identical"


@pytest.mark.parametrize("plant", ["active", "inactive", "active_rcdu"])
def test_planted_bit_change_exits_red(tmp_path: Path, plant: str) -> None:
    reference = tmp_path / "reference.bin"
    candidate = tmp_path / "candidate.bin"
    _write(reference)
    _write(candidate, active_plant=plant == "active",
           inactive_plant=plant == "inactive",
           active_rcdu_plant=plant == "active_rcdu")
    with pytest.raises(gate.GateError, match="not byte-identical"):
        gate.require_identical(reference, candidate)


def test_inactive_change_is_fully_accounted(tmp_path: Path) -> None:
    reference = tmp_path / "reference.bin"
    candidate = tmp_path / "candidate.bin"
    _write(reference)
    _write(candidate, inactive_plant=True)
    result = gate.compare_streams(reference, candidate)
    assert result["classification"] == "inactive/uninitialized"
    assert result["differing_bytes"] == result["accounted_differences"]
    assert result["active_differences"] == 0


def test_active_rcdu_change_is_real_field(tmp_path: Path) -> None:
    reference = tmp_path / "reference.bin"
    candidate = tmp_path / "candidate.bin"
    _write(reference)
    _write(candidate, active_rcdu_plant=True)
    result = gate.compare_streams(reference, candidate)
    assert result["classification"] == "real field"
    assert result["active_differences"] == 1
