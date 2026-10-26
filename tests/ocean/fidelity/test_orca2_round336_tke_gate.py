"""Fail-closed controls for the round-336 production-arithmetic TKE record."""

from __future__ import annotations

import copy
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round336_tke_gate as gate,
)

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition"
LAYOUT = (7, 6, 5, 3, 5, 3, 4)


def _checksum(array: np.ndarray) -> int:
    words = np.ascontiguousarray(array).view(np.uint64).reshape(-1)
    return int(np.bitwise_xor.reduce(words, initial=np.uint64(0)))


def _write_record(path: Path, rank: int) -> None:
    header = dict(zip(gate.HEADER_NAMES, (2, 2, 3, 3, rank, *LAYOUT, 64, len(gate.FIELDS)),
                      strict=True))
    with path.open("wb") as stream:
        stream.write(gate.MAGIC.ljust(16).encode("ascii"))
        stream.write(struct.pack(f"={len(gate.HEADER_NAMES)}i",
                                 *(header[name] for name in gate.HEADER_NAMES)))
        for number, name in enumerate(gate.FIELDS, start=1):
            shape = gate._extent(name, header)
            array = np.full(shape, float(number + rank), dtype="=f8", order="F")
            if name == "tke_wmask":
                array.fill(1.0)
                array[-1, -1, -1] = -0.0
            if name == "entry_nmln":
                array.fill(2.0)
            if name == "tke_pdlr":
                array.fill(0.5)
            if name in {"final_mxl", "final_mxld"}:
                array.fill(2.0 if name == "final_mxl" else 3.0)
            if name in gate.NEGATIVE_ZERO_EDGE:
                array[..., -1] = -0.0
            origin = gate._origin(name, header)
            stream.write(name.ljust(24).encode("ascii"))
            stream.write(struct.pack("=7i", 2 if name in gate.INTERIOR_2D else 3,
                                     *shape, *origin))
            stream.write(struct.pack("=Q", _checksum(array)))
            stream.write(array.tobytes(order="F"))


@pytest.fixture()
def record(tmp_path: Path) -> dict:
    path = tmp_path / gate.record_name(0)
    _write_record(path, 0)
    return gate.read_record(path, expected_layout=LAYOUT)


def test_real_declared_extents_and_semantics(record: dict) -> None:
    header = record["header"]
    assert record["fields"]["entry_en"].shape == (3, 2, 5)
    assert record["fields"]["entry_avm"].shape == (7, 6, 5)
    assert gate.verify_record(record)["wet_process_interfaces"] > 0
    assert gate._layout(header) == LAYOUT


@pytest.mark.parametrize("plant", [
    "entry", "rn2", "langmuir", "pre-sweep", "zav", "pdlr", "final",
    "signed-zero",
])
def test_semantic_plants_hit_recorded_wet_boundaries(record: dict, plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.verify_record(copy.deepcopy(record), plant=plant)


@pytest.mark.parametrize("plant", [
    "header", "field-order", "truncation", "checksum", "extent", "time-level",
])
def test_layout_plants_refuse(tmp_path: Path, plant: str) -> None:
    path = tmp_path / gate.record_name(0)
    _write_record(path, 0)
    with pytest.raises(gate.GateError):
        gate.read_record(path, plant=plant, expected_layout=LAYOUT)


def test_patch_keeps_calls_outside_tke_loops_and_captures_missing_boundaries() -> None:
    patch = (TOOL / "zdftke_round336.patch").read_text()
    assert patch.count("CALL r336_tke_") == 3
    assert patch.count("R336_TKE_COPYOUT") == 12
    for name in ("r336_after_langmuir", "r336_pre_sweep_mxl", "r336_pre_pdlr_zav"):
        assert name in patch
    writer = (TOOL / "tke_round336_writer.F90").read_text()
    for name in ("entry_en", "entry_avt", "entry_rn2", "entry_rn2b"):
        assert f"'{name}'" in writer


def test_wrapper_pins_distinct_build_record_and_exit_retention() -> None:
    wrapper = (TOOL / "run.sh").read_text()
    assert "R328_ACQ_TARGET=tke336" in wrapper
    assert "ORCA2_ORCA1ICE_OMIP_L4_R336TKE" in wrapper
    assert "orca1ice_tke_internal_kt2_np2" in wrapper
    assert "trap cleanup EXIT" in wrapper
