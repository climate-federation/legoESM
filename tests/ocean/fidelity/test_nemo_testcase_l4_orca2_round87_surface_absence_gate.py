"""Controls for the round-87 ORCA2 rung-0 surface absence gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
GATE_PATH = ROOT / (
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round87_surface_absence_gate.py"
)
SPEC = importlib.util.spec_from_file_location("orca2_r87_surface_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _write_switches(root: Path) -> tuple[Path, Path]:
    namelist = root / "namelist_cfg"
    namelist.write_text(
        "&namsbc\n ln_rnf = .false.\n nn_ice = 0\n/\n"
        "&namberg\n ln_icebergs = .false.\n/\n"
    )
    output = root / "ocean.output"
    output.write_text(
        " runoff / runoff mouths                     ln_rnf        =  F\n"
        " ice management in the sbc (=0/1/2/3)       nn_ice        = 0\n"
        "    ==>>>   No icebergs used\n"
    )
    return namelist, output


def _write_record(path: Path, extra_absent: str | None = None) -> None:
    nx, ny = 3, 4
    with path.open("wb") as handle:
        handle.write(f"{gate.MAGIC:<16}".encode("ascii"))
        handle.write(gate.HEADER.pack(2, 1, 1, nx, ny, 2, 2, 64, 35, 0, 0, 0))
        for index, name in enumerate(gate.FIELDS):
            handle.write(f"{name:<16}".encode("ascii"))
            absent = name in gate.RUNOFF_FIELDS | gate.ICEBERG_FIELDS
            absent = absent or name == extra_absent
            if absent:
                handle.write(gate.FIELD_HEADER.pack(0, 0, 0, 0))
            else:
                handle.write(gate.FIELD_HEADER.pack(2, nx, ny, 1))
                values = np.arange(nx * ny, dtype="=f8") + index
                handle.write(values.tobytes())


def test_preflight_is_additions_only_and_combines_with_frames() -> None:
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["removed_source_lines"] == 0
    assert report["surface_fields"] == 35
    assert report["explicit_absent_calls"] == 10
    assert report["frame_calls"] == 4


def test_absence_census_and_corruption_plants(tmp_path: Path) -> None:
    record = tmp_path / "surface.bin"
    _write_record(record)
    namelist, output = _write_switches(tmp_path)
    report = gate.read_surface(record, namelist, output)
    assert report["present_count"] == 25
    assert report["absent_count"] == 10
    assert set(report["absent"]) == gate.RUNOFF_FIELDS | gate.ICEBERG_FIELDS
    for name in ("snwice_mass", "snwice_mass_b", "snwice_fmass", "rCdU_ice"):
        assert report["fields"][name]["status"] == "PRESENT"
    for plant in (
        "header", "field-name", "truncation", "nonfinite",
        "absent-as-zero", "owner-on",
    ):
        with pytest.raises(gate.GateError):
            gate.read_surface(record, namelist, output, plant)


def test_absent_core_field_is_refused(tmp_path: Path) -> None:
    record = tmp_path / "surface.bin"
    _write_record(record, extra_absent="qsr")
    namelist, output = _write_switches(tmp_path)
    with pytest.raises(gate.GateError, match="owner is on but field is ABSENT"):
        gate.read_surface(record, namelist, output)
