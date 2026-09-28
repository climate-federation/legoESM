"""Controls for the round-69 ORCA2 month surface acquisition."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_round69_month_surface_gate.py"
)
SPEC = importlib.util.spec_from_file_location("round69_surface_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _arrays(kt: int, rank: int) -> dict[str, np.ndarray]:
    result = {}
    for index, name in enumerate(gate.FIELDS):
        shape = (3, 2, 2) if name == "rnf_tsc" else (3, 2)
        result[name] = (
            np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
            + 1000 * kt + 100 * rank + index
        )
    return result


def _surface(path: Path, kt: int, rank: int) -> dict[str, np.ndarray]:
    fields = _arrays(kt, rank)
    with path.open("wb") as handle:
        handle.write(gate.MAGIC.encode().ljust(16, b" "))
        handle.write(gate.HEADER.pack(1, kt, 1 if kt % 2 else 3, rank,
                                      len(gate.FIELDS), 2, 3, 64))
        for name in gate.FIELDS:
            values = fields[name]
            n3 = 2 if name == "rnf_tsc" else 1
            handle.write(name.encode().ljust(16, b" "))
            handle.write(gate.FIELD_HEADER.pack(3 if n3 == 2 else 2, 2, 3, n3))
            source_order = values.transpose(1, 0, 2) if n3 == 2 else values.T[..., None]
            handle.write(np.asarray(source_order, dtype="=f8").tobytes(order="F"))
    return fields


def test_preflight_patch_is_additions_only_and_applies():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["patch_removed_lines"] == 0


def test_self_describing_record_round_trip(tmp_path):
    path = tmp_path / gate._record_name(7, 1)
    expected = _surface(path, 7, 1)
    record = gate.read_surface(path, kt=7, rank=1)
    assert tuple(record["fields"]) == gate.FIELDS
    for name in gate.FIELDS:
        assert gate._raw_equal(record["fields"][name], expected[name])


@pytest.mark.parametrize("plant", ("field-name", "truncated"))
def test_schema_plants_refuse(tmp_path, plant):
    path = tmp_path / gate._record_name(1, 0)
    _surface(path, 1, 0)
    with pytest.raises(gate.GateError):
        gate.read_surface(path, kt=1, rank=0, plant=plant)


@pytest.mark.parametrize("plant", ("missing-frame", "extra-stream"))
def test_inventory_plants_refuse(tmp_path, plant):
    for rank in (0, 1):
        _surface(tmp_path / gate._record_name(1, rank), 1, rank)
    with pytest.raises(gate.GateError, match="inventory mismatch"):
        gate.validate_inventory(tmp_path, steps=range(1, 2), plant=plant)


def test_calibration_ulp_plant_refuses(tmp_path, monkeypatch):
    expected = {}
    for kt in (1, 2):
        for rank in (0, 1):
            expected[kt, rank] = _surface(
                tmp_path / gate._record_name(kt, rank), kt, rank
            )
    monkeypatch.setattr(
        gate, "_old_surface", lambda _root, kt, rank: expected[kt, rank]
    )
    assert gate.validate_calibration(
        tmp_path, tmp_path, steps=range(1, 3)
    )["field_comparisons"] == 40
    with pytest.raises(gate.GateError, match="calibration differs"):
        gate.validate_calibration(
            tmp_path, tmp_path, steps=range(1, 3), plant="calibration-ulp"
        )


def _restart(path: Path) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 2)
        dataset.createVariable("field", "f8", ("x",))[:] = (1.0, -0.0)


def test_restart_ulp_plant_changes_raw_payload(tmp_path):
    source = tmp_path / "source.nc"
    changed = tmp_path / "changed.nc"
    _restart(source)
    gate._mutated_copy(source, changed)
    with pytest.raises(gate.phase1.GateError):
        gate.phase1._netcdf_equal_except_timestamp(source, changed)
