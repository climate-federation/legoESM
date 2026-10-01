"""Controls for the ORCA2 hierarchy rung-10 acquisition gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round1_gate.py"
)
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round1_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_real_rung10_preflight_is_exact():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["deck_difference_lines"] == 0
    assert report["cpp_keys"] == list(gate.CPP_KEYS)
    assert report["ln_spc_dyn_compiled_scope"] == "INERT_WITHOUT_key_agrif"
    assert report["ln_spc_dyn_compiled_sites"] == []


@pytest.mark.parametrize("plant", ("deck-byte", "manifest-field"))
def test_preflight_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.preflight(plant=plant)


def _fake_frame(kt: int, rank: int) -> dict[str, object]:
    fields = {
        name: np.asarray([[kt + rank + index]], dtype=np.float64)
        for index, name in enumerate(gate.surface.FIELDS)
    }
    return {"fields": fields}


def test_frame_inventory_and_ulp_plants_refuse(tmp_path, monkeypatch):
    for kt in range(1, 241):
        for rank in (0, 1):
            (tmp_path / gate.surface._record_name(kt, rank)).touch()

    def read_surface(_path, *, kt, rank, plant="none"):
        if plant in ("field-name", "truncated"):
            raise gate.surface.GateError(plant)
        return _fake_frame(kt, rank)

    monkeypatch.setattr(gate.surface, "read_surface", read_surface)
    report = gate.validate_frames(tmp_path)
    assert report["frames"] == 480
    assert report["finite_field_payloads"] == 4800
    assert report["bit_exact_field_comparisons"] == 200
    for plant in ("missing-frame", "field-name", "truncated", "operand-ulp"):
        with pytest.raises((gate.GateError, gate.surface.GateError)):
            gate.validate_frames(tmp_path, plant=plant)


def _ocean_restart(path: Path) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 2)
        dataset.createVariable("kt", "f8")[:] = 240.0
        for index, name in enumerate(gate.RESTART_FIELDS):
            dataset.createVariable(name, "f8", ("x",))[:] = (index + 1.0, -0.0)


def _ice_restart(path: Path) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 1)
        dataset.createVariable("ice", "f8", ("x",))[:] = (1.0,)


def test_terminal_controls_refuse(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    record = tmp_path / "record"
    reference.mkdir()
    record.mkdir()
    for root in (reference, record):
        for rank in (0, 1):
            _ocean_restart(root / f"ORCA2_00000240_restart_{rank:04d}.nc")
            _ice_restart(root / f"ORCA2_00000240_restart_ice_{rank:04d}.nc")
    monkeypatch.setattr(gate, "SOURCE", reference)
    assert gate.validate_terminal(record)["status"] == "BIT_EXACT_FINITE_FP64"
    with pytest.raises(gate.phase1.GateError):
        gate.validate_terminal(record, plant="restart-ulp")
    with pytest.raises(gate.GateError, match="non-finite"):
        gate.validate_terminal(record, plant="terminal-nonfinite")


def test_resolved_switch_plant_refuses(tmp_path):
    (tmp_path / "ocean.output").write_text(
        "number of the last time step nn_itend = 240\n"
        "frequency of restart file nn_stock = 240\n"
        "restart logical ln_rstart = F\n"
        "ice management in the sbc nn_ice = 2\n"
        "number of ice categories jpl = 1\n"
    )
    (tmp_path / "run.user.stdout.log").write_text("STOP 0\n")
    (tmp_path / "run.user.time.log").write_text("RUN_DONE\n")
    assert gate.validate_resolved(tmp_path)["status"] == "PASS"
    with pytest.raises(gate.GateError, match="resolved deck"):
        gate.validate_resolved(tmp_path, plant="resolved-switch")


def test_sha_inventory_plant_refuses(tmp_path):
    (tmp_path / "a").write_text("alpha")
    (tmp_path / "b").write_text("beta")
    (tmp_path / "SHA256SUMS").write_text(
        f"{gate.sha256(tmp_path / 'a')}  a\n{gate.sha256(tmp_path / 'b')}  b\n"
    )
    assert gate.validate_sha_inventory(tmp_path)["regular_files"] == 2
    with pytest.raises(gate.GateError, match="inventory differs"):
        gate.validate_sha_inventory(tmp_path, plant="sha-inventory")


def test_month_product_inventory_is_finite(tmp_path):
    for grid in gate.MONTH_GRIDS:
        for rank in (0, 1):
            path = tmp_path / f"ORCA2_30d_00010101_00010130_grid_{grid}_{rank:04d}.nc"
            with Dataset(path, "w") as dataset:
                dataset.createDimension("x", 1)
                dataset.createVariable("value", "f8", ("x",))[:] = (1.0,)
    report = gate.validate_month_products(tmp_path)
    assert report["status"] == "FINITE"
    assert len(report["files"]) == 8
