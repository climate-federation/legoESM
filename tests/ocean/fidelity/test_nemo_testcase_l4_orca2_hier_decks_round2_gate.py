"""Controls for the ORCA2 hierarchy rung-9 no-ice acquisition gate."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from netCDF4 import Dataset


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round2_gate.py"
)
RUNNER = SCRIPT.parent / "nemo_testcase_l4_orca2_hier_decks_round2_acquisition/run.sh"
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round2_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_real_rung9_preflight_has_exact_one_module_delta():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS_RUNG9"
    assert report["assignment_delta"] == {"namsbc.nn_ice": ["2", "0"]}
    assert len(report["line_delta"]) == 1
    assert report["line_delta"][0][0] == 86
    assert len(report["retained_ice_namelist_assignments_inert"]) == 91
    assert report["compiled_consequences"] == {
        "ice_fraction": 0,
        "ice_init_called": False,
        "ice_stp_called": False,
        "namelist_ice_cfg_loaded": False,
        "nn_mxlice": 0,
        "ln_drgice_imp": False,
        "nn_fwb_voltype": 2,
    }


@pytest.mark.parametrize("plant", ("deck-extra", "build-pin", "ice-artifact"))
def test_preflight_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.preflight(plant=plant)


def test_stage_deck_is_exact_and_idempotent(tmp_path):
    gate.stage_deck(tmp_path)
    gate.stage_deck(tmp_path)
    assert gate.sha256_bytes((tmp_path / "namelist_cfg").read_bytes()) == gate.RUNG9_CFG_SHA
    assert (tmp_path / "namelist_ice_cfg").read_bytes() == gate.RUNG10_ICE_CFG.read_bytes()
    assert "nn_ice      = 0" in (tmp_path / "namelist_cfg").read_text()
    assert len((tmp_path / "SHA256SUMS").read_text().splitlines()) == 3


def _fake_frame(kt: int, rank: int) -> dict[str, object]:
    return {
        "fields": {
            name: np.asarray([[kt + rank + index]], dtype=np.float64)
            for index, name in enumerate(gate.rung10.surface.FIELDS)
        }
    }


def test_frame_parser_and_inventory_plants_refuse(tmp_path, monkeypatch):
    for kt in range(1, 241):
        for rank in (0, 1):
            (tmp_path / gate.rung10.surface._record_name(kt, rank)).touch()

    def read_surface(_path, *, kt, rank, plant="none"):
        if plant in ("field-name", "truncated"):
            raise gate.rung10.surface.GateError(plant)
        return _fake_frame(kt, rank)

    monkeypatch.setattr(gate.rung10.surface, "read_surface", read_surface)
    report = gate.validate_frames(tmp_path)
    assert report == {
        "status": "SELF_DESCRIBING_FINITE",
        "frames": 480,
        "finite_field_payloads": 4800,
    }
    for plant in ("field-name", "truncated", "missing-frame", "frame-nonfinite"):
        with pytest.raises((gate.GateError, gate.rung10.surface.GateError)):
            gate.validate_frames(tmp_path, plant=plant)


def _ocean_restart(path: Path) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 2)
        dataset.createVariable("kt", "f8")[:] = 240.0
        for index, name in enumerate(gate.rung10.RESTART_FIELDS):
            dataset.createVariable(name, "f8", ("x",))[:] = (index + 1.0, -0.0)


def test_terminal_is_ocean_only_and_plants_refuse(tmp_path, capsys):
    for rank in (0, 1):
        _ocean_restart(tmp_path / f"ORCA2_00000240_restart_{rank:04d}.nc")
    assert gate.validate_terminal(tmp_path)["status"] == "FINITE_FP64_NO_ICE_PRODUCTS"
    for plant in ("terminal-nonfinite", "terminal-step"):
        with pytest.raises(gate.GateError):
            gate.validate_terminal(tmp_path, plant=plant)
        assert "STATUS PLANT-FIRED:" in capsys.readouterr().out
    (tmp_path / "ORCA2_00000240_restart_ice_0000.nc").touch()
    with pytest.raises(gate.GateError, match="ice products"):
        gate.validate_terminal(tmp_path)


def _resolved_record(root: Path, *, fwb_active: bool = True) -> None:
    (root / "namelist_ice_cfg").write_bytes(gate.SENTINEL.read_bytes())
    (root / "ocean.output").write_text(
        "number of the last time step nn_itend = 240\n"
        "frequency of restart file nn_stock = 240\n"
        "restart logical ln_rstart = F\n"
        "ice management in the sbc nn_ice = 0\n"
        "type of scaling under sea-ice nn_mxlice = 0\n"
        "implicit ice-ocean drag ln_drgice_imp = F\n"
        f"FreshWater Budget control nn_fwb = {2 if fwb_active else 0}\n"
        + ("nn_fwb_voltype = 2: Control OCEAN volume\n" if fwb_active else "")
    )
    (root / "run.user.stdout.log").write_text("STOP 0\n")
    (root / "run.user.time.log").write_text("RUN_DONE\n")


def test_unread_sentinel_and_resolved_consequence_plants_refuse(tmp_path):
    _resolved_record(tmp_path)
    assert gate.validate_resolved(tmp_path)["status"] == "PASS_NO_ICE_SENTINEL_UNREAD"
    for plant in ("ice-sentinel-read", "resolved-consequence"):
        with pytest.raises(gate.GateError):
            gate.validate_resolved(tmp_path, plant=plant)


def test_inactive_freshwater_budget_requires_volume_print_absent(tmp_path):
    _resolved_record(tmp_path, fwb_active=False)
    report = gate.validate_resolved(tmp_path)
    assert report["freshwater_budget_print_matches_activity"] is True

    with (tmp_path / "ocean.output").open("a") as stream:
        stream.write("nn_fwb_voltype = 2: Control OCEAN volume\n")
    with pytest.raises(gate.GateError, match="resolved no-ice checks failed"):
        gate.validate_resolved(tmp_path)


def test_runner_stages_committed_unread_sentinel():
    runner = RUNNER.read_text()
    assert 'namelist_ice_cfg) cp -a "$SENTINEL" "$RUN/$name"' in runner
    assert "makenemo" not in runner
    assert "/usr/bin/time" not in runner
    assert "REFUSE:" in runner


HIERARCHY_GATES = tuple(
    importlib.import_module(
        "scripts.validate.ocean_fidelity.orca2_l4."
        f"nemo_testcase_l4_orca2_hier_decks_round{round_number}_gate"
    )
    for round_number in range(1, 16)
)
HIERARCHY_PLANTS = tuple(
    (module, plant)
    for module in HIERARCHY_GATES
    for plant in module.PLANTS
    if plant != "none"
)


@pytest.mark.parametrize(("module", "plant"), HIERARCHY_PLANTS)
def test_every_hierarchy_cli_plant_reports_marker(module, plant, monkeypatch, capsys):
    def fail(*_args, **_kwargs):
        # Use a foreign hierarchy GateError to reproduce the inherited-class
        # failure that escaped round 14's explicit exception inventory.
        raise gate.GateError(f"deterministic {plant} failure")

    if module.__name__.endswith("round12_gate"):
        monkeypatch.setattr(module, "evaluate", fail)
        argv = [module.__file__, "--plant", plant]
    else:
        monkeypatch.setattr(module, "validate_record", fail)
        argv = [
            module.__file__,
            "--record",
            "/nonexistent",
            "--expect-commit",
            "0" * 40,
            "--plant",
            plant,
        ]
        if module.__name__.endswith("round7_gate"):
            argv.extend(("--calibration", "/nonexistent"))
    monkeypatch.setattr(sys, "argv", argv)

    assert module.main() != 0
    assert "STATUS PLANT-FIRED:" in capsys.readouterr().out
