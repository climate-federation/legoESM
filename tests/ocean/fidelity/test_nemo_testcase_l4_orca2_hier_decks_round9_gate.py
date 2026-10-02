"""Controls for the ORCA2 hierarchy rung-3 deck and acquisition."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round9_gate.py"
)
RUNNER = SCRIPT.parent / "nemo_testcase_l4_orca2_hier_decks_round9_acquisition/run.sh"
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round9_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_real_preflight_selects_exact_zero_flux_arm_only():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS_RUNG3"
    assert report["compiled_consequences"] == {
        "surface_formulation": "flx",
        "usr_gyre_forcing_selected": False,
        "bulk_forcing_selected": False,
        "sea_surface_restoring": False,
        "freshwater_budget": 0,
        "zero_flux_fields": ["utau", "vtau", "qtot", "qsr", "emp"],
    }
    assert report["assignment_delta"]["namsbc.ln_flx"] == ["ABSENT", ".true."]
    assert report["assignment_delta"]["namsbc.ln_blk"] == [".true.", ".false."]


@pytest.mark.parametrize("plant", gate.PRECHECK_PLANTS)
def test_preflight_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.preflight(plant=plant)


def test_staged_deck_is_exact_and_idempotent(tmp_path):
    gate.stage_deck(tmp_path)
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    gate.stage_deck(tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == first
    assert (tmp_path / "namelist_cfg").read_bytes() == gate.rung3_cfg_bytes()


def test_runner_reuses_main_lane_zero_flux_schema_and_is_fail_closed():
    runner = RUNNER.read_text()
    assert "makenemo" not in runner
    assert "/usr/bin/time" not in runner
    assert "REFUSE:" in runner
    assert 'for name in ("utau", "vtau", "qtot", "qsr", "emp")' in runner
    assert "variable[:] = 0.0" in runner
    assert "--admit-existing" in runner
    assert "sha256sum -c input_files.sha256" in runner


def test_main_marks_inherited_terminal_gate_error_as_plant_fired(
    monkeypatch, capsys
):
    inherited_error = (
        gate.rung4.rung5_deck.rung6.rung7.rung8.rung9.GateError
    )

    def fail_record(*args, **kwargs):
        raise inherited_error("planted inherited terminal failure")

    monkeypatch.setattr(gate, "validate_record", fail_record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--record",
            "/nonexistent",
            "--expect-commit",
            "0" * 40,
            "--plant",
            "terminal-nonfinite",
        ],
    )

    assert gate.main() == 1
    assert capsys.readouterr().out == (
        "STATUS PLANT-FIRED: planted inherited terminal failure\n"
    )
