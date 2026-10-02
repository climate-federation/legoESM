"""Controls for the ORCA2 hierarchy rung-1 deck and acquisition."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round11_gate.py"
)
RUNNER = SCRIPT.parent / "nemo_testcase_l4_orca2_hier_decks_round11_acquisition/run.sh"
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round11_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_real_preflight_removes_only_bbl_and_geothermal():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS_RUNG1"
    assert report["assignment_delta"] == {
        "nambbc.ln_trabbc": [".true.", ".false."],
        "nambbl.ln_trabbl": [".true.", ".false."],
    }
    assert report["compiled_consequences"] == {
        "bottom_boundary_layer": False,
        "geothermal_heating": False,
        "stage3_bbl_coefficient_call": False,
        "stage3_bbl_tracer_call": False,
        "stage3_geothermal_tracer_call": False,
    }


@pytest.mark.parametrize("plant", gate.PRECHECK_PLANTS)
def test_preflight_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.preflight(plant=plant)


def test_staged_deck_is_exact_and_idempotent(tmp_path):
    gate.stage_deck(tmp_path)
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    gate.stage_deck(tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == first
    assert (tmp_path / "namelist_cfg").read_bytes() == gate.rung1_cfg_bytes()


def test_resolved_gate_requires_both_inactive_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gate.rung2,
        "validate_resolved",
        lambda root, plant="none": {"status": "PASS_RUNG2_RESOLVED"},
    )
    (tmp_path / "ocean.output").write_text(
        "Apply a geothermal heating at ocean bottom ln_trabbc = F\n"
        "==>>>   no geothermal heat flux\n"
        "bottom boundary layer flag ln_trabbl = F\n"
    )
    assert gate.validate_resolved(tmp_path)["status"] == "PASS_RUNG1_RESOLVED"
    with pytest.raises(gate.GateError):
        gate.validate_resolved(tmp_path, plant="bbl-bbc-consequence")


def test_main_rung0_comparison_names_damping_boundary():
    diff = gate.preflight()["main_rung0_comparison"]["semantic_differences"]
    assert diff["namtra_dmp.ln_tradmp"] == [".true.", ".false."]
    assert diff["namtsd.ln_tsd_dmp"] == [".true.", ".false."]


def test_runner_reuses_binary_and_is_fail_closed():
    runner = RUNNER.read_text()
    assert "makenemo" not in runner
    assert "/usr/bin/time" not in runner
    assert "REFUSE:" in runner
    assert "--admit-existing" in runner
    assert "sha256sum -c input_files.sha256" in runner
    assert "mpirun -np 2 --oversubscribe ./nemo" in runner
