"""Controls for the ORCA2 hierarchy rung-2 deck and acquisition."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round10_gate.py"
)
RUNNER = SCRIPT.parent / "nemo_testcase_l4_orca2_hier_decks_round10_acquisition/run.sh"
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round10_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_real_preflight_removes_only_gm_and_mle():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS_RUNG2"
    assert report["assignment_delta"] == {
        "namtra_eiv.ln_ldfeiv": [".true.", ".false."],
        "namtra_mle.ln_mle": [".true.", ".false."],
    }
    assert report["compiled_consequences"] == {
        "gm_eddy_induced_velocity": False,
        "mixed_layer_eddies": False,
        "stage3_eiv_transport_called": False,
        "stage3_mle_transport_called": False,
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
    assert (tmp_path / "namelist_cfg").read_bytes() == gate.rung2_cfg_bytes()


def test_resolved_gate_requires_both_inactive_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gate.rung3,
        "validate_resolved",
        lambda root, plant="none": {"status": "PASS_RUNG3_RESOLVED"},
    )
    (tmp_path / "ocean.output").write_text(
        "Eddy Induced Velocity (eiv) param. ln_ldfeiv = F\n"
        "==>>> eddy induced velocity param is NOT used\n"
        "use mixed layer eddy (MLE, i.e. Fox-Kemper param) (T/F) ln_mle = F\n"
        "==>>> Mixed Layer Eddy parametrisation NOT used\n"
    )
    assert gate.validate_resolved(tmp_path)["status"] == "PASS_RUNG2_RESOLVED"
    with pytest.raises(gate.GateError):
        gate.validate_resolved(tmp_path, plant="gm-mle-consequence")


def test_runner_reuses_binary_and_is_fail_closed():
    runner = RUNNER.read_text()
    assert "makenemo" not in runner
    assert "/usr/bin/time" not in runner
    assert "REFUSE:" in runner
    assert "--admit-existing" in runner
    assert "sha256sum -c input_files.sha256" in runner
    assert "mpirun -np 2 --oversubscribe ./nemo" in runner
