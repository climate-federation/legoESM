"""Controls for the ORCA2 hierarchy rung-4 deck and acquisition."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round8_gate.py"
)
RUNNER = SCRIPT.parent / "nemo_testcase_l4_orca2_hier_decks_round8_acquisition/run.sh"
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round8_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_real_preflight_selects_nonpenetrative_shortwave_only():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS_RUNG4"
    assert report["assignment_delta"] == {
        "namsbc.ln_traqsr": [".true.", ".false."],
        "namtra_qsr.ln_qsr_rgb": [".true.", ".false."],
        "namtra_qsr.nn_chldta": ["1", "0"],
    }
    assert report["compiled_consequences"] == {
        "shortwave_penetration": False,
        "two_band_fallback": False,
        "surface_qsr_moved_to_qns": True,
        "fraqsr_1lev": 1.0,
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
    assert (tmp_path / "namelist_cfg").read_bytes() == gate.rung4_cfg_bytes()


def test_runner_reuses_binary_and_is_fail_closed():
    runner = RUNNER.read_text()
    assert "makenemo" not in runner
    assert "/usr/bin/time" not in runner
    assert "REFUSE:" in runner
    assert "record_absent_v2" in runner
    assert "rung5_round8_admission.json" in runner
    assert "--admit-existing" in runner
    assert "sha256sum -c input_files.sha256" in runner
