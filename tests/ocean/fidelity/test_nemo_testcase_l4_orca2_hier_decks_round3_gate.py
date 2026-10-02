"""Controls for the ORCA2 hierarchy rung-8 acquisition gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round3_gate.py"
)
RUNNER = SCRIPT.parent / "nemo_testcase_l4_orca2_hier_decks_round3_acquisition/run.sh"
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round3_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_real_rung8_preflight_is_exact_three_selector_delta():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS_RUNG8"
    assert report["assignment_delta"] == {
        "namsbc_rnf.ln_rnf_mouth": [".true.", ".false."],
        "namzdf.ln_zdfddm": [".true.", ".false."],
        "namzdf_iwm.ln_tsdiff": [".true.", ".false."],
    }
    assert [row[0] for row in report["line_delta"]] == [162, 394, 425]
    assert report["retained_parameters"] == gate.RETAINED_PARAMETERS
    assert report["compiled_consequences"] == {
        "river_mouth_mask": 0,
        "river_mouth_levels": 0,
        "double_diffusion_called": False,
        "avs_equals_avt_before_waves": True,
        "internal_wave_salt_heat_ratio": 1,
        "internal_wave_background_reset": True,
    }


@pytest.mark.parametrize(
    "plant",
    ("deck-extra", "missing-selector", "retained-parameter", "build-pin"),
)
def test_preflight_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.preflight(plant=plant)


def test_stage_deck_is_exact_and_idempotent(tmp_path):
    gate.stage_deck(tmp_path)
    gate.stage_deck(tmp_path)
    assert gate.rung9.sha256_bytes((tmp_path / "namelist_cfg").read_bytes()) == gate.RUNG8_CFG_SHA
    assert (tmp_path / "namelist_ice_cfg").read_bytes() == gate.rung10.RUNG_ICE_CFG.read_bytes()
    text = (tmp_path / "namelist_cfg").read_text()
    assert "ln_rnf_mouth = .false." in text
    assert "ln_zdfddm   = .false." in text
    assert "ln_tsdiff   = .false." in text
    assert len((tmp_path / "SHA256SUMS").read_text().splitlines()) == 3


def _resolved_record(root: Path) -> None:
    (root / "namelist_ice_cfg").write_bytes(gate.SENTINEL.read_bytes())
    (root / "ocean.output").write_text(
        "number of the last time step nn_itend = 240\n"
        "frequency of restart file nn_stock = 240\n"
        "restart logical ln_rstart = F\n"
        "ice management in the sbc nn_ice = 0\n"
        "type of scaling under sea-ice nn_mxlice = 0\n"
        "implicit ice-ocean drag ln_drgice_imp = F\n"
        "FreshWater Budget control nn_fwb = 2\n"
        "nn_fwb_voltype = 2: Control OCEAN volume\n"
        "specific river mouths treatment ln_rnf_mouth = F\n"
        "No specific treatment at river mouths\n"
        "double diffusive mixing ln_zdfddm = F\n"
        "No  double diffusive mixing: avs = avt\n"
        "internal wave (de Lavergne et al 2017) ln_zdfiwm = T\n"
        "Differential internal wave-driven mixing (T) or not (F) = F\n"
        "Force the background value applied to avm & avt in TKE\n"
    )
    (root / "run.user.stdout.log").write_text("STOP 0\n")
    (root / "run.user.time.log").write_text("RUN_DONE\n")


def test_resolved_consequences_and_plants_refuse(tmp_path):
    _resolved_record(tmp_path)
    assert gate.validate_resolved(tmp_path)["status"] == "PASS_RUNG8_RESOLVED"
    for plant in ("ice-sentinel-read", "resolved-consequence"):
        with pytest.raises((gate.GateError, gate.rung9.GateError)):
            gate.validate_resolved(tmp_path, plant=plant)


def test_compiled_symbol_inventory_covers_river_and_wave_selectors():
    report = gate.preflight()
    assert report["compiled_symbol_files"] == {
        "ln_rnf_mouth": ["sbcrnf.f90", "zdfphy.f90"],
        "rn_hrnf": ["sbcrnf.f90"],
        "rn_avt_rnf": ["sbcrnf.f90", "zdfphy.f90"],
        "ln_tsdiff": ["zdfiwm.f90"],
    }


def test_runner_is_rebuild_free_and_fail_closed():
    runner = RUNNER.read_text()
    assert 'readonly MODE=${1:---run}' in runner
    assert "makenemo" not in runner
    assert "/usr/bin/time" not in runner
    assert "REFUSE:" in runner
    assert "compiled_zdfphy.f90" not in runner
    assert 'cp -a "$BUILD/BLD/ppsrc/nemo/${source}.f90"' in runner
