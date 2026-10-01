"""Controls for the ORCA2 hierarchy rung-6 acquisition gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round5_gate.py"
)
RUNNER = SCRIPT.parent / "nemo_testcase_l4_orca2_hier_decks_round5_acquisition/run.sh"
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round5_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_real_rung6_preflight_is_exact_constant_closure_delta():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS_RUNG6"
    assert report["assignment_delta"] == {
        "namzdf.ln_zdfcst": ["<reference:.false.>", ".true."],
        "namzdf.ln_zdftke": [".true.", ".false."],
    }
    assert [row[0] for row in report["line_delta"]] == [390, 391]
    assert report["retained_parameters"] == gate.RETAINED_PARAMETERS
    assert report["compiled_consequences"] == {
        "closure": "np_CST",
        "tke_initializer_called": False,
        "tke_step_called": False,
        "shear_production_computed": False,
        "tke_restart_written": False,
        "rn_avm0": 1.2e-4,
        "rn_avt0": 1.2e-5,
        "nn_avb": 0,
        "nn_havtb": 1,
    }


@pytest.mark.parametrize(
    "plant",
    (
        "deck-extra",
        "missing-constant-selector",
        "missing-tke-selector",
        "retained-coefficient",
        "build-pin",
    ),
)
def test_preflight_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.preflight(plant=plant)


def test_stage_deck_and_execution_sentinel_are_exact(tmp_path):
    gate.stage_deck(tmp_path)
    gate.stage_deck(tmp_path)
    assert gate.rung7.rung8.rung9.sha256_bytes((tmp_path / "namelist_cfg").read_bytes()) == gate.RUNG6_CFG_SHA
    assert gate.rung7.rung8.rung9.sha256_bytes(gate.execution_cfg_bytes()) == gate.EXECUTION_CFG_SHA
    assert (tmp_path / "namelist_ice_cfg").read_bytes() == gate.rung10.RUNG_ICE_CFG.read_bytes()
    text = (tmp_path / "namelist_cfg").read_text()
    assert "ln_zdfcst   = .true." in text
    assert "ln_zdftke   = .false." in text
    assert "THIS_GROUP_MUST_NOT_BE_READ" not in text
    assert "THIS_GROUP_MUST_NOT_BE_READ" in gate.execution_cfg_bytes().decode()
    assert len((tmp_path / "SHA256SUMS").read_text().splitlines()) == 3


def _resolved_record(root: Path) -> None:
    (root / "namelist_ice_cfg").write_bytes(gate.rung7.rung8.rung9.SENTINEL.read_bytes())
    (root / "ocean.output").write_text(
        "number of the last time step nn_itend = 240\n"
        "frequency of restart file nn_stock = 240\n"
        "restart logical ln_rstart = F\n"
        "ice management in the sbc nn_ice = 0\n"
        "type of scaling under sea-ice nn_mxlice = 0\n"
        "implicit ice-ocean drag ln_drgice_imp = F\n"
        "nn_fwb_voltype = 2: Control OCEAN volume\n"
        "constant vertical mixing coefficient ln_zdfcst = T\n"
        "Turbulent Kinetic Energy closure (TKE) ln_zdftke = F\n"
        "vertical eddy viscosity rn_avm0 = 1.2000000000000000E-004\n"
        "vertical eddy diffusivity rn_avt0 = 1.2000000000000000E-005\n"
        "constant background or profile nn_avb = 0\n"
        "horizontal variation for avtb nn_havtb = 1\n"
    )
    (root / "run.user.stdout.log").write_text("STOP 0\n")
    (root / "run.user.time.log").write_text("RUN_DONE\n")


def test_resolved_consequences_and_plants_refuse(tmp_path):
    _resolved_record(tmp_path)
    assert gate.validate_resolved(tmp_path)["status"] == "PASS_RUNG6_RESOLVED"
    for plant in ("ice-sentinel-read", "tke-sentinel-read", "resolved-consequence"):
        with pytest.raises((gate.GateError, gate.rung7.rung8.rung9.GateError)):
            gate.validate_resolved(tmp_path, plant=plant)


def test_compiled_selector_inventory_is_closed():
    report = gate.preflight()
    assert report["compiled_selector_files"] == {
        "ln_zdfcst": ["zdf_oce.f90", "zdfphy.f90"],
        "ln_zdftke": ["asmbkg.f90", "dia25h.f90", "zdf_oce.f90", "zdfphy.f90"],
    }


def test_runner_is_rebuild_free_and_fail_closed():
    runner = RUNNER.read_text()
    assert 'readonly MODE=${1:---run}' in runner
    assert "makenemo" not in runner
    assert "/usr/bin/time" not in runner
    assert "REFUSE:" in runner
    assert 'cp -a "$BUILD/BLD/ppsrc/nemo/${source}.f90"' in runner
    assert "THIS_GROUP_MUST_NOT_BE_READ" in runner
