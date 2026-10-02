"""Controls for Decision 83's replacement ORCA2 rung-6 record gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round13_gate.py"
)
RUNNER = (
    SCRIPT.parent
    / "nemo_testcase_l4_orca2_hier_decks_round13_acquisition/run.sh"
)
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round13_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_real_preflight_is_exact_decision83_delta():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS_RUNG6_HAVTB0"
    assert report["replacement_delta"] == {"namzdf.nn_havtb": ["1", "0"]}
    assert report["line_delta"] == [[403, gate.OLD_LINE, gate.NEW_LINE]]
    assert report["rung7_boundary_delta"] == {
        "namzdf.ln_zdfcst": ["<reference:.false.>", ".true."],
        "namzdf.ln_zdftke": [".true.", ".false."],
        "namzdf.nn_havtb": ["1", "0"],
    }
    assert report["compiled_consequences"]["avtb_2d"] == "uniform_1"


@pytest.mark.parametrize("plant", gate.PRECHECK_PLANTS)
def test_preflight_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.preflight(plant=plant)


def test_stage_deck_and_execution_sentinel_are_exact(tmp_path):
    gate.stage_deck(tmp_path)
    gate.stage_deck(tmp_path)
    assert gate.sha256(tmp_path / "namelist_cfg") == gate.NEW_CFG_SHA
    assert gate.sha256(tmp_path / "manifest.json") == gate.sha256(gate.MANIFEST)
    assert (
        gate.hashlib.sha256(gate.execution_cfg_bytes()).hexdigest()
        == gate.NEW_EXECUTION_CFG_SHA
    )
    text = (tmp_path / "namelist_cfg").read_text()
    assert gate.NEW_LINE in text
    assert gate.OLD_LINE not in text
    assert "THIS_GROUP_MUST_NOT_BE_READ" in gate.execution_cfg_bytes().decode()


def _resolved_record(root: Path) -> None:
    (root / "namelist_ice_cfg").write_bytes(
        gate.legacy.rung7.rung8.rung9.SENTINEL.read_bytes()
    )
    (root / "ocean.output").write_text(
        "number of the last time step nn_itend = 240\n"
        "frequency of restart file nn_stock = 240\n"
        "restart logical ln_rstart = F\n"
        "ice management in the sbc nn_ice = 0\n"
        "implicit ice-ocean drag ln_drgice_imp = F\n"
        "FreshWater Budget control nn_fwb = 2\n"
        "nn_fwb_voltype = 2: Control OCEAN volume\n"
        "constant vertical mixing coefficient ln_zdfcst = T\n"
        "Turbulent Kinetic Energy closure (TKE) ln_zdftke = F\n"
        "vertical eddy viscosity rn_avm0 = 1.2000000000000000E-004\n"
        "vertical eddy diffusivity rn_avt0 = 1.2000000000000000E-005\n"
        "constant background or profile nn_avb = 0\n"
        "horizontal variation for avtb nn_havtb = 0\n"
    )
    (root / "run.user.stdout.log").write_text("STOP 0\n")
    (root / "run.user.time.log").write_text("RUN_DONE\n")


def test_resolved_consequences_and_plants_refuse(tmp_path):
    _resolved_record(tmp_path)
    assert gate.validate_resolved(tmp_path)["status"] == "PASS_RUNG6_HAVTB0_RESOLVED"
    for plant in (
        "ice-sentinel-read",
        "tke-sentinel-read",
        "resolved-consequence",
        "resolved-havtb",
    ):
        with pytest.raises((gate.GateError, gate.legacy.rung7.rung8.rung9.GateError)):
            gate.validate_resolved(tmp_path, plant=plant)


def test_runner_is_fail_closed_and_preserves_superseded_record():
    runner = RUNNER.read_text()
    assert 'readonly MODE=${1:---run}' in runner
    assert "makenemo" not in runner
    assert "/usr/bin/time" not in runner
    assert "REFUSE:" in runner
    assert 'mv "$RUN" "$SUPERSEDED_RECORD"' in runner
    assert 'mv "$CURRENT_ADMISSION" "$SUPERSEDED_ADMISSION"' in runner
    assert "--admit-existing" in runner
