"""Controls for the ORCA2 hierarchy rung-5 acquisition gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round6_gate.py"
)
RUNNER = SCRIPT.parent / "nemo_testcase_l4_orca2_hier_decks_round6_acquisition/run.sh"
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round6_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_real_rung5_preflight_is_exact_runoff_delta():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS_RUNG5"
    assert report["assignment_delta"] == {"namsbc.ln_rnf": [".true.", ".false."]}
    assert report["line_delta"] == [[92, gate.RUNOFF_SELECTOR[0], gate.RUNOFF_SELECTOR[1]]]
    assert report["retained_assignments"] == gate.RETAINED
    assert report["compiled_consequences"] == {
        "runoff": False,
        "runoff_namelist_read": True,
        "runoff_initializer_active_body": False,
        "runoff_surface_call": False,
        "runoff_tracer_source": False,
        "runoff_continuity_source": False,
        "runoff_external_mode_source": False,
        "ln_rnf_mouth": False,
    }


@pytest.mark.parametrize(
    "plant", ("deck-extra", "missing-selector", "retained-selector", "build-pin")
)
def test_preflight_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.preflight(plant=plant)


def test_stage_deck_and_execution_namelist_are_exact(tmp_path):
    gate.stage_deck(tmp_path)
    gate.stage_deck(tmp_path)
    assert (
        gate.rung6.rung7.rung8.rung9.sha256_bytes((tmp_path / "namelist_cfg").read_bytes())
        == gate.RUNG5_CFG_SHA
    )
    assert (
        gate.rung6.rung7.rung8.rung9.sha256_bytes(gate.execution_cfg_bytes())
        == gate.EXECUTION_CFG_SHA
    )
    assert (
        tmp_path / "namelist_ice_cfg"
    ).read_bytes() == gate.rung6.rung10.RUNG_ICE_CFG.read_bytes()
    text = (tmp_path / "namelist_cfg").read_text()
    assert "ln_rnf      = .false." in text
    assert "ln_zdfcst   = .true." in text
    assert "ln_zdftke   = .false." in text
    assert "THIS_GROUP_MUST_NOT_BE_READ" not in text
    assert "THIS_GROUP_MUST_NOT_BE_READ" in gate.execution_cfg_bytes().decode()


def _resolved_record(root: Path) -> None:
    (root / "namelist_ice_cfg").write_bytes(gate.rung6.rung7.rung8.rung9.SENTINEL.read_bytes())
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
        "horizontal variation for avtb nn_havtb = 1\n"
        "runoff / runoff mouths ln_rnf = F\n"
    )
    (root / "output.namelist.dyn").write_text(
        "&NAMSBC_RNF\n LN_RNF_MOUTH=F,\n LN_RNF_TEM=F,\n LN_RNF_SAL=F,\n LN_RNF_ICB=F,\n/\n"
    )
    (root / "run.user.stdout.log").write_text("STOP 0\n")
    (root / "run.user.time.log").write_text("RUN_DONE\n")


def test_resolved_consequences_and_plants_refuse(tmp_path):
    _resolved_record(tmp_path)
    report = gate.validate_resolved(tmp_path)
    assert report["status"] == "PASS_RUNG5_RESOLVED"
    assert report["runoff_group_echoed"]
    for plant in (
        "ice-sentinel-read",
        "tke-sentinel-read",
        "resolved-consequence",
        "runoff-group-unread",
        "active-runoff-print",
    ):
        with pytest.raises(
            (gate.GateError, gate.rung6.GateError, gate.rung6.rung7.rung8.rung9.GateError)
        ):
            gate.validate_resolved(tmp_path, plant=plant)


def test_active_runoff_print_refuses_clean_validation(tmp_path):
    _resolved_record(tmp_path)
    output = tmp_path / "ocean.output"
    output.write_text(output.read_text() + "sbc_rnf_init : runoff\n")
    with pytest.raises(gate.GateError):
        gate.validate_resolved(tmp_path)


def test_compiled_runoff_symbol_inventory_is_closed():
    assert gate.preflight()["compiled_symbol_files"] == gate.RUNOFF_SYMBOL_FILES


def test_runner_is_rebuild_free_and_fail_closed():
    runner = RUNNER.read_text()
    assert "readonly MODE=${1:---run}" in runner
    assert "makenemo" not in runner
    assert "/usr/bin/time" not in runner
    assert "REFUSE:" in runner
    assert 'cp -a "$BUILD/BLD/ppsrc/nemo/${source}.f90"' in runner
    assert "--admit-existing" in runner
