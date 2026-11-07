"""Direct tests for the GYRE card-reconciliation gate.

The gate exists because two receipts on this branch reported step 2 of one
card eleven orders apart.  Its binding assertion is that the kt=1..10 ladder
and the from-rest YEAR harness resolve the SAME model-config object from the
SAME card -- so every assertion below is paired with a synthetic violation
that must make it fail, and the unification assertion is run through the
gate's own ``--require-unified`` exit code rather than by re-deriving it here.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GATE = (REPO / "scripts" / "validate" / "ocean_fidelity" / "testcases"
        / "nemo_testcase_l2_gyre_card_reconciliation_gate.py")


ORACLE = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10")
# Capturing a harness's program RUNS that harness up to its model
# construction, and both of them read the oracle tree on the way.  That is the
# price of executing the program instead of parsing it, and the review that
# killed the parsing reader made the price worth paying -- so the dependency is
# declared and skipped on, never worked around by going back to source text.
needs_oracle = pytest.mark.skipif(
    not (ORACLE / "output.namelist.dyn").is_file(),
    reason=f"no NEMO oracle tree at {ORACLE}")


@pytest.fixture(autouse=True)
def _allow_dirty_stamp():
    """Tests run on whatever tree the developer has, committed or not.

    The gate stamps provenance and FAILS CLOSED on a dirty tree, which is
    correct for a recorded measurement and wrong for a test -- a test that only
    passes on a clean tree is a test nobody runs while working.  The escape is
    the documented one and it is SCOPED, because this latch is process-global
    and leaking it would silently disarm the fail-closed stamp for every later
    gate in the same process.
    """
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps

    with allow_dirty_stamps(True):
        yield


@pytest.fixture(scope="module")
def gate():
    assert GATE.is_file(), GATE
    spec = importlib.util.spec_from_file_location(
        "gyre_card_reconciliation_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args):
    env = {"PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu",
           "JAX_ENABLE_X64": "1",
           "LEGOESM_GATE_ALLOW_DIRTY": "1",
           "PYTHONPATH": ":".join(str(REPO / "packages" / name)
                                  for name in ("core", "ocean", "atmosphere",
                                               "coupler", "ice", "land", "ml",
                                               "tools"))
           + ":" + str(REPO / "src"),
           "HOME": str(Path.home())}
    return subprocess.run([sys.executable, str(GATE), *args], env=env,
                          capture_output=True, text=True, cwd=str(REPO))


# --------------------------------------------- the display-only source read --
def test_construction_site_reader_names_a_line_for_each_harness(gate):
    """DISPLAY ONLY: nothing is asserted from it, but it must not be blank."""
    for path, function in ((gate.LADDER_GATE, "run"),
                           (gate.YEAR_HARNESS, "run_member"),
                           (gate.OWNERS_HARNESS, "equal_input_step")):
        text = gate.model_config_argument(path, function)
        assert text, f"{path.name}:{function} construction site unnamed"


def test_construction_site_reader_raises_when_there_is_no_model(gate, tmp_path):
    source = tmp_path / "fake_harness.py"
    source.write_text("def run():\n    return 1\n")
    with pytest.raises(gate.GateError):
        gate.model_config_argument(source, "run")


def test_capture_refuses_a_call_that_builds_no_model(gate, tmp_path):
    """A capture that returned nothing would compare nothing."""
    source = tmp_path / "fake_harness.py"
    source.write_text("def run():\n    return 1\n")
    with pytest.raises(gate.GateError):
        gate.capture_model_config("fake_harness_nomodel", source,
                                  lambda module: module.run())


# ------------------------------------------------------- the resolved object --
@needs_oracle
def test_the_three_harnesses_resolve_the_same_program(gate):
    """THE UNIFICATION GATE.  Empty table, or the campaign has two cards."""
    report = gate.config_diff()
    assert report["differing_fields"] == [], (
        "the kt=1..10 ladder and the from-rest year resolve DIFFERENT model "
        f"configurations from one card: {report['differing_fields']}")
    assert report["differing_fields_year_owners"] == [], (
        "the year-owners harness resolves a THIRD program: "
        f"{report['differing_fields_year_owners']}")


@needs_oracle
def test_a_drifted_NESTED_field_is_seen(gate):
    """Synthetic violation for the WALK: the drifted field is nested."""
    report = gate.config_diff(plant="config-drift")
    assert [row["field"] for row in report["differing_fields"]] == [
        "lateral_viscosity.A_h"]


@needs_oracle
def test_a_harness_handed_a_different_program_is_seen(gate):
    """Synthetic violation for the CAPTURE, which is the one that matters.

    A review defeated the previous source-reading version of this gate by
    rebinding the config through a local, so no plant that perturbs the
    resolved object AFTER the reader ran is worth anything.  This one changes
    what the YEAR HARNESS is handed, leaving its construction statement
    untouched, and the pre-unification table must come back.
    """
    report = gate.config_diff(plant="program-drift")
    # The legacy program is (virtual_salt_flux, fix_eta_drift=False).  Since
    # decision 35 the card itself runs fix_eta_drift=False, so the drift the
    # plant hands over is the closure row alone -- still a visible drift.
    assert sorted(row["field"] for row in report["differing_fields"]) == [
        "freshwater_closure"]


@needs_oracle
def test_flatten_walks_into_nested_configs(gate):
    """A flattener that stopped at the top level would hide most of the card."""
    _card, ladder, _year, _owners = gate.resolve_programs()
    leaves = gate.flatten(ladder)
    assert any("." in key for key in leaves), "nested configs were not walked"
    assert "freshwater_closure" in leaves and "fix_eta_drift" in leaves
    assert "lateral_viscosity.A_h" in leaves
    # Decision 35 (user, 2026-09-11): the certified card carries NO global eta
    # projection -- NEMO has none (sshwzv.f90:137 is local) -- and E-P stays a
    # real volume source (usrdef_sbc.f90:160 sets sfx = 0).  A silent flip of
    # either row puts a fixer back on the card, so both are pinned here.
    assert leaves["freshwater_closure"] == "real_freshwater"
    assert leaves["fix_eta_drift"] is False


# ----------------------------------------------------------- the oracle floor --
@needs_oracle
def test_oracle_floor_refuses_a_root_compared_with_itself(gate):
    """A floor measured against the same record is zero by construction."""
    with pytest.raises(gate.GateError):
        gate.oracle_floor(steps=2, roots=(gate.ORACLE_V1_ROOT,
                                          gate.ORACLE_V1_ROOT))


# ---------------------------------------------------------------- end to end --
@needs_oracle
@pytest.mark.slow
def test_require_unified_exits_zero_as_ci_would_run_it():
    result = _run("--config-diff", "--require-unified")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "UNIFIED" in result.stdout


@needs_oracle
@pytest.mark.slow
def test_require_unified_exits_nonzero_under_the_plant():
    """It must fail for the RIGHT reason, not by crashing.

    The previous version of this test passed vacuously: its plant raised
    AttributeError and the subprocess exited non-zero because it had died.
    The stdout assertion is what separates a refusal from a crash.
    """
    result = _run("--config-diff", "--require-unified", "--plant",
                  "program-drift")
    assert result.returncode != 0, result.stdout
    assert "DIFFERENT programs" in (result.stdout + result.stderr), (
        result.stdout + result.stderr)


@needs_oracle
def test_rk3_card_steps_with_nemo_face_shear_and_live_face_metric(gate):
    """Decision 36: the GYRE NEMO-identity card (RK3, no carried before-state)
    selects NEMO's face-native shear (now2 variant), face avm weighting and
    the live QCO face metric.  NEMO RK3 calls zdf_phy(kstp, Nbb, Nbb, Nrhs)
    (stprk3.f90:168), so the Kbb face metric is the step-entry one and the
    model must step twice (kt=1 seeds avm_k; kt=2 evaluates the step-entry
    shear) instead of refusing for a missing eta_before.  Fails on the code
    that raised."""
    import numpy as np
    card, ladder, _year, _owners = gate.resolve_programs()
    tke = ladder.physics.vertical_mixing.tke
    assert (tke.tke_shear_production, tke.tke_shear_avm_weighting,
            tke.tke_shear_metric_source, tke.tke_shear_evaluation_stage) == (
        "nemo_face_native_now2", "nemo_face", "nemo_qco_live_face", "step_entry")
    phase3 = gate._load("gyre_phase3_gate", gate.LADDER_GATE)
    model = gate._model(card, ladder)
    state = card.recipe.initial_state
    assert state.eta_before is None
    for kt in (1, 2):
        freshwater, surface = phase3._surface_forcings(card, state, kt)
        state = model.step(state, dt=card.dt_s, freshwater=freshwater,
                           surface_forcing=surface)
    fields = phase3.lego_fields(state)
    assert all(np.all(np.isfinite(np.asarray(fields[k]))) for k in ("T", "S", "u", "v", "ssh"))
