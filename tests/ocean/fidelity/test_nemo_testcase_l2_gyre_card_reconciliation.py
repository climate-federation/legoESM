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
           "PYTHONPATH": ":".join(str(REPO / "packages" / name)
                                  for name in ("core", "ocean", "atmosphere",
                                               "coupler", "ice", "land", "ml",
                                               "tools"))
           + ":" + str(REPO / "src"),
           "HOME": str(Path.home())}
    return subprocess.run([sys.executable, str(GATE), *args], env=env,
                          capture_output=True, text=True, cwd=str(REPO))


# ------------------------------------------------------------ the AST reader --
def test_reader_finds_each_harness_own_model_construction(gate):
    """Both statements must be READ, or the table is a transcription."""
    ladder = gate.model_config_argument(gate.LADDER_GATE, "run")
    year = gate.model_config_argument(gate.YEAR_HARNESS, "run_member")
    assert ladder and year
    # Whatever they are, the gate must be able to name them; a reader that
    # silently returned "" would make every row unattributable.
    assert "config" in ladder or "cfg" in ladder
    assert "config" in year or "cfg" in year


def test_reader_raises_on_a_function_that_does_not_exist(gate):
    """Synthetic violation for the reader itself."""
    with pytest.raises(gate.GateError):
        gate.model_construction_kwargs(gate.LADDER_GATE, "no_such_function")


def test_reader_raises_when_the_function_builds_no_model(gate, tmp_path):
    source = tmp_path / "fake_harness.py"
    source.write_text("def run():\n    return 1\n")
    with pytest.raises(gate.GateError):
        gate.model_config_argument(source, "run")


# ------------------------------------------------------- the resolved object --
def test_the_two_harnesses_resolve_the_same_program(gate):
    """THE UNIFICATION GATE.  Empty table, or the campaign has two cards."""
    report = gate.config_diff()
    assert report["differing_fields"] == [], (
        "the kt=1..10 ladder and the from-rest year resolve DIFFERENT model "
        f"configurations from one card: {report['differing_fields']}")


def test_a_drifted_field_is_seen(gate):
    """Synthetic violation: the unification assertion CAN fail."""
    report = gate.config_diff(plant="config-drift")
    assert [row["field"] for row in report["differing_fields"]] == ["A_h"]


def test_flatten_walks_into_nested_configs(gate):
    """A flattener that stopped at the top level would hide most of the card."""
    card, ladder, _, _, _ = gate.resolve_programs()
    leaves = gate.flatten(ladder)
    assert any("." in key for key in leaves), "nested configs were not walked"
    assert "freshwater_closure" in leaves and "fix_eta_drift" in leaves


# ----------------------------------------------------------- the oracle floor --
def test_oracle_floor_refuses_a_root_compared_with_itself(gate):
    """A floor measured against the same record is zero by construction."""
    with pytest.raises(gate.GateError):
        gate.oracle_floor(steps=2, roots=(gate.ORACLE_V1_ROOT,
                                          gate.ORACLE_V1_ROOT))


# ---------------------------------------------------------------- end to end --
@pytest.mark.slow
def test_require_unified_exits_zero_as_ci_would_run_it():
    result = _run("--config-diff", "--require-unified")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "UNIFIED" in result.stdout


@pytest.mark.slow
def test_require_unified_exits_nonzero_under_the_plant():
    result = _run("--config-diff", "--require-unified", "--plant",
                  "config-drift")
    assert result.returncode != 0, result.stdout
