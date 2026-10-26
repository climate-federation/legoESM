"""Guards for the ORCA2 fourth-card HPG probe.

The probe's numeric row needs ORCA2's oracle records and a disposable model
worktree, so it runs as a script with its planted control.  What is guarded
here is fail-closed behaviour that would otherwise turn a missing input into a
silently wrong row: the model-root loader must refuse a checkout that has no
ORCA2 gate module, and the scorer must call bit-inequality by bits rather than
by tolerance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TESTCASES = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
)
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l4_orca2_hpg_model_arm_probe",
    TESTCASES / "nemo_testcase_l4_orca2_hpg_model_arm_probe.py",
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_model_root_without_the_orca2_gate_is_refused(tmp_path):
    with pytest.raises(probe.ProbeError, match="missing ORCA2 gate module"):
        probe.load_orca2_gate(tmp_path)


def test_scorer_counts_a_one_ulp_difference_as_debt():
    """Tolerance would hide it; the bar here is bit equality."""
    oracle = np.full(8, 1.0e-3)
    mask = np.ones(8, dtype=bool)
    assert probe.score(oracle.copy(), oracle, mask)["status"] == "AT-BAR"
    perturbed = oracle.copy()
    perturbed[3] = np.nextafter(perturbed[3], np.inf)
    row = probe.score(perturbed, oracle, mask)
    assert row["status"] == "DEBT" and row["n_unequal"] == 1


def test_scorer_refuses_an_empty_or_non_finite_comparison():
    oracle = np.ones(4)
    with pytest.raises(probe.ProbeError, match="empty or mismatched"):
        probe.score(oracle, oracle, np.zeros(4, dtype=bool))
    bad = oracle.copy()
    bad[0] = np.nan
    with pytest.raises(probe.ProbeError, match="non-finite"):
        probe.score(bad, oracle, np.ones(4, dtype=bool))
