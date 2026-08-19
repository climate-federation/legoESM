"""The backend-parity job, and what it is allowed to call a pass.

Three defects codex found in the evidence for this lane's headline claim.

1. The job that produces the parity table always exited zero. It printed the
   worst backend-to-backend difference and stopped: a one-percent wind error
   and bit-identical agreement produced the same job state. A missing result
   file was also success.
2. It scored the EAGER step while the model deploys the COMPILED one.
   Compilation is not neutral in this port -- fused multiply-adds and
   reassociation can flip an upwind selector bit, which is why one of its
   recurrences had to be unrolled -- so the published parity covered a lane
   nobody runs.
3. The array comparator checked that the two lanes had non-finite values in
   the SAME CELLS and then excluded those cells. A NaN in one lane and an
   infinity in the other therefore compared equal.

These are static checks: running the parity job needs the pinned Fortran
oracle output, which is not in the tree.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_JOB = _ROOT / "scripts" / "cluster" / "fv3_native" / "full_step_backend_parity.sbatch"
_SCORER = _ROOT / "scripts" / "validate" / "fv3_native" / "full_step_oracle_parity.py"


def test_the_parity_job_can_report_a_failure():
    body = _JOB.read_text()
    assert "MAX_REL" in body, (
        "the job compares the two backends and exits zero whatever it finds, "
        "so the parity table it produces cannot distinguish agreement from a "
        "one-percent error")
    assert "exit $BKRC" in body, "the job discards its comparison's status"
    assert "SystemExit(0)" not in body, (
        "a missing or empty result file still exits zero, i.e. 'nothing was "
        "compared' is reported as 'the backends agree'")


def test_a_dead_backend_stops_the_job():
    body = _JOB.read_text()
    assert 'exit "$RC"' in body, (
        "a backend that died leaves no result file; the job then found no "
        "shared metrics and finished successfully anyway")


def test_the_scorer_can_run_the_compiled_path():
    src = _SCORER.read_text()
    assert '"--jit"' in src, (
        "the scorer has no way to run the compiled step, so a parity claim "
        "about the shipped solver cannot be measured at all")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_make_jax_step")
    called = {n.func.attr for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "make_fv_dynamics_step_jit" in called, (
        "the jax adapter never builds the compiled step the model deploys")
    assert "fv_dynamics_step" in called, (
        "the eager path is gone; the established score would no longer be "
        "reproducible")


def test_the_compiled_flag_is_refused_on_the_other_backend():
    """A flag that silently does nothing is how a lane goes unmeasured."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "full_step_oracle_parity", _SCORER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["full_step_oracle_parity"] = mod
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit, match="backend jax"):
        mod.main(["--backend", "numpy", "--jit"])


def test_two_lanes_that_failed_differently_do_not_compare_equal():
    """The comparator's own defect, exercised.

    A tripwire left as NaN in one lane and leaked as an infinity in the other
    sits in the same cell. The masks match, so the cell was skipped and two
    fields that had diverged into different failure modes passed.
    """
    import importlib.util
    import sys

    stepper = _ROOT / "tests" / "grids" / "test_fv3_duo_stepper.py"
    spec = importlib.util.spec_from_file_location("_fv3_duo_stepper", stepper)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fv3_duo_stepper"] = mod
    spec.loader.exec_module(mod)
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, np.inf, 3.0])
    with pytest.raises(AssertionError, match="non-finite VALUES differ"):
        mod._cmp(a, b, "field", 0.0)

    # and two lanes that failed the SAME way still compare fine
    mod._cmp(a, np.array([1.0, np.nan, 3.0]), "field", 0.0)
