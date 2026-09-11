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
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_JOB_DIR = _ROOT / "scripts" / "cluster" / "fv3_native"
_JOB = _JOB_DIR / "full_step_backend_parity.sbatch"
_BATCHED_JOB = _JOB_DIR / "full_step_batched_parity.sbatch"
_JOBS = [_JOB, _BATCHED_JOB]
_ARM_CEILINGS = [(_JOB, "MAX_REL"),
                 (_BATCHED_JOB, "BATCHED_VS_LOOP_MAX_REL")]
_JIT_SELECTORS = [(_JOB,
                   '[ "$BK" = jax ] && [ "$JIT" = 1 ] && JFLAG="--jit"'),
                  (_BATCHED_JOB,
                   '[ "$JIT" = 1 ] && JFLAG="--jit"')]
_SCORER = _ROOT / "scripts" / "validate" / "fv3_native" / "full_step_oracle_parity.py"


@pytest.mark.parametrize(("job", "arm_ceiling"), _ARM_CEILINGS)
def test_the_parity_job_can_report_a_failure(job, arm_ceiling):
    body = job.read_text()
    arm_args = re.findall(r'EXTRA="([^"\n]*)"', body)
    assert arm_args and all("--max-rel" in args for args in arm_args), (
        "the job compares the two backends and exits zero whatever it finds, "
        "so the parity table it produces cannot distinguish agreement from a "
        "one-percent error")
    assert re.search(r"--backend [^\n]*\$EXTRA", body), (
        "the per-arm oracle ceiling is defined but not passed to the scorer")
    assert f'os.environ["{arm_ceiling}"]' in body, (
        "the job does not gate its two arms against each other")
    assert "if worst[0] > bound:" in body, (
        "the arm-to-arm ceiling is read but does not gate the comparison")
    assert re.search(r"^if not math\.isfinite\(bound\):$", body, re.MULTILINE), (
        "a NaN or infinite value is not a ceiling")
    assert "exit $BKRC" in body, "the job discards its comparison's status"
    assert "SystemExit(0)" not in body, (
        "a missing or empty result file still exits zero, i.e. 'nothing was "
        "compared' is reported as 'the backends agree'")


@pytest.mark.parametrize(("job", "arm_ceiling"), _ARM_CEILINGS)
def test_the_parity_job_refuses_an_unset_arm_ceiling(job, arm_ceiling):
    env = os.environ.copy()
    env.pop(arm_ceiling, None)
    env["REPO"] = str(_ROOT / "missing-refusal-control-repo")
    result = subprocess.run(["bash", str(job)], cwd=_ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert arm_ceiling in result.stderr
    assert ("a comparison with no ceiling reports a number and certifies "
            "nothing" in result.stderr)


@pytest.mark.parametrize(("job", "arm_ceiling"), _ARM_CEILINGS)
def test_arm_delta_scan_excludes_bookkeeping(job, arm_ceiling, tmp_path):
    body = job.read_text()
    delimiter = "PYEOF" if job == _BATCHED_JOB else "EOF"
    full_program = body.rsplit(f"<<'{delimiter}'\n", 1)[1].split(
        f"\n{delimiter}", 1)[0]
    program = full_program.split("\nimport math, os\n", 1)[0]
    left = {"batched": False, "compiled": False,
            "residuals": {"face": {"field": {
                "rel": 1.0, "vacuous_constant_ic": False,
                "cells_over_10pct": 1}}}}
    right = {"batched": True, "compiled": False,
             "residuals": {"face": {"field": {
                 "rel": 1.5, "vacuous_constant_ic": True,
                 "cells_over_10pct": 1}}}}
    paths = [tmp_path / "left.json", tmp_path / "right.json"]
    for path, payload in zip(paths, (left, right)):
        path.write_text(json.dumps(payload))
    result = subprocess.run([sys.executable, "-", *map(str, paths), "0"],
                            input=program, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "at residuals.face.field.rel" in result.stdout
    env = os.environ.copy()
    env[arm_ceiling] = str(np.finfo(float).tiny)
    result = subprocess.run([sys.executable, "-", *map(str, paths), "0"],
                            input=full_program, env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "exceeds" in result.stdout
    for invalid in ("nan", "inf"):
        env[arm_ceiling] = invalid
        result = subprocess.run(
            [sys.executable, "-", *map(str, paths), "0"],
            input=full_program, env=env, capture_output=True, text=True)
        assert result.returncode != 0
        assert "must be finite" in result.stdout
    for path, payload in zip(paths, (
            {"batched": False, "compiled": False},
            {"batched": True, "compiled": False})):
        path.write_text(json.dumps(payload))
    result = subprocess.run([sys.executable, "-", *map(str, paths), "0"],
                            input=program, capture_output=True, text=True)
    assert result.returncode != 0
    assert "no shared numeric" in result.stdout


@pytest.mark.parametrize("job", _JOBS)
def test_a_dead_backend_stops_the_job(job):
    body = job.read_text()
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


def test_main_forwards_the_batched_arm_to_the_jax_step():
    tree = ast.parse(_SCORER.read_text())
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    call = next(n for n in ast.walk(main)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_make_jax_step")
    passed = {kw.arg: kw.value for kw in call.keywords}.get("batched")
    assert (isinstance(passed, ast.Attribute)
            and isinstance(passed.value, ast.Name)
            and passed.value.id == "args" and passed.attr == "batched"), (
        "main must forward args.batched; silently ignoring --batched would "
        "certify the wrong arm, which is worse than crashing")


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


def test_batched_flag_is_refused_on_the_numpy_backend():
    """Face batching is a JAX implementation arm, not a NumPy option."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "full_step_oracle_parity", _SCORER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["full_step_oracle_parity"] = mod
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit, match="only in the JAX dynamics lane"):
        mod.main(["--backend", "numpy", "--batched"])


def test_batched_arm_dispatch_and_compiled_cache_identity(monkeypatch):
    """Both dispatches receive the arm, and compiled arms cannot alias."""
    import importlib.util
    import sys

    from legoesm.core import fv3_cgrid_phase_3d, fv3_duo_stepper, fv3_dynamics

    spec = importlib.util.spec_from_file_location(
        "full_step_oracle_parity", _SCORER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["full_step_oracle_parity"] = mod
    spec.loader.exec_module(mod)

    monkeypatch.setattr(fv3_duo_stepper, "build_jax_duo_stepper_context",
                        lambda ctx: ctx)
    monkeypatch.setattr(fv3_cgrid_phase_3d, "state_3d_to_jax",
                        lambda state: {"x": np.stack([f["x"] for f in state])})
    seen = []

    def result(jstate, jpress, jq):
        return {"state": jstate, "press": jpress, "q": jq}

    def eager(_ctx, jstate, jpress, *, q, **kw):
        seen.append(("eager", kw["batched"]))
        return result(jstate, jpress, q)

    def builder(_ctx, _km, **kw):
        seen.append(("jit", kw["batched"]))
        return lambda jstate, jpress, jq, **_dyn: result(jstate, jpress, jq)

    monkeypatch.setattr(fv3_dynamics, "fv_dynamics_step", eager)
    monkeypatch.setattr(fv3_dynamics, "make_fv_dynamics_step_jit", builder)
    state = [{"x": np.zeros(1)} for _ in range(6)]
    press = [{k: np.zeros(1) for k in ("ps", "pe", "peln", "pk", "pkz")}
             for _ in range(6)]
    q = [[np.zeros(1)] for _ in range(6)]

    mod._make_jax_step({}, batched=True)({}, state, press, q=q, km=1)
    keys = []
    for batched in (False, True):
        step = mod._make_jax_step({}, jit=True, batched=batched)
        step({}, state, press, q=q, km=1)
        keys.append(next(iter(inspect.getclosurevars(step).nonlocals["_cache"])))

    assert seen == [("eager", True), ("jit", False), ("jit", True)]
    assert keys[0] != keys[1]
    assert ("batched", False) in keys[0]
    assert ("batched", True) in keys[1]


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


@pytest.mark.parametrize(("job", "selector"), _JIT_SELECTORS)
def test_the_job_certifies_the_compiled_path(job, selector):
    """Both jobs expose the compiled arm without changing their defaults."""
    body = job.read_text()
    assert ('JIT="${JIT:-0}"' in body and selector in body
            and re.search(r"--backend [^\n]*\$JFLAG", body)), (
        "the parity job cannot opt in to scoring the compiled solver")
