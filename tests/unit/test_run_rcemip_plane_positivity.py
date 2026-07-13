"""run_rcemip_plane state-positivity guard is wired into the integration loop.

Campaign finding: under weno5-θ' convective overshoot the advective-form tracer
advection leaves q<0 in the STATE (minTr ≈ −1.9e-4); the microphysics-read clip
(PR #966) does not repair the stored state. The serial runner now applies
``apply_positive_filter_state(state, mode="clip")`` after every ``model.step`` —
matching run_rce_mpi_long. The filter's own correctness is covered by
``tests/unit/test_tracer_positivity.py``; the end-to-end effect (minTr→0 in f64
and f32, ocean and land, all advection schemes) is covered by the CRM RCEMIP
dycore campaign. This AST guard pins that the driver keeps CALLING the filter
after the step so a refactor cannot silently drop it (mirrors
``test_run_rce_mpi_long_driver_defaults`` for the MPI driver).
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "scripts" / "run" / "run_rcemip_plane.py"


def _driver_ast() -> ast.Module:
    assert DRIVER.exists(), f"driver missing: {DRIVER}"
    return ast.parse(DRIVER.read_text())


def test_driver_imports_positivity_filter():
    """Top-level import of the shared positivity filter (not a re-derived clip)."""
    tree = _driver_ast()
    found = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "legoesm.atmosphere.dynamics.tracer_positivity"
        and any(a.name == "apply_positive_filter_state" for a in n.names)
        for n in ast.walk(tree)
    )
    assert found, (
        "run_rcemip_plane must import apply_positive_filter_state from "
        "legoesm.atmosphere.dynamics.tracer_positivity (shared, tested filter)."
    )


def _step_calls(scope):
    return [n for n in ast.walk(scope) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "step"
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "model"]


def _filter_calls(scope):
    return [n for n in ast.walk(scope) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "apply_positive_filter_state"]


def test_loop_applies_positivity_filter_after_step():
    """The state-positivity guard must run INSIDE the integration ``for`` loop,
    AFTER ``model.step(...)`` — the guard only works on the post-step state and
    only if it fires every iteration. Assert on the for-loop that contains
    ``model.step`` (not just source order anywhere in the file, per codex LOW):
    a refactor that drops it, moves it out of the loop, or above the step
    re-opens the campaign's negative-water bug (minTr −1.9e-4)."""
    tree = _driver_ast()
    loops_with_step = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.For) and _step_calls(n)
    ]
    assert loops_with_step, (
        "no for-loop containing model.step(...) — the integration loop "
        "structure changed; re-audit the positivity-guard placement."
    )
    ok = False
    for loop in loops_with_step:
        steps, filts = _step_calls(loop), _filter_calls(loop)
        if filts and max(f.lineno for f in filts) > min(s.lineno for s in steps):
            ok = True
    assert ok, (
        "the integration for-loop must call apply_positive_filter_state AFTER "
        "model.step INSIDE the loop body — the state-positivity guard is "
        "missing, moved out of the loop, or ordered before the step."
    )


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
