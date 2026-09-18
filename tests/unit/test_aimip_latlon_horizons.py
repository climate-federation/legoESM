"""The training horizon and the evaluation horizon must each match their own data.

``run_aimip_latlon`` supervises at one lead and scores at another.  Before
2026-09-18 both were the same constant, so nothing could go out of step.  They
are now deliberately different — training rolls 24 h, evaluation still scores
the 6 h free forecast the canonical AIMIP scorecard is defined on — and that
makes a specific silent failure possible for the first time:

    ``train_physics_params``' own docstring warns that ``rollout_hours`` MUST
    equal the lead the target carries were loaded at, because
    ``single_day_rollout`` defaults to 24 h and "leaving it implicit silently
    scores a 24 h forecast against a 6 h target".

A mismatch produces no error and no NaN.  It produces a loss, and a wrong one.
These tests read the launcher's source and pin each horizon to the loader call
that supplies its data, so a future edit that changes one and forgets the other
fails here rather than in a training run whose numbers merely look poor.

Source-level (AST) rather than behavioural on purpose: the alternative needs
ERA5 on disk and a GPU, and the invariant being protected is precisely a
wiring one.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_aimip_latlon.py")


@pytest.fixture(scope="module")
def tree():
    return ast.parse(SCRIPT.read_text())


def _module_constant(tree, name):
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == name for t in node.targets)):
            return ast.literal_eval(node.value)
    raise AssertionError(f"module constant {name} not found in {SCRIPT.name}")


def _loader_calls(tree):
    """Every ``load_window_pairs(...)`` call, as {kwarg: source text}."""
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "load_window_pairs"):
            out.append({kw.arg: ast.unparse(kw.value)
                        for kw in node.keywords if kw.arg})
    return out


def test_training_horizon_is_24h_and_eval_horizon_is_6h(tree):
    """The values themselves, pinned.

    24 h was adopted after measuring that this lane's adjoint is bounded out to
    144 steps (jobs 9829660 / 9829661).  6 h for evaluation is metric parity
    with the canonical scorecard; raising it would redefine every number
    already on record rather than improve anything.
    """
    assert _module_constant(tree, "_ROLLOUT_HOURS") == 24
    assert _module_constant(tree, "_EVAL_HOURS") == 6


def test_every_loader_call_uses_one_of_the_two_horizon_constants(tree):
    """No literal lead may be passed to the loader.

    A hardcoded ``rollout_hours=6`` next to a 24 h trainer is exactly the
    mismatch this module exists to prevent, and it would read as perfectly
    ordinary code.
    """
    calls = _loader_calls(tree)
    assert len(calls) >= 2, "expected a training and an evaluation loader call"
    leads = [c.get("rollout_hours") for c in calls]
    assert all(lead in ("_ROLLOUT_HOURS", "_EVAL_HOURS") for lead in leads), (
        f"load_window_pairs called with a lead that is not one of the two "
        f"horizon constants: {leads}")
    # Both horizons must actually be used: if every call moved to one constant
    # the two-horizon design has silently collapsed back to one.
    assert set(leads) == {"_ROLLOUT_HOURS", "_EVAL_HOURS"}, (
        f"expected one training loader and one evaluation loader, got {leads}")


def test_evaluate_defaults_to_the_eval_horizon(tree):
    """``evaluate``'s default lead must be the one its targets were loaded at."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            defaults = {a.arg: ast.unparse(d) for a, d in zip(
                node.args.args[-len(node.args.defaults):], node.args.defaults)}
            assert defaults.get("hours") == "_EVAL_HOURS", (
                f"evaluate(hours=...) defaults to {defaults.get('hours')!r}; it "
                "must be _EVAL_HOURS, or the scorecard scores a forecast at one "
                "lead against a target loaded at another")
            return
    raise AssertionError("evaluate() not found")


def test_trainers_are_given_the_training_horizon(tree):
    """Both variants' trainers must receive ``_ROLLOUT_HOURS`` explicitly.

    ``train_physics_params`` / ``train_neural_gcm`` default ``rollout_hours`` to
    24 h internally.  That happens to equal the training horizon today, so an
    omitted argument would be invisible now and wrong the moment either value
    moves.  Requiring it explicitly keeps the launcher, not a library default,
    the place the horizon is decided.
    """
    trainers = {"train_physics_params", "train_neural_gcm"}
    seen = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in trainers):
            seen[node.func.id] = {kw.arg: ast.unparse(kw.value)
                                  for kw in node.keywords if kw.arg}
    assert seen.keys() == trainers, f"trainer calls not found: {trainers - seen.keys()}"
    for name, kwargs in seen.items():
        assert kwargs.get("rollout_hours") == "_ROLLOUT_HOURS", (
            f"{name}() is called with rollout_hours="
            f"{kwargs.get('rollout_hours')!r}; it must be _ROLLOUT_HOURS")
