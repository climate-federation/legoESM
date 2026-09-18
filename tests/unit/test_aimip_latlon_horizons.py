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


def _loader_assignments(tree):
    """``{assigned name: lead expression}`` for every ``x = load_window_pairs(...)``.

    Binding the horizon to the NAME the loader's output is stored under is what
    makes the check meaningful.  Asserting only that both constants appear
    somewhere passes just as happily when the two are swapped — training on 6 h
    data while scoring against 24 h targets — which is the failure this module
    exists to catch.
    """
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "load_window_pairs"):
            continue
        lead = next((ast.unparse(kw.value) for kw in call.keywords
                     if kw.arg == "rollout_hours"), None)
        for target in node.targets:
            names = ([e.id for e in target.elts] if isinstance(target, ast.Tuple)
                     else [target.id])
            for n in names:
                out[n] = lead
    return out


def test_each_loader_is_bound_to_the_horizon_its_consumer_uses(tree):
    """The training loader feeds the trainers; the eval loader feeds evaluate().

    Swapping the two leads is a silent, plausible-looking edit that produces a
    loss rather than an error, so it is pinned by name and not merely by
    presence.
    """
    assigned = _loader_assignments(tree)
    assert assigned, "no `x = load_window_pairs(...)` assignment found"

    # The training triple is what the trainers receive.
    for name in ("ics", "targets", "forcings"):
        assert assigned.get(name) == "_ROLLOUT_HOURS", (
            f"{name} is loaded at {assigned.get(name)!r}; the trainers roll "
            "_ROLLOUT_HOURS and would score a forecast at one lead against a "
            "target loaded at another")

    # The eval bundle is what evaluate() receives.
    assert assigned.get("eval_data") == "_EVAL_HOURS", (
        f"eval_data is loaded at {assigned.get('eval_data')!r}; evaluate() "
        "scores at _EVAL_HOURS for scorecard parity")

    # No literal lead anywhere.
    for name, lead in assigned.items():
        assert lead in ("_ROLLOUT_HOURS", "_EVAL_HOURS"), (
            f"load_window_pairs -> {name} uses a lead that is not one of the "
            f"two horizon constants: {lead!r}")


def test_no_evaluate_call_overrides_the_eval_horizon(tree):
    """A call site may not re-raise the eval lead past its targets.

    ``evaluate``'s default is pinned elsewhere, but a default protects nothing
    if a caller passes ``hours=_ROLLOUT_HOURS`` explicitly — which is exactly
    the edit someone makes when "making eval match training".
    """
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "evaluate"):
            passed = next((ast.unparse(kw.value) for kw in node.keywords
                           if kw.arg == "hours"), None)
            assert passed in (None, "_EVAL_HOURS"), (
                f"evaluate(hours={passed}) at line {node.lineno}: the eval lead "
                "must stay _EVAL_HOURS, or the scorecard is no longer "
                "comparable to any number already on record")


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
