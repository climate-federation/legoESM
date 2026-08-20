"""A physical constant a run PINS must not be silently defaulted (#1627).

``compute_hydrostatic_pressure`` takes ``g`` with a default of
``constants.g``.  Two calls inside the convection trigger omitted it while the
run's own configuration pinned a different value, so the equation of state was
handed a pressure built on the wrong gravity -- and ten lines away, in the same
function, the same call passed it correctly.  Nothing failed; the number was
just wrong by five parts in a hundred thousand on the NEMO-comparison cards.

THE RULE, and it is narrow on purpose: in a function that has a constants
configuration in scope -- i.e. one that KNOWS what gravity this run uses --
every call must pass it.  A call in a helper with no such configuration cannot
comply and is not asked to; those are listed below so the exemption is a
decision rather than an oversight.

This is a tripwire, not a proof: it does not check that the value passed is the
right one, only that the caller which could know did not fall back.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_OCEAN = Path(__file__).resolve().parents[2] / "packages" / "ocean"
_GUARDED = "compute_hydrostatic_pressure"
# How a constants configuration is spelled where it is in scope.
_CONFIG_NAMES = ("constants_config", "cc")

# Call sites in helpers that take no constants configuration, so they CANNOT
# pass the run's gravity and fall back to the library default. Each is a
# decision recorded here, not an oversight; giving them the run's gravity means
# threading a configuration through their signatures, which is its own change.
_NO_CONFIG_IN_SCOPE = {
    "eos.py": "compute_ocean_rho / compute_ocean_rho_and_pressure take a state, "
              "a vertical coordinate and a jacobian -- no constants config",
    "gm_redi_latlon_cgrid.py": "the Visbeck and slope helpers take raw arrays; "
                               "no constants config reaches them",
}


def _calls_with_scope():
    """(file, line, supplies_g, config_in_scope) for every guarded call."""
    out = []
    for path in sorted(_OCEAN.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:                      # pragma: no cover
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
            has_cfg = bool(names & set(_CONFIG_NAMES))
            for call in ast.walk(fn):
                if not (isinstance(call, ast.Call) and getattr(
                        call.func, "id",
                        getattr(call.func, "attr", None)) == _GUARDED):
                    continue
                kw = {k.arg for k in call.keywords}
                # signature: rho, eta, dz, jacobian, rho_ref, g, h_actual
                supplies = ("g" in kw) or len(call.args) >= 6
                out.append((path, call.lineno, supplies, has_cfg))
    return out


def test_the_guarded_call_is_actually_present():
    """Non-vacuity: a rule over an empty set passes for the wrong reason."""
    calls = _calls_with_scope()
    assert calls, f"no {_GUARDED} calls found under {_OCEAN}; has it moved?"
    assert any(cfg for *_, cfg in calls), (
        "no call site has a constants configuration in scope, so this rule "
        "cannot bind on anything")


def test_a_caller_that_knows_the_runs_gravity_passes_it():
    calls = _calls_with_scope()
    offenders = [f"{p.name}:{ln}" for p, ln, supplies, cfg in calls
                 if cfg and not supplies]
    assert not offenders, (
        f"{offenders} call {_GUARDED} with a constants configuration in scope "
        f"but no gravity, so a run that pins its own value silently gets "
        f"{'constants.g'} instead. Pass it explicitly.")


@pytest.mark.parametrize("filename", sorted(_NO_CONFIG_IN_SCOPE))
def test_every_exemption_is_still_earned(filename):
    """An exemption that stops being true is a defect wearing a waiver."""
    calls = [(p, ln, supplies, cfg) for p, ln, supplies, cfg
             in _calls_with_scope() if p.name == filename]
    assert calls, (
        f"{filename} no longer calls {_GUARDED}; drop its exemption")
    assert not any(cfg for *_, cfg in calls), (
        f"{filename} now HAS a constants configuration in scope, so it can "
        f"pass the run's gravity and its exemption is no longer earned: "
        f"{_NO_CONFIG_IN_SCOPE[filename]}")
