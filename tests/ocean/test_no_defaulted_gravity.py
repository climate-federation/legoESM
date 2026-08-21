"""Every hydrostatic-pressure and buoyancy-frequency call passes gravity (#1627).

A helper takes gravity with a default of the library constant. Calls that
omitted it handed the equation of state a pressure built on the DEFAULT while
the run's configuration pinned another value -- and in one function the
pressure two lines above had been built with the right one, so a single routine
disagreed with itself.

THE RULE IS PURELY SYNTACTIC, and deliberately so. Every guarded call passes
gravity explicitly; anything that does not is listed below as debt. Two earlier
versions of this test tried to be clever -- to work out from the code whether a
function COULD know the run's gravity, and excuse it if not -- and both were
wrong, in opposite directions. The first keyed on the configuration's spelling,
missed three real offenders whose functions take a plain ``g``, and then waived
their files on a written reason that was false. The second counted the library
module itself as knowing the run's value and invented two offenders that cannot
comply. Inferring where a VALUE came from by looking at the NAMES around it does
not work; "did this call pass the argument" is a question with an answer.

The consequence is that "cannot comply locally" becomes tracked debt rather
than a silent exemption. Each entry below needs a signature change to clear --
the value has to be threaded in from a caller that holds the configuration --
and that is exactly the work the list exists to keep visible.

SHRINK-ONLY. Adding an entry means shipping the defect; the list may only get
shorter. It does not check that the value passed is the RIGHT one -- that is a
value-level test, not a tripwire.

THE RULE WAS VALIDATED AGAINST THE DEFECT IT WAS WRITTEN FOR, which is the only
reason to believe a tripwire that has already been wrong twice. Run against the
tree as it stood before any of this work (commit f9bd6101a), it flags ELEVEN
call sites, including both the two this issue reports and every one a reviewer
found afterwards. Against this tree it flags four, and all four are listed
below. A rule that could not reproduce the known answer would not be worth
running on the unknown one.
"""

from __future__ import annotations

import ast
from pathlib import Path

_OCEAN = Path(__file__).resolve().parents[2] / "packages" / "ocean"

# call name -> positional index of gravity, or None if it can only be a keyword
_GUARDED = {
    "compute_hydrostatic_pressure": 5,   # rho, eta, dz, jacobian, rho_ref, g
    "compute_buoyancy_frequency_adiabatic": None,
}

# "<path>::<function>" -> why it cannot pass the run's gravity today.
# SHRINK-ONLY.  Each needs the constants configuration threaded in from its
# caller; none of them can be fixed where it stands.
_DEBT: dict[str, str] = {
    "legoesm/ocean/physics/vertical_mixing/richardson.py::"
    "richardson_vertical_mixing":
        "takes no constants configuration; its caller in the k-profile module "
        "holds one, so clearing this means adding gravity and reference "
        "density to this signature and passing them there",
    "legoesm/ocean/physics/combined.py::_make_mle":
        "the mixed-layer-eddy factory builds its pressure from the library "
        "constant and receives no configuration",
    "legoesm/ocean/physics/combined.py::physics_fn":
        "the closure inside that factory, same reason",
    "legoesm/ocean/physics/lateral_mixing/mle_mpas.py::"
    "mle_tracer_tendency_mpas":
        "the unstructured mixed-layer tendency builds its pressure from the "
        "library constant and receives no configuration",
}


def _non_compliant() -> set[str]:
    """Every guarded call that does not pass gravity, as '<path>::<function>'."""
    out = set()
    for path in sorted(_OCEAN.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:                       # pragma: no cover
            continue
        rel = path.relative_to(_OCEAN)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(fn):
                if not isinstance(call, ast.Call):
                    continue
                name = getattr(call.func, "id",
                               getattr(call.func, "attr", None))
                if name not in _GUARDED:
                    continue
                pos = _GUARDED[name]
                if "g" in {k.arg for k in call.keywords}:
                    continue
                if pos is not None and len(call.args) > pos:
                    continue
                out.add(f"{rel}::{fn.name}")
    return out


def test_the_rule_binds_on_something():
    """Non-vacuity: a rule matching no call passes for free."""
    seen = 0
    for path in _OCEAN.rglob("*.py"):
        seen += sum(path.read_text(errors="replace").count(n) for n in _GUARDED)
    assert seen >= 10, (
        f"only {seen} mentions of the guarded calls under {_OCEAN}; this rule "
        f"is not covering the code it was written for")


def test_no_new_call_defaults_the_runs_gravity():
    new = sorted(_non_compliant() - set(_DEBT))
    assert not new, (
        f"{new} call a pressure or buoyancy-frequency routine without passing "
        f"gravity, so a run pinning its own value silently gets the library "
        f"constant. Pass it, or add an entry to _DEBT with the signature "
        f"change needed to clear it (#1627).")


def test_the_debt_list_only_shrinks():
    stale = sorted(set(_DEBT) - _non_compliant())
    assert not stale, (
        f"{stale} now pass gravity — delete their _DEBT entries. The list is "
        f"shrink-only, and a stale entry is a waiver covering nothing.")
