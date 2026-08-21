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

# call name -> (positional index of gravity or None, positional index of the
# reference density or None). BOTH are gated, not just gravity: these routines
# take the pair, and a call that passes the run's gravity while defaulting the
# reference density rebuilds the very self-inconsistency this exists to stop --
# a pressure on one convention and a buoyancy frequency on another, one layer
# down (GLM-5.2).
_GUARDED = {
    # rho, eta, dz, jacobian, rho_ref, g
    "compute_hydrostatic_pressure": (5, 4),
    "compute_buoyancy_frequency_adiabatic": (None, None),
    # The eddy-closure chain. A reviewer found the hole these close: one path
    # passed the run's gravity into its PRESSURE and then computed the
    # buoyancy frequency through here, where gravity was hardcoded to the
    # library constant -- the same self-inconsistency, one level deeper, and
    # invisible to a rule that only watched the two calls above.
    "compute_visbeck_kappa_gm": (None, None),
    "compute_eke_kappa_gm": (None, None),
    "compute_treguier_kappa_gm": (None, None),
    # The density that is fed INTO the pressure. A reviewer's point on #1627,
    # and the reason a fix that stopped at the pressure calls produced a
    # DIFFERENTLY wrong number rather than a right one: these build their own
    # hydrostatic integral internally, so a call that omits gravity hands the
    # equation of state a pressure the run never used, and the corrected
    # pressure downstream is then integrated over a density that is still
    # wrong.
    "compute_ocean_rho": (None, None),
    "compute_ocean_rho_and_pressure": (None, None),
}

#: Calls for which only GRAVITY is demanded. The two density helpers take a
#: reference density as well, but it is consulted ONLY on the geometric-depth
#: path and is documented as ignored on the default one, so requiring it
#: everywhere would force an argument that changes nothing and mean nothing --
#: the opposite of what a tripwire is for. Gravity enters both paths.
_GRAVITY_ONLY = frozenset({"compute_ocean_rho", "compute_ocean_rho_and_pressure"})
_DENSITY_KW = "rho_ref"

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
    "legoesm/ocean/physics/lateral_mixing/gm_redi.py::gm_redi_lateral_mixing":
        "the cubed-sphere GM/Redi entry point receives no constants "
        "configuration, so it reaches the shared eddy closure on library "
        "constants; clearing this means threading the pair from the model",
    "legoesm/ocean/physics/lateral_mixing/gm_redi_mpas.py::"
    "_visbeck_kappa_gm_mpas":
        "the unstructured Visbeck helper hardcodes its pair and has no "
        "configuration; needs the pair on its signature and at the model call",
    "legoesm/ocean/physics/lateral_mixing/mle_mpas.py::"
    "mle_tracer_tendency_mpas":
        "the unstructured mixed-layer tendency builds its pressure from the "
        "library constant and receives no configuration",
    # Exposed once the rule started resolving import aliases: this file imports
    # the density helper under another name, so its calls had never been seen.
    # Checked rather than assumed: the factory's only parameter is a
    # LateralMixingConfig, which carries no constants, and its sole caller is
    # the physics combiner -- so clearing this means adding the pair to the
    # factory's signature and to that call, the same shape as the entries above.
    "legoesm/ocean/physics/lateral_mixing/integration.py::_make_gm_redi":
        "the cubed-sphere GM/Redi factory takes only a lateral-mixing "
        "configuration, which holds no constants; clearing it means threading "
        "the pair from the combiner that builds it",
    "legoesm/ocean/physics/lateral_mixing/integration.py::physics_fn":
        "the closure inside that factory, same reason",
}


def _local_names(tree) -> dict[str, str]:
    """Map every local name back to the guarded function it was imported as.

    A file may import a guarded helper under another name -- ``compute_ocean_rho
    as _compute_rho`` is in the tree today -- and a rule that matches only the
    original spelling reports that file clean while four of its calls default
    the run's gravity. Found by a reviewer after the first version of this
    extension shipped; the alias had hidden them from the audit as well.
    """
    alias = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name in _GUARDED:
                    alias[a.asname or a.name] = a.name
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.asname and a.name.rsplit(".", 1)[-1] in _GUARDED:
                    alias[a.asname] = a.name.rsplit(".", 1)[-1]
    return alias


def _non_compliant() -> set[str]:
    """Every guarded call that does not pass gravity, as '<path>::<function>'."""
    out = set()
    for path in sorted(_OCEAN.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:                       # pragma: no cover
            continue
        aliases = _local_names(tree)
        rel = path.relative_to(_OCEAN)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(fn):
                if not isinstance(call, ast.Call):
                    continue
                name = getattr(call.func, "id",
                               getattr(call.func, "attr", None))
                name = aliases.get(name, name)
                if name not in _GUARDED:
                    continue
                g_pos, rho_pos = _GUARDED[name]
                kw = {k.arg for k in call.keywords}
                has_g = ("g" in kw) or (g_pos is not None
                                        and len(call.args) > g_pos)
                has_rho = (_DENSITY_KW in kw) or (rho_pos is not None
                                                  and len(call.args) > rho_pos)
                if has_g and (has_rho or name in _GRAVITY_ONLY):
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
        f"gravity AND a reference density, so a run pinning its own values "
        f"silently gets the library constants. Pass it, or add an entry to _DEBT with the signature "
        f"change needed to clear it (#1627).")


def test_the_debt_list_only_shrinks():
    stale = sorted(set(_DEBT) - _non_compliant())
    assert not stale, (
        f"{stale} now pass gravity — delete their _DEBT entries. The list is "
        f"shrink-only, and a stale entry is a waiver covering nothing.")
