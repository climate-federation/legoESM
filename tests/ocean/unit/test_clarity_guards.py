"""Deterministic CI clarity guards (Phase G, Q5).

Mechanizable structural-clarity checks that run every PR — the enforceable
counterpart to the ``lego-modularity-tester`` dimension 9 (structural /
decomposition modularity). See ``docs/ocean_fidelity/oracle_recipe_strategy.md``.

Currently enforces a **function-LOC ceiling**: no new monolithic functions. The
allow-list is seeded with today's known offenders so the suite is green now; it
must only ever SHRINK (each decomposition removes an entry). Adding an entry
requires an explicit, reviewed change to ``LOC_ALLOW_LIST`` here.

(The two-source-of-truth and deprecated-but-live config detectors from the
strategy doc are handled in Q7 / the constant-discipline audit, where curation
avoids false positives — e.g. g/rho_0 legitimately appear in several configs.)
"""

from __future__ import annotations

import ast
import pathlib

OCEAN_ROOT = (
    pathlib.Path(__file__).resolve().parents[3] / "src" / "legoesm" / "ocean"
)
MAX_FUNCTION_LOC = 400

# (function_name, path relative to src/legoesm) of functions that currently
# exceed MAX_FUNCTION_LOC. BASELINE ONLY — this set must SHRINK as functions
# are decomposed (e.g. Q8 splits latlon_cgrid_ocean_baroclinic_tendencies into
# named substages). Growing it requires a reviewed change here.
LOC_ALLOW_LIST = {
    ("latlon_cgrid_ocean_baroclinic_tendencies",
     "ocean/dynamics/ocean_pe_latlon_cgrid.py"),
    ("mpas_ocean_baroclinic_tendencies", "ocean/dynamics/ocean_pe_mpas.py"),
    ("_step_impl", "ocean/dynamics/ocean_model_latlon_cgrid.py"),
    ("_step_impl", "ocean/dynamics/ocean_model_mpas.py"),
    ("spectral_ocean_tendencies", "ocean/dynamics/spectral_ocean_pe.py"),
}


def _oversized_functions(root: pathlib.Path, ceiling: int):
    """Return [(name, relpath, loc)] for every function/method over ``ceiling``."""
    out = []
    for p in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        rel = "ocean/" + str(p.relative_to(root))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                loc = (node.end_lineno or node.lineno) - node.lineno + 1
                if loc > ceiling:
                    out.append((node.name, rel, loc))
    return out


def test_no_new_monolithic_ocean_functions():
    """No ocean function may exceed MAX_FUNCTION_LOC unless allow-listed.
    Decompose it (preferred), or — with review — add it to LOC_ALLOW_LIST.
    Enforces the dimension-9 structural-modularity rule."""
    offenders = _oversized_functions(OCEAN_ROOT, MAX_FUNCTION_LOC)
    new = [(n, r, loc) for (n, r, loc) in offenders if (n, r) not in LOC_ALLOW_LIST]
    assert not new, (
        f"New function(s) exceed the {MAX_FUNCTION_LOC}-LOC ceiling — decompose, "
        f"or add to LOC_ALLOW_LIST with review:\n"
        + "\n".join(f"  {loc} LOC  {n}  {r}" for n, r, loc in new)
    )


def test_loc_allow_list_has_no_stale_entries():
    """The allow-list must not name functions that are no longer oversized (or
    no longer exist) — it should only shrink. Forces removing an entry once its
    function is decomposed below the ceiling."""
    offenders = {(n, r) for (n, r, _loc) in _oversized_functions(OCEAN_ROOT, MAX_FUNCTION_LOC)}
    stale = LOC_ALLOW_LIST - offenders
    assert not stale, (
        f"LOC_ALLOW_LIST has stale entries (function decomposed/removed — drop "
        f"from the list): {sorted(stale)}"
    )
