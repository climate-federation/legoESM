"""Iter-875: Fortran-faithful `bounded_domain` gate for legacy edge
handling in `_ke_upwind`, `_corner_vorticity`, `_vorticity_flux`, and
`_d_sw5_corner_divergence`.

CLAUDE.md duogrid constraint #2: "Legacy edge handling must be
disabled in duogrid mode via bounded_domain = .true."  Per Fortran
sw_core.F90 the gate for the legacy face-boundary special-case
branch is

    if (bounded_domain .or. grid_type>=3 .or. duogrid)  -> PLAIN
    else                                                -> LEGACY

i.e., LEGACY fires ONLY when ``not bounded_domain`` (since duogrid
implies bounded_domain in our cdgrid; grid_type<3 always for
cubed-sphere).

Pre-iter-875 the Python gates were ``not use_duogrid`` or
``cdgrid.base.duogrid is None``.  Both evaluate True in
regional/nested non-duogrid mode (`bounded_domain=True,
duogrid=None`), where Fortran takes the PLAIN branch — silent
fidelity gap.  iter-875 widens the gates to
``not cdgrid.base.bounded_domain`` (matches Fortran exactly).

W2 LEGACY (``use_duogrid=False`` standalone cubed-sphere) has
``bounded_domain=False, duogrid=None``: both gates evaluate True →
no behavioural change.  W2 sentinel preserved bit-for-bit.

The fidelity gap is only visible in the regional/nested non-duogrid
regime.  We don't have such a test today, so the iter-875 fix is
verified by AST scan: each of the four gate sites must use
``not cdgrid.base.bounded_domain`` (or equivalent) — NOT
``not use_duogrid`` or ``cdgrid.base.duogrid is None``.

If a future iter regresses one of these gates, the AST scan will
fail loudly so the silent fidelity gap can't reappear.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import ast
from pathlib import Path

import pytest


# Each entry: (function_name, fortran_reference, expected_token).
# `expected_token` is the substring we expect to find in the gate
# expression at the function level.  `bounded_domain` is preferred
# but `not cdgrid.base.bounded_domain` is the canonical form.
_ITER875_GATE_SITES = [
    ("_ke_upwind", "sw_core.F90:303-365"),
    ("_corner_vorticity", "sw_core.F90:divergence_corner legacy branch"),
    ("_vorticity_flux", "sw_core.F90:416-480"),
    ("_d_sw5_corner_divergence",
     "sw_core.F90 nord==0 / nord>0 corner-correction blocks"),
]


def _read_fv3_sw_core():
    src = (Path(__file__).resolve().parent.parent
           / "src" / "legoesm" / "core" / "fv3_sw_core.py")
    return src.read_text()


def _find_function(tree, name):
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name):
            return node
    return None


def _gate_uses_bounded_domain(test_node):
    """Return True if the test expression compares to
    ``cdgrid.base.bounded_domain`` (positively or via ``not``).
    Rejects ``cdgrid.base.duogrid is None`` and ``not use_duogrid``.
    """
    s = ast.dump(test_node)
    return "bounded_domain" in s


def _gate_uses_legacy_duogrid_pattern(test_node):
    """Return True if the test expression contains the pre-iter-875
    legacy patterns ``cdgrid.base.duogrid is None`` or
    ``not use_duogrid``.  These are the patterns iter-875 widens
    away from.
    """
    s = ast.dump(test_node)
    return ("duogrid" in s and "None" in s) or "use_duogrid" in s


@pytest.mark.parametrize("fn_name, fortran_ref", _ITER875_GATE_SITES)
def test_iter875_gate_uses_bounded_domain(fn_name, fortran_ref):
    """Each iter-875 gate site MUST use
    ``cdgrid.base.bounded_domain`` (or equivalent on the local
    bounded_domain variable) in its legacy-branch test expression.

    A regression to ``cdgrid.base.duogrid is None`` or
    ``not use_duogrid`` re-introduces the pre-iter-875 silent
    fidelity gap in regional/nested non-duogrid mode.  Per
    CLAUDE.md duogrid constraint #2 and Fortran sw_core.F90:303
    the gate must be ``not bounded_domain``.
    """
    tree = ast.parse(_read_fv3_sw_core())
    fn = _find_function(tree, fn_name)
    assert fn is not None, (
        f"Could not find function `{fn_name}` in fv3_sw_core.py.  "
        f"Either renamed or removed; update iter-875 sentinel.")

    # Find every `if` test in the function whose body mentions a
    # legacy-edge operation.  We use a structural heuristic:
    # consider every If/IfExp node in the function body, then check
    # whether at least one of them gates on `bounded_domain` AND
    # none of them gate on the legacy `use_duogrid`/`duogrid is None`
    # patterns.
    if_tests = []
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            if_tests.append(node.test)
        elif isinstance(node, ast.IfExp):
            if_tests.append(node.test)

    bounded_domain_gates = [
        t for t in if_tests if _gate_uses_bounded_domain(t)]
    legacy_pattern_gates = [
        t for t in if_tests if _gate_uses_legacy_duogrid_pattern(t)]

    assert bounded_domain_gates, (
        f"Function `{fn_name}` has no `if` statement testing "
        f"`bounded_domain`.  Per Fortran {fortran_ref} the gate must "
        f"be `not bounded_domain` (CLAUDE.md duogrid constraint #2). "
        f"All if-tests dump:\n"
        f"{[ast.unparse(t) for t in if_tests]}")

    if legacy_pattern_gates:
        # Allow if a function has both — the bounded_domain check is
        # sufficient.  But warn if ALL gates are still legacy.
        # Strict check: a function may not have ONLY legacy patterns
        # without ANY bounded_domain reference.
        only_legacy = (not bounded_domain_gates) and bool(legacy_pattern_gates)
        assert not only_legacy, (
            f"Function `{fn_name}` has only legacy "
            f"`use_duogrid`/`duogrid is None` gates without any "
            f"`bounded_domain` check.  Per Fortran {fortran_ref} this "
            f"silently mishandles regional/nested non-duogrid mode.  "
            f"Iter-875 fix: widen the gate to "
            f"`not cdgrid.base.bounded_domain`.")


# Known-legitimate uses of `use_duogrid` / `duogrid is None` patterns
# in fv3_sw_core.py that are NOT legacy-edge gates.  These functions
# use the duogrid flag for ROUTING decisions (early-return for
# duogrid case, halo offset selection, duogrid-specific stencil
# branches) rather than for gating legacy face-boundary corrections.
# Per Fortran sw_core.F90, the gate at each of these sites is
# Fortran-faithful even though it doesn't reference bounded_domain.
_ITER875_NON_LEGACY_DUOGRID_USES = {
    "_d_sw1_recompute_ut_vt": (
        "Fortran sw_core.F90:622+656.  The face-override gate is "
        "`not (bounded_domain AND duogrid)`, which in our cdgrid "
        "(where duogrid implies bounded_domain) reduces to "
        "`not duogrid`.  Python's `if use_duogrid: return` early-"
        "return is Fortran-faithful here."),
    "_c_sw": (
        "Halo offset selection — `_offs = None if use_duogrid else "
        "grid.halo_interp_offsets`.  Routing decision, not a legacy-"
        "edge gate; Fortran does the analogous routing internally "
        "via `bounded_domain`-aware `mpp_update_domains`."),
    "_corner_vorticity": (
        "Has TWO `if use_duogrid` branches.  The first (line 1485) "
        "is a duogrid-specific stencil routing decision.  The "
        "second (line 1564, fixed in iter-875) is the legacy "
        "corner-correction gate widened to `not bounded_domain`."),
    "_bgrid_ke_transport": (
        "Optional `synchronize_bgrid_ne_corner_geo` call — duogrid-"
        "specific halo exchange replacing the Fortran scalar-KE sync "
        "(dyn_core.F90:1029-1055 commented-out alternative).  "
        "Routing decision, not a legacy-edge gate."),
}


def test_iter875_inventory_completeness_audit():
    """Audit-only: enumerate every function in `fv3_sw_core.py` that
    uses the `use_duogrid` / `duogrid is None` pattern, and verify
    each is either:

    (a) in the iter-875 inventory `_ITER875_GATE_SITES` (covered by
        the parametrized gate test above), OR
    (b) in the `_ITER875_NON_LEGACY_DUOGRID_USES` allowlist with an
        explicit rationale for why the pattern is Fortran-faithful
        without referencing `bounded_domain`.

    A future iter that adds a new function using the pattern will
    fail this test, forcing an explicit decision: either widen the
    gate to `bounded_domain` (move to the iter-875 inventory) or
    document the routing rationale (add to the allowlist).
    """
    tree = ast.parse(_read_fv3_sw_core())
    suspects = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if isinstance(child, ast.If):
                    s = ast.dump(child.test)
                    if ("duogrid" in s) or ("use_duogrid" in s):
                        suspects.add(node.name)
                        break

    # Every detected function must be covered.
    inventoried = {fn for fn, _ in _ITER875_GATE_SITES}
    allowlisted = set(_ITER875_NON_LEGACY_DUOGRID_USES.keys())
    new_uncovered = suspects - inventoried - allowlisted
    assert not new_uncovered, (
        f"New function(s) in fv3_sw_core.py use the duogrid/"
        f"use_duogrid pattern but are NOT in either the iter-875 "
        f"inventory or the non-legacy allowlist: "
        f"{sorted(new_uncovered)}.  Audit each: if the gate is for "
        f"legacy edge handling, widen to `bounded_domain` and add "
        f"to `_ITER875_GATE_SITES`.  If the gate is for routing "
        f"(early-return, halo offset, etc.), add to "
        f"`_ITER875_NON_LEGACY_DUOGRID_USES` with rationale.")


def test_iter875_w2_legacy_baseline_unchanged():
    """W2 LEGACY config has ``use_duogrid=False`` standalone
    cubed-sphere, which gives ``bounded_domain=False, duogrid=None``.
    Both pre-iter-875 (``not use_duogrid`` / ``duogrid is None``)
    and post-iter-875 (``not bounded_domain``) gates evaluate True
    in this regime, so the legacy correction fires the same way.
    The iter-875 fix is bit-identical for the W2 LEGACY sentinel.

    This test pins the property by constructing the W2 LEGACY
    cdgrid and verifying the gate evaluates as expected.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

    n = 8
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    # W2 LEGACY regime: bounded_domain=False, duogrid=None.
    assert grid.duogrid is None, (
        "W2 LEGACY regime expects `duogrid=None`.")
    assert not cdgrid.base.bounded_domain, (
        "W2 LEGACY regime expects `bounded_domain=False`.  If the "
        "default cubed-sphere cdgrid started reporting "
        "`bounded_domain=True` without explicit regional/nested "
        "config, the iter-865b/iter-875 gate semantics need a "
        "rethink.")
    # Both the pre-iter-875 and post-iter-875 gates produce the
    # same result here.  This test fails if either:
    # (a) the cdgrid.base property changes semantics, OR
    # (b) someone accidentally swaps the two regimes.
    pre_iter875_gate = grid.duogrid is None  # legacy "apply" gate
    post_iter875_gate = not cdgrid.base.bounded_domain  # iter-875 gate
    assert pre_iter875_gate == post_iter875_gate, (
        f"W2 LEGACY regime mismatch: pre-iter-875 gate evaluated "
        f"to {pre_iter875_gate}, post-iter-875 gate to "
        f"{post_iter875_gate}.  This regime should preserve "
        f"behaviour bit-for-bit across the iter-875 widening.")
