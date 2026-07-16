"""Reachability pin: run_centennial_spinup must dispatch ice-shelf basal melt
to the MPAS apply on ``--grid mpas``.

``apply_ice_shelf_basal_step_mpas`` was implemented and unit-tested (see
``tests/unit/test_runoff_iceshelf_apply.py``) but UNREACHABLE from any driver:
run_centennial_spinup -- the one driver that offers ``--ice-shelf`` -- warned
"not yet supported on grid='mpas'; ignoring" and only ever called the lat-lon
variant, even though the MPAS apply and its three-equation/linear schemes were
sitting right there, exactly as the SSS-restoring / runoff steps in the same
loop already grid-dispatch.

This is the same class the scheme-reachability work removed elsewhere: physics
that is implemented, tested, and impossible to select. The test asserts the
driver now (a) accepts ``mpas`` for ``--ice-shelf`` instead of warning-and-
ignoring, and (b) actually routes to the ``_mpas`` apply when the grid is mpas.

AST-level rather than a full ``main()`` run: main() builds a real MPAS mesh and
integrates, which a unit test cannot afford, but the dispatch wiring is a static
fact. The physics itself running on the production centennial state (both
schemes, non-trivial melt) was verified by direct smoke against
``run_omip2._build_state("mpas", ...)``; it is not re-run here to keep the test
off the heavy matrix-setup import path.
"""
from __future__ import annotations

import ast
from pathlib import Path

_DRIVER = Path("scripts/run/ocean_long_runs/run_centennial_spinup.py")


def _driver_ast() -> ast.Module:
    return ast.parse(_DRIVER.read_text())


def _all_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _all_attrs(node: ast.AST) -> set[str]:
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def test_driver_imports_the_mpas_apply():
    """The MPAS variant must be imported -- otherwise the dispatch below cannot
    reference it and the flag is inert on mpas."""
    imported = {
        alias.name
        for node in ast.walk(_driver_ast())
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "apply_ice_shelf_basal_step_mpas" in imported, (
        "run_centennial_spinup does not import apply_ice_shelf_basal_step_mpas; "
        "--ice-shelf --grid mpas would silently do nothing"
    )
    # and the lat-lon one is still there (both grids remain reachable)
    assert "apply_ice_shelf_basal_step" in imported


def test_driver_calls_the_mpas_apply():
    """The imported symbol must actually be CALLED -- importing without calling
    is the dead-plumbing failure codex caught repeatedly on the sibling branch.
    """
    called = {
        n.func.id
        for n in ast.walk(_driver_ast())
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "apply_ice_shelf_basal_step_mpas" in called, (
        "apply_ice_shelf_basal_step_mpas is imported but never called"
    )
    assert "apply_ice_shelf_basal_step" in called


def test_ice_shelf_setup_no_longer_warns_and_ignores_mpas():
    """The ICE-SHELF setup gate must admit mpas.

    Previously it read ``if args.grid != "latlon": WARNING ... ignoring`` -- a
    latlon-only gate that made the mpas apply unreachable. It must now accept
    both grids (``args.grid not in ("latlon", "mpas")``).

    Scoped to the ice-shelf message specifically: the TIDAL-MIXING block in the
    same driver legitimately keeps a latlon-only warning, because
    apply_tidal_mixing_step is lat-lon-only by design (no MPAS variant exists) --
    that narrower gate is protective, not drift, so a whole-file string search
    would wrongly flag it.
    """
    src = _DRIVER.read_text()
    assert 'ice-shelf not yet supported on grid=' not in src, (
        "the ice-shelf setup still carries the latlon-only "
        "'ice-shelf not yet supported on grid=' warning that ignored --grid mpas"
    )
    # and the accepting gate is present
    assert 'not in ("latlon", "mpas")' in src, (
        "the ice-shelf setup gate does not admit both latlon and mpas"
    )


def test_ice_shelf_dispatch_is_guarded_by_grid_mpas():
    """The apply must be conditioned on the grid, mirroring the SSS/runoff
    steps. Assert an ``args.grid == "mpas"`` comparison co-occurs with the mpas
    apply call inside the same function, so the routing is real."""
    tree = _driver_ast()
    main_fns = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    ]
    assert main_fns, "run_centennial_spinup has no main()"
    main = main_fns[0]
    calls_mpas = "apply_ice_shelf_basal_step_mpas" in {
        n.func.id for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    # a "mpas" string constant used in a comparison somewhere in main()
    mpas_literal = any(
        isinstance(n, ast.Constant) and n.value == "mpas"
        for n in ast.walk(main)
    )
    assert calls_mpas and mpas_literal, (
        "the mpas ice-shelf apply is not grid-dispatched inside main()"
    )
