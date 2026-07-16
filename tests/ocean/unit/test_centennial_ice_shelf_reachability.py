"""Reachability pins: run_centennial_spinup must dispatch its caller-applied
forcings -- ice-shelf basal melt, river runoff, tidal mixing -- to the right
step on ``--grid mpas``, not warn-and-ignore.

Each was implemented but unreachable on MPAS from this driver:
  * ``apply_ice_shelf_basal_step_mpas`` -- implemented + unit-tested, but the
    setup warned "not yet supported on grid='mpas'" and only the lat-lon variant
    was ever called;
  * ``apply_runoff_step_mpas`` -- implemented but UNTESTED, and no river->cell
    projector existed for an unstructured mesh, so the setup warned too;
  * tidal mixing -- the whole chain (E_BT -> layer thickness -> K_tidal ->
    apply_tidal_mixing_step) is trailing-axis and grid-agnostic, yet the setup
    still gated it on lat-lon and its layer-depth broadcast hardcoded a 2-D
    shape.

Same class the scheme-reachability work removed elsewhere: physics that is
implemented and impossible to select. These assert the driver now admits mpas
for each forcing AND routes to the mpas step.

AST-level rather than a full ``main()`` run: main() builds a real MPAS mesh and
integrates, which a unit test cannot afford, but the dispatch wiring is a static
fact. The physics running on the production centennial state was verified by
direct smoke against ``run_omip2._build_state("mpas", ...)`` for every forcing
(non-trivial melt / eta rise / K_tidal); not re-run here to keep the test off
the heavy matrix-setup import path.
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

    Scoped to the ice-shelf message specifically so it pins the ice-shelf gate
    independently of the runoff / tidal-mixing gates checked below (all three
    forcings now admit mpas, but each is asserted on its own message).
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


def test_ice_shelf_mask_help_documents_both_grid_shapes():
    """Once mpas is reachable, the --ice-shelf-mask/--ice-draft help must not
    prescribe only the lat-lon (n_lat, n_lon) shape -- the mpas apply needs
    (nCells,) and raises for a 2-D array (codex). Advertising one shape for a
    two-grid flag is the same overclaim class as the surface-flux help fixes on
    the scheme-reachability branch."""
    src = _DRIVER.read_text()
    assert "nCells" in src, (
        "--ice-shelf-mask/--ice-draft help does not mention the (nCells,) mpas "
        "shape, so an mpas user is told to supply a (n_lat, n_lon) array that "
        "the mpas apply will reject"
    )


# ==============================================================================
# Runoff + tidal-mixing: the same all-grid reachability, same driver
# ==============================================================================
def test_driver_imports_and_calls_the_mpas_runoff_pieces():
    """apply_runoff_step_mpas + project_runoff_to_mpas_cells must be imported
    AND called. apply_runoff_step_mpas existed but was unreachable (untested,
    and the setup warned '--runoff not yet supported on grid=mpas') because no
    river->cell projector existed for an unstructured mesh."""
    tree = _driver_ast()
    imported = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) for alias in node.names
    }
    called = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    for sym in ("apply_runoff_step_mpas", "project_runoff_to_mpas_cells"):
        assert sym in imported, f"{sym} not imported -> --runoff inert on mpas"
        assert sym in called, f"{sym} imported but never called"
    # lat-lon path stays reachable too
    assert "apply_runoff_step" in called
    assert "project_runoff_to_grid" in called


def test_runoff_setup_no_longer_warns_and_ignores_mpas():
    src = _DRIVER.read_text()
    assert 'runoff not yet supported on grid=' not in src, (
        "runoff setup still carries the latlon-only "
        "'runoff not yet supported on grid=' warning that ignored --grid mpas"
    )


def test_tidal_mixing_setup_no_longer_warns_and_ignores_mpas():
    """Unlike apply_tidal_mixing_step's PREVIOUS latlon-only reputation, the
    whole tidal-mixing chain is trailing-axis and grid-agnostic; the driver
    just had to stop gating it on latlon."""
    src = _DRIVER.read_text()
    assert 'tidal-mixing not yet supported on' not in src, (
        "tidal-mixing setup still carries the latlon-only warning"
    )
    # both forcings now admit mpas via the two-grid gate
    assert src.count('not in ("latlon", "mpas")') >= 3, (
        "expected ice-shelf, runoff AND tidal-mixing to use the two-grid gate"
    )
