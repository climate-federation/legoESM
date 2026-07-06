"""Catalog gate for the colliding-modons standard case (#521).

Lives OUTSIDE the x64-gated prognostic module (codex 2026-07-03 Medium):
the registry check needs no float64, so it must run in default fp32 CI —
``--test colliding_modons`` selecting a real case on EVERY grid type may
never silently regress.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_matrix_module():
    """Spec-load the matrix-runner script without sys.path pollution."""
    script = (Path(__file__).resolve().parents[2]
              / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
    name = "_modons_matrix_catalog_unit"
    spec = importlib.util.spec_from_file_location(name, script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_matrix_registers_all_four_grids():
    """``colliding_modons`` must be a REAL matrix case on every grid type
    (cube, latlon, icosahedral, spectral) — a standard cross-grid SW
    case; an exact selector matching nothing is a silent no-op."""
    M = _load_matrix_module()
    mat = M._build_test_matrix()
    cm = [t for t in mat if t.case == "colliding_modons"]
    grids = sorted(t.grid_type for t in cm)
    assert grids == ["cubed_sphere", "icosahedral", "latlon",
                     "spectral"], grids
    for t in cm:
        assert t.run_kwargs.get("test_num") == 8, t.run_kwargs
    assert M.RUNNERS.get("colliding_modons") is M.run_shallow_water
