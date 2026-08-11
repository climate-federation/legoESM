"""FESOM matrix arms must be registered ONLY when `fesom_jax` is importable.

`fesom_jax` is an out-of-tree package and NOT a declared dependency of this
repo. Registering its cases unconditionally made
``run_ocean_test_matrix.py --quick`` record an ERROR per fesom case and exit 1
on every normal installation — the arms are optional, the ocean matrix is not.

The test drives the real module twice, with the dependency probe forced both
ways, so removing the gate fails it in the installed-everywhere direction.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[3]
           / "scripts" / "matrix" / "run_ocean_test_matrix.py")


def _load(monkeypatch, *, fesom_present: bool):
    """Import the matrix module with `find_spec('fesom_jax')` forced."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *a, **kw):
        if name == "fesom_jax":
            return object() if fesom_present else None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    spec = importlib.util.spec_from_file_location("_om_fesom_opt", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


def _fesom_case_count(mod) -> int:
    import dataclasses
    return sum(1 for c in mod._build_test_matrix()
               if "fesom" in str(dataclasses.astuple(c)))


def test_fesom_arms_registered_when_the_package_is_importable(monkeypatch):
    mod = _load(monkeypatch, fesom_present=True)
    assert "fesom" in mod._OPTIONAL_GRIDS
    assert _fesom_case_count(mod) > 0


def test_fesom_arms_absent_without_the_package(monkeypatch):
    mod = _load(monkeypatch, fesom_present=False)
    assert "fesom" not in mod._OPTIONAL_GRIDS
    assert _fesom_case_count(mod) == 0
    # ... and the rest of the matrix is untouched: gating the optional arms
    # must not silently drop the grids the repo actually ships.
    assert "tripole" in mod._OPTIONAL_GRIDS
    assert len(mod._build_test_matrix()) > 0
