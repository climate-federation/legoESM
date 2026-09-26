"""The long ocean-only drivers must not run on analytic stand-in forcing unless
asked: the loaders fail by default, the drivers forward an explicit
``--allow-synthetic`` flag to every loader they call, and the climate
post-processor never scores SST against a synthetic WOA.

Static (AST) checks, like test_centennial_ice_shelf_reachability: main() builds
a real model and integrates, which a unit test cannot afford.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parents[3] / "scripts" / "run" / "ocean_long_runs"
_LOADERS = {"load_jra55_do", "load_woa_sss", "load_dai_trenberth", "load_woa_sst"}


def _loader_calls(path):
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _LOADERS):
            kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
            out.append((node.func.id, kw.get("allow_synthetic")))
    return out


def _has_flag(path):
    return "--allow-synthetic" in path.read_text()


@pytest.mark.parametrize("script,expected", [
    ("run_omip2.py", {"load_jra55_do"}),
    ("run_centennial_spinup.py", {"load_jra55_do", "load_woa_sss", "load_dai_trenberth"}),
])
def test_drivers_forward_allow_synthetic(script, expected):
    calls = _loader_calls(_DIR / script)
    assert {name for name, _ in calls} == expected
    assert all(kw == "args.allow_synthetic" for _, kw in calls), calls
    assert _has_flag(_DIR / script)


def test_postprocess_never_scores_against_synthetic_woa():
    calls = _loader_calls(_DIR / "postprocess_climate.py")
    assert calls == [("load_woa_sst", "False")]


@pytest.mark.parametrize("modname,fn", [
    ("legoesm.ocean.forcing.jra55_do", "load_jra55_do"),
    ("legoesm.ocean.forcing.woa", "load_woa_sst"),
    ("legoesm.ocean.forcing.woa_sss", "load_woa_sss"),
    ("legoesm.ocean.forcing.dai_trenberth", "load_dai_trenberth"),
])
def test_loaders_fail_loud_by_default(modname, fn):
    import importlib
    sig = inspect.signature(getattr(importlib.import_module(modname), fn))
    assert sig.parameters["allow_synthetic"].default is False
