"""Direct test of scripts/validate/canopy_fallback_gradient_fd.py (no solve)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest

_PY = (Path(__file__).resolve().parents[2]
       / "scripts" / "validate" / "canopy_fallback_gradient_fd.py")

pytestmark = pytest.mark.skipif(not jax.config.read("jax_enable_x64"),
                                reason="the probe asserts x64 at import")


def _load():
    spec = importlib.util.spec_from_file_location("canopy_fallback_gradient_fd", _PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_forcing_perturbations_reach_the_fields():
    mod = _load()
    f0 = mod._forcing(0.0, 0.0)
    f1 = mod._forcing(0.1, 2.0)
    np.testing.assert_allclose(np.asarray(f1.sw_down / f0.sw_down), 1.1)
    np.testing.assert_allclose(np.asarray(f1.T_lowest - f0.T_lowest), 2.0)


def test_measure_restores_solver_and_tolerances(monkeypatch):
    mod = _load()
    tol = (mod.canopy_solver._LM_RTOL, mod.canopy_solver._LM_ATOL)
    monkeypatch.setattr(mod, "_setup", lambda *a: (_ for _ in ()).throw(RuntimeError))
    with pytest.raises(RuntimeError):
        mod._measure(1, 307.0, True, near_root=True)
    assert mod.canopy_solver._make_implicit_newton_solver is mod._ORIG
    assert (mod.canopy_solver._LM_RTOL, mod.canopy_solver._LM_ATOL) == tol
