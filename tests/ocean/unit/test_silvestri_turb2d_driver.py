"""Smoke test for the 2D-turbulence driver (scripts/run/...): it integrates the
harness, saves the Fig-4/5 metric arrays, and reproduces the qualitative §4
result (WENO retains energy; Leith over-damps it)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_DRIVER = (Path(__file__).resolve().parents[3]
           / "scripts" / "run" / "run_silvestri_turbulence_2d.py")


def _load():
    spec = importlib.util.spec_from_file_location("_turb2d_driver", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_driver_runs_saves_and_ranks_energy(tmp_path):
    drv = _load()
    out = {}
    for sch in ("W9V", "Leith2"):
        verdict = drv.run(sch, N=48, t_end=1.0, cfl=0.2, Re=3.3e4,
                          out=str(tmp_path), seed=0, tag="_smoke")
        assert verdict == "STABLE"
        f = list(tmp_path.glob(f"turb2d_{sch}_N48_smoke.npz"))
        assert len(f) == 1
        d = np.load(f[0])
        for key in ("t", "ke", "enstrophy", "k_energy", "P_energy",
                    "k_enstrophy", "P_enstrophy", "zeta_36"):
            assert key in d, key
        out[sch] = float(d["ke"][-1]) / float(d["ke"][0])
    # Paper §4 headline: WENO retains far more energy than Leith (C=2 over-damps).
    assert out["W9V"] > 0.8
    assert out["Leith2"] < 0.5
    assert out["W9V"] > out["Leith2"]
