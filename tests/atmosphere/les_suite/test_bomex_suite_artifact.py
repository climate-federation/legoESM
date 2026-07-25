"""Unit test for run_bomex_les._emit_suite_artifact — the moist LES → LESReferenceArtifact
converter (assembles the tuner's self-describing moist artifact from les_record prof_NNN.npz
+ the run's forcing). Uses synthetic prof files so it needs no GPU / moist LES run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_MOD = Path(__file__).resolve().parents[3] / "scripts" / "run" / "run_bomex_les.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("run_bomex_les", _MOD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_bomex_les"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_emit_suite_artifact_builds_valid_moist_artifact(tmp_path):
    mod = _load_converter()
    nz, nt = 40, 3
    z = np.linspace(0.0, 3000.0, nz)
    prof_dir = tmp_path / "profiles"
    prof_dir.mkdir()
    rng = np.random.default_rng(0)
    for k in range(nt):
        np.savez(
            prof_dir / f"prof_{k:03d}.npz", t_hours=0.5 * k, z=z,
            theta=298.0 + 0.004 * z, u=-8.0 + 1e-3 * z, v=np.zeros(nz),
            wtheta=0.01 * np.exp(-z / 500.0), qt=0.017 - 4e-6 * z,
            wqt=5e-5 * np.exp(-z / 500.0) + 1e-9 * rng.normal(size=nz))

    args = SimpleNamespace(
        output=tmp_path, case_label="bomex", sgs_model="lasd")
    forc = dict(ug=-8.0 + 1e-3 * z, vg=np.zeros(nz), w_ls=-0.01 * np.ones(nz),
                tls=np.full(nz, -1e-5), qls=np.full(nz, -1e-8),
                th_flux=0.00806, qv_flux=5.27e-5)
    out = tmp_path / "bomex__lasd.npz"
    mod._emit_suite_artifact(args, forc, out)

    # load_artifact runs the bridge's validation (moist artifact must carry wqt_resolved etc.)
    from legoesm.atmosphere.les_suite.bridge import load_artifact
    a = load_artifact(out)
    assert a.is_moist and a.case_name == "bomex" and a.sgs == "lasd"
    assert a.theta.shape == (nt, nz) and a.qt.shape == (nt, nz)
    assert a.wqt_resolved.shape == (nt, nz) and a.wtheta_resolved.shape == (nt, nz)
    assert np.isclose(a.f_c, mod._FCOR) and np.isclose(float(a.w_theta_s[0]), 0.00806)
    assert np.isclose(float(a.w_qv_s[0]), 5.27e-5)
    assert a.u_geo is not None and float(a.subsidence_w[0]) < 0.0  # subsidence is downward
