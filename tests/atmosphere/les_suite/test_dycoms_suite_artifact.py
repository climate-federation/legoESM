"""Unit test for run_dycoms_les._emit_suite_artifact — the moist DYCOMS-II RF01 LES →
LESReferenceArtifact converter. Unlike BOMEX, DYCOMS's non-turbulent θ forcing is the
parameterized RF01 LW COOLING (placed in the theta_adv channel), and there is no q_v
advection. Synthetic prof files + rad tendency so it needs no GPU / stratocumulus LES run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_MOD = Path(__file__).resolve().parents[3] / "scripts" / "run" / "run_dycoms_les.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("run_dycoms_les", _MOD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_dycoms_les"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_emit_suite_artifact_builds_valid_moist_dycoms_artifact(tmp_path):
    mod = _load_converter()
    nz, nt = 48, 3
    z = np.linspace(0.0, 1500.0, nz)
    prof_dir = tmp_path / "profiles"
    prof_dir.mkdir()
    rng = np.random.default_rng(0)
    for k in range(nt):
        # DYCOMS θ_l well-mixed then a sharp inversion; q_t ~9 g/kg subcloud
        np.savez(
            prof_dir / f"prof_{k:03d}.npz", t_hours=1.0 * k, z=z,
            theta=289.0 + np.clip((z - 840.0) / 100.0, 0.0, None) * 8.0,
            u=7.0 + 0.0 * z, v=-5.5 + 0.0 * z,
            wtheta=-0.01 * np.exp(-(z - 840.0) ** 2 / 1e4),
            qt=np.clip(9.0e-3 - 3.0e-6 * z, 1.5e-3, None),
            wqt=5e-5 * np.exp(-z / 800.0) + 1e-9 * rng.normal(size=nz))

    args = SimpleNamespace(
        output=tmp_path, case_label="dycoms_rf01_sc", sgs_model="smagorinsky",
        dynamic=False)
    forc = dict(ug=7.0 + 0.0 * z, vg=-5.5 + 0.0 * z, th_flux=0.0113, qv_flux=6.5e-5)
    # RF01 LW cooling: negative dθ/dt peaking at cloud top (~840 m); subsidence −D·z
    rad_tend = -8.0e-4 * np.exp(-(z - 840.0) ** 2 / 2e4)
    subsidence_w = -3.75e-6 * z
    out = tmp_path / "dycoms_rf01_sc__smagorinsky.npz"
    mod._emit_suite_artifact(args, forc, rad_tend, subsidence_w, out)

    from legoesm.atmosphere.les_suite.bridge import load_artifact
    a = load_artifact(out)
    assert a.is_moist and a.case_name == "dycoms_rf01_sc" and a.sgs == "smagorinsky"
    assert a.theta.shape == (nt, nz) and a.qt.shape == (nt, nz)
    assert a.wqt_resolved.shape == (nt, nz)
    # the RF01 cooling is carried in theta_adv, negative at cloud top (radiative COOLING)
    assert a.theta_adv is not None and float(np.min(a.theta_adv)) < 0.0
    assert float(a.subsidence_w[np.argmax(z)]) < 0.0    # subsidence sinks (−D·z, +up conv)
    assert np.isclose(float(a.w_qv_s[0]), 6.5e-5) and np.isclose(float(a.w_theta_s[0]), 0.0113)
    assert a.qv_adv is not None and float(np.max(np.abs(a.qv_adv))) == 0.0  # no q_v advection


def test_dycoms_case_label_resolves_to_registry_regime():
    # the emit's default case-label must resolve to the registry's stratocumulus regime
    # (else the scorecard silently skips DYCOMS records, as BOMEX did before the fix).
    from legoesm.atmosphere.les_suite import (
        get_case, list_cases, register_default_catalog)
    if not list_cases():
        register_default_catalog()
    assert get_case("dycoms_rf01_sc").regime == "stratocumulus"
