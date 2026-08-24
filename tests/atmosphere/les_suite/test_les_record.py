"""Tests for les_record's planar-mean profiles — the moist-artifact fields (⟨w'θ'⟩, q_t,
⟨w'q_t'⟩) added so a moist BOMEX/DYCOMS LES run can be converted to a LESReferenceArtifact.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_MOD = Path(__file__).resolve().parents[3] / "scripts" / "run" / "les_record.py"
_spec = importlib.util.spec_from_file_location("les_record", _MOD)
les_record = importlib.util.module_from_spec(_spec)
sys.modules["les_record"] = les_record
_spec.loader.exec_module(les_record)


def _fields(ny=8, nx=8, nz=20, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(ny, nx, nz))
    v = rng.normal(size=(ny, nx, nz))
    w = rng.normal(size=(ny, nx, nz))
    th = 300.0 + rng.normal(size=(ny, nx, nz))
    z = np.linspace(0.0, 2000.0, nz)
    return z, u, v, w, th, rng


def test_wtheta_always_present_and_matches_manual_flux():
    z, u, v, w, th, _ = _fields()
    p = les_record._profiles(z, u, v, w, th, 0.1, "test")
    assert "wtheta" in p and p["wtheta"].shape == (len(z),)
    wm = w.mean((0, 1))
    thm = th.mean((0, 1))
    ref = ((w - wm) * (th - thm)).mean((0, 1))  # ⟨w'θ'⟩
    assert np.allclose(p["wtheta"], ref, atol=0, rtol=1e-12)


def test_dry_profile_has_no_qt():
    z, u, v, w, th, _ = _fields()
    p = les_record._profiles(z, u, v, w, th, 0.1, "test")  # qt3=None
    assert "qt" not in p and "wqt" not in p  # dry set unchanged


def test_moist_profile_qt_and_wqt():
    z, u, v, w, th, rng = _fields()
    qt3 = 0.016 + 0.001 * rng.normal(size=w.shape)  # total water ~16 g/kg
    p = les_record._profiles(z, u, v, w, th, 0.1, "test", qt3=qt3)
    assert np.allclose(p["qt"], qt3.mean((0, 1)), atol=0, rtol=1e-12)
    wm = w.mean((0, 1))
    qtm = qt3.mean((0, 1))
    ref = ((w - wm) * (qt3 - qtm)).mean((0, 1))  # ⟨w'q_t'⟩
    assert np.allclose(p["wqt"], ref, atol=0, rtol=1e-12)


def test_record_frame_builds_qt_from_qv_qc_qr(tmp_path):
    z, u, v, w, th, rng = _fields()
    qv = 0.015 + 0.001 * rng.normal(size=w.shape)
    qc = np.maximum(0.0, rng.normal(size=w.shape) * 1e-4)
    qr = np.maximum(0.0, rng.normal(size=w.shape) * 1e-5)
    h_idx = [0, 5, 10, 15]
    les_record.record_frame(
        tmp_path, 0, 0.5, "bomex", z, u, v, w, th, 3200.0, 3200.0, h_idx,
        z[h_idx], 0.1, qc3=qc, qr3=qr, qv3=qv)
    prof = np.load(tmp_path / "profiles" / "prof_000.npz")
    assert "qt" in prof and "wqt" in prof and "wtheta" in prof
    # q_t profile = mean of (q_v + q_c + q_r)
    assert np.allclose(prof["qt"], (qv + qc + qr).mean((0, 1)), atol=0, rtol=1e-10)
