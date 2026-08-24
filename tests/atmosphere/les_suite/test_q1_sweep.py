"""Test the Q1a counter-gradient flux-sweep validator."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact, save_artifact

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "validate" / "les_suite"))

NZ, NT = 40, 3


def _cbl_artifact(q0: float, counter_gradient: bool):
    z = np.linspace(10.0, 1600.0, NZ)
    # well-mixed θ (slightly stable) + a localized inversion; flux linear surface->
    # entrainment. counter_gradient=True gives an up-gradient (positive product) layer.
    theta = 300.0 + (1.0e-3 if counter_gradient else 5.0e-2) * z
    theta = np.broadcast_to(theta, (NT, NZ)).copy()
    flux = q0 * np.clip(1.0 - z / 900.0, -0.3, 1.0)  # upward, decreasing
    wtheta = np.broadcast_to(flux, (NT, NZ)).copy()
    return LESReferenceArtifact(
        case_name="cbl_nieuwstadt", sgs="lasd", heights_m=z,
        times_s=np.linspace(0.0, 1800.0, NT), theta=theta,
        u=np.zeros((NT, NZ)), v=np.zeros((NT, NZ)),
        wtheta_resolved=wtheta, wtheta_sgs=np.zeros((NT, NZ)),
        prescribe="fluxes", w_theta_s=np.full(NT, q0), f_c=0.0,
    )


def _sweep_mod():
    import q1_counter_gradient_sweep  # noqa: PLC0415
    return q1_counter_gradient_sweep


def test_sweep_sorts_by_flux_and_flags_counter_gradient(tmp_path):
    m = _sweep_mod()
    # two fluxes: the weakly-stable one has a counter-gradient layer, the strongly
    # stable one does not (flux and gradient opposite sign there).
    save_artifact(_cbl_artifact(0.02, counter_gradient=False), tmp_path / "a__q0_0.02.npz")
    save_artifact(_cbl_artifact(0.08, counter_gradient=True), tmp_path / "b__q0_0.08.npz")
    rows = m.sweep(tmp_path)
    assert [r["Q0_K_m_s"] for r in rows] == [0.02, 0.08]  # sorted ascending
    assert all("has_counter_gradient_layer" in r for r in rows)
    cg08 = next(r for r in rows if r["Q0_K_m_s"] == 0.08)
    assert cg08["has_counter_gradient_layer"] is True
    assert cg08["layer_base_m"] is not None


def test_sweep_skips_non_flux_artifacts(tmp_path):
    m = _sweep_mod()
    save_artifact(_cbl_artifact(0.06, counter_gradient=True), tmp_path / "c__q0_0.06.npz")
    rows = m.sweep(tmp_path)
    assert len(rows) == 1
    assert rows[0]["Q0_K_m_s"] == 0.06


def test_sweep_empty_dir_returns_empty(tmp_path):
    m = _sweep_mod()
    assert m.sweep(tmp_path) == []
