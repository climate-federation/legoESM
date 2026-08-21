"""Test the Q1b diagnostic-flux margin validator (scripts/validate/les_suite)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact

_SPEC = importlib.util.spec_from_file_location(
    "q1b_diagnostic_margin",
    Path(__file__).resolve().parents[3]
    / "scripts/validate/les_suite/q1b_diagnostic_margin.py",
)
q1b = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(q1b)

NZ, NT = 24, 3


def _windy_cbl_npz(tmp_path: Path, sgs: str) -> Path:
    """Write a well-mixed sheared dry-CBL artifact (enough wind to carry the surface flux)."""
    z = np.linspace(10.0, 1500.0, NZ)
    theta = np.broadcast_to(
        np.where(z > 800.0, 300.0 + 0.006 * (z - 800.0), 300.0), (NT, NZ)).copy()
    art = LESReferenceArtifact(
        case_name="cbl_nieuwstadt", sgs=sgs, heights_m=z,
        times_s=np.linspace(0.0, 1800.0, NT), theta=theta,
        u=np.full((NT, NZ), 5.0), v=np.zeros((NT, NZ)),
        wtheta_resolved=np.broadcast_to(
            0.06 * np.clip(1.0 - z / 900.0, -0.2, 1.0), (NT, NZ)).copy(),
        wtheta_sgs=np.zeros((NT, NZ)), prescribe="fluxes",
        w_theta_s=np.full(NT, 0.06), f_c=0.0)
    path = tmp_path / f"cbl_nieuwstadt__{sgs}__ug8.npz"
    # LESReferenceArtifact round-trips through the bridge saver used by run_les_suite.
    from legoesm.atmosphere.les_suite.bridge import save_artifact
    save_artifact(art, path)
    return path


def test_margin_nonlocal_beats_local(tmp_path):
    for sgs in q1b.SGS_VARIANTS:
        _windy_cbl_npz(tmp_path, sgs)
    # No tuned JSONs in tmp → default configs (had_tuned all False); the STRUCTURAL margin
    # (local ≈0 mixed-layer flux) does not depend on tuning, so the sign is robust.
    res = q1b.measure_margin(
        tmp_path, tmp_path, "cbl_nieuwstadt", "ug8", nlev=32, use_tuned=True)
    assert res["n_sgs_scored"] == len(q1b.SGS_VARIANTS)
    assert not any(res["had_tuned"].values())  # none present ⇒ default fallback logged
    assert res["margin_min"] > 0.1            # nonlocal beats local by a clear margin
    assert res["significant"]                 # margin clears its own σ_LES spread


def test_moist_artifact_is_skipped(tmp_path):
    # A moist artifact raises inside diagnostic_scheme_flux ⇒ reported skipped, not scored.
    from legoesm.atmosphere.les_suite.bridge import save_artifact
    z = np.linspace(10.0, 2000.0, NZ)
    art = LESReferenceArtifact(
        case_name="cbl_nieuwstadt", sgs="lasd", heights_m=z,
        times_s=np.linspace(0.0, 1800.0, NT),
        theta=np.broadcast_to(298.0 + 0.004 * z, (NT, NZ)).copy(),
        u=np.full((NT, NZ), 5.0), v=np.zeros((NT, NZ)),
        wtheta_resolved=np.broadcast_to(
            0.05 * np.clip(1.0 - z / 900.0, -0.2, 1.0), (NT, NZ)).copy(),
        wtheta_sgs=np.zeros((NT, NZ)),
        qt=np.broadcast_to(np.clip(0.016 - 6e-6 * z, 0.003, None), (NT, NZ)).copy(),
        wqt_resolved=np.broadcast_to(
            5e-5 * np.clip(1.0 - z / 1500.0, -0.2, 1.0), (NT, NZ)).copy(),
        wqt_sgs=np.zeros((NT, NZ)), prescribe="fluxes",
        w_theta_s=np.full(NT, 0.05), w_qv_s=np.full(NT, 5e-5), f_c=0.0)
    save_artifact(art, tmp_path / "cbl_nieuwstadt__lasd__ug8.npz")
    res = q1b.measure_margin(tmp_path, tmp_path, "cbl_nieuwstadt", "ug8", nlev=32)
    row = res["per_sgs"][0]
    assert "margin" not in row and row["skipped"]  # all schemes skipped (moist)
    assert res["n_sgs_scored"] == 0
