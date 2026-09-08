"""Test the σ_LES CLI driver: load SGS-variant artifacts from disk → σ_LES."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact, save_artifact

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "validate" / "les_suite"))

NZ, NT = 24, 3


def _driver():
    import compute_sigma_les  # noqa: PLC0415
    return compute_sigma_les


def _write_variant(d: Path, case: str, sgs: str, dtheta: float) -> None:
    z = np.linspace(10.0, 1500.0, NZ)
    theta = np.broadcast_to(300.0 + 0.004 * z + dtheta, (NT, NZ)).copy()
    art = LESReferenceArtifact(
        case_name=case, sgs=sgs, heights_m=z, times_s=np.linspace(0.0, 600.0, NT),
        theta=theta, u=np.zeros((NT, NZ)), v=np.zeros((NT, NZ)),
        wtheta_resolved=np.zeros((NT, NZ)), wtheta_sgs=np.zeros((NT, NZ)),
        prescribe="fluxes", w_theta_s=np.full(NT, 0.06), f_c=0.0)
    save_artifact(art, d / f"{case}__{sgs}.npz")


def test_cli_computes_sigma_from_variants(tmp_path, capsys):
    m = _driver()
    for sgs, dth in (("lasd", 0.0), ("smagorinsky", 0.3), ("vreman", -0.2)):
        _write_variant(tmp_path, "cbl_x", sgs, dth)
    rc = m.main(["--artifacts-dir", str(tmp_path), "--case", "cbl_x"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "σ_LES" in out and "combined" in out


def test_cli_errors_with_one_variant(tmp_path):
    m = _driver()
    _write_variant(tmp_path, "cbl_x", "lasd", 0.0)  # only one → cannot form a spread
    rc = m.main(["--artifacts-dir", str(tmp_path), "--case", "cbl_x"])
    assert rc == 2
