"""Unit tests for scripts/validate/compare_rce_vs_rcemip_sam.py — the RCE-vs-RCEMIP
faithfulness comparison tool. Exercises the pure ``compute_metrics`` +
``evaluate`` functions on synthetic RCE-like profiles with KNOWN bulk
magnitudes, so a regression in the averaging / pressure / CWV / cold-point /
ascending-z logic is caught without needing a live GPU run."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from scripts.validate import compare_rce_vs_rcemip_sam as cmp  # noqa: E402
from legoesm import constants as C  # noqa: E402

_NLEV = 30


def _synthetic_snap(surface_first: bool = True) -> dict:
    """An RCE-like single-snapshot profile dict with a known 195 K cold point,
    exponential-decay q_v, and a mid-troposphere-peaked w_RMS."""
    z = np.linspace(50.0, 20000.0, _NLEV)            # ascending, surface-first
    # T: tropospheric lapse to a 195 K cold point (idx 22, ~15 km), strat rise.
    T = np.concatenate([np.linspace(297.0, 195.0, 23),
                        np.linspace(196.0, 207.0, 7)])
    qv = 0.016 * np.exp(-z / 3000.0)                 # kg/kg, exponential decay
    # theta s.t. T/theta = exp(-kappa z/8000) => p = p_ref exp(-z/8000) (realistic).
    theta = T * np.exp(C.kappa * z / 8000.0)
    w_RMS = 0.5 * np.exp(-((z - 7000.0) / 4000.0) ** 2)        # peak 0.5 @ 7 km
    cloud = (0.10 * np.exp(-((z - 1000.0) / 800.0) ** 2)
             + 0.15 * np.exp(-((z - 13000.0) / 2000.0) ** 2))  # bimodal, peak 0.15
    rho = 1.1 * np.exp(-z / 8000.0)
    qc = 1e-4 * np.exp(-((z - 9000.0) / 3000.0) ** 2)
    qprecip = 5e-5 * np.exp(-((z - 4000.0) / 3000.0) ** 2)
    mid = 0.5 * (z[:-1] + z[1:])
    z_half = np.concatenate([[2 * z[0] - mid[0]], mid, [2 * z[-1] - mid[-1]]])
    snap = dict(step=5000, z=z, z_half=z_half, T_mean=T, theta_mean=theta,
                qv_mean=qv, qc_mean=qc, w_RMS=w_RMS, cloud_fraction=cloud,
                rho_mean=rho, q_precip_mean=qprecip)
    if not surface_first:               # reverse ALL profiles -> surface-last
        snap = {k: (v[::-1].copy() if isinstance(v, np.ndarray) and v.ndim == 1
                    and v.shape[0] in (_NLEV, _NLEV + 1) else v)
                for k, v in snap.items()}
    return snap


def test_compute_metrics_known_profile():
    m = cmp.compute_metrics([_synthetic_snap()])
    got, shapes = m["got"], m["shapes"]
    assert got["T near-surface"] == pytest.approx(297.0, abs=1e-6)
    assert got["qv near-surface"] == pytest.approx(
        0.016 * np.exp(-50 / 3000) * 1e3, rel=1e-3)
    assert got["T cold-point"] == pytest.approx(195.0, abs=1e-6)
    assert 13.5 <= got["z cold-point"] <= 17.5      # ~15.2 km
    assert got["w_RMS mid-trop peak"] == pytest.approx(0.5, rel=0.02)
    assert got["cloud frac peak"] == pytest.approx(0.15, rel=0.05)
    assert 5.0 < got["CWV (PW)"] < 100.0            # sane PW magnitude (units OK)
    assert all(shapes.values())                     # all 3 shape signatures hold


def test_ascending_z_robustness():
    """Surface-first vs surface-last input must give identical metrics."""
    a = cmp.compute_metrics([_synthetic_snap(surface_first=True)])["got"]
    b = cmp.compute_metrics([_synthetic_snap(surface_first=False)])["got"]
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-9), k


def test_time_mean_averages_snapshots():
    """Two snaps offset by +/-2 K in T_mean must average back to the base."""
    s0 = _synthetic_snap()
    s_hot = dict(s0); s_hot["T_mean"] = s0["T_mean"] + 2.0
    s_cold = dict(s0); s_cold["T_mean"] = s0["T_mean"] - 2.0
    mean = cmp.compute_metrics([s_hot, s_cold])["got"]
    base = cmp.compute_metrics([s0])["got"]
    assert mean["T near-surface"] == pytest.approx(base["T near-surface"], abs=1e-6)
    assert mean["T cold-point"] == pytest.approx(base["T cold-point"], abs=1e-6)


def test_evaluate_edge_and_out_of_range():
    rows, n_pass, n_edge = cmp.evaluate({
        "CWV (PW)": 35.5,             # just above 35 low bound -> lo-edge, PASS
        "T near-surface": 296.7,      # mid
        "qv near-surface": 16.0,      # mid
        "T cold-point": 205.0,        # above 198 -> ABOVE, CHECK
        "z cold-point": 15.0,         # mid
        "w_RMS mid-trop peak": 0.55,  # mid
        "cloud frac peak": 0.20,      # mid
    })
    by_name = {r[0]: r for r in rows}
    assert by_name["CWV (PW)"][6] == "lo-edge" and by_name["CWV (PW)"][5] is True
    assert by_name["T cold-point"][6] == "ABOVE" and by_name["T cold-point"][5] is False
    assert n_pass == 6
    assert n_edge == 1
