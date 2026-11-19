"""Known-answer checks for the stable-LES formula test (les_stable.py)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import les_stable as ls  # noqa: E402


@pytest.fixture(autouse=True)
def baseline():
    ls.cf.register(ls.lz.BASE, "J3", ls.lz.J3_V5, "test")


def test_h_tau_matches_closed_form():
    zf = np.linspace(0, 400, 4001)
    tau = (1 - np.minimum(zf / 200.0, 1)) ** 1.5  # u* = 1
    exact = 200.0 * (1 - 0.05 ** (2 / 3)) / 0.95
    assert ls.h_tau(zf, tau, 0 * zf, 1.0) == pytest.approx(exact, rel=1e-4)


def _dyer_case(L=50.0, ustar=0.25, q0=-0.002, theta0=265.0):
    zm = np.arange(2.0, 160.0, 0.1)  # discrete error (dz/z)^2/12 < 2e-4
    zf = 0.5 * (zm[1:] + zm[:-1])
    U = 0.6 * ustar / ls.KAPPA * (np.log(zm / 0.01) + 5 * zm / L)
    V = 0.8 * ustar / ls.KAPPA * (np.log(zm / 0.01) + 5 * zm / L)
    ts = -q0 / ustar
    th = theta0 + ts / ls.KAPPA * (np.log(zm / 0.01) + 5 * zm / L)
    uw = -ustar**2 * (1 - zf / 150.0) * 0.6
    vw = -ustar**2 * (1 - zf / 150.0) * 0.8
    wt = q0 * (1 - zf / 150.0)
    L_true = -ustar**3 * theta0 / (ls.KAPPA * ls.lz.constants.g * q0)
    return ls._case(zm, U, V, th, zf, uw, vw, wt, 0 * zf, ustar, q0, theta0, 0.1), L_true


def test_phi_at_midpoint_recovers_dyer_and_ranks_it_first():
    d, L_true = _dyer_case(L=-0.25**3 * 265.0 / (ls.KAPPA * ls.lz.constants.g * -0.002))
    assert d["L"] == pytest.approx(L_true) and d["L"] > 0
    expect = 1 + 5 * d["zeta"]
    assert np.allclose(d["phi_m"], expect, rtol=2e-3) and np.allclose(d["phi_h"], expect, rtol=2e-3)
    s = ls.score(d, [("dyer1974", None), ("hogstrom1988", None), ("J3_global", None)])
    assert s["m"]["best"] == "dyer1974" and s["m"]["log_rmse"]["dyer1974"] < 2e-3
    assert s["m"]["log_rmse"]["J3_global"] > 0.01


def test_local_scaling_arm_uses_local_flux():
    d, _ = _dyer_case(L=-0.25**3 * 265.0 / (ls.KAPPA * ls.lz.constants.g * -0.002))
    frac = 1 - d["z"] / 150.0
    assert np.allclose(d["phi_m_loc"], d["phi_m"] / np.sqrt(frac), rtol=1e-9)  # u*_loc = u* sqrt(frac)
    assert np.allclose(d["zeta_loc"], d["zeta"] / np.sqrt(frac), rtol=1e-9)    # L_loc = L frac^(3/2) / frac


def test_score_uses_one_level_set_for_every_formula():
    d, _ = _dyer_case(L=-0.25**3 * 265.0 / (ls.KAPPA * ls.lz.constants.g * -0.002))
    d["phi_m"] = d["phi_m"].copy()
    d["phi_m"][3] = -1.0  # planted invalid level
    s = ls.score(d, [("dyer1974", None), ("J3_global", None)])
    assert s["m"]["n"] == d["z"].size - 1
    assert s["m"]["n_scored"]["dyer1974"] == s["m"]["n_scored"]["J3_global"] == s["m"]["n"]


def test_paired_counts_ties():
    runs = {c: {"score": {"m": {"log_rmse": {"a": v, "J3_global": 0.1, "dyer1974": 0.2}}, "h": {"log_rmse": {"a": v, "J3_global": 0.1, "dyer1974": 0.2}}}}
            for c, v in (("r1", 0.05), ("r2", 0.1), ("r3", 0.15))}
    p = ls.paired(runs, [("a", None)])["m|a|vs_J3_global"]
    assert (p["better"], p["worse"], p["tie"], p["n_runs"]) == (1, 1, 1, 3)
