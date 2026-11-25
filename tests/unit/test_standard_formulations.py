"""Known-answer checks for the weather/climate-model stability-function benchmark."""
import pathlib
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import standard_formulations as sf  # noqa: E402

KAPPA = constants.kappa_von_karman


def _rational_phi(z, a0, a1, b1, b2):
    """1 - z dpsi/dz for the GFS near-neutral rational psi = (a0 + a1 z) z / (1 + (b1 + b2 z) z)."""
    n, d = (a0 + a1 * z) * z, 1 + (b1 + b2 * z) * z
    return 1 - z * ((a0 + 2 * a1 * z) * d - n * (b1 + 2 * b2 * z)) / d**2


@pytest.mark.parametrize("name,kind,zeta,expected", [
    ("businger1971", "m", 0.5, 1 + 4.7 * 0.5),                       # Businger 1971 phi_m stable
    ("businger1971", "h", -1.0, 0.74 * 10.0 ** -0.5),                 # Businger 1971 phi_h unstable
    ("hogstrom1988", "m", -1.0, 20.3 ** -0.25),                       # Hogstrom 1988 eq 12
    ("hogstrom1988", "h", 0.5, 0.95 + 7.8 * 0.5),                     # Hogstrom 1988 Table VII
    ("wrf_revmm5", "m", 1.0, 1 + 6.1 * (1 + 2 ** -0.6) / (1 + 2 ** 0.4)),   # Cheng-Brutsaert 2005 phi_m, a=6.1 b=2.5
    ("wrf_revmm5", "h", 1.0, 1 + 5.3 * (1 + 2 ** (-0.1 / 1.1)) / (1 + 2 ** (1 / 1.1))),  # c=5.3 d=1.1
    ("mynn_sfc", "m", 1.0, 1 + 6.1 * (1 + 2 ** -0.6) / (1 + 2 ** 0.4)),
    ("gfs", "m", -2.0, 0.5 * 2.0 ** -0.25),                           # 1 - zeta dpsi/dzeta of ln h + 2 h^-1/4 - 0.8776
    ("gfs", "h", -2.0, 0.25 * 2.0 ** -0.5),                           # of ln h + 0.5 h^-1/2 + 1.386
    ("gfs", "m", -0.2, _rational_phi(-0.2, -3.975, 12.32, -7.755, 6.041)),
    ("gfs", "h", -0.5, _rational_phi(-0.5, -7.941, 24.75, -8.705, 7.899)),
    ("zeng1998_clm", "h", -2.0, 0.8 * 0.333 * 2.0 ** -0.333),         # CTSM coded free-convection heat
    ("gfs", "m", 1.0, 1 + 10.0 / (1 + np.sqrt(21.0))),               # d/dzeta of -a+ln(a+1), a=sqrt(1+20 zeta)
    ("zeng1998_clm", "m", -2.0, 1.14 * 0.333 * 2 ** 0.333),           # CTSM coded free convection
    ("zeng1998_clm", "h", 2.0, 5.0 + 2.0),                            # Zeng 1998 very stable
    ("dyer1974", "m", 0.5, 1 + 5 * 0.5),
])
def test_hand_values(name, kind, zeta, expected):
    assert sf.phi(name, kind, np.array([zeta]))[0] == pytest.approx(expected, rel=1e-6)


# legoESM's gryanik2020 psi_h gives phi_h(0) = 1, not the paper's Pr0 = 0.98 (reported as a finding)
NEUTRAL_PR = {"businger1971": 0.74, "hogstrom1988": 0.95}


@pytest.mark.parametrize("name", list(sf.SCHEMES))
def test_neutral_limit(name):
    z = np.array([-1e-6, 1e-6])
    assert sf.phi(name, "m", z) == pytest.approx([1.0, 1.0], abs=1e-3)
    pr = NEUTRAL_PR.get(name, 1.0)
    assert sf.phi(name, "h", z) == pytest.approx([pr, pr], abs=2e-3)


@pytest.mark.parametrize("name,kind,branch,tol", [
    ("zeng1998_clm", "m", -1.574, 2e-3), ("zeng1998_clm", "h", -0.465, 2e-3), ("zeng1998_clm", "m", 1.0, 1e-6),
    ("wrf_revmm5", "m", -0.3, 1e-6), ("businger1971", "m", 0.0, 1e-6),
])
def test_continuity_at_branch_points(name, kind, branch, tol):
    lo, hi = sf.phi(name, kind, np.array([branch - 1e-10, branch + 1e-10]))
    assert abs(lo - hi) <= tol * abs(hi)


def test_integral_matches_closed_form_and_higher_order():
    z1, z2, inv_l = np.array([2.0]), np.array([8.0]), np.array([1 / 20.0])
    exact = np.log(4.0) + 5.0 * (z2 - z1) * inv_l
    assert sf.integral("dyer1974", "m", inv_l, z1, z2) == pytest.approx(exact, rel=1e-6)
    inv_u = np.array([-1 / 5.0])
    eight = sf.integral("zeng1998_clm", "m", inv_u, z1, z2)
    sf.GAUSS_NODES, keep = 64, sf.GAUSS_NODES
    try:
        assert eight == pytest.approx(sf.integral("zeng1998_clm", "m", inv_u, z1, z2), rel=1e-3)
    finally:
        sf.GAUSS_NODES = keep


@pytest.mark.parametrize("name,inv_l", [("dyer1974", 1 / 30.0), ("dyer1974", -1 / 15.0), ("businger1971", 0.5),
                                         ("gfs", 0.3), ("zeng1998_clm", -0.5)])
def test_inversion_recovers_fluxes(name, inv_l):
    ustar, theta = 0.35, 285.0
    k = sf.SCHEMES[name][2]
    tstar = inv_l * ustar**2 * theta / (k * constants.g)       # theta* > 0 and L > 0 when stable
    zu = (np.array([2.0]), np.array([8.0]))
    zt = (np.array([1.5]), np.array([6.0]))
    l = np.array([inv_l])
    du = ustar / k * sf.integral(name, "m", l, *zu)
    dt = tstar / k * sf.integral(name, "h", l, *zt)
    assert np.sign(dt[0]) == np.sign(inv_l)                     # warmer aloft when stable
    us, ts, no_root = sf.invert(name, du, dt, zu, zt, np.array([theta]))
    assert not no_root[0]
    assert us[0] == pytest.approx(ustar, rel=1e-6) and ts[0] == pytest.approx(tstar, rel=1e-6)


def test_inversion_finds_root_when_endpoint_signs_agree():
    # codex round-2 fixture: different wind and temperature pairs give an even number of roots in the domain
    ustar, theta, inv_l = 0.3, 285.0, np.log(2) / 15
    tstar = inv_l * ustar**2 * theta / (KAPPA * constants.g)
    zu, zt, l = (np.array([1.0]), np.array([64.0])), (np.array([4.0]), np.array([8.0])), np.array([inv_l])
    du = ustar / KAPPA * sf.integral("dyer1974", "m", l, *zu)
    dt = tstar / KAPPA * sf.integral("dyer1974", "h", l, *zt)
    us, ts, no_root = sf.invert("dyer1974", du, dt, zu, zt, np.array([theta]))
    assert not no_root[0] and us[0] == pytest.approx(ustar, rel=1e-6)


def test_inversion_reports_no_root_beyond_critical_richardson():
    # linear 1 + 5 z/L has no solution once the bulk Richardson number exceeds about 0.2
    zu = zt = (np.array([2.0]), np.array([8.0]))
    us, ts, no_root = sf.invert("dyer1974", np.array([0.5]), np.array([2.0]), zu, zt, np.array([285.0]))
    assert no_root[0] and np.isnan(us[0])


def test_stats_keep_zero_predictions_and_signed_values():
    r = sf._stats(np.array([1.0, 2.0]), np.array([0.0, 2.0]))
    assert r["n"] == 2 and r["rmse"] == pytest.approx(np.sqrt(0.5))
    r = sf._stats(np.array([-1.0, -2.0]), np.array([-1.0, -2.0]))
    assert r["median_ratio"] == 1.0 and r["log_rmse"] == 0.0 and r["frac_sign_mismatch"] == 0.0


def test_businger_scored_in_own_kappa():
    """A world obeying Businger 1971 with kappa 0.35 scores ratio 1 for it, in gradient and integrated space."""
    s = sf.BUSINGER_KAPPA / KAPPA
    n, ustar = 200, 0.3
    inv_l = np.linspace(0.01, 0.2, n)                    # our-convention 1/L (kappa 0.40)
    z1, z2 = np.full(n, 2.0), np.full(n, 8.0)
    z_g = np.sqrt(z1 * z2)
    phi_obs = sf.phi("businger1971", "m", z_g * inv_l * s) / s
    diff = ustar / sf.BUSINGER_KAPPA * sf.integral("businger1971", "m", inv_l * s, z1, z2)
    pair = {"z1": z1, "z2": z2, "diff": diff, "scale": np.full(n, ustar), "inv_l": inv_l, "zeta": z_g * inv_l,
            "phi": phi_obs, "keep": np.ones(n, bool)}
    res = sf.score_site({"m": [pair], "h": []}, None, 30)["m_stable"]
    assert res["businger1971"]["gradient"]["median_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert res["businger1971"]["integrated"]["median_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert abs(res["hogstrom1988"]["gradient"]["median_ratio"] - 1.0) > 0.05  # a different law does not score 1


def test_inversion_common_subset_is_shared():
    """Schemes that fail on a sample must not score on an easier subset than the others."""
    z1, z2 = np.array([2.0, 2.0]), np.array([8.0, 8.0])
    du, dt = np.array([2.0, 0.5]), np.array([0.2, 0.5])  # second sample: bulk Ri ~0.4, beyond the linear scheme's critical Ri
    pair = lambda diff: {"key": "m:2-8", "z_g": 4.0, "z1": z1, "z2": z2, "diff": diff, "zeta": np.array([0.1, 0.1]),
                         "keep": np.ones(2, bool)}
    flux = {"theta": np.full(2, 285.0), "rho": np.full(2, 1.2), "ustar": np.full(2, 0.3), "h": np.full(2, -20.0)}
    res = next(iter(sf.score_inversion({"m": [pair(du)], "h": [dict(pair(dt), key="h:2-8")]}, flux, 1).values()))["stable"]
    solved_both = [s for s in sf.SCHEMES if res[s]["no_root_frac"] == 0.0]
    assert res["dyer1974"]["no_root_frac"] == 0.5 and solved_both
    assert res[solved_both[0]]["ustar"]["n"] == 2
    assert {res[s]["common"]["ustar"]["n"] for s in sf.SCHEMES} == {1}


def test_holtslag_debruin_hand_value():
    # Gryanik et al. 2020 eq A1: 1 + a z + b z (1 + c - d z) exp(-d z), a=0.7 b=0.75 c=5 d=0.35, phi_m = phi_h
    expected = 1 + 0.7 + 0.75 * (6 - 0.35) * np.exp(-0.35)
    assert sf.phi("holtslag_debruin1988", "m", np.array([1.0]))[0] == pytest.approx(expected, rel=1e-9)
    assert sf.phi("holtslag_debruin1988", "h", np.array([1.0]))[0] == pytest.approx(expected, rel=1e-9)


# Algebraic one-sided slopes at zeta = 0 (left = unstable branch, right = stable branch), from each closed form.
SLOPES_AT_ZERO = {
    ("businger1971", "m"): (15 / 4, 4.7), ("businger1971", "h"): (0.74 * 9 / 2, 4.7),
    ("hogstrom1988", "m"): (19.3 / 4, 6.0), ("hogstrom1988", "h"): (0.95 * 11.6 / 2, 7.8),
    ("dyer1974", "m"): (4.0, 5.0), ("dyer1974", "h"): (8.0, 5.0),
    ("holtslag_debruin1988", "m"): (4.0, 0.7 + 0.75 * 6), ("holtslag_debruin1988", "h"): (8.0, 0.7 + 0.75 * 6),
    ("beljaars_holtslag1991", "m"): (4.0, 1 + 0.667 * 6), ("beljaars_holtslag1991", "h"): (8.0, 1 + 0.667 * 6),
    ("grachev2007_sheba", "m"): (4.0, 5.0), ("grachev2007_sheba", "h"): (8.0, 5.0),
    ("gryanik2020", "m"): (4.0, 5.0), ("gryanik2020", "h"): (0.98 * 8.0, 0.98 * 5.0),
    ("coare30", "m"): (15 / 4, 1 + 0.6667 * (1 + 0.35 * 14.28)), ("coare30", "h"): (15 / 2, 1 + 0.6667 * (1 + 0.35 * 14.28)),
    ("wrf_revmm5", "m"): (4.0, 6.1), ("wrf_revmm5", "h"): (8.0, None),     # stable heat slope -> 5.3 only as z^0.1 -> 0
    ("mynn_sfc", "m"): (4.0, 6.1), ("mynn_sfc", "h"): (8.0, None),
    ("gfs", "m"): (3.975, 5.0), ("gfs", "h"): (7.941, 5.0),
    ("zeng1998_clm", "m"): (4.0, 5.0), ("zeng1998_clm", "h"): (8.0, 5.0),
}
# Measured outcome of the user's rule (phi continuous everywhere, C1 except at zeta = 0), recorded, not fixed.
C1_FAILURES = {  # the model's hard |zeta| = 10 clip became a smooth cap in #2511, removing those kinks
    "gfs": [-0.5],
    "zeng1998_clm_m": [-1.574, 1.0], "zeng1998_clm_h": [-0.465, 1.0],
}


@pytest.mark.parametrize("name,kind", sorted(SLOPES_AT_ZERO))
def test_continuity_recorded(name, kind):
    c = sf.continuity(name, kind)
    assert c["zeta0"]["phi_continuous"], "phi must be continuous at zeta = 0"
    left, right = SLOPES_AT_ZERO[(name, kind)]
    assert c["zeta0"]["slope_algebraic"][0] == pytest.approx(left, rel=1e-3)
    if right is not None:
        assert c["zeta0"]["slope_algebraic"][1] == pytest.approx(right, rel=1e-3)
    failing = sorted(b["point"] for b in c["branches"] if not b["c1"])
    assert failing == C1_FAILURES.get(f"{name}_{kind}", C1_FAILURES.get(name, []))
    assert c["pass"] == (not failing)


def test_louis_hand_values_and_continuity():
    ri = jnp.asarray(0.1)
    assert float(sf.louis_f(ri, 4.0, "m")) == pytest.approx(1 / (1 + 10 * 0.1 / np.sqrt(1.5)), rel=1e-9)
    assert float(sf.louis_f(ri, 4.0, "h")) == pytest.approx(1 / (1 + 15 * 0.1 * np.sqrt(1.5)), rel=1e-9)
    for kind, slope in (("m", -10.0), ("h", -15.0)):
        c = sf.louis_continuity(kind)
        assert c["phi"][0] == pytest.approx(c["phi"][1], abs=1e-6)
        assert c["slope_algebraic"][0] == pytest.approx(slope, rel=1e-3)
        assert c["slope_algebraic"][1] == pytest.approx(slope, rel=1e-3)


def test_louis_neutral_inversion_is_log_law():
    zu = zt = (np.array([2.0]), np.array([8.0]))
    us, ts, bad = sf.louis_invert(np.array([2.0]), np.array([0.0]), zu, zt, np.array([285.0]))
    assert not bad[0] and us[0] == pytest.approx(KAPPA * 2.0 / np.log(4.0)) and ts[0] == 0.0


def test_common_scores_respect_min_count():
    n = 31
    z1, z2 = np.full(n, 2.0), np.full(n, 8.0)
    du, dt = np.r_[2.0, np.full(n - 1, 0.5)], np.r_[0.2, np.full(n - 1, 0.5)]  # one solvable by all, 30 not by Dyer
    pair = lambda diff, key: {"key": key, "z_g": 4.0, "z1": z1, "z2": z2, "diff": diff, "zeta": np.full(n, 0.1),
                              "keep": np.ones(n, bool)}
    flux = {"theta": np.full(n, 285.0), "rho": np.full(n, 1.2), "ustar": np.full(n, 0.3), "h": np.full(n, -20.0)}
    res = next(iter(sf.score_inversion({"m": [pair(du, "m:2-8")], "h": [pair(dt, "h:2-8")]}, flux, 30).values()))["stable"]
    assert res["gfs"]["common"] == {"n": 1}


def test_louis_unstable_hand_value():
    # ED2 sfclyrfun.r: c = C*m b a^2 ee sqrt(ee |Ri|), ee = zratio^(1/3) - 1, Fm = 1 - 2 b Ri / (1 + 2 c)
    ri, zr = -0.1, 4.0
    a2 = KAPPA**2 / np.log(zr) ** 2
    ee = zr ** (1 / 3) - 1
    c = 7.5 * 5 * a2 * ee * np.sqrt(ee * 0.1)
    assert float(sf.louis_f(jnp.asarray(ri), zr, "m")) == pytest.approx(1 + 10 * 0.1 / (1 + 2 * c), rel=1e-9)


@pytest.mark.parametrize("name,kind", sorted(SLOPES_AT_ZERO))
def test_numeric_and_algebraic_slopes_agree(name, kind):
    c = sf.continuity(name, kind)
    checks = [c["zeta0"]] + c["branches"]
    for i, r in enumerate(checks):
        for side in (0, 1):
            if i == 0 and side == 1 and SLOPES_AT_ZERO[(name, kind)][1] is None:
                continue  # z^0.1 term: slope converges to its limit too slowly for a finite-difference step
            alg, num = r["slope_algebraic"][side], r["slope_numeric"][side]
            assert num == pytest.approx(alg, rel=1e-3, abs=1e-3)
