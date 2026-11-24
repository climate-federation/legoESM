"""Known-answer checks for the z/zi tower test (zi_scaling.py)."""
import pathlib
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import candidate_formulations as cf  # noqa: E402
import zi_scaling as zs  # noqa: E402


@pytest.fixture(autouse=True)
def baseline():
    cf.register(zs.BASE, "J3", [4.0, 0.47, 1.22, 1.0], "test")


def test_fe_slope_recovers_known_eta_slope_and_ignores_zeta_offsets():
    rng = np.random.default_rng(1)
    n = 20000
    zeta = rng.uniform(0.02, 3.0, n)
    eta = 10 ** rng.uniform(-3, -0.1, n)
    r = 0.3 * np.digitize(zeta, zs.fs.EDGES) + 0.07 * np.log(eta) + rng.normal(0, 0.05, n)
    d = zs.fe_slope(zeta, eta, r, 30)
    assert d["slope"] == pytest.approx(0.07, abs=3 * d["se"] + 1e-3)
    assert d["partial_r2"] > 0.5
    null = zs.fe_slope(zeta, eta, 0.3 * np.digitize(zeta, zs.fs.EDGES) + rng.normal(0, 0.05, n), 30)
    assert abs(null["slope"]) < 4 * null["se"]


def test_sonde_match_window(tmp_path):
    t = np.array(["2020-01-01T00:00", "2020-01-01T12:00"], "datetime64[s]")
    xr.Dataset({"launch_time": ("sonde", t), "zi": ("sonde", np.float32([300, 900])),
                "distance_km": ("sonde", np.float32([5, 5]))}).to_netcdf(tmp_path / "x_sondes.nc")
    t0 = t[0].astype(np.int64)
    mid = t0 + np.array([0, 5400, 5401, 12 * 3600 - 100])
    got = zs.sonde_zi(str(tmp_path / "x_sondes.nc"), mid)
    assert got[0] == 300 and got[1] == 300 and np.isnan(got[2]) and got[3] == 900


def test_forms_recover_baseline_as_eta_vanishes():
    z = np.array([-2.0, -0.3, 0.4, 2.0])
    e = np.full(4, 1e-12)
    for form, p in (("H2_cap", [-1.0]), ("HC_exp", [3.0]), ("J19", [2.0, 1.0])):
        for kind in "mh":
            sel = np.isin(np.sign(z), [1 if r == "stable" else -1 for r in zs.FORMS[form][0]])
            assert np.allclose(zs.model(form, kind, z[sel], e[sel], np.array(p), 1.0), zs.base(kind, z[sel]), rtol=1e-9)


def test_strict_passes_shared_factor_and_catches_a_one_sided_one():
    assert zs.strict("HC_exp", np.array([2.0]), 1.0, "m")["pass"]
    zs.FORMS["_jump"] = (("unstable",), "m", False, ("c",), (0.0,), (-1.0,), (1.0,),
                         lambda k, z, e, p, pr0: zs.base(k, z) * np.exp(p[0] * e))
    try:
        assert not zs.strict("_jump", np.array([2.0]), 1.0, "m")["pass"]  # factor only below 0: jump at zeta=0
    finally:
        del zs.FORMS["_jump"]


def test_placebo_is_a_fixed_permutation():
    s = {"m": {"zeta": -np.linspace(0.01, 2, 50), "phi": np.ones(50), "eta": np.linspace(0.01, 0.5, 50)}}
    a = zs._blocks([("x", s)], "H8", "placebo", "m")[0][0][3]
    b = zs._blocks([("x", s)], "H8", "placebo", "m")[0][0][3]
    assert np.array_equal(a, b) and sorted(a) == sorted(s["m"]["eta"]) and not np.array_equal(a, s["m"]["eta"])


def test_limit_mode_switches_eta_off_in_every_unstable_form():
    x = np.array([0.05, 0.5, 3.0])
    z, e = -x, np.full(3, zs.LIMIT_ETA)
    assert np.allclose(zs.model("H8", "m", z, e, np.array([10.0, 0.7]), 1.0), (1 + 10 * x) ** (-1 / 3), rtol=1e-6)
    assert np.allclose(zs.model("H8", "m", z, e, np.array([200.0, 0.05]), 1.0), (1 + 200 * x) ** (-1 / 3), rtol=1e-6)
    assert np.allclose(zs.model("E1", "m", z, e, np.array([10.0, 3.0]), 1.0), (1 + 10 * x) ** (-1 / 3), rtol=1e-6)
    assert np.allclose(zs.model("H1", "h", z, e, np.array([10.0, 3.0]), 0.9), 0.9 * (1 + 10 * x) ** (-1 / 3), rtol=1e-6)
    assert np.allclose(zs.model("H9", "m", z, e, np.array([12.0, 30.0, 1.0]), 1.0), (1 + 12 * x) ** -0.25, rtol=1e-6)
    assert np.allclose(zs.model("H2_cap", "m", z, e, np.array([-1.0]), 1.0), zs.base("m", z), rtol=1e-6)


def test_invalid_prediction_invalidates_the_primary_score():
    s = {"m": {"zeta": -np.linspace(0.1, 2, 40), "phi": np.ones(40), "eta": np.linspace(0.01, 0.5, 40)}}
    sc = zs.score("H8", np.array([10.0, -1.0]), s, "m", "unstable", 1.0)  # p < 0: 1 - eta^p < 0, phi non-finite
    assert sc["n_fail"] > 0 and np.isnan(sc["log_rmse"])


def test_fe_slope_day_clustered_se_exceeds_iid_for_day_shared_noise():
    rng = np.random.default_rng(2)
    days = np.repeat(np.arange(200), 50)
    zeta = rng.uniform(0.02, 3.0, days.size)
    eta = 10 ** (rng.uniform(-3, -0.1, 200)[days] + rng.normal(0, 0.05, days.size))
    r = rng.normal(0, 0.3, 200)[days] + rng.normal(0, 0.05, days.size)
    d = zs.fe_slope(zeta, eta, r, 30, days)
    assert d["se"] > 3 * d["se_iid"]


def test_strict_rejects_a_log_divergent_neutral_slope():
    zs.FORMS["_log"] = (("stable", "unstable"), "m", False, ("c",), (0.0,), (-1.0,), (1.0,),
                        lambda k, z, e, p, pr0: 1 + z + p[0] * z * np.log(np.abs(z)))
    try:
        assert not zs.strict("_log", np.array([1e-5]), 1.0, "m")["pass"]
        assert zs.strict("_log", np.array([0.0]), 1.0, "m")["pass"]
    finally:
        del zs.FORMS["_log"]


def test_placebo_permutes_fit_and_score_days_independently():
    eta = np.linspace(0.01, 0.5, 200)
    fit_ = zs.eta_for("x", "m", "unstable", eta, "placebo", 1)
    score_ = zs.eta_for("x", "m", "unstable", eta, "placebo", 0)
    assert not np.array_equal(fit_, score_) and sorted(fit_) == sorted(score_) == sorted(eta)


def test_fe_split_separates_height_from_zi():
    rng = np.random.default_rng(3)
    n = 30000
    days = rng.integers(0, 300, n)
    zeta = rng.uniform(0.02, 3.0, n)
    zg = 10 ** rng.uniform(0, 1.6, n)
    zi = 10 ** rng.uniform(2, 3, 300)[days]
    r = 0.1 * np.log(zg / zi) + rng.normal(0, 0.05, n)
    d = zs.fe_split(zeta, zg, zi, r, 30, days)
    assert d["b_z"] == pytest.approx(0.1, abs=0.01) and d["b_zi"] == pytest.approx(-0.1, abs=0.01)
    h = zs.fe_split(zeta, zg, zi, 0.1 * np.log(zg) + rng.normal(0, 0.05, n), 30, days)
    assert h["b_zi"] == pytest.approx(0.0, abs=0.01) and h["b_z_plus_b_zi"] == pytest.approx(0.1, abs=0.01)


def _rows(means, z):
    lv = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    return [{"levels": k, "ln_z": float(np.log(np.sqrt(z[k[0]] * z[k[1]]))), "n": 100, "mean": float(m)}
            for k, m in zip(lv, means)]


def test_pair_test_tells_level_bias_from_height():
    z = np.array([2.0, 5.0, 10.0, 20.0])
    lv = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    height = [0.2 * np.log(np.sqrt(z[i] * z[j])) for i, j in lv]
    bias = np.array([0.0, 0.3, -0.2, 0.1])
    sensor = [bias[j] - bias[i] for i, j in lv]
    noise = np.random.default_rng(4).normal(0, 0.005, (2, 6))
    h = zs.pair_test(_rows(height + noise[0], z), _rows(height + noise[1], z), 4)
    s_ = zs.pair_test(_rows(sensor + noise[0], z), _rows(sensor + noise[1], z), 4)
    assert h["height"]["r2"] > 0.95 and s_["sensor"]["r2"] > 0.95 and s_["height"]["r2"] < 0.5
    assert not h["sensor_saturated"]
