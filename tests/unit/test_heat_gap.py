import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import heat_gap as uh  # noqa: E402


def test_ratio_is_unstable_heat_against_dyer_on_even_days():
    z = -np.repeat([0.3, 0.4], 15)
    dyer = (1 - 16 * z) ** -0.5
    stable = np.full(30, 0.3)
    data = {"a": {"h": {0: (np.r_[z, stable], np.r_[1.3 * dyer, np.full(30, 9.0)]), 1: (z, 5 * dyer)},
                  "m": {0: (z, 9 * dyer), 1: (z, 9 * dyer)}},
            "b": {"h": {0: (z, 1.5 * dyer), 1: (z, dyer)}, "m": {0: (z, dyer), 1: (z, dyer)}}}
    med, rows, _, _ = uh.ratio(data)
    i = np.searchsorted(uh.EDGES, 0.3) - 1
    assert np.allclose(rows[:, i], [1.3, 1.5]) and abs(med[i] - 1.4) < 1e-12
    assert np.isfinite(rows).sum() == 2


def test_fixed_site_set_needs_every_bin_in_span():
    span = (uh.EDGES[:-1] >= uh.FIXED_BINS[0]) & (uh.EDGES[1:] <= uh.FIXED_BINS[1])
    assert span.sum() == 5  # -z/L 0.1-0.2, 0.2-0.5, 0.5-1, 1-2, 2-5
    full = np.where(span, 1.2, np.nan)
    gap = full.copy()
    gap[np.where(span)[0][-1]] = np.nan
    outside = np.where(span, 1.2, 3.0)
    assert uh.fixed_site_set(["a", "b", "c"], np.array([full, gap, outside])) == ["a", "c"]


def test_paired_keeps_only_cells_finite_in_both_arms():
    nan = np.nan
    x = (None, np.array([[1.0, 2.0], [3.0, nan]]), None, np.array([[0.1, 0.2], [0.3, 0.4]]))
    y = (None, np.array([[5.0, nan], [7.0, 8.0]]), None, np.array([[0.5, 0.6], [0.7, 0.8]]))
    (mx, rx, zx, _), (my, ry, zy, _) = uh.paired(x, y)
    assert np.allclose(zx[0], 0.2) and np.isnan(zx[1]) and np.allclose(zy[0], 0.6)  # coordinates from kept cells only
    assert np.allclose(mx[0], 2.0) and np.isnan(mx[1]) and np.allclose(my[0], 6.0) and np.isnan(my[1])
    assert np.isfinite(rx).sum() == 2 and np.isfinite(ry).sum() == 2


def test_oracle_scale_gain_uses_odd_days_to_fit_and_even_days_to_score():
    z = -np.linspace(0.15, 3.0, 40)
    d = (1 - 16 * z) ** -0.5
    wiggle = np.exp(0.1 * np.sin(np.arange(40)))
    data = {"a": {"h": {1: (z, 1.5 * d * wiggle), 0: (z, 1.5 * d * wiggle)}},
            "b": {"h": {1: (z, 1.0 * d * wiggle), 0: (z, 2.0 * d * wiggle)}}}
    g = uh.oracle_scale_gain(data)
    rmse = lambda r: np.sqrt(np.mean(np.log(r) ** 2))  # noqa: E731
    c_a = 1.5 * np.exp(np.mean(np.log(wiggle)))
    assert abs(g["a"] - (rmse(1.5 * wiggle / c_a) - rmse(1.5 * wiggle))) < 1e-12 and g["a"] < -0.3
    c_b = np.exp(np.mean(np.log(wiggle)))
    assert abs(g["b"] - (rmse(2.0 * wiggle / c_b) - rmse(2.0 * wiggle))) < 1e-12


def test_stable_side_and_reference_are_honoured():
    z = np.repeat([0.3, 0.4], 15)
    ref = lambda x: 2.0 + x  # noqa: E731
    data = {"a": {"h": {0: (np.r_[z, -z], np.r_[1.25 * ref(z), np.full(30, 9.0)]), 1: (z, ref(z) * 0.5)},
                  "m": {0: (z, z), 1: (z, z)}}}
    med, rows, _, _ = uh.ratio(data, 1, ref)
    i = np.searchsorted(uh.EDGES, 0.3) - 1
    assert abs(med[i] - 1.25) < 1e-12 and np.isfinite(rows).sum() == 1
    w = np.exp(0.1 * np.sin(np.arange(30)))
    g = uh.oracle_scale_gain({"a": {"h": {1: (np.r_[z, -z], np.r_[ref(z) * w, np.full(30, 9.0)]),
                                          0: (np.r_[z, -z], np.r_[2 * ref(z) * w, np.full(30, 9.0)])}}}, 1, ref)
    c = np.exp(np.mean(np.log(w)))
    rmse = lambda r: np.sqrt(np.mean(np.log(r) ** 2))  # noqa: E731
    assert abs(g["a"] - (rmse(2 * w / c) - rmse(2 * w))) < 1e-12


def test_reference_j3_reads_the_benchmark_global_fit(tmp_path):
    import json
    (tmp_path / "b.json").write_text(json.dumps({"global_fit": {"J3": {"a0": 4.0, "a_inf": 0.5, "zc": 1.2, "pr0": 0.9}}}))
    f = uh.reference("j3", str(tmp_path / "b.json"))
    z = np.array([-0.5, 0.3, 2.0])
    st = 0.9 + 0.5 * z + 3.5 * z / (1 + z / 1.2)
    assert np.allclose(f(z), np.where(z > 0, st, 0.9 * (1 - 16 * z) ** -0.5))
    assert uh.reference("dyer") is uh.dyer


def test_sign_flips_join_on_time_and_heights_even_days_one_side():
    a = {"time": np.array([1.0, 2, 3, 4]), "z1": np.array([2.0, 2, 4, 2]), "z2": np.array([4.0, 4, 8, 4]),
         "pair": np.array([0, 0, 1, 0]), "zeta": np.array([0.2, 0.3, 0.4, -0.2]), "phi": np.array([1.0, 1.0, 1.0, 1.0])}
    b = {"time": np.array([3.0, 2, 1, 4]), "z1": np.array([4.0, 2, 2, 2]), "z2": np.array([8.0, 4, 4, 4]),
         "pair": np.array([0, 0, 0, 0]), "zeta": np.array([0.4, 0.3, 0.2, -0.2]), "phi": np.array([-1.0, 2.0, 1.0, -1.0])}
    odd = {k: v.copy() for k, v in b.items()}
    odd["phi"] = -odd["phi"]
    corr = {("s", "h", 0): a, ("s", "h", 1): a, ("s", "m", 0): a}
    plain = {("s", "h", 0): b, ("s", "h", 1): odd, ("s", "m", 0): b}
    assert uh.sign_flips(corr, plain, 1) == (1, 0, 3)  # pair numbers differ at t=3; heights match
    assert uh.sign_flips(corr, plain, -1) == (1, 0, 1)
    b0 = dict(b, phi=np.array([0.0, 2.0, 1.0, -1.0]))
    assert uh.sign_flips(corr, {("s", "h", 0): b0}, 1) == (0, 1, 3)  # positive to zero is a zero change, not a reversal


def test_matched_keeps_observations_present_and_positive_in_both():
    a = {"time": np.array([1.0, 2, 3]), "z1": np.array([2.0, 2, 2]), "z2": np.array([4.0, 4, 4]),
         "zeta": np.array([0.2, 0.3, 0.4]), "phi": np.array([1.0, 1.0, -1.0])}
    b = {"time": np.array([1.0, 2, 2, 3]), "z1": np.array([2.0, 2, 4, 2]), "z2": np.array([4.0, 4, 8, 4]),
         "zeta": np.array([0.2, 0.3, 0.3, 0.4]), "phi": np.array([5.0, -6.0, 7.0, 8.0])}
    ca, cb = uh.matched({("s", "h", 0): a}, {("s", "h", 0): b, ("s", "h", 1): b})
    assert np.allclose(ca["s"]["h"][0][1], [1.0]) and np.allclose(cb["s"]["h"][0][1], [5.0])
    assert cb["s"]["h"][1][1].size == 0 and ca["s"]["h"][1][1].size == 0


def test_check_caches_refuses_swapped_or_mismatched_caches(tmp_path):
    import json
    import pytest
    import hashlib
    base = {"tables": "/t", "min_ustar": 0.1, "max_zg": 50.0, "sites": ["a"], "table_sha256": {"a.nc": "x"}}
    digest = {}
    for name, off in (("c", True), ("p", False)):
        (tmp_path / f"{name}.npz").write_bytes(name.encode())
        digest[name] = hashlib.sha256(name.encode()).hexdigest()
        (tmp_path / f"{name}.npz.json").write_text(json.dumps(dict(base, offset_correct=off, sha256=digest[name])))
    c, p = str(tmp_path / "c.npz"), str(tmp_path / "p.npz")
    uh.check_caches(c, p)
    with pytest.raises(SystemExit):
        uh.check_caches(p, c)
    (tmp_path / "c.npz").write_bytes(b"p")  # payload swapped under its record
    with pytest.raises(SystemExit, match="provenance"):
        uh.check_caches(c, p)
    (tmp_path / "c.npz").write_bytes(b"c")
    (tmp_path / "p.npz.json").write_text(json.dumps(dict(base, offset_correct=False, min_ustar=0.2, sha256=digest["p"])))
    with pytest.raises(SystemExit, match="min_ustar"):
        uh.check_caches(c, p)
    (tmp_path / "p.npz.json").write_text(json.dumps(dict(base, offset_correct=False, sha256=digest["p"], table_sha256={"a.nc": "y"})))
    with pytest.raises(SystemExit, match="table_sha256"):
        uh.check_caches(c, p)
    for name in ("c", "p"):
        (tmp_path / f"{name}.npz.json").write_text(json.dumps(dict(base, offset_correct=name == "c", sha256=digest[name], table_sha256={})))
    with pytest.raises(SystemExit, match="table hashes"):
        uh.check_caches(c, p)
