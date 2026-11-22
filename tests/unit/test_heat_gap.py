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
    med, rows, _ = uh.ratio(data)
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
    x = (None, np.array([[1.0, 2.0], [3.0, nan]]), np.zeros(2))
    y = (None, np.array([[5.0, nan], [7.0, 8.0]]), np.zeros(2))
    (mx, rx, _), (my, ry, _) = uh.paired(x, y)
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
    med, rows, _ = uh.ratio(data, 1, ref)
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
