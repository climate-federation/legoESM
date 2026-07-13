"""Unit tests for :mod:`legoesm.land.evaluation.metrics`.

Covers the elementary error metrics, the PLUMBER2 distribution metrics, the
ILAMB unit-interval scores, sentinel/NaN masking, the metric registry
dispatch guard, and the backward-compatible ``scalar_stats`` shim.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.land.evaluation import metrics as m


def test_perfect_match_scores_are_ideal():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=500)
    mod = ref.copy()
    assert m.bias(ref, mod) == pytest.approx(0.0)
    assert m.rmse(ref, mod) == pytest.approx(0.0)
    assert m.centered_rmse(ref, mod) == pytest.approx(0.0)
    assert m.nme(ref, mod) == pytest.approx(0.0)
    assert m.pearson_r(ref, mod) == pytest.approx(1.0)
    assert m.bias_score(ref, mod) == pytest.approx(1.0)
    assert m.rmse_score(ref, mod) == pytest.approx(1.0)
    assert m.taylor_score(ref, mod) == pytest.approx(1.0)
    assert m.pdf_overlap(ref, mod) == pytest.approx(1.0, abs=1e-9)


def test_constant_offset_is_pure_bias():
    ref = np.linspace(0.0, 10.0, 200)
    mod = ref + 2.0
    assert m.bias(ref, mod) == pytest.approx(2.0)
    # A pure offset has zero centralised RMSE and perfect correlation.
    assert m.centered_rmse(ref, mod) == pytest.approx(0.0, abs=1e-9)
    assert m.pearson_r(ref, mod) == pytest.approx(1.0)
    assert m.rmse_score(ref, mod) == pytest.approx(1.0, abs=1e-9)
    # rmse includes the bias -> nonzero.
    assert m.rmse(ref, mod) == pytest.approx(2.0)


def test_relative_error_score_mapping():
    assert m.relative_error_score(0.0) == pytest.approx(1.0)
    assert m.relative_error_score(0.5) == pytest.approx(np.exp(-0.5))
    assert m.relative_error_score(1.0) == pytest.approx(np.exp(-1.0))
    assert np.isnan(m.relative_error_score(np.nan))
    # Score is monotonically decreasing in error.
    assert m.relative_error_score(0.2) > m.relative_error_score(0.8)


def test_nme_normalises_by_reference_variability():
    ref = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # Predict the mean everywhere -> NME == 1 by construction.
    mod = np.full_like(ref, ref.mean())
    assert m.nme(ref, mod) == pytest.approx(1.0)


def test_pdf_overlap_disjoint_and_identical():
    a = np.linspace(0.0, 1.0, 1000)
    b = np.linspace(10.0, 11.0, 1000)
    assert m.pdf_overlap(a, b) == pytest.approx(0.0, abs=1e-6)
    assert m.pdf_overlap(a, a) == pytest.approx(1.0, abs=1e-9)


def test_phase_score_diurnal():
    n = 48
    x = np.arange(n)
    ref = np.cos(2 * np.pi * (x - 24) / n)  # peak at bin 24
    same = ref.copy()
    assert m.phase_score(ref, same) == pytest.approx(1.0)
    half = np.roll(ref, n // 2)  # peak shifted half a period
    assert m.phase_score(ref, half) == pytest.approx(0.0, abs=1e-9)
    quarter = np.roll(ref, n // 4)
    assert 0.0 < m.phase_score(ref, quarter) < 1.0


def test_masking_sentinels_and_nan():
    ref = np.array([1.0, 2.0, 1e36, np.nan, 4.0])
    mod = np.array([1.0, 2.0, 3.0, 5.0, 4.0])
    r, mm = m.finite_pair(ref, mod)
    assert r.tolist() == [1.0, 2.0, 4.0]
    assert mm.tolist() == [1.0, 2.0, 4.0]
    # Metric only sees the 3 valid pairs -> perfect.
    assert m.rmse(ref, mod) == pytest.approx(0.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        m.finite_pair(np.zeros(3), np.zeros(4))


def test_too_few_points_returns_nan():
    # Variance-based scores need >= 2 valid pairs.
    assert np.isnan(m.taylor_score(np.array([1.0]), np.array([1.0])))
    assert np.isnan(m.nrmse(np.array([1.0]), np.array([1.0])))
    # rmse/bias/mae are well-defined on a single pair.
    assert m.rmse(np.array([1.0]), np.array([1.0])) == 0.0
    # Zero valid pairs (all masked) -> nan.
    assert np.isnan(m.rmse(np.array([np.nan]), np.array([1.0])))


def test_diurnal_cycle_bins():
    # Two days at exact half-hourly cadence, value == hour-of-day.  Use an
    # exact integer/48 grid so every bin gets exactly 2 samples (a raw
    # np.arange(0, 2, 1/48) drifts and can leave a bin empty).
    t = np.arange(96) / 48.0
    hours = (t % 1.0) * 24.0
    cyc = m.diurnal_cycle(t, hours, n_bins=48)
    assert cyc.shape == (48,)
    assert np.all(np.isfinite(cyc))
    assert cyc[0] == pytest.approx(0.0, abs=1e-9)
    # Monotone increasing across the day.
    assert cyc[24] > cyc[0]


def test_registry_and_get_metric_guard():
    assert set(["bias", "rmse", "corr", "taylor_score"]).issubset(
        m.METRIC_REGISTRY
    )
    assert "bias_score" in m.SCORE_METRICS
    with pytest.raises(ValueError, match="Unknown metric"):
        m.get_metric("not_a_metric")


def test_scalar_stats_backward_compatible_keys():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    mod = np.array([1.1, 2.1, 2.9, 4.2])
    s = m.scalar_stats(ref, mod)
    assert set(s) == {"n", "rmse", "mae", "bias", "r2", "corr", "nrmse"}
    assert s["n"] == 4
    assert s["r2"] == pytest.approx(s["corr"] ** 2)


def test_percentile_error_registry_lambdas():
    ref = np.linspace(0, 100, 1000)
    mod = ref + 5.0
    assert m.get_metric("p5_error")(ref, mod) == pytest.approx(5.0, abs=0.5)
    assert m.get_metric("p95_error")(ref, mod) == pytest.approx(5.0, abs=0.5)
