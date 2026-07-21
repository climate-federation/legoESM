"""Unit tests for evaluations/aimip_metrics.py (AIMIP E2 trend helpers)."""

import numpy as np
import pytest

from evaluations.aimip_metrics import (
    HOLDOUT_ERA,
    TRAIN_ERA,
    annual_trend_k_per_decade,
    e2_trend_metrics,
)


def test_linear_ramp_recovers_exact_trend():
    years = np.arange(1979, 2025)
    values = 0.02 * (years - 1979.0)  # 0.02 K/yr = 0.2 K/decade
    assert annual_trend_k_per_decade(years, values) == pytest.approx(0.2)
    m = e2_trend_metrics(years, values)
    assert m["trend_train_K_per_decade"] == pytest.approx(0.2)
    assert m["trend_holdout_K_per_decade"] == pytest.approx(0.2)


def test_windows_are_independent():
    years = np.arange(1979, 2025)
    # flat during training era, +0.5 K/decade afterwards
    values = np.where(years <= 2014, 1.0, 1.0 + 0.05 * (years - 2014.0))
    m = e2_trend_metrics(years, values)
    assert m["trend_train_K_per_decade"] == pytest.approx(0.0, abs=1e-12)
    assert m["trend_holdout_K_per_decade"] == pytest.approx(0.5)


def test_nan_safe_and_min_points():
    years = np.arange(1979, 2025)
    values = 0.02 * (years - 1979.0)
    values[::2] = np.nan  # half missing -> still a clean trend
    assert annual_trend_k_per_decade(years, values) == pytest.approx(0.2)
    # < 3 finite points -> nan, not a fake 2-point trend
    assert np.isnan(annual_trend_k_per_decade([2015, 2016], [1.0, 2.0]))
    assert np.isnan(annual_trend_k_per_decade(years, np.full_like(values, np.nan)))


def test_protocol_windows():
    assert TRAIN_ERA == (1979, 2014)
    assert HOLDOUT_ERA == (2015, 2024)
