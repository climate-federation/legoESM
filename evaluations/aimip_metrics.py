"""AIMIP Phase-1 scalar metrics (E-battery helpers, design D8).

E2 of the AIMIP evaluation battery (arXiv:2605.06944): linear trends of the
annual global-mean anomaly time series, computed SEPARATELY for the training
era (1979-2014) and the out-of-sample decade (2015-2024) — the Phase-1
finding is that weather RMSE and holdout-trend fidelity are uncorrelated
failure axes, so the trend split is reported next to bias/RMSE (E1).

Pure NumPy, jax-free, import-light.
"""

from __future__ import annotations

import numpy as np

# AIMIP Phase-1 protocol windows (train on ERA5 1979-2014; 2015-2024 is the
# out-of-sample decade).
TRAIN_ERA = (1979, 2014)
HOLDOUT_ERA = (2015, 2024)


def annual_trend_k_per_decade(years, values, y0=None, y1=None) -> float:
    """OLS trend of an annual series in K/decade over [y0, y1] inclusive.

    NaN-safe (non-finite values dropped). Returns ``nan`` when fewer than 3
    finite points fall in the window (a 2-point "trend" is noise).
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    m = np.isfinite(values) & np.isfinite(years)
    if y0 is not None:
        m &= years >= y0
    if y1 is not None:
        m &= years <= y1
    if m.sum() < 3:
        return float("nan")
    slope_per_year = np.polyfit(years[m], values[m], 1)[0]
    return float(slope_per_year * 10.0)


def e2_trend_metrics(years, values) -> dict:
    """Train-era + holdout-era trends [K/decade] for an annual series."""
    return {
        "trend_train_K_per_decade": annual_trend_k_per_decade(
            years, values, *TRAIN_ERA
        ),
        "trend_holdout_K_per_decade": annual_trend_k_per_decade(
            years, values, *HOLDOUT_ERA
        ),
    }
