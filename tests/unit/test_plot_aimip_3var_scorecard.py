"""Direct tests for the AIMIP 3-variant scorecard plotter's pure helpers.

Exercises the anomaly / statistics math and the MAX_YEAR forcing-coverage
cap without invoking matplotlib rendering.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_PLOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "plot"
sys.path.insert(0, str(_PLOT_DIR))
import plot_aimip_3var_scorecard as sc  # noqa: E402


def test_anomaly_subtracts_baseline_mean():
    # Absolute T rising 1 K/yr over the 1979-2014 window; anomaly must be
    # centered on that window's mean, independent of the model's climatology.
    absT = {y: 300.0 + (y - 1979) for y in range(1979, 2015)}
    anom = sc._anomaly(absT)
    base = [anom[y] for y in range(sc.BASE_LO, sc.BASE_HI + 1)]
    assert abs(sum(base) / len(base)) < 1e-9      # baseline mean ~ 0
    assert anom[1979] == absT[1979] - (sum(absT.values()) / len(absT))


def test_stats_rms_and_bias():
    anom = {1979: -1.0, 1980: 1.0, 2013: 2.0, 2014: -2.0}
    y0, y1, n, mean, rms, rms_late = sc._stats(anom)
    assert (y0, y1, n) == (1979, 2014, 4)
    assert abs(mean - 0.0) < 1e-12
    assert abs(rms - math.sqrt((1 + 1 + 4 + 4) / 4)) < 1e-12
    # 2013+ RMS uses only the >=2013 subset (2.0, -2.0) -> 2.0
    assert abs(rms_late - 2.0) < 1e-12


def test_load_caps_at_max_year(tmp_path, monkeypatch):
    # A CSV with a spurious post-MAX_YEAR drift row must be dropped.
    csv = tmp_path / "legoesm_demo_annual.csv"
    csv.write_text(
        "year,annual_global_mean_surfT_K\n"
        "2021,285.0\n2022,285.1\n2023,281.0\n2024,281.0\n"
    )
    monkeypatch.setattr(sc, "FLEET", tmp_path)
    out = sc._load("legoesm_demo_annual")
    assert set(out) == {2021, 2022}          # 2023-2024 dropped
    assert sc.MAX_YEAR == 2022


def test_load_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "FLEET", tmp_path)
    assert sc._load("does_not_exist") is None
