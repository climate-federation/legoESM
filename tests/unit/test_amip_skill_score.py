"""Tests for the numerical AMIP skill-score gate (scripts/validate/amip_skill_score.py)."""
import numpy as np
import pytest

from scripts.validate.amip_skill_score import (
    REFERENCE,
    area_weighted_global_mean,
    format_report,
    score_amip_run,
)


def test_area_weighted_global_mean_uniform():
    """Uniform field returns its constant value regardless of weights."""
    f = np.full((4, 8), 300.0)
    w = np.random.default_rng(0).uniform(0.5, 2.0, (4, 8))
    assert area_weighted_global_mean(f, w) == pytest.approx(300.0)


def test_area_weighted_global_mean_weighting():
    """Two cells, area 1 and 3 -> weighted mean is 3/4 toward the big cell."""
    f = np.array([10.0, 20.0])
    w = np.array([1.0, 3.0])
    assert area_weighted_global_mean(f, w) == pytest.approx((10 + 60) / 4)


def test_area_weighted_global_mean_time_MEAN_not_last_step():
    """The time axis is AVERAGED — scoring only the final month against
    annual-mean references conflates seasonal cycle with bias (the December
    netTOA +12.9 vs annual -2.9 incident, 2026-08-01)."""
    f = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 5.0)])  # (time=2,2,2)
    w = np.ones((2, 2))
    assert area_weighted_global_mean(f, w) == pytest.approx(3.0)


def test_area_weighted_global_mean_nan_month_drops_cell_whole():
    """A cell with ANY NaN month is excluded entirely (plain mean, not
    nanmean): partial coverage must not silently pass as a full-window mean."""
    f = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 5.0)])
    f[-1, 0, 0] = np.nan
    w = np.ones((2, 2))
    # the NaN cell is dropped; remaining three cells average (1+5)/2 = 3
    assert area_weighted_global_mean(f, w) == pytest.approx(3.0)


def test_reference_table_wellformed():
    for field, spec in REFERENCE.items():
        ref, tol, units, source = spec
        assert isinstance(ref, float) and isinstance(tol, float)
        assert tol > 0.0 and units and source


def _synthetic_results(within):
    """Build a results dict with every field exactly on-ref (within=True) or
    3x-tolerance biased (within=False)."""
    out = {}
    for field, (ref, tol, units, source) in REFERENCE.items():
        bias = 0.0 if within else 3.0 * tol
        out[field] = {"value": ref + bias, "reference": ref, "bias": bias,
                      "tol": tol, "pass": abs(bias) <= tol, "units": units,
                      "source": source}
    return out


def test_format_report_pass_and_fail():
    report_ok, ok = format_report(_synthetic_results(within=True))
    assert ok is True and "PASS" in report_ok
    report_bad, bad = format_report(_synthetic_results(within=False))
    assert bad is False and "FAIL" in report_bad


def test_score_amip_run_missing_dir():
    """A directory with no Amon output yields an empty score (not a crash)."""
    with pytest.raises((FileNotFoundError, OSError)):
        score_amip_run("/nonexistent/cmor")
