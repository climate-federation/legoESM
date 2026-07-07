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


def test_area_weighted_global_mean_drops_time_axis_and_nan():
    f = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 5.0)])  # (time=2,2,2)
    f[-1, 0, 0] = np.nan  # NaN excluded
    w = np.ones((2, 2))
    # last step used; NaN cell dropped -> mean of the three 5.0 cells
    assert area_weighted_global_mean(f, w) == pytest.approx(5.0)


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
