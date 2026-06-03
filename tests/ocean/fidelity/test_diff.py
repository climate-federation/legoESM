"""Unit tests for legoesm.ocean.fidelity.diff."""

from __future__ import annotations

import math

import pytest

from legoesm.ocean.fidelity import diff


def test_measurement_inside_window_passes():
    rec = diff.diff_metric(
        case_name="case", tier=1, metric="m",
        measured=0.5, tolerance_window=(0.0, 1.0),
    )
    assert rec.passed is True
    assert rec.triage_hint is None


def test_measurement_below_window_fails_with_hint():
    rec = diff.diff_metric(
        case_name="case", tier=2, metric="m",
        measured=-0.1, tolerance_window=(0.0, 1.0),
    )
    assert rec.passed is False
    assert rec.triage_hint == diff.TIER_TRIAGE_HINTS[2]


def test_measurement_above_window_fails():
    rec = diff.diff_metric(
        case_name="case", tier=0, metric="m",
        measured=1.5, tolerance_window=(0.0, 1.0),
    )
    assert rec.passed is False


def test_nonfinite_measurement_fails_with_notes():
    rec = diff.diff_metric(
        case_name="case", tier=3, metric="m",
        measured=float("nan"), tolerance_window=(0.0, 1.0),
    )
    assert rec.passed is False
    assert "non-finite" in rec.notes
    assert rec.triage_hint == diff.TIER_TRIAGE_HINTS[3]
    assert math.isnan(rec.measured)


def test_inf_measurement_fails_with_notes():
    rec = diff.diff_metric(
        case_name="case", tier=5, metric="m",
        measured=float("inf"), tolerance_window=(0.0, 1.0),
    )
    assert rec.passed is False
    assert "non-finite" in rec.notes


def test_window_lo_gt_hi_raises():
    with pytest.raises(ValueError, match="lo <= hi"):
        diff.diff_metric(
            case_name="case", tier=1, metric="m",
            measured=0.0, tolerance_window=(1.0, 0.0),
        )


def test_record_format_includes_verdict():
    rec = diff.diff_metric(
        case_name="igw", tier=1, metric="dispersion_rms_error",
        measured=0.01, tolerance_window=(0.0, 0.02),
    )
    s = rec.format()
    assert "PASS" in s
    assert "T1" in s
    assert "igw" in s
    assert "dispersion_rms_error" in s


def test_record_format_includes_fail_and_notes_when_nonfinite():
    rec = diff.diff_metric(
        case_name="x", tier=0, metric="m",
        measured=float("nan"), tolerance_window=(0.0, 1.0),
    )
    s = rec.format()
    assert "FAIL" in s
    assert "non-finite" in s


def test_all_nine_tiers_have_triage_hints():
    for tier in range(9):
        assert tier in diff.TIER_TRIAGE_HINTS
        assert isinstance(diff.TIER_TRIAGE_HINTS[tier], str)
        assert len(diff.TIER_TRIAGE_HINTS[tier]) > 30


def test_record_is_frozen():
    rec = diff.diff_metric(
        case_name="c", tier=0, metric="m",
        measured=0.5, tolerance_window=(0.0, 1.0),
    )
    with pytest.raises(Exception):
        rec.passed = False  # type: ignore[misc]
