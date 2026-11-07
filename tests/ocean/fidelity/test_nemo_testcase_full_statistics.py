"""Direct controls for the preregistered full-duration testcase scorer."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_full_statistics.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_testcase_full_statistics", SCRIPT)
assert SPEC and SPEC.loader
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


@pytest.mark.parametrize(
    ("candidate", "floor", "spread", "expected"),
    [
        (1.0, 1.0, 0.0, "INDISTINGUISHABLE-AT-FLOOR"),
        (2.0, 1.0, 2.0, "WITHIN-SCHEME-SPREAD"),
        (3.0, 1.0, 2.0, "OUTSIDE"),
    ],
)
def test_exact_three_way_verdict(candidate, floor, spread, expected):
    assert stats.verdict(candidate, floor, spread) == expected


def test_crossing_uses_linear_subcell_position():
    crossings = stats.ascending_crossings(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([10.0, 14.0, 18.0]),
        15.0,
    )
    np.testing.assert_array_equal(crossings, np.asarray([1.25]))


def test_connected_front_reducer_selects_rightmost_of_multiple_crossings():
    # Two cold-to-warm transitions behind/at the leading edge.  The frozen
    # preregistration names the rightmost crossing, not "exactly one".
    front, count = stats.rightmost_ascending_crossing(
        np.arange(5.0), np.asarray([10.0, 20.0, 10.0, 20.0, 20.0]),
        15.0, label="planted multi-crossing")
    assert count == 2
    assert front == 2.5


def test_connected_front_reducer_rejects_missing_crossing():
    with pytest.raises(stats.StatisticalError, match="missing ascending front"):
        stats.rightmost_ascending_crossing(
            np.arange(3.0), np.asarray([10.0, 10.0, 10.0]), 15.0,
            label="planted missing")


def test_endpoint_roundoff_debt_stays_loud_but_does_not_change_membership():
    values = np.asarray([14.0, 20.0 + 3.0e-12], dtype=np.float64)
    snapped, receipt = stats._snap_temperature(values, "OVERFLOW-zps", values.dtype)
    assert receipt["roundoff_status"] == "UNMEASURED"
    assert receipt["excess_relative"] < receipt["gross_guard_relative"]
    assert snapped[-1] == 20.0


def test_gross_temperature_control_hard_fails():
    values = np.asarray([10.0, 20.001], dtype=np.float64)
    with pytest.raises(stats.StatisticalError, match="gross T excursion"):
        stats._snap_temperature(values, "OVERFLOW-zps", values.dtype)


def test_precision_scaled_guard_requires_pinned_accumulation_artifact(tmp_path):
    artifact = tmp_path / "fp32_trace.json"
    artifact.write_text(
        json.dumps(
            {
                "preregistration_commit": stats.FP32_DISCRIMINATOR_PREREG_SHA,
                "case": "OVERFLOW-zps",
                "precision": "fp32",
                "all_finite": True,
                "classification": {
                    "classification": "PRECISION_ACCUMULATION",
                    "n_steps": 3060,
                },
            }
        )
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    guard, evidence = stats.precision_floor_guard(
        "OVERFLOW-zps",
        "L32",
        np.dtype(np.float32),
        discriminator_path=artifact,
        expected_sha256=digest,
    )
    assert guard == 6120 * np.finfo(np.float32).eps
    assert evidence["formula"] == "max(1e-6, N_steps * eps(dtype))"
    values = np.asarray([10.0, 20.0022], dtype=np.float32)
    _, receipt = stats._snap_temperature(
        values,
        "OVERFLOW-zps",
        values.dtype,
        gross_guard_relative=guard,
        guard_evidence=evidence,
    )
    assert receipt["roundoff_status"] == "UNMEASURED"


def test_precision_scaled_guard_rejects_tampered_artifact(tmp_path):
    artifact = tmp_path / "fp32_trace.json"
    artifact.write_text("{}")
    with pytest.raises(stats.StatisticalError, match="hash mismatch"):
        stats.precision_floor_guard(
            "OVERFLOW-zps",
            "L32",
            np.dtype(np.float32),
            discriminator_path=artifact,
            expected_sha256="0" * 64,
        )


def test_fp32_trace_classifier_accepts_gradual_precision_accumulation():
    excess = np.linspace(0.0, 1.0e-3, 3061)
    report = stats.classify_fp32_temperature_trace(excess, all_finite=True)
    assert report["classification"] == "PRECISION_ACCUMULATION"
    assert report["largest_jump_fraction_of_peak"] < 0.05
    assert report["peak_ulp_per_step"] < 1.0


def test_fp32_trace_classifier_detects_planted_limiter_jump():
    excess = np.zeros(3061)
    excess[2000:] = 1.0e-3
    report = stats.classify_fp32_temperature_trace(excess, all_finite=True)
    assert report["classification"] == "LIMITER_EVENT"
    assert report["largest_jump_fraction_of_peak"] == 1.0


def test_fp32_trace_classifier_fails_closed_on_nonfinite_state():
    excess = np.linspace(0.0, 1.0e-3, 3061)
    report = stats.classify_fp32_temperature_trace(excess, all_finite=False)
    assert report["classification"] == "NONFINITE"


def test_planted_census_distance_goes_outside():
    baseline = np.asarray([0.2, 0.5, 0.3])
    planted = np.asarray([1.0, 0.0, 0.0])
    candidate = stats._curve_distance(planted, baseline)
    assert stats.verdict(candidate, floor=0.0, spread=0.1) == "OUTSIDE"


def test_metric_registry_and_planted_controls_are_wired():
    source = SCRIPT.read_text()
    assert 'score_parser.add_argument("--plant-state"' in source
    assert 'score_parser.add_argument("--plant-census"' in source
    assert 'score_parser.add_argument("--plant-unregistered"' in source
    assert 'run_parser.add_argument("--check-finite-every-step"' in source
    assert '"first_nonfinite_completed_step": completed + 1' in source
    assert "names == REGISTERED_METRICS[case]" in source
    assert stats.REGISTERED_METRICS["LOCK_EXCHANGE-zco"] == {
        "front_position_km",
        "front_speed_anchor_ratio",
        "rpe_relative",
        "temperature_variance_fraction",
        "temperature_linf",
        "instantaneous_u_linf",
    }
