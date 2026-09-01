"""Direct controls for the preregistered full-duration testcase scorer."""

from __future__ import annotations

import importlib.util
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


def test_metric_registry_and_planted_controls_are_wired():
    source = SCRIPT.read_text()
    assert 'score_parser.add_argument("--plant-state"' in source
    assert 'score_parser.add_argument("--plant-census"' in source
    assert 'score_parser.add_argument("--plant-unregistered"' in source
    assert "names == REGISTERED_METRICS[case]" in source
    assert stats.REGISTERED_METRICS["LOCK_EXCHANGE-zco"] == {
        "front_position_km",
        "front_speed_anchor_ratio",
        "rpe_relative",
        "temperature_variance_fraction",
        "temperature_linf",
        "instantaneous_u_linf",
    }
