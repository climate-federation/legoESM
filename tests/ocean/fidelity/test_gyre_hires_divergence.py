import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / (
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_divergence.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_hires_divergence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIVERGENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIVERGENCE)


def test_normalized_rms_and_correlation_known_answer_and_one_ulp_plant():
    reference = np.array([[1.0, 2.0], [3.0, 4.0]])
    candidate = reference + 1.0
    mask = np.ones_like(reference, dtype=bool)
    row = DIVERGENCE.normalized_rms_and_correlation(
        reference, candidate, mask)
    assert row["normalized_rms_difference"] == pytest.approx(1.0 / np.std(reference))
    assert row["pattern_correlation"] == pytest.approx(1.0)

    exact = DIVERGENCE.normalized_rms_and_correlation(reference, reference, mask)
    planted = reference.copy()
    planted[0, 0] = np.nextafter(planted[0, 0], np.inf)
    moved = DIVERGENCE.normalized_rms_and_correlation(reference, planted, mask)
    assert exact["normalized_rms_difference"] == 0.0
    assert moved["normalized_rms_difference"] > 0.0


def test_divergence_curve_preserves_daily_order():
    reference = np.stack([
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[2.0, 4.0], [6.0, 8.0]]),
    ])
    candidate = reference.copy()
    candidate[1, 0, 0] += 1.0
    curve = DIVERGENCE.divergence_curve(
        reference, candidate, np.ones((2, 2), dtype=bool))
    assert np.array_equal(curve["normalized_rms_difference"][:1], [0.0])
    assert curve["normalized_rms_difference"][1] > 0.0
    assert curve["absolute_rms_difference"][1] > 0.0


def test_wet_span_regions_and_signed_excess_localize_western_seed():
    mask = np.ones((2, 10), dtype=bool)
    bins = DIVERGENCE.wet_span_bins(mask)
    assert np.array_equal(bins[0], np.repeat(np.arange(5), 2))
    reference = np.zeros((4, 2, 10))
    twin = reference.copy()
    candidate = reference.copy()
    candidate[:, :, :2] = 3.0
    rows = DIVERGENCE.regional_excess(
        reference, candidate, twin, mask, start_day=1, stop_day=4)
    assert rows[0]["region"] == "western_boundary"
    assert rows[0]["signed_excess_sse"] == 144.0
    assert rows[0]["signed_excess_share"] == 1.0
    assert all(row["signed_excess_sse"] == 0.0 for row in rows[1:])


def test_first_factor_excess_names_earliest_field_and_tie():
    def curve(values):
        return {"normalized_rms_difference": np.asarray(values, dtype=float)}

    twin = {"ssh": curve([1, 1, 1]), "sst": curve([1, 1, 1])}
    candidate = {"ssh": curve([1, 5, 6]), "sst": curve([1, 5, 3])}
    result = DIVERGENCE.first_factor_excess(
        candidate, twin, factor=4.0, start_day=1, stop_day=3)
    assert result["day"] == 2
    assert result["fields"] == ["ssh", "sst"]


def test_first_factor_excess_serializes_a_zero_twin_floor_without_infinity():
    def curve(values):
        return {"normalized_rms_difference": np.asarray(values, dtype=float)}

    result = DIVERGENCE.first_factor_excess(
        {"sst": curve([1.0])}, {"sst": curve([0.0])},
        factor=4.0, start_day=1, stop_day=1)
    assert result["rows"] == [{
        "field": "sst", "candidate": 1.0, "twin": 0.0,
        "ratio": None, "twin_zero": True,
    }]


def test_growth_law_controls_separate_exponential_and_algebraic():
    days = np.arange(30.0, 181.0)
    exponential = DIVERGENCE.fit_growth_laws(days, np.exp(0.02 * days))
    algebraic = DIVERGENCE.fit_growth_laws(days, days ** 2.5)
    assert exponential["classification"] == "exponential"
    assert algebraic["classification"] == "algebraic"
    with pytest.raises(Exception, match="positive fitted growth"):
        DIVERGENCE.fit_growth_laws(days, np.exp(-0.02 * days))


def test_thickness_weighted_depth_mean_known_answer_and_refusals():
    velocity = np.array([[[[1.0]], [[3.0]]], [[[2.0]], [[4.0]]]])
    weights = np.array([[[1.0]], [[3.0]]])
    result = DIVERGENCE.thickness_weighted_depth_mean(velocity, weights)
    assert np.array_equal(result[:, 0, 0], [2.5, 3.5])
    with pytest.raises(Exception, match="negative"):
        DIVERGENCE.thickness_weighted_depth_mean(velocity, -weights)
    with pytest.raises(Exception, match="no active"):
        DIVERGENCE.thickness_weighted_depth_mean(
            velocity, np.zeros_like(weights))


def test_time_tile_assembly_places_prefix_dimensions():
    left = np.ones((3, 2, 2))
    right = np.full((3, 2, 2), 2.0)
    assembled = DIVERGENCE._assemble_time_tiles(
        [(np.array([1, 1]), left), (np.array([3, 1]), right)],
        shape=(2, 4),
    )
    assert assembled.shape == (3, 2, 4)
    assert np.all(assembled[:, :, :2] == 1.0)
    assert np.all(assembled[:, :, 2:] == 2.0)


def test_yearly_eddy_fraction_uses_full_series_and_all_years():
    day = np.arange(730, dtype=np.float64)
    signal = np.sin(2.0 * np.pi * day / 7.0)[:, None, None]
    u = np.broadcast_to(2.0 + signal, (730, 2, 3)).copy()
    v = np.broadcast_to(signal, (730, 2, 3)).copy()
    rows = DIVERGENCE.yearly_eddy_statistics(
        u, v, np.ones((2, 3), dtype=bool))
    assert [row["year"] for row in rows] == [1, 2]
    assert all(row["eddy_fraction"] > 0.0 for row in rows)
    zero_eddy = DIVERGENCE.yearly_eddy_statistics(
        np.full_like(u, 2.0), np.zeros_like(v), np.ones((2, 3), dtype=bool))
    assert [row["eddy_fraction"] for row in zero_eddy] == [0.0, 0.0]


def test_twin_benchmark_refuses_a_shifted_curve():
    curve = {
        "ssh": {"normalized_rms_difference": np.zeros(365)},
        "u": {"normalized_rms_difference": np.zeros(365)},
    }
    for day, value in DIVERGENCE.OPERATOR_TWIN_SSH_BENCHMARK.items():
        curve["ssh"]["normalized_rms_difference"][day - 1] = float(value)
    assert DIVERGENCE._validate_twin(curve)["status"] == "PASS"
    curve["ssh"]["normalized_rms_difference"][29] *= 2.0
    with pytest.raises(Exception, match="benchmark"):
        DIVERGENCE._validate_twin(curve)
