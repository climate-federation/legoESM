"""Known-answer checks for ERA5 boundary-layer height on flux-table time axes."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import zi_era5 as zi  # noqa: E402

GRID_LAT = np.array([0.0, 0.0, 45.0])
GRID_LON = np.array([0.0, 90.0, 0.0])


def fake_day(job):
    _, hours, points = job
    return hours / zi.HOUR + 1000.0 * points


def test_midpoint_hours():
    h0, w = zi.midpoint_hours(np.array([3600, 5400, 7200]), 30)
    np.testing.assert_array_equal(h0, [0, 3600, 3600])
    np.testing.assert_allclose(w, [0.75, 0.25, 0.75])
    h0, w = zi.midpoint_hours(np.array([7200]), 60)
    np.testing.assert_array_equal(h0, [3600])
    np.testing.assert_allclose(w, [0.5])


def test_track_crosses_dateline_without_sweeping_the_globe():
    v = zi.track_vectors(np.array([0.0, 2.0]), np.array([80.0, 80.0]), np.array([179.0, -179.0]), np.array([1.0]))
    lon = np.rad2deg(np.arctan2(v[0, 1], v[0, 0]))
    assert abs(abs(lon) - 180.0) < 1e-6


def test_era5_zi_nearest_point_and_time_interpolation(monkeypatch):
    monkeypatch.setattr(zi, "_grid", lambda: (GRID_LAT, GRID_LON))
    monkeypatch.setattr(zi, "_read_day", fake_day)
    t = np.array([zi.DAY + 3600, zi.DAY + 5400, 2 * zi.DAY])  # last interval straddles midnight
    near_b = zi.unit_vectors(np.array([1.0]), np.array([89.0]))
    near_c = zi.unit_vectors(np.array([44.0]), np.array([1.0]))
    sites = [("a", t, 30, np.repeat(near_b, 3, 0)), ("b", t[:2], 30, np.repeat(near_c, 2, 0))]
    out = zi.era5_zi(sites, workers=2)
    h0, w = zi.midpoint_hours(t, 30)
    expect = lambda p, h0, w: (1 - w) * (h0 / zi.HOUR + 1000 * p) + w * (h0 / zi.HOUR + 1 + 1000 * p)
    np.testing.assert_allclose(out["a"]["zi"], expect(1, h0, w))
    np.testing.assert_allclose(out["b"]["zi"], expect(2, h0[:2], w[:2]))
    assert np.all(out["a"]["distance_km"] < 200) and np.all(out["a"]["grid_lon"] == 90.0)


def test_track_is_nan_outside_its_time_span():
    v = zi.track_vectors(np.array([0.0, 1.0]), np.array([80.0, 81.0]), np.array([0.0, 0.0]), np.array([0.5, 86400.0]))
    assert np.isfinite(v[0]).all() and np.isnan(v[1]).all()


def test_era5_zi_missing_positions_and_hour_rows(monkeypatch):
    monkeypatch.setattr(zi, "_grid", lambda: (GRID_LAT, GRID_LON))
    monkeypatch.setattr(zi, "_read_day", fake_day)
    t = np.array([3600, 5400, 7200])
    vec = np.repeat(zi.unit_vectors(np.array([1.0]), np.array([89.0])), 3, 0)
    vec[2] = np.nan
    hours = np.array([0, 3600])
    hvec = np.vstack([zi.unit_vectors(np.array([44.0]), np.array([1.0])), zi.unit_vectors(np.array([1.0]), np.array([89.0]))])
    out = zi.era5_zi([("a", t, 30, vec), ("a@hour", hours, 0, hvec)], workers=1)
    assert np.isnan(out["a"]["zi"][2]) and np.isfinite(out["a"]["zi"][:2]).all()
    np.testing.assert_allclose(out["a@hour"]["zi"], [2000.0, 1001.0])


def test_track_without_valid_positions_gives_nan():
    v = zi.track_vectors(np.array([0.0, 1.0]), np.array([np.nan, np.nan]), np.array([0.0, 0.0]), np.array([0.5]))
    assert v.shape == (1, 3) and np.isnan(v).all()
