"""window_diff: paired-arm window means from a partial-month CMOR file and the
restart sidecar.

Synthetic inputs with a known window mean must be reproduced exactly; a
sidecar/file orientation mismatch, a wrong cadence and a two-month bucket must
be refused rather than produce a number.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest
import xarray as xr

_P = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate" / "amip_bias" / "window_diff.py"
LAT = np.array([-45.0, 45.0])
LON = np.array([0.0, 120.0, 240.0])


@pytest.fixture
def wd():
    spec = importlib.util.spec_from_file_location("window_diff", _P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sidecar(path, sums, counts, months=((0, 3),), extra=None):
    """``extra`` adds further buckets as {(year, month): {var: (sum, count)}},
    e.g. a CLOSED month sitting alongside the open one."""
    data_2d, arrays = [], {}
    for i, (var, s) in enumerate(sums.items()):
        for y, m in months:
            key = f"arr_{i}_{m}"
            data_2d.append([y, m, var, counts[var], key])
            arrays[f"monthly.{key}"] = s
    for j, ((y, m), per_var) in enumerate(sorted((extra or {}).items())):
        for i, (var, (s, c)) in enumerate(per_var.items()):
            key = f"xtra_{j}_{i}_{m}"
            data_2d.append([y, m, var, c, key])
            arrays[f"monthly.{key}"] = s
    man = {"version": 1, "type": "SpatialMonthlyAccumulator", "nlat": 2,
           "nlon": 3, "nlev": 1, "data_2d": data_2d, "data_3d": []}
    np.savez(path, **{"monthly.__manifest__": np.array(json.dumps(man))}, **arrays)


def _amon(run_dir, var, mean, tag="197903-197903"):
    d = run_dir / "cmor" / "Amon"
    d.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({var: (("time", "lat", "lon"), mean[None])},
                    coords={"time": [74.5], "lat": LAT, "lon": LON})
    ds.to_netcdf(d / f"{var}_Amon_legoESM-1-0_amip_r1i1p1f1_gn_{tag}.nc")


def _pattern(a, b):
    """(2, 3) field with a hemispheric asymmetry so a flipped lat is visible."""
    return np.array([[a, a, a], [b, b, b]], dtype=float)


def test_window_mean_recovers_the_arm_days(wd, tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "FIELDS", ("rsut", "rlut"))
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    run = tmp_path / "arm"; run.mkdir()
    # 20 days of history at (100, 200) per day, then 6 arm days at (130, 260)
    hist = {"rsut": 20 * _pattern(100.0, 200.0), "rlut": 20 * _pattern(10.0, 20.0)}
    _sidecar(run / "cmor_accum_day_0080.npz", hist, {"rsut": 20, "rlut": 20})
    _amon(run, "rsut", (20 * _pattern(100.0, 200.0) + 6 * _pattern(130.0, 260.0)) / 26)
    _amon(run, "rlut", (20 * _pattern(10.0, 20.0) + 6 * _pattern(40.0, 50.0)) / 26)
    m, lat, _lon = wd.window_means("arm", 80, 86)
    np.testing.assert_allclose(m["rsut"], _pattern(130.0, 260.0), rtol=1e-12)
    np.testing.assert_allclose(m["rlut"], _pattern(40.0, 50.0), rtol=1e-12)
    assert m["rsut__n"] == 6
    np.testing.assert_allclose(lat, LAT)
    # cos-weighted band mean of a symmetric pair is the plain mean
    assert wd.band_mean(m["rsut"], lat, _lon, -90, 90) == pytest.approx(195.0)


def test_band_mean_weights_by_area_and_selects_the_longitude_box():
    """Both halves of the reduction get their own failing case.

    The area weighting needs latitudes whose cosines DIFFER (a symmetric pair
    weights equally, so dropping the weights entirely still gives the right
    answer), and the longitude box needs a field that varies with longitude
    (a zonally uniform one is unchanged by selecting every column).
    """
    spec = importlib.util.spec_from_file_location("window_diff", _P)
    wd = importlib.util.module_from_spec(spec); spec.loader.exec_module(wd)
    lat = np.array([0.0, 60.0])                  # cos = 1.0 and 0.5
    lon = np.array([10.0, 100.0, 200.0])
    field = np.array([[0.0, 30.0, 60.0], [0.0, 30.0, 60.0]], dtype=float)

    # zonal mean is 30 at both rows, so area weighting cannot change it...
    assert wd.band_mean(field, lat, lon, -90, 90) == pytest.approx(30.0)
    # ...but a field that varies with LATITUDE exposes the weights: rows 0 and
    # 100 under weights 1.0 and 0.5 average to 33.33, not the unweighted 50.
    tilted = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]], dtype=float)
    assert wd.band_mean(tilted, lat, lon, -90, 90) == pytest.approx(100.0 / 3.0)
    # and the longitude box must actually select: one column, not all three
    assert wd.band_mean(field, lat, lon, -90, 90, 150.0, 250.0) == pytest.approx(60.0)
    # a box that wraps past 360 keeps the columns either side of the meridian
    assert wd.band_mean(field, lat, lon, -90, 90, 350.0, 380.0) == pytest.approx(0.0)


def test_orientation_mismatch_is_refused(wd, tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "FIELDS", ("rsut",))
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    run = tmp_path / "arm"; run.mkdir()
    _sidecar(run / "cmor_accum_day_0080.npz", {"rsut": 20 * _pattern(100.0, 200.0)}, {"rsut": 20})
    _amon(run, "rsut", _pattern(200.0, 100.0))          # lat flipped
    with pytest.raises(SystemExit, match="orientation"):
        wd.window_means("arm", 80, 86)


def test_cadence_is_asserted(wd, tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    run = tmp_path / "parent"; run.mkdir()
    s = {"rsut": np.ones((2, 3))}
    _sidecar(run / "cmor_accum_day_0070.npz", s, {"rsut": 10})
    _sidecar(run / "cmor_accum_day_0080.npz", s, {"rsut": 20})
    wd.check_cadence("parent", (70, 80))                 # 1/day: OK
    _sidecar(run / "cmor_accum_day_0080.npz", s, {"rsut": 30})
    with pytest.raises(SystemExit, match="cadence"):
        wd.check_cadence("parent", (70, 80))


def test_closed_month_is_ignored(wd, tmp_path):
    """A run past a month boundary keeps the finished month in the same bucket;
    only the open (latest) month may enter the window arithmetic."""
    run = tmp_path / "arm"; run.mkdir()
    march = {"rsut": (31 * _pattern(999.0, 999.0), 31)}
    _sidecar(run / "cmor_accum_day_0100.npz", {"rsut": 10 * _pattern(100.0, 200.0)},
             {"rsut": 10}, months=((0, 4),), extra={(0, 3): march})
    sums, month = wd.sidecar_sums(run / "cmor_accum_day_0100.npz")
    assert month == (0, 4)
    assert sums["rsut"][1] == 10
    np.testing.assert_allclose(sums["rsut"][0], 10 * _pattern(100.0, 200.0))


@pytest.mark.parametrize("start,end", [(80, 110), (85, 95)])
def test_window_crossing_a_month_boundary_is_refused(wd, tmp_path, monkeypatch,
                                                     start, end):
    """Both a long window and a SHORT one that merely straddles 31 March must
    be refused; the short one passes any elapsed-sample heuristic."""
    monkeypatch.setattr(wd, "FIELDS", ("rsut",))
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    run = tmp_path / "arm"; run.mkdir()
    _sidecar(run / f"cmor_accum_day_{start:04d}.npz",
             {"rsut": 26 * _pattern(100.0, 200.0)}, {"rsut": 26}, months=((0, 3),))
    _amon(run, "rsut", _pattern(110.0, 210.0), tag="197903-197903")
    with pytest.raises(SystemExit, match="inside one calendar month"):
        wd.window_means("arm", start, end)


def test_window_inside_the_open_month_is_accepted(wd, tmp_path, monkeypatch):
    """The mirror case: days 92..102 lie wholly in April and must NOT be
    refused just because the month has only two earlier samples."""
    monkeypatch.setattr(wd, "FIELDS", ("rsut",))
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    run = tmp_path / "arm"; run.mkdir()
    _sidecar(run / "cmor_accum_day_0092.npz", {"rsut": 2 * _pattern(100.0, 200.0)},
             {"rsut": 2}, months=((0, 4),))
    _amon(run, "rsut", (2 * _pattern(100.0, 200.0) + 10 * _pattern(130.0, 260.0)) / 12,
          tag="197904-197904")
    m, _, _lon = wd.window_means("arm", 92, 102)
    np.testing.assert_allclose(m["rsut"], _pattern(130.0, 260.0), rtol=1e-12)


def test_published_variable_missing_from_the_open_month_is_refused(wd, tmp_path,
                                                                  monkeypatch):
    """A day-90 start finds an April bucket that has cloud but no shortwave
    yet, for a run that DOES write shortwave; dropping the field silently would
    hide a scored column from the table."""
    monkeypatch.setattr(wd, "FIELDS", ("rsut", "clt"))
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    run = tmp_path / "arm"; run.mkdir()
    _sidecar(run / "cmor_accum_day_0092.npz", {"clt": 2 * _pattern(50.0, 60.0)},
             {"clt": 2}, months=((0, 4),))
    _amon(run, "clt", _pattern(50.0, 60.0), tag="197904-197904")
    _amon(run, "rsut", _pattern(100.0, 200.0), tag="197904-197904")
    with pytest.raises(SystemExit, match="no samples yet"):
        wd.window_means("arm", 92, 102)


def test_unpublished_variable_is_skipped_not_refused(wd, tmp_path, monkeypatch):
    """The clear-sky fluxes are OPTIONAL output -- the driver's 2-D writer
    skips a field whose source is None -- so a run that never writes one must
    still score on the fields it does write.  Refusing here once cost this tool
    the ability to analyse exactly those runs."""
    monkeypatch.setattr(wd, "FIELDS", ("clt", "rsutcs"))
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    run = tmp_path / "arm"; run.mkdir()
    _sidecar(run / "cmor_accum_day_0092.npz", {"clt": 2 * _pattern(50.0, 60.0)},
             {"clt": 2}, months=((0, 4),))
    _amon(run, "clt", (2 * _pattern(50.0, 60.0) + 10 * _pattern(80.0, 90.0)) / 12,
          tag="197904-197904")
    m, _lat, _lon = wd.window_means("arm", 92, 102)
    np.testing.assert_allclose(m["clt"], _pattern(80.0, 90.0), rtol=1e-12)
    assert "rsutcs" not in m


def test_ambiguous_year_is_refused(wd, tmp_path, monkeypatch):
    """Two files ending in the same month means the relative sidecar year
    cannot pick one; refuse rather than take the newest."""
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    run = tmp_path / "arm"; run.mkdir()
    _amon(run, "rsut", _pattern(1.0, 2.0), tag="197903-197904")
    _amon(run, "rsut", _pattern(3.0, 4.0), tag="198003-198004")
    with pytest.raises(SystemExit, match="ambiguous"):
        wd.partial_month("arm", "rsut", 0, 4)


def test_bucket_key_matches_the_accumulator(wd):
    """The window's month must be binned by the driver's own helpers."""
    from legoesm.diagnostics.monthly_means import MonthlyAccumulator
    from legoesm.forcing.time_utils import day_to_calendar
    for day in (0, 30, 59, 60, 89, 90, 100, 110, 364, 365, 400):
        doy, _ = day_to_calendar(float(day))
        assert wd.bucket_key(day) == (day // 365, MonthlyAccumulator.day_to_month(doy))


def test_p_minus_e_is_derived_in_mm_per_day_and_the_cap_bands_exist(wd):
    assert wd.DERIVED["P-E"] == ("pr", "evspsbl") and wd.UNITS["P-E"] == "mm/d"
    assert "evspsbl" in wd.FIELDS and wd.SCALE["evspsbl"] == wd.SCALE["pr"] == 86400.0
    assert wd.BANDS["Arctic 72.5-90N"] == (72.5, 90.0, 0.0, 360.0)
    assert wd.BANDS["Arctic 75-90N"] == (75.0, 90.0, 0.0, 360.0)
