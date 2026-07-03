"""Unit tests for the gap-free EC-site driver producer
(scripts/data/build_ec_gapfree_driver.py): the FLUXNET sentinel cleanup and the
short(linear)/long(diurnal-climatology) gap-fill logic.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[3]
_PY = _REPO / "scripts" / "data" / "build_ec_gapfree_driver.py"


def _mod():
    spec = importlib.util.spec_from_file_location("build_ec_gapfree_driver", _PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _synthetic_hourly(days: int = 40):
    """Hourly series with a clean diurnal sine over `days` days (Jan)."""
    n = days * 24
    hours = np.tile(np.arange(24), days)
    months = np.ones(n, dtype=int)            # all January
    diurnal = 10.0 + 8.0 * np.sin((hours - 6) / 24.0 * 2 * np.pi)
    return n, hours, months, diurnal.astype(float)


def test_clean_fluxnet_sentinel():
    m = _mod()
    out = m._clean_fluxnet(np.array([1.0, -9999.0, 3.0, -9999.0]))
    assert np.isnan(out[1]) and np.isnan(out[3])
    assert out[0] == 1.0 and out[2] == 3.0


def test_is_observed_rejects_finite_out_of_range():
    """The producer's 'observed' definition (reader.is_observed) must reject a
    finite but out-of-physical-range value, not just NaN (Codex iter-4 #2)."""
    from legoesm.land.boundary_data.ec_site import is_observed
    # SWC physical range is (0, 75) %
    vals = np.array([28.0, np.nan, -9999.0, 999.0, 0.0, 75.0])
    obs = is_observed("SWC", vals)
    assert list(obs) == [True, False, False, False, True, True]


def test_long_gap_filled_by_climatology_preserves_diurnal():
    m = _mod()
    n, hours, months, vals = _synthetic_hourly(days=40)
    v = vals.copy()
    # carve a 5-day (120 h) leading gap -> boundary+long, must use climatology
    v[:120] = np.nan
    filled, n_long, n_short = m.fill_gaps(v, months, hours, 24, 3600.0, long_gap_hours=24)

    assert np.isfinite(filled).all()           # gap-free
    assert n_long == 120 and n_short == 0
    # climatology reproduces the clean diurnal cycle within the gap
    assert np.allclose(filled[:120], vals[:120], atol=1e-9)
    # observed steps untouched
    assert np.array_equal(filled[120:], vals[120:])


def test_interior_short_gap_filled_linearly():
    m = _mod()
    n, hours, months, vals = _synthetic_hourly(days=10)
    v = vals.copy()
    v[50:52] = np.nan                          # 2 h interior gap -> linear
    filled, n_long, n_short = m.fill_gaps(v, months, hours, 24, 3600.0, long_gap_hours=24)

    assert np.isfinite(filled).all()
    assert n_long == 0 and n_short == 2
    lo, hi = sorted((vals[49], vals[52]))
    assert lo - 1e-9 <= filled[50] <= hi + 1e-9
    assert lo - 1e-9 <= filled[51] <= hi + 1e-9


def test_leading_short_gap_uses_climatology_not_backfill():
    """A short gap at the record START has no left anchor, so it must be filled
    by climatology (diurnal-preserving), NOT future back-filled (Codex #5)."""
    m = _mod()
    n, hours, months, vals = _synthetic_hourly(days=10)
    v = vals.copy()
    v[:2] = np.nan                             # 2 h LEADING gap (touches boundary)
    filled, n_long, n_short = m.fill_gaps(v, months, hours, 24, 3600.0, long_gap_hours=24)

    assert np.isfinite(filled).all()
    assert n_long == 2 and n_short == 0        # boundary -> climatology, not linear
    # climatology recovers the diurnal value, NOT a flat backfill of vals[2]
    assert np.allclose(filled[:2], vals[:2], atol=1e-9)
    assert not np.allclose(filled[0], vals[2])


def test_gap_classification_is_timestep_aware():
    """A 30-sample interior gap is 'long' at hourly dt (30 h >= 24 h) but 'short'
    at half-hourly dt (15 h < 24 h) — the threshold is in hours, not samples
    (Codex #4)."""
    m = _mod()
    n, hours, months, vals = _synthetic_hourly(days=10)
    v = vals.copy()
    v[100:130] = np.nan                        # 30-sample interior gap
    _, n_long_hourly, n_short_hourly = m.fill_gaps(
        v.copy(), months, hours, 24, 3600.0, long_gap_hours=24)
    _, n_long_halfhr, n_short_halfhr = m.fill_gaps(
        v.copy(), months, hours, 24, 1800.0, long_gap_hours=24)

    assert n_long_hourly == 30 and n_short_hourly == 0   # 30 h -> long
    assert n_long_halfhr == 0 and n_short_halfhr == 30   # 15 h -> short


def test_empty_climatology_cell_falls_back_to_global_not_backfill():
    """If a boundary gap's (month,hour) AND hour cells have no observations, the
    fill must fall back to the global observed mean — never a positional
    future-backfill (Codex iter-2 #3)."""
    m = _mod()
    n, hours, months, vals = _synthetic_hourly(days=3)
    v = vals.copy()
    v[hours == 0] = np.nan                       # hour 0 observed NOWHERE
    filled, _, _ = m.fill_gaps(v, months, hours, 24, 3600.0, long_gap_hours=24)

    assert np.isfinite(filled).all()
    global_mean = float(np.nanmean(v))           # mean of the observed (non-hour-0)
    # index 0 is a leading (boundary) gap at the empty hour-0 cell -> global mean
    assert np.isclose(filled[0], global_mean, atol=1e-9)
    # and NOT the next observed value (which a bfill would have produced)
    assert not np.isclose(filled[0], v[1])


def test_halfhourly_climatology_keeps_slots_distinct():
    """Half-hourly climatology must bin by slot-of-day (48 slots), keeping :00
    and :30 as DISTINCT bins instead of averaging them into one hourly value
    (Codex iter-3 #2)."""
    m = _mod()
    days = 20
    n = days * 48
    slots = np.tile(np.arange(48), days)          # 0..47 (half-hourly)
    months = np.ones(n, dtype=int)
    # neighbouring slots differ sharply: :00 (even) vs :30 (odd) offset by 100
    vals = (slots % 2) * 100.0 + slots.astype(float)
    v = vals.copy()
    v[:96] = np.nan                               # 2-day leading gap -> climatology
    filled, n_long, n_short = m.fill_gaps(v, months, slots, 48, 1800.0, long_gap_hours=24)

    assert np.isfinite(filled).all()
    assert n_long == 96
    # climatology recovers the distinct per-slot values, not a flattened hourly avg
    assert np.allclose(filled[:96], vals[:96], atol=1e-9)
    assert abs(filled[0] - filled[1]) > 50.0       # :00 and :30 stay distinct


def test_no_gap_is_noop():
    m = _mod()
    n, hours, months, vals = _synthetic_hourly(days=5)
    filled, n_long, n_short = m.fill_gaps(vals.copy(), months, hours, 24, 3600.0)
    assert n_long == 0 and n_short == 0
    assert np.array_equal(filled, vals)


def test_soil_field_uses_shallowest_sensor_when_driver_is_depth_averaged(tmp_path):
    """SOIL fields (TS/SWC) must take the SHALLOWEST FULLSET sensor (_F_MDS_1) as
    the model's top-layer field, even when the driver's own field is a DIFFERENT
    definition (preprocess_v2 sets TS = mean of 3 depths).  A maxdiff there is
    EXPECTED and must NOT raise; MET forcing keeps the strict units check.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    m = _mod()
    sub = m._SUBSTITUTIONS                     # {driver_var: fullset_col}
    n = 150
    starts = pd.date_range("2020-01-01 00:00", periods=n, freq="1h")
    ends = starts + pd.Timedelta("1h")
    centre = starts + pd.Timedelta("30min")    # driver time coordinate

    met = {"TA": 15.0, "VPD": 5.0, "SW_IN": 200.0, "LW_IN": 300.0, "PA": 100.0}
    driver_ts, shallow_ts, swc = 15.0, 18.0, 30.0   # driver TS depth-avg vs shallowest

    def _val(dv, shallow):
        if dv == "TS":
            return shallow_ts if shallow else driver_ts
        if dv == "SWC":
            return swc
        return met[dv]

    # driver NC: TS is the depth-average (15), differing from the shallowest (18)
    ds = xr.Dataset(
        {dv: ("time", np.full(n, _val(dv, shallow=False))) for dv in sub},
        coords={"time": centre.values})
    drv = tmp_path / "SITE_driver_v2.nc"
    ds.to_netcdf(drv)

    # FULLSET: TS_F_MDS_1 is the shallowest sensor (18); MET matches the driver
    csv_cols = {"TIMESTAMP_START": [int(x) for x in starts.strftime("%Y%m%d%H%M")],
                "TIMESTAMP_END": [int(x) for x in ends.strftime("%Y%m%d%H%M")]}
    for dv, fv in sub.items():
        csv_cols[fv] = np.full(n, _val(dv, shallow=True))
    csv = tmp_path / "FULLSET.csv"
    pd.DataFrame(csv_cols).to_csv(csv, index=False)

    res = m.build(str(drv), str(csv), str(tmp_path / "out.nc"))  # must NOT raise

    # TS output == the SHALLOWEST sensor (18), NOT the driver's depth-average (15)
    np.testing.assert_allclose(np.asarray(res["TS"].values), shallow_ts)
    # MET forcing passed the strict check and is unchanged
    np.testing.assert_allclose(np.asarray(res["TA"].values), met["TA"])
