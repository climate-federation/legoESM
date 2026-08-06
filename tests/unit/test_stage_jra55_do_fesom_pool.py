"""Direct tests for ``scripts/data/stage_jra55_do_fesom_pool.py``.

The staging script decides what the model is actually forced with, so the
pieces that could silently corrupt a field get their own checks:

* CF-bounds -> edges, including the contiguity guard and the descending-axis
  flip (a mirrored latitude axis would put Arctic runoff in the Southern
  Ocean and still look plausible);
* the conservative river-grid -> atmospheric-grid regrid, checked on
  conservation of a localised source AND on preservation of a constant, which
  fail in different ways (a missing seam column breaks the second, a wrong
  area weight breaks the first);
* the zero-order hold that puts daily runoff on the 3-hourly axis, whose
  ``searchsorted`` boundary is the easy off-by-one.

The synthetic grids reproduce the real mismatch this script exists for: a
source whose first cell edge is at 0 deg and a destination whose first cell
edge is negative, so the seam padding is exercised rather than bypassed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest

# The conservative overlap weights are area ratios summed to 1; in float32
# their roundoff lands at 1e-8 relative, which is exactly the size of the
# conservation error these tests are meant to detect.  x64 is required, and
# the script itself raises rather than downgrade silently.
jax.config.update("jax_enable_x64", True)

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "data" / "stage_jra55_do_fesom_pool.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("stage_jra55", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage = _load_module()

_MIDNIGHT = "days since 1900-01-01 00:00:00"


# ---------------------------------------------------------------------------
# CF bounds -> edges
# ---------------------------------------------------------------------------

def test_edges_from_cf_bounds_contiguous_ascending():
    bnds = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
    np.testing.assert_allclose(
        stage._edges_from_cf_bounds(bnds, "x"), [0.0, 1.0, 2.0, 4.0])


def test_edges_from_cf_bounds_flips_descending_axis():
    """A descending axis must come back ASCENDING, not reversed-in-place."""
    bnds = np.array([[4.0, 2.0], [2.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(
        stage._edges_from_cf_bounds(bnds, "x"), [0.0, 1.0, 2.0, 4.0])


def test_edges_from_cf_bounds_rejects_gaps():
    bnds = np.array([[0.0, 1.0], [1.5, 2.0]])
    with pytest.raises(ValueError, match="not contiguous"):
        stage._edges_from_cf_bounds(bnds, "x")


def test_edges_from_cf_bounds_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        stage._edges_from_cf_bounds(np.zeros((3,)), "x")


def test_axis_edges_prefers_bounds_over_uniform_inference():
    """On a NON-uniform axis the two paths differ; the bounds must win."""
    xr = pytest.importorskip("xarray")
    centres = np.array([-60.0, -10.0, 20.0, 70.0])          # non-uniform
    bnds = np.array([[-90.0, -35.0], [-35.0, 5.0],
                     [5.0, 45.0], [45.0, 90.0]])
    ds = xr.Dataset(
        {"lat_bnds": (("lat", "bnds"), bnds)},
        coords={"lat": ("lat", centres)},
    )
    got = stage._axis_edges(ds, "lat", centres, "test")
    np.testing.assert_allclose(got, [-90.0, -35.0, 5.0, 45.0, 90.0])

    # Without bounds the fallback infers UNIFORM spacing, which on this axis
    # is a different (and wrong) answer -- so the preference above is real.
    ds_nb = xr.Dataset(coords={"lat": ("lat", centres)})
    fallback = stage._axis_edges(ds_nb, "lat", centres, "test")
    assert not np.allclose(fallback, got)


# ---------------------------------------------------------------------------
# conservative runoff regrid
# ---------------------------------------------------------------------------

def _tiny_grids():
    """A 0.25 deg-like source and a coarser, origin-shifted destination."""
    n_src_lat, n_src_lon = 8, 16
    src_lat_edges = np.linspace(-90.0, 90.0, n_src_lat + 1)
    src_lon_edges = np.linspace(0.0, 360.0, n_src_lon + 1)
    n_dst_lat, n_dst_lon = 4, 8
    dst_lat_edges = np.linspace(-90.0, 90.0, n_dst_lat + 1)
    dst_dlon = 360.0 / n_dst_lon
    # Shift the destination half a cell west, reproducing the real seam
    # mismatch (destination starts at a NEGATIVE longitude edge).
    dst_lon_edges = np.linspace(0.0, 360.0, n_dst_lon + 1) - 0.5 * dst_dlon
    lon_dst_centres = 0.5 * (dst_lon_edges[:-1] + dst_lon_edges[1:])
    return (src_lat_edges, src_lon_edges, dst_lat_edges, dst_lon_edges,
            lon_dst_centres, n_src_lat, n_src_lon)


def _src_dataarray(values, src_lat_edges, src_lon_edges):
    xr = pytest.importorskip("xarray")
    lat = 0.5 * (src_lat_edges[:-1] + src_lat_edges[1:])
    lon = 0.5 * (src_lon_edges[:-1] + src_lon_edges[1:])
    return xr.DataArray(values, dims=("time", "lat", "lon"),
                        coords={"lat": lat, "lon": lon})


def _cell_area(lat_edges, lon_edges):
    from legoesm import constants
    dsin = np.sin(np.deg2rad(lat_edges[1:])) - np.sin(np.deg2rad(lat_edges[:-1]))
    dlon = np.deg2rad(np.diff(lon_edges))
    return dsin[:, None] * dlon[None, :] * constants.R_earth ** 2


def test_runoff_regrid_conserves_a_localised_source():
    """A single hot cell must keep its global integral across the regrid."""
    (sle, slo, dle, dlo, lon_c, nlat, nlon) = _tiny_grids()
    vals = np.zeros((1, nlat, nlon))
    vals[0, 5, 3] = 7.0          # off-centre, away from every seam
    out = stage._regrid_runoff_to_atmos_grid(
        _src_dataarray(vals, sle, slo), sle, slo, dle, dlo, lon_c)
    a = float((vals[0] * _cell_area(sle, slo)).sum())
    b = float((out[0] * _cell_area(dle, dlo)).sum())
    assert abs(b - a) / a < 1e-12, f"{a} -> {b}"


def test_runoff_regrid_preserves_a_constant_across_the_seam():
    """Constant in, constant out -- catches a half-weighted seam column,
    which conservation of a localised source would not see."""
    (sle, slo, dle, dlo, lon_c, nlat, nlon) = _tiny_grids()
    vals = np.full((1, nlat, nlon), 3.0)
    out = stage._regrid_runoff_to_atmos_grid(
        _src_dataarray(vals, sle, slo), sle, slo, dle, dlo, lon_c)
    np.testing.assert_allclose(out, 3.0, rtol=1e-12, atol=0.0)


def test_runoff_regrid_flips_a_descending_source_with_its_data():
    """The regrid must give the SAME answer whether the source file stores
    latitude ascending or descending."""
    (sle, slo, dle, dlo, lon_c, nlat, nlon) = _tiny_grids()
    vals = np.zeros((1, nlat, nlon))
    vals[0, 6, 2] = 5.0
    asc = stage._regrid_runoff_to_atmos_grid(
        _src_dataarray(vals, sle, slo), sle, slo, dle, dlo, lon_c)

    xr = pytest.importorskip("xarray")
    lat = 0.5 * (sle[:-1] + sle[1:])[::-1]
    lon = 0.5 * (slo[:-1] + slo[1:])
    da_desc = xr.DataArray(vals[:, ::-1, :], dims=("time", "lat", "lon"),
                           coords={"lat": lat, "lon": lon})
    desc = stage._regrid_runoff_to_atmos_grid(da_desc, sle, slo, dle, dlo,
                                              lon_c)
    np.testing.assert_allclose(desc, asc, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# daily -> 3-hourly zero-order hold
# ---------------------------------------------------------------------------

def test_daily_hold_covers_whole_calendar_days():
    """Each daily mean must apply through ITS OWN day, 00:00-24:00.

    Calls the PRODUCTION helper, not a reimplementation of it.  The stamps are
    interval MIDPOINTS (12:00); binning against them directly runs the hold
    noon-to-noon, giving the first day 12 three-hourly slots and the last 4.
    """
    run_time = np.array([0.5, 1.5, 2.5])            # daily, stamped at 12:00
    ref_time = np.arange(0.0, 3.0, 0.125)           # 3-hourly, 24 slots
    idx = stage._daily_hold_index(run_time, ref_time, label="t",
                            units=_MIDNIGHT, ref_units=_MIDNIGHT)

    counts = np.bincount(idx, minlength=3)
    assert list(counts) == [8, 8, 8], counts
    assert list(idx[:8]) == [0] * 8                 # all of day 0
    assert idx[8] == 1                              # 00:00 on day 1 -> day 1
    assert idx[16] == 2
    assert np.all(np.diff(idx) >= 0)                # monotone, no rewind

    # The WRONG binning is genuinely different, so this test discriminates.
    bad = np.clip(np.searchsorted(run_time, ref_time, side="right") - 1,
                  0, run_time.size - 1)
    assert list(np.bincount(bad, minlength=3)) != [8, 8, 8]


def test_daily_hold_rejects_an_irregular_axis():
    """A missing day must FAIL in production code, not be absorbed.

    Without the guard the old ``np.clip`` held the previous day across the
    gap -- a plausible field, silently wrong.
    """
    run_time = np.array([0.5, 1.5, 3.5, 4.5])       # 3 Jan missing
    ref_time = np.arange(0.0, 5.0, 0.125)
    with pytest.raises(ValueError, match="uniform spacing"):
        stage._daily_hold_index(run_time, ref_time, label="friver.1958",
                            units=_MIDNIGHT, ref_units=_MIDNIGHT)


def test_daily_hold_rejects_a_non_increasing_axis():
    with pytest.raises(ValueError, match="strictly increasing"):
        stage._daily_hold_index(np.array([2.5, 1.5, 0.5]),
                                np.arange(0.0, 3.0, 0.125), label="t",
                                units=_MIDNIGHT, ref_units=_MIDNIGHT)


def test_daily_hold_rejects_a_period_mismatch():
    """A 3-hourly axis reaching outside the daily records' coverage must
    raise with BOTH spans, not silently clamp to the edge day."""
    run_time = np.array([0.5, 1.5])                 # covers days 0-1 only
    ref_time = np.arange(0.0, 4.0, 0.125)           # asks for days 0-3
    with pytest.raises(ValueError, match="do not describe the same period"):
        stage._daily_hold_index(run_time, ref_time, label="friver.1958",
                            units=_MIDNIGHT, ref_units=_MIDNIGHT)


def test_daily_hold_needs_two_records_to_infer_the_interval():
    with pytest.raises(ValueError, match="need >= 2 records"):
        stage._daily_hold_index(np.array([0.5]), np.array([0.0]), label="t",
                                units=_MIDNIGHT, ref_units=_MIDNIGHT)


def test_flux_midpoint_offset_is_half_the_sampling_interval():
    """The rebasing constant must stay tied to the 3-hourly cadence."""
    assert stage._FLUX_MIDPOINT_OFFSET_DAYS * 24.0 == pytest.approx(1.5)
    assert set(stage.STATE_VARS).isdisjoint(stage.FLUX_VARS)
    assert stage.RUNOFF_VAR not in stage.STATE_VARS + stage.FLUX_VARS


def test_daily_hold_rejects_a_target_axis_starting_mid_day():
    """codex round 9 counterexample. A target axis starting at 03:00 gives
    7/8/8-slot days and passes EVERY other guard -- uniform daily spacing,
    uniform target spacing, in-range indices, coverage. Only the per-day
    count sees it. This is the case for which the count check was briefly
    deleted, on the wrong argument that it could never fire."""
    run_time = np.array([0.5, 1.5, 2.5])
    ref_time = np.arange(0.125, 3.0, 0.125)         # starts at 03:00
    with pytest.raises(ValueError, match="do not tile"):
        stage._daily_hold_index(run_time, ref_time, label="friver.1958",
                            units=_MIDNIGHT, ref_units=_MIDNIGHT)


def test_daily_hold_rejects_records_that_are_not_one_day_apart():
    """codex round 9 counterexample. 1.125 d spacing is perfectly uniform and
    would hand out 9/9/6-slot 'days'."""
    run_time = np.array([0.5625, 1.6875, 2.8125])   # 1.125 d apart
    ref_time = np.arange(0.0, 3.0, 0.125)
    with pytest.raises(ValueError, match="not\\n?\\s*1 d|not 1 d"):
        stage._daily_hold_index(run_time, ref_time, label="friver.1958",
                            units=_MIDNIGHT, ref_units=_MIDNIGHT)


def test_daily_hold_rejects_a_stamp_phase_offset():
    """Stamps at 10:30 tile evenly and pass every span check, but displace
    the hold by 1.5 h. The pool contract is noon stamps, so tolerating this
    would only hide a malformed source."""
    run_time = np.array([0.4375, 1.4375, 2.4375])   # 10:30
    ref_time = np.arange(0.0, 3.0, 0.125)
    with pytest.raises(ValueError, match="BETWEEN target samples"):
        stage._daily_hold_index(run_time, ref_time, label="friver.1958",
                            units=_MIDNIGHT, ref_units=_MIDNIGHT)


def test_daily_hold_rejects_a_non_uniform_target_axis():
    run_time = np.array([0.5, 1.5, 2.5])
    ref_time = np.array([0.0, 0.125, 0.5, 1.0])
    with pytest.raises(ValueError, match="target axis is not uniformly"):
        stage._daily_hold_index(run_time, ref_time, label="t",
                            units=_MIDNIGHT, ref_units=_MIDNIGHT)


def test_daily_hold_rejects_a_sample_at_the_exclusive_upper_edge():
    """Coverage is half-open: a sample exactly at the end of the last day
    belongs to a day this file does not have."""
    run_time = np.array([0.5, 1.5])                 # covers [0, 2)
    ref_time = np.array([0.0, 1.0, 2.0])            # 2.0 is out
    with pytest.raises(ValueError, match="do not describe the same period"):
        stage._daily_hold_index(run_time, ref_time, label="friver.1958",
                            units=_MIDNIGHT, ref_units=_MIDNIGHT)


# --- codex round 10: the phase check must be ABSOLUTE, not just relative ---


def test_daily_hold_rejects_a_shared_calendar_phase_offset():
    """codex round 10 counterexample. Shift BOTH axes by the same 3 h and the
    relative phase check still passes -- every "day" then runs 03:00-03:00.
    Only an ABSOLUTE check against a midnight epoch sees it."""
    run_time = np.array([0.625, 1.625, 2.625])       # 15:00 stamps
    ref_time = np.arange(0.125, 3.125, 0.125)        # 03:00 phase
    # Every arithmetic guard passes -- the relative phase is consistent and
    # each day still gets 8 slots.  Only the ABSOLUTE midnight check sees it.
    with pytest.raises(ValueError, match="not a calendar midnight"):
        stage._daily_hold_index(run_time, ref_time, label="friver.1958",
                                units=_MIDNIGHT, ref_units=_MIDNIGHT)


def test_daily_hold_accepts_the_real_convention_with_units():
    """Noon stamps on a midnight-epoch axis: the pool contract."""
    run_time = np.array([0.5, 1.5, 2.5])
    ref_time = np.arange(0.0, 3.0, 0.125)
    idx = stage._daily_hold_index(run_time, ref_time, label="t",
                                  units=_MIDNIGHT, ref_units=_MIDNIGHT)
    assert list(np.bincount(idx, minlength=3)) == [8, 8, 8]


def test_daily_hold_rejects_a_non_midnight_epoch():
    """A day boundary cannot be identified against a non-midnight epoch."""
    with pytest.raises(ValueError, match="midnight epoch"):
        stage._daily_hold_index(np.array([0.5, 1.5, 2.5]),
                                np.arange(0.0, 3.0, 0.125), label="t",
                                units="days since 1900-01-01 06:00:00",
                                ref_units="days since 1900-01-01 06:00:00")


@pytest.mark.parametrize("units,expected", [
    ("days since 1900-01-01 00:00:00", True),
    ("days since 1900-01-01", True),
    ("days since 1900-01-01 00:00", True),
    ("days since 1900-01-01 06:00:00", False),
    ("days since 1900-01-01 00:30:00", False),
    (None, False),
])
def test_epoch_is_midnight(units, expected):
    assert stage._epoch_is_midnight(units) is expected


# --- codex round 11: CF time zones and mismatched units --------------------

@pytest.mark.parametrize("units,expected", [
    ("days since 1900-01-01 00:00:00", True),
    ("days since 1900-01-01", True),
    ("days since 1900-01-01 00:00", True),
    ("days since 1900-01-01 00:00:00Z", True),      # attached UTC marker
    ("days since 1900-01-01 00:00:00 Z", True),
    ("days since 1900-01-01 00:00:00+00:00", True),
    ("days since 1900-01-01 00:00:00 +00:00", True),
    ("days since 1900-01-01 00:00:00+03:00", False),   # 21:00 UTC, NOT midnight
    ("days since 1900-01-01 00:00:00 +03:00", False),
    ("days since 1900-01-01 00:00:00-05:30", False),
    ("days since 1900-01-01 06:00:00", False),
    ("days since 1900-01-01 00:30:00", False),
    ("", False),
    (None, False),
])
def test_epoch_is_midnight_handles_cf_time_zones(units, expected):
    assert stage._epoch_is_midnight(units) is expected


def test_daily_hold_rejects_mismatched_time_units():
    """A runoff axis on +03:00 against a UTC target passes every arithmetic
    check while applying local-midnight days to a UTC axis."""
    run_time = np.array([0.5, 1.5, 2.5])
    ref_time = np.arange(0.0, 3.0, 0.125)
    with pytest.raises(ValueError, match="DIFFERENT time units"):
        stage._daily_hold_index(
            run_time, ref_time, label="friver.1958",
            units="days since 1900-01-01 00:00:00 +03:00",
            ref_units="days since 1900-01-01 00:00:00Z")


@pytest.mark.parametrize("units,ref_units", [
    (None, _MIDNIGHT), (_MIDNIGHT, None), (None, None), ("", _MIDNIGHT),
])
def test_daily_hold_requires_both_units(units, ref_units):
    """Missing CF units must FAIL, not fall open past the absolute check."""
    with pytest.raises(ValueError, match="need CF `units`"):
        stage._daily_hold_index(np.array([0.5, 1.5, 2.5]),
                                np.arange(0.0, 3.0, 0.125), label="t",
                                units=units, ref_units=ref_units)
