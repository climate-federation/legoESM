"""Tests for ``scripts/make_ryf.py``.

The Stewart 2020 splice rule is non-trivial enough (leap-year logic,
calendar slice arithmetic, multi-cadence harmonisation) that we
exercise it on synthetic 2-year IAF directories where we control the
input values cell-by-cell and can verify the spliced output bit-by-bit.

The synthetic data is small (4 lat × 8 lon, daily cadence at the
3hrPt and 3hr tables, which is unphysically coarse but exercises the
exact code paths the real 23 GB inputs hit).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")
pd = pytest.importorskip("pandas")

# Load the script as a module by file path.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_ryf.py"
_spec = importlib.util.spec_from_file_location("make_ryf", _SCRIPT)
make_ryf_mod = importlib.util.module_from_spec(_spec)
sys.modules["make_ryf"] = make_ryf_mod
_spec.loader.exec_module(make_ryf_mod)


# ============================================================================
# Splice plan rules
# ============================================================================

def test_plan_for_two_non_leap_years_takes_year2_as_template():
    """1990-1991: 1991 non-leap → template = 1991, overwrite May-Dec."""
    plan = make_ryf_mod._build_splice_plan(1990, 1991)
    assert plan.template_year == 1991
    assert plan.overwrite_source_year == 1990
    assert plan.overwrite_start == (5, 1)
    assert plan.overwrite_end == (12, 31)


def test_plan_when_year2_is_leap_takes_year1_as_template():
    """1991-1992: 1992 leap → template = 1991, overwrite Jan-Apr."""
    plan = make_ryf_mod._build_splice_plan(1991, 1992)
    assert plan.template_year == 1991
    assert plan.overwrite_source_year == 1992
    assert plan.overwrite_start == (1, 1)
    assert plan.overwrite_end == (4, 30)


def test_plan_rejects_non_consecutive_years():
    with pytest.raises(ValueError, match="consecutive"):
        make_ryf_mod._build_splice_plan(1990, 1992)


# ============================================================================
# _slice_by_calendar
# ============================================================================

def test_slice_by_calendar_inclusive_range():
    n = 365
    times = pd.date_range("1991-01-01", periods=n, freq="1D")
    da = xr.DataArray(np.arange(n, dtype=np.float64),
                      dims=("time",), coords={"time": times})
    sub = make_ryf_mod._slice_by_calendar(da, 1991, (5, 1), (12, 31))
    # May 1 (day-of-year 121, 0-indexed = 120) through Dec 31 (day 365, idx 364)
    assert sub.sizes["time"] == 365 - 120
    assert int(sub.values[0]) == 120  # May 1
    assert int(sub.values[-1]) == 364  # Dec 31


def test_slice_by_calendar_handles_3hourly_cadence():
    n = 8 * 365
    times = pd.date_range("1991-01-01 00:00", periods=n, freq="3h")
    da = xr.DataArray(np.arange(n, dtype=np.float64),
                      dims=("time",), coords={"time": times})
    sub = make_ryf_mod._slice_by_calendar(da, 1991, (5, 1), (12, 31))
    # May 1 → Dec 31 = 245 days × 8 records/day
    assert sub.sizes["time"] == 245 * 8


# ============================================================================
# Synthetic IAF dir + end-to-end splice
# ============================================================================

def _write_iaf_var(iaf_dir: Path, var, year: int, base_value: float):
    """Write a synthetic IAF NetCDF with constant value = base_value
    in the layout download_jra55_iaf produces."""
    # Native cadence per ts_scheme:
    if var.ts_scheme == "instant":
        n_per_day = 8
        first_offset = pd.Timedelta(0)
    elif var.ts_scheme == "mean3h":
        n_per_day = 8
        first_offset = pd.Timedelta(minutes=90)  # HH:30
    else:  # daily
        n_per_day = 1
        first_offset = pd.Timedelta(hours=12)

    n_days = 366 if (var.ts_scheme != "daily" and pd.Timestamp(year, 12, 31).is_leap_year) else 365
    # For test simplicity, always use 365 days (we test non-leap).
    n_days = 365

    n = n_days * n_per_day
    if n_per_day == 8:
        times = pd.date_range(
            f"{year}-01-01 {int(first_offset.total_seconds()/60//60):02d}"
            f":{int(first_offset.total_seconds()/60%60):02d}",
            periods=n, freq="3h",
        )
    else:
        times = pd.date_range(f"{year}-01-01 12:00", periods=n, freq="1D")

    n_lat, n_lon = 4, 8
    # Distinguishable per-record values: base_value + record_index
    data = (base_value + np.arange(n, dtype=np.float64)[:, None, None]
            * np.ones((1, n_lat, n_lon)))

    ds = xr.Dataset(
        {var.name: (("time", "lat", "lon"), data)},
        coords={
            "time": times,
            "lat": np.linspace(-89, 89, n_lat),
            "lon": np.linspace(0, 360, n_lon, endpoint=False),
        },
    )
    path = make_ryf_mod._local_path(iaf_dir, var, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(str(path))


@pytest.fixture
def synthetic_iaf(tmp_path):
    """A tiny but complete 2-year IAF directory."""
    iaf = tmp_path / "iaf"
    for v in make_ryf_mod.VARS:
        _write_iaf_var(iaf, v, 1990, base_value=10000.0)
        _write_iaf_var(iaf, v, 1991, base_value=20000.0)
    return iaf


def test_make_ryf_end_to_end(synthetic_iaf, tmp_path):
    out = tmp_path / "RYF9091.zarr"
    make_ryf_mod.make_ryf(
        synthetic_iaf, 1990, 1991, out, progress=False,
    )
    assert out.exists()

    ds = xr.open_zarr(str(out), decode_times=False)
    # All 10 variables present
    for v in make_ryf_mod.VARS:
        assert v.name in ds.data_vars

    # Common time axis is 365-day × 8 = 2920 records (3-hourly).
    assert ds.sizes["time"] == 365 * 8

    # When read with decode_times=True, the axis decodes to 1900-based
    # noleap dates.  Re-open to verify.
    ds2 = xr.open_zarr(str(out), decode_times=True)
    t0 = ds2["time"].values[0]
    # Different cftime / numpy combinations stringify slightly
    # differently; just check the year/month/day fields.
    if hasattr(t0, "year"):
        assert (t0.year, t0.month, t0.day) == (1900, 1, 1)
    else:
        assert "1900-01-01" in str(t0)


def test_make_ryf_splice_boundary_uses_year1_for_may_dec(synthetic_iaf, tmp_path):
    """For RYF9091 (year2=1991 non-leap), May-Dec values must come
    from year1=1990 (base 10000) not from year2=1991 (base 20000)."""
    out = tmp_path / "RYF9091.zarr"
    make_ryf_mod.make_ryf(synthetic_iaf, 1990, 1991, out, progress=False)
    ds = xr.open_zarr(str(out), decode_times=False)

    uas = ds["uas"].values   # 3hrPt — exercises the no-shift path
    # May 1 1991 in the template = day-of-year 121, 0-indexed slot
    # idx 120*8 = 960. After splice, this must be from year1.
    # In year1's file, May 1 0:00 was at slot 120*8 = 960, with
    # base_value 10000 + 960 = 10960.
    assert uas[960, 0, 0] == 10960.0
    # Apr 30 1991 last slot: idx 119*8 + 7 = 959. From the template
    # year2, base = 20000 + 959 = 20959.
    assert uas[959, 0, 0] == 20959.0


def test_make_ryf_3hr_mean_path_shifts_timestamp(synthetic_iaf, tmp_path):
    """3hr-mean variables (rsds, rlds, prra, prsn) get re-stamped from
    HH:30 to HH:00 — verify that path executes and produces the
    expected number of records."""
    out = tmp_path / "RYF9091.zarr"
    make_ryf_mod.make_ryf(synthetic_iaf, 1990, 1991, out, progress=False)
    ds = xr.open_zarr(str(out), decode_times=False)
    # rsds joins the common axis; if the timestamp shift were wrong
    # by an integer multiple of 3h the reindex would create NaN rows.
    rsds = ds["rsds"].values
    assert not np.any(np.isnan(rsds))
    assert rsds.shape == (365 * 8, 4, 8)


def test_make_ryf_daily_friver_broadcast_to_8_slots(synthetic_iaf, tmp_path):
    """Daily friver should be broadcast 8x per noleap day."""
    out = tmp_path / "RYF9091.zarr"
    make_ryf_mod.make_ryf(synthetic_iaf, 1990, 1991, out, progress=False)
    ds = xr.open_zarr(str(out), decode_times=False)
    fr = ds["friver"].values
    assert fr.shape == (365 * 8, 4, 8)
    # Slots 0..7 of any given day must be identical (the daily value
    # broadcast 8x).
    for day in (0, 50, 200, 364):
        slot0 = fr[day * 8, 0, 0]
        for slot in range(1, 8):
            assert fr[day * 8 + slot, 0, 0] == slot0


def test_make_ryf_rejects_missing_variable_file(tmp_path):
    """A missing per-variable file is a hard error, not silent."""
    iaf = tmp_path / "iaf_partial"
    # Only write uas; other vars missing.
    _write_iaf_var(iaf, make_ryf_mod.VARS[0], 1990, 1.0)
    _write_iaf_var(iaf, make_ryf_mod.VARS[0], 1991, 2.0)
    out = tmp_path / "RYF9091.zarr"
    with pytest.raises(FileNotFoundError, match="not found"):
        make_ryf_mod.make_ryf(iaf, 1990, 1991, out, progress=False)


def test_make_ryf_handles_friver_on_different_grid(tmp_path):
    """Real JRA55-do v1.6.0 publishes friver on a 0.25° regular grid
    while the atmospheric vars are on TL319 (~0.55°). This test
    reproduces that layout in miniature: 9 atmos vars on 4×8, friver
    on 8×16. make_ryf must interpolate friver onto the atmos grid
    rather than fail at the merge step."""
    iaf = tmp_path / "iaf"
    # 9 atmos vars on the small grid
    for v in make_ryf_mod.VARS:
        if v.name == "friver":
            continue
        _write_iaf_var(iaf, v, 1990, base_value=10000.0)
        _write_iaf_var(iaf, v, 1991, base_value=20000.0)

    # friver on a finer grid (8×16) — must be regridded
    _write_friver_alt_grid = (
        lambda year, base: _write_iaf_var_alt_grid(iaf, year, base)
    )
    _write_friver_alt_grid(1990, base=10000.0)
    _write_friver_alt_grid(1991, base=20000.0)

    out = tmp_path / "RYF9091.zarr"
    make_ryf_mod.make_ryf(iaf, 1990, 1991, out, progress=False)

    ds = xr.open_zarr(str(out), decode_times=False)
    # friver should now be on the same grid as atmos (4×8) after
    # bilinear interp.
    assert ds["friver"].sizes == {"time": 365 * 8, "lat": 4, "lon": 8}
    assert not np.any(np.isnan(ds["friver"].values))


def _write_iaf_var_alt_grid(iaf_dir, year, base):
    """Write friver on a finer 8x16 grid (vs the 4x8 default in
    _write_iaf_var) to exercise the regrid path."""
    var = next(v for v in make_ryf_mod.VARS if v.name == "friver")
    n_per_day = 1
    first_offset = pd.Timedelta(hours=12)
    n_days = 365
    n = n_days * n_per_day
    times = pd.date_range(f"{year}-01-01 12:00", periods=n, freq="1D")

    n_lat, n_lon = 8, 16  # different from atmos grid
    data = (base + np.arange(n, dtype=np.float64)[:, None, None]
            * np.ones((1, n_lat, n_lon)))

    half_lat = 90.0 / n_lat
    half_lon = 180.0 / n_lon
    lat = np.linspace(-90.0 + half_lat, 90.0 - half_lat, n_lat)
    lon = np.linspace(half_lon, 360.0 - half_lon, n_lon)

    ds = xr.Dataset(
        {var.name: (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    path = make_ryf_mod._local_path(iaf_dir, var, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(str(path))


def test_make_ryf_cli_entry_point(synthetic_iaf, tmp_path):
    """Smoke test the argv parser + main()."""
    out = tmp_path / "out.zarr"
    rc = make_ryf_mod.main([
        "--iaf-dir", str(synthetic_iaf),
        "--year1", "1990", "--year2", "1991",
        "--out", str(out),
        "--quiet",
    ])
    assert rc == 0
    assert out.exists()
