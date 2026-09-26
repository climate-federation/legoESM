"""Tests for ``legoesm.forcing.jra55_do`` — JRA55-do cache builder + loader.

The tests construct a small synthetic JRA55-do Zarr that mimics the
real distribution's structure (CMOR variable names,
``"days since 1958-01-01"`` time encoding, descending latitude, mixed
3-hourly / 6-hourly / daily cadence). This covers the module
end-to-end without requiring access to the real ~30 GB JRA55-do store.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.forcing.jra55_do import (
    JRA55_VARIABLES,
    JRA55DoConfig,
    JRA55Slice,
    RECORDS_PER_DAY,
    _floor_indices_and_alpha,
    _gregorian_to_jdn,
    _grid_edges_from_centers,
    _jdn_to_gregorian,
    _jra55_cos_zenith,
    _parse_time_axis_to_gregorian,
    _build_noleap_record_index,
    build_jra55_cache,
    jra55_to_atm_surface,
    jra55_to_freshwater,
    load_jra55_slice,
)
from legoesm.forcing.time_utils import date_to_day
from legoesm.ocean.freshwater import FreshwaterForcing


jax.config.update("jax_enable_x64", True)


# ============================================================================
# Synthetic source dataset builder
# ============================================================================

def _make_synthetic_jra55_zarr(
    out_path: Path,
    year_start: int = 1958,
    year_end: int = 1958,
    n_lat: int = 8,
    n_lon: int = 16,
    cadence_hours: int = 6,
):
    """Write a tiny synthetic JRA55-do-like Zarr.

    Includes:
      - All required CMOR variables
      - ``"days since 1958-01-01"`` time encoding
      - Descending latitude (real distribution convention)
      - Gregorian dates with leap days actually present (so the
        leap-day-drop machinery is exercised)
    """
    # Build Gregorian time axis at cadence_hours, including leap days.
    days_per_record = cadence_hours / 24.0
    # Number of days between Gregorian Jan-1 of year_start and Jan-1 of
    # (year_end+1), counting leap days correctly.
    jdn_start = _gregorian_to_jdn(year_start, 1, 1)
    jdn_end = _gregorian_to_jdn(year_end + 1, 1, 1)
    n_days_real = int(jdn_end - jdn_start)
    n_records = int(n_days_real * 24 / cadence_hours)
    time = np.arange(n_records, dtype=np.float64) * days_per_record

    # Latitude centers at half-cell offsets from the poles, descending —
    # JRA55-do convention. With n_lat=8 → centers at [78.75, 56.25, ...,
    # -78.75] so cell edges land exactly at ±90°.
    half = 90.0 / n_lat
    lat = np.linspace(90.0 - half, -90.0 + half, n_lat)   # descending
    half_lon = 180.0 / n_lon
    lon = np.linspace(half_lon, 360.0 - half_lon, n_lon)

    # Synthetic field generators — keep values inside _PLAUSIBLE_RANGE.
    def _scaled(low, high):
        rng = np.random.default_rng(int((low * 1000 + high) % 2**31))
        return rng.uniform(low, high, size=(n_records, n_lat, n_lon))

    data_vars = {
        "uas":    (("time", "lat", "lon"), _scaled(-15.0, 15.0)),
        "vas":    (("time", "lat", "lon"), _scaled(-15.0, 15.0)),
        "tas":    (("time", "lat", "lon"), _scaled(260.0, 300.0)),
        "huss":   (("time", "lat", "lon"), _scaled(0.001, 0.020)),
        "psl":    (("time", "lat", "lon"), _scaled(98000.0, 103000.0)),
        "rsds":   (("time", "lat", "lon"), _scaled(0.0, 800.0)),
        "rlds":   (("time", "lat", "lon"), _scaled(200.0, 450.0)),
        "prra":   (("time", "lat", "lon"), _scaled(0.0, 1e-4)),
        "prsn":   (("time", "lat", "lon"), _scaled(0.0, 1e-5)),
        "friver": (("time", "lat", "lon"), _scaled(0.0, 1e-5)),
    }
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["time"].attrs["units"] = f"days since {year_start}-01-01 00:00:00"
    ds["time"].attrs["calendar"] = "gregorian"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"
    ds.to_zarr(str(out_path), mode="w", consolidated=True)


# ============================================================================
# Gregorian / JDN arithmetic primitives
# ============================================================================

def test_jdn_round_trip_known_dates():
    # Known JDN values:
    #   1958-01-01 (epoch 1) = 2436205
    #   2000-01-01            = 2451545
    #   2018-12-31            = 2458484
    for y, m, d, expected in [
        (1958, 1, 1, 2436205),
        (2000, 1, 1, 2451545),
        (2018, 12, 31, 2458484),
    ]:
        jdn = int(_gregorian_to_jdn(y, m, d))
        assert jdn == expected, f"JDN({y}-{m}-{d}) = {jdn}, expected {expected}"
        y2, m2, d2 = _jdn_to_gregorian(jdn)
        assert (int(y2), int(m2), int(d2)) == (y, m, d)


def test_jdn_handles_feb_29_2016():
    """2016 is a leap year — Feb 29 must round-trip."""
    jdn = int(_gregorian_to_jdn(2016, 2, 29))
    y, m, d = _jdn_to_gregorian(jdn)
    assert (int(y), int(m), int(d)) == (2016, 2, 29)


def test_jdn_vectorized():
    years = np.array([1958, 1960, 2000])
    months = np.array([1, 2, 12])
    days = np.array([1, 29, 31])
    jdn = _gregorian_to_jdn(years, months, days)
    y2, m2, d2 = _jdn_to_gregorian(jdn)
    np.testing.assert_array_equal(y2, years)
    np.testing.assert_array_equal(m2, months)
    np.testing.assert_array_equal(d2, days)


# ============================================================================
# Time-axis parsing
# ============================================================================

def test_parse_time_axis_basic():
    time_vals = np.array([0.0, 0.25, 1.0, 1.5])  # days since 1958-01-01
    y, m, d, h = _parse_time_axis_to_gregorian(
        time_vals, "days since 1958-01-01 00:00:00",
    )
    assert (int(y[0]), int(m[0]), int(d[0])) == (1958, 1, 1)
    assert h[0] == pytest.approx(0.0)
    assert h[1] == pytest.approx(6.0)   # 0.25 day = 6 hours
    assert (int(y[2]), int(m[2]), int(d[2])) == (1958, 1, 2)
    assert h[2] == pytest.approx(0.0)
    assert h[3] == pytest.approx(12.0)


def test_parse_time_axis_rejects_unrecognised_units():
    """Anything not in {days,hours,minutes,seconds} since YYYY-MM-DD
    is a hard error."""
    with pytest.raises(ValueError, match="Cannot parse"):
        _parse_time_axis_to_gregorian(np.array([0.0]),
                                       "fortnights since 1958-01-01")
    with pytest.raises(ValueError, match="Cannot parse"):
        _parse_time_axis_to_gregorian(np.array([0.0]),
                                       "no time units here")


def test_parse_time_axis_handles_hours_since():
    """xarray often serialises 3-hourly data as 'hours since ...';
    the cache builder must accept it (this is exactly what make_ryf
    produces)."""
    # 24.0 hours since 1958-01-01 00:00 → 1958-01-02
    y, mo, d, h = _parse_time_axis_to_gregorian(
        np.array([0.0, 3.0, 24.0, 27.5]),
        "hours since 1958-01-01 00:00:00",
    )
    assert (int(y[0]), int(mo[0]), int(d[0])) == (1958, 1, 1)
    assert h[0] == pytest.approx(0.0)
    assert h[1] == pytest.approx(3.0)
    assert (int(y[2]), int(mo[2]), int(d[2])) == (1958, 1, 2)
    assert h[2] == pytest.approx(0.0)
    assert h[3] == pytest.approx(3.5)


# ============================================================================
# Cache-index construction (leap-day drop + window filter)
# ============================================================================

def test_record_index_drops_feb_29():
    """Feb 29 of any leap year must not land on the cache axis."""
    # Build src records: Jan 1 2016 -> Feb 29 2016 -> Mar 1 2016, all 00:00.
    src_year = np.array([2016, 2016, 2016])
    src_month = np.array([1, 2, 3])
    src_day = np.array([1, 29, 1])
    src_hour = np.array([0.0, 0.0, 0.0])
    keep, idx = _build_noleap_record_index(
        src_year, src_month, src_day, src_hour,
        years=(2016, 2016), ref_year=1958,
    )
    assert keep.tolist() == [True, False, True]


def test_record_index_filters_to_year_window():
    src_year = np.array([1957, 1958, 2018, 2019])
    src_month = np.array([12, 1, 12, 1])
    src_day = np.array([31, 1, 31, 1])
    src_hour = np.array([0.0, 0.0, 0.0, 0.0])
    keep, _ = _build_noleap_record_index(
        src_year, src_month, src_day, src_hour,
        years=(1958, 2018), ref_year=1958,
    )
    assert keep.tolist() == [False, True, True, False]


def test_record_index_maps_to_correct_cache_slot():
    """1958-01-01 06:00 should land at cache index 2 (3-hourly slot 2)."""
    src_year = np.array([1958])
    src_month = np.array([1])
    src_day = np.array([1])
    src_hour = np.array([6.0])
    keep, idx = _build_noleap_record_index(
        src_year, src_month, src_day, src_hour,
        years=(1958, 1958), ref_year=1958,
    )
    assert keep[0]
    assert int(idx[0]) == 2  # day 0, slot 6/3 = 2


# ============================================================================
# Grid-edge inference
# ============================================================================

def test_grid_edges_from_centers_uniform():
    centers = np.array([0.5, 1.5, 2.5, 3.5])  # 1° spacing, edges at 0..4
    edges = _grid_edges_from_centers(centers)
    expected = np.deg2rad(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(edges, expected, atol=1e-14)


# ============================================================================
# Slice loader floor-and-alpha math
# ============================================================================

def test_floor_indices_and_alpha_exact_slot():
    # day = 1.0 → cache slot 8 exactly (3-hourly).  alpha = 0.
    i_lo, i_hi, alpha = _floor_indices_and_alpha(1.0)
    assert (i_lo, i_hi, alpha) == (8, 8, 0.0)


def test_floor_indices_and_alpha_midway():
    # day = 0.0625 = 0.5 × 3-hour slot.  Slot 0 → 1 with alpha=0.5.
    i_lo, i_hi, alpha = _floor_indices_and_alpha(0.0625)
    assert (i_lo, i_hi) == (0, 1)
    assert alpha == pytest.approx(0.5, abs=1e-14)


def test_floor_indices_and_alpha_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        _floor_indices_and_alpha(-0.1)


# ============================================================================
# End-to-end cache build + slice load
# ============================================================================

@pytest.fixture
def synthetic_cache(tmp_path: Path):
    """Build a 1958–1958 noleap cache from a synthetic 6-hourly Zarr."""
    # ``build_jra55_cache`` writes the cache via ``cache_ds.chunk(...).to_zarr``,
    # which requires the optional ``dask`` chunk manager. Skip (don't error)
    # when dask is absent so the rest of the module still runs.
    pytest.importorskip("dask")
    src_path = tmp_path / "synthetic_jra55.zarr"
    _make_synthetic_jra55_zarr(
        src_path, year_start=1958, year_end=1958,
        n_lat=8, n_lon=16, cadence_hours=6,
    )

    target_lat_edges = np.deg2rad(np.linspace(-90.0, 90.0, 5))   # 4 cells
    target_lon_edges = np.deg2rad(np.linspace(0.0, 360.0, 9))    # 8 cells

    cache_dir = tmp_path / "cache"
    cfg = JRA55DoConfig(
        source_path=str(src_path),
        years=(1958, 1958),
        target_lat_edges=target_lat_edges,
        target_lon_edges=target_lon_edges,
        cache_dir=cache_dir,
    )
    cache_path = build_jra55_cache(cfg, overwrite=True, progress=False)
    return cache_path, cfg


def test_build_cache_creates_zarr_with_expected_structure(synthetic_cache):
    cache_path, cfg = synthetic_cache
    assert Path(cache_path).exists()

    ds = xr.open_zarr(str(cache_path), decode_times=False)
    # All required variables present
    for var in JRA55_VARIABLES:
        assert var in ds.data_vars, f"missing {var}"
    # Dimensions
    n_records_expected = 365 * RECORDS_PER_DAY
    assert ds.sizes["time"] == n_records_expected
    assert ds.sizes["lat"] == cfg.target_lat_edges.size - 1
    assert ds.sizes["lon"] == cfg.target_lon_edges.size - 1
    # Attrs
    assert ds.attrs["calendar"] == "noleap"
    assert int(ds.attrs["records_per_day"]) == RECORDS_PER_DAY
    assert int(ds.attrs["ref_year"]) == 1958


def test_build_cache_skips_when_exists(synthetic_cache, capsys):
    cache_path, cfg = synthetic_cache
    # Re-build with overwrite=False should be a no-op.
    out = build_jra55_cache(cfg, overwrite=False, progress=True)
    captured = capsys.readouterr()
    assert "skipping build" in captured.out
    assert out == cache_path


def test_build_cache_raises_on_missing_variable(tmp_path):
    # Make a synthetic zarr that's missing 'friver'.
    src_path = tmp_path / "incomplete.zarr"
    _make_synthetic_jra55_zarr(src_path, year_start=1958, year_end=1958,
                                n_lat=4, n_lon=8, cadence_hours=24)
    ds = xr.open_zarr(str(src_path), decode_times=False)
    ds = ds.drop_vars("friver")
    src2 = tmp_path / "incomplete2.zarr"
    ds.to_zarr(str(src2), mode="w", consolidated=True)

    cfg = JRA55DoConfig(
        source_path=str(src2),
        years=(1958, 1958),
        target_lat_edges=np.deg2rad(np.linspace(-90.0, 90.0, 3)),
        target_lon_edges=np.deg2rad(np.linspace(0.0, 360.0, 5)),
        cache_dir=tmp_path / "cache",
    )
    with pytest.raises(KeyError, match="friver"):
        build_jra55_cache(cfg, overwrite=True, progress=False)


def test_load_slice_at_day_zero_returns_first_record(synthetic_cache):
    cache_path, _ = synthetic_cache
    slc = load_jra55_slice(cache_path, day=0.0)

    ds = xr.open_zarr(str(cache_path), decode_times=False)
    for var in JRA55_VARIABLES:
        np.testing.assert_allclose(
            np.asarray(getattr(slc, var)),
            ds[var].isel(time=0).values,
            atol=1e-12,
        )


def test_load_slice_interpolates_linearly_between_records(synthetic_cache):
    """At fractional day 1/16 (midway between 3-hourly slots 0 and 1),
    the loader must return the average of the two records."""
    cache_path, _ = synthetic_cache
    midway_day = 0.5 / RECORDS_PER_DAY  # halfway between slot 0 and slot 1
    slc = load_jra55_slice(cache_path, day=midway_day)

    ds = xr.open_zarr(str(cache_path), decode_times=False)
    for var in JRA55_VARIABLES:
        expected = 0.5 * (
            ds[var].isel(time=0).values + ds[var].isel(time=1).values
        )
        np.testing.assert_allclose(
            np.asarray(getattr(slc, var)),
            expected,
            atol=1e-12,
        )


def test_load_slice_rejects_out_of_range_day(synthetic_cache):
    cache_path, _ = synthetic_cache
    with pytest.raises(IndexError, match="exceeds cache length"):
        load_jra55_slice(cache_path, day=10000.0)


# ============================================================================
# RYF cycling — wrap modulo cache length
# ============================================================================

def test_load_slice_cycle_wraps_past_cache_end(synthetic_cache):
    """With cycle=True a 365-day cache repeats every 365 days."""
    cache_path, _ = synthetic_cache
    s_year0 = load_jra55_slice(cache_path, day=0.0, cycle=True)
    s_year1 = load_jra55_slice(cache_path, day=365.0, cycle=True)
    s_year2 = load_jra55_slice(cache_path, day=730.0, cycle=True)
    for var in JRA55_VARIABLES:
        np.testing.assert_array_equal(
            np.asarray(getattr(s_year0, var)),
            np.asarray(getattr(s_year1, var)),
        )
        np.testing.assert_array_equal(
            np.asarray(getattr(s_year0, var)),
            np.asarray(getattr(s_year2, var)),
        )


def test_load_slice_cycle_preserves_seasonal_phase(synthetic_cache):
    """day=10 in year 0 must equal day=375 in year 1 under cycling
    (i.e. the same simulation day-of-year)."""
    cache_path, _ = synthetic_cache
    a = load_jra55_slice(cache_path, day=10.0, cycle=True)
    b = load_jra55_slice(cache_path, day=375.0, cycle=True)
    for var in JRA55_VARIABLES:
        np.testing.assert_array_equal(
            np.asarray(getattr(a, var)), np.asarray(getattr(b, var)),
        )


def test_load_slice_no_cycle_still_raises_past_end(synthetic_cache):
    """The default (cycle=False) behaviour is unchanged."""
    cache_path, _ = synthetic_cache
    with pytest.raises(IndexError):
        load_jra55_slice(cache_path, day=400.0, cycle=False)


def test_load_slice_returns_jax_arrays(synthetic_cache):
    cache_path, _ = synthetic_cache
    slc = load_jra55_slice(cache_path, day=0.0)
    for var in JRA55_VARIABLES:
        arr = getattr(slc, var)
        assert isinstance(arr, jax.Array)
        assert arr.dtype == jnp.float64


def test_cache_drops_feb_29_from_2016_window(tmp_path):
    """Sanity: build a 1958–2016 cache (3 years incl. one leap year)
    and verify the cache length excludes Feb 29 from 2016."""
    pytest.importorskip("dask")  # cache write requires the dask chunk manager
    src_path = tmp_path / "leap_window.zarr"
    _make_synthetic_jra55_zarr(
        src_path, year_start=2014, year_end=2016,
        n_lat=4, n_lon=8, cadence_hours=24,   # daily for speed
    )
    cfg = JRA55DoConfig(
        source_path=str(src_path),
        years=(2014, 2016),
        target_lat_edges=np.deg2rad(np.linspace(-90.0, 90.0, 3)),
        target_lon_edges=np.deg2rad(np.linspace(0.0, 360.0, 5)),
        cache_dir=tmp_path / "cache_leap",
        ref_year=2014,
    )
    cache_path = build_jra55_cache(cfg, overwrite=True, progress=False)
    ds = xr.open_zarr(str(cache_path), decode_times=False)
    # 3 noleap years × 365 days × 8 records/day
    assert ds.sizes["time"] == 3 * 365 * RECORDS_PER_DAY


def test_cache_conservation_for_constant_field(tmp_path):
    """A spatially constant source field must regrid to the same constant
    on the model grid, end-to-end through the cache builder."""
    pytest.importorskip("dask")  # cache write requires the dask chunk manager
    src_path = tmp_path / "constant_jra55.zarr"
    n_lat, n_lon = 8, 16
    n_t = 8  # one day at 3-hourly cadence
    time = np.arange(n_t, dtype=np.float64) * (3.0 / 24.0)
    half = 90.0 / n_lat
    lat = np.linspace(90.0 - half, -90.0 + half, n_lat)
    half_lon = 180.0 / n_lon
    lon = np.linspace(half_lon, 360.0 - half_lon, n_lon)
    const_values = {
        "uas": 5.0, "vas": -3.0, "tas": 290.0, "huss": 0.01,
        "psl": 1.013e5, "rsds": 250.0, "rlds": 350.0,
        "prra": 1e-5, "prsn": 0.0, "friver": 0.0,
    }
    data_vars = {
        var: (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), val))
        for var, val in const_values.items()
    }
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["time"].attrs["units"] = "days since 1958-01-01 00:00:00"
    ds.to_zarr(str(src_path), mode="w", consolidated=True)

    cfg = JRA55DoConfig(
        source_path=str(src_path),
        years=(1958, 1958),
        target_lat_edges=np.deg2rad(np.linspace(-90.0, 90.0, 5)),
        target_lon_edges=np.deg2rad(np.linspace(0.0, 360.0, 9)),
        cache_dir=tmp_path / "cache_const",
    )
    cache_path = build_jra55_cache(cfg, overwrite=True, progress=False)

    # Load the first slot (the only one populated by source); it must
    # match the input constants on the model grid.
    slc = load_jra55_slice(cache_path, day=0.0)
    for var, val in const_values.items():
        arr = np.asarray(getattr(slc, var))
        np.testing.assert_allclose(arr, val, atol=1e-12, err_msg=f"{var}")


# ============================================================================
# Driver-side glue: JRA55Slice -> AtmToSurface, FreshwaterForcing
# ============================================================================


def _make_synthetic_slice(shape=(4, 8)) -> JRA55Slice:
    """A tiny in-memory JRA55Slice with realistic-magnitude values."""
    return JRA55Slice(
        uas=jnp.full(shape, 5.0),
        vas=jnp.full(shape, -3.0),
        tas=jnp.full(shape, 290.0),
        huss=jnp.full(shape, 0.01),
        psl=jnp.full(shape, 1.013e5),
        rsds=jnp.full(shape, 250.0),
        rlds=jnp.full(shape, 350.0),
        prra=jnp.full(shape, 1e-5),
        prsn=jnp.full(shape, 1e-6),
        friver=jnp.full(shape, 1e-7),
    )


# ----------------------------------------------------------------------------
# _jra55_cos_zenith
# ----------------------------------------------------------------------------

def test_cos_zenith_returns_finite_values():
    lat = jnp.deg2rad(jnp.linspace(-89.0, 89.0, 9))[:, None]
    lon = jnp.deg2rad(jnp.linspace(0.0, 350.0, 12))[None, :]
    cos_z = _jra55_cos_zenith(lat, lon, day=0.5, ref_year=1958)
    assert jnp.all(jnp.isfinite(cos_z))
    assert float(jnp.min(cos_z)) >= -1.0 - 1e-12
    assert float(jnp.max(cos_z)) <= 1.0 + 1e-12


def test_cos_zenith_diurnal_signal_at_equator():
    """At the equator on the equinox (day-of-year ~80 = Mar 21),
    cos(θ_z) at noon (hour=12) should be near +1; at midnight near -1."""
    lat = jnp.zeros((1, 1))    # equator
    lon = jnp.zeros((1, 1))    # 0° longitude
    # Mar 21 = day 80 of noleap (Jan 31 + Feb 28 + Mar 21 = 31+28+21=80)
    # day 79 = Mar 21 0:00 (since day 0 is Jan 1 00:00)
    day_noon = 79.0 + 12.0 / 24.0  # Mar 21 noon
    day_midnight = 79.0            # Mar 21 00:00 → at lon=0 (UTC), this IS midnight
    cz_noon = float(_jra55_cos_zenith(lat, lon, day=day_noon)[0, 0])
    cz_mid = float(_jra55_cos_zenith(lat, lon, day=day_midnight)[0, 0])
    # Sun overhead at noon at the equator on equinox.
    assert cz_noon > 0.95, f"expected ~+1 at noon, got {cz_noon}"
    # Sun directly opposite at midnight.
    assert cz_mid < -0.95, f"expected ~-1 at midnight, got {cz_mid}"


# ----------------------------------------------------------------------------
# jra55_to_atm_surface
# ----------------------------------------------------------------------------

def test_atm_surface_field_shapes_and_dtypes():
    slc = _make_synthetic_slice(shape=(4, 8))
    lat = jnp.deg2rad(jnp.linspace(-89.0, 89.0, 4))[:, None]
    lon = jnp.deg2rad(jnp.linspace(0.0, 315.0, 8))[None, :]
    atm = jra55_to_atm_surface(slc, lat, lon, day=10.5)

    assert isinstance(atm, AtmToSurface)
    expected_shape = (4, 8)
    for fname in [
        "sw_down", "lw_down", "precip_total", "precip_snow",
        "T_lowest", "q_lowest", "u_lowest", "v_lowest",
        "p_lowest", "p_surface", "rho_lowest", "cos_zenith",
    ]:
        arr = getattr(atm, fname)
        assert arr.shape == expected_shape, f"{fname}: got {arr.shape}"
        assert arr.dtype == slc.tas.dtype, f"{fname} dtype mismatch"
    # Scalar flags
    assert atm.has_radiation.shape == ()
    assert atm.has_precipitation.shape == ()
    assert atm.co2_ppmv.shape == ()


def test_atm_surface_passes_through_winds_and_state():
    slc = _make_synthetic_slice()
    lat = jnp.zeros((4, 8))
    lon = jnp.zeros((4, 8))
    atm = jra55_to_atm_surface(slc, lat, lon, day=0.0)
    np.testing.assert_array_equal(np.asarray(atm.u_lowest), np.asarray(slc.uas))
    np.testing.assert_array_equal(np.asarray(atm.v_lowest), np.asarray(slc.vas))
    np.testing.assert_array_equal(np.asarray(atm.T_lowest), np.asarray(slc.tas))
    np.testing.assert_array_equal(np.asarray(atm.q_lowest), np.asarray(slc.huss))
    np.testing.assert_array_equal(np.asarray(atm.sw_down), np.asarray(slc.rsds))
    np.testing.assert_array_equal(np.asarray(atm.lw_down), np.asarray(slc.rlds))
    np.testing.assert_array_equal(np.asarray(atm.p_surface), np.asarray(slc.psl))
    np.testing.assert_array_equal(np.asarray(atm.p_lowest), np.asarray(slc.psl))


def test_atm_surface_combines_precip_total_correctly():
    """precip_total = liquid + solid; precip_snow = snow only."""
    slc = _make_synthetic_slice()
    atm = jra55_to_atm_surface(slc, jnp.zeros((4, 8)), jnp.zeros((4, 8)), day=0.0)
    np.testing.assert_allclose(
        np.asarray(atm.precip_total),
        np.asarray(slc.prra + slc.prsn),
        atol=1e-14,
    )
    np.testing.assert_array_equal(
        np.asarray(atm.precip_snow), np.asarray(slc.prsn),
    )


def test_atm_surface_density_from_ideal_gas():
    """ρ = p / (R_d · T_v), T_v = T (1 + (1/ε − 1) q) with the canonical coeff."""
    slc = _make_synthetic_slice()
    atm = jra55_to_atm_surface(slc, jnp.zeros((4, 8)), jnp.zeros((4, 8)), day=0.0)
    T_v = slc.tas * (1.0 + (1.0 / constants.epsilon - 1.0) * slc.huss)
    expected_rho = slc.psl / (constants.R_d * T_v)
    np.testing.assert_allclose(
        np.asarray(atm.rho_lowest), np.asarray(expected_rho), rtol=1e-12,
    )
    # Sanity: density at 290 K, 1013 hPa, q=0.01 should be ~1.21 kg/m³.
    assert 1.15 < float(atm.rho_lowest[0, 0]) < 1.25


def test_atm_surface_co2_default():
    slc = _make_synthetic_slice()
    atm = jra55_to_atm_surface(slc, jnp.zeros((4, 8)), jnp.zeros((4, 8)), day=0.0)
    assert float(atm.co2_ppmv) == 400.0
    # Override
    atm2 = jra55_to_atm_surface(
        slc, jnp.zeros((4, 8)), jnp.zeros((4, 8)), day=0.0, co2_ppmv=420.0,
    )
    assert float(atm2.co2_ppmv) == 420.0


def test_atm_surface_cos_zenith_varies_with_day_at_fixed_point():
    """At a fixed (lat, lon) on the equator at 0° longitude, cos_zenith
    must differ measurably between noon and midnight UTC.

    Globally averaging is a poor test because the sun is overhead
    *somewhere* at every UTC hour — the spatial pattern rotates with
    longitude but the global mean stays constant. Checking a single
    cell removes that symmetry.
    """
    slc = _make_synthetic_slice(shape=(1, 1))
    lat = jnp.zeros((1, 1))   # equator
    lon = jnp.zeros((1, 1))   # 0° longitude
    atm_noon = jra55_to_atm_surface(slc, lat, lon, day=79.5)   # Mar 21 12:00
    atm_mid = jra55_to_atm_surface(slc, lat, lon, day=79.0)    # Mar 21 00:00
    cz_noon = float(atm_noon.cos_zenith[0, 0])
    cz_mid = float(atm_mid.cos_zenith[0, 0])
    assert cz_noon > 0.95, f"expected ~+1 at equator/equinox/noon, got {cz_noon}"
    assert cz_mid < -0.95, f"expected ~-1 at equator/equinox/midnight, got {cz_mid}"


# ----------------------------------------------------------------------------
# jra55_to_freshwater
# ----------------------------------------------------------------------------

def test_freshwater_field_shapes_and_types():
    slc = _make_synthetic_slice(shape=(6, 12))
    lhflx = jnp.full((6, 12), 100.0)  # W/m²
    fw = jra55_to_freshwater(slc, lhflx)
    assert isinstance(fw, FreshwaterForcing)
    for fname in ("precip", "evap", "runoff", "ice_fw"):
        arr = getattr(fw, fname)
        assert arr.shape == (6, 12), f"{fname}: got {arr.shape}"


def test_freshwater_precip_is_rain_plus_snow():
    slc = _make_synthetic_slice()
    fw = jra55_to_freshwater(slc, jnp.zeros((4, 8)))
    np.testing.assert_allclose(
        np.asarray(fw.precip),
        np.asarray(slc.prra + slc.prsn),
        atol=1e-14,
    )


def test_freshwater_evap_from_lhflx():
    """E [kg/m²/s] = L_h [W/m²] / L_v [J/kg]."""
    slc = _make_synthetic_slice()
    lhflx = jnp.full((4, 8), 250.0)  # W/m² (typical tropical ocean)
    fw = jra55_to_freshwater(slc, lhflx)
    expected_evap = 250.0 / constants.L_v
    np.testing.assert_allclose(np.asarray(fw.evap), expected_evap, rtol=1e-12)
    # Sanity: a typical tropical L_h ~250 W/m² gives E ~1e-4 kg/m²/s
    # (= ~8.6 mm/day) — within an order of magnitude of climatology.
    assert 5e-5 < float(fw.evap[0, 0]) < 2e-4


def test_freshwater_runoff_passthrough():
    slc = _make_synthetic_slice()
    fw = jra55_to_freshwater(slc, jnp.zeros((4, 8)))
    np.testing.assert_array_equal(
        np.asarray(fw.runoff), np.asarray(slc.friver),
    )


def test_freshwater_ice_fw_zero_in_tropical_omip():
    slc = _make_synthetic_slice()
    fw = jra55_to_freshwater(slc, jnp.full((4, 8), 100.0))
    np.testing.assert_allclose(np.asarray(fw.ice_fw), 0.0, atol=0.0)


def test_freshwater_custom_L_v():
    """Caller can override L_v, e.g. for sublimation."""
    slc = _make_synthetic_slice()
    lhflx = jnp.full((4, 8), 300.0)
    custom_L = constants.L_s  # latent heat of sublimation
    fw = jra55_to_freshwater(slc, lhflx, L_v=custom_L)
    np.testing.assert_allclose(
        np.asarray(fw.evap), 300.0 / custom_L, rtol=1e-12,
    )


# ----------------------------------------------------------------------------
# AD compatibility
# ----------------------------------------------------------------------------

def test_glue_functions_are_differentiable():
    """jax.grad through both glue functions produces finite gradients —
    keeps the door open to differentiable forcing experiments later."""
    slc = _make_synthetic_slice()
    lat = jnp.deg2rad(jnp.linspace(-45.0, 45.0, 4))[:, None]
    lon = jnp.zeros((1, 8))

    def loss(tas):
        slc2 = slc._replace(tas=tas)
        atm = jra55_to_atm_surface(slc2, lat, lon, day=10.0)
        # Mock lhflx that depends on tas (for AD chain).
        lhflx = 10.0 * (slc2.tas - 280.0)
        fw = jra55_to_freshwater(slc2, lhflx)
        return jnp.sum(atm.rho_lowest ** 2) + jnp.sum(fw.evap ** 2)

    grads = jax.grad(loss)(slc.tas)
    assert bool(jnp.all(jnp.isfinite(grads)))
    assert not bool(jnp.allclose(grads, 0.0))


# ----------------------------------------------------------------------------
# JIT compatibility
# ----------------------------------------------------------------------------

def test_glue_jit_compiles():
    slc = _make_synthetic_slice()
    lat = jnp.deg2rad(jnp.linspace(-45.0, 45.0, 4))[:, None]
    lon = jnp.zeros((1, 8))

    @jax.jit
    def go(slice_obj, lhflx):
        atm = jra55_to_atm_surface(slice_obj, lat, lon, day=10.0)
        fw = jra55_to_freshwater(slice_obj, lhflx)
        return atm.rho_lowest.sum() + fw.evap.sum()

    out = go(slc, jnp.full((4, 8), 100.0))
    assert jnp.isfinite(out)


def test_cache_polar_coverage_after_lat_clamp(tmp_path):
    """A constant source whose uniform-inferred lat edge does NOT land on
    the pole (outer centre 86 deg -> inferred outer edge ~98 deg, whose
    sin caps at ~0.9897 < 1) leaves the polar destination cell only ~96.5%
    covered: without the lat clamp the polar rows regrid the constant to
    LESS than its value. RED before the clamp (polar tas ~279.8 K != 290);
    GREEN after np.clip(src/target lat edges, +/- pi/2) restores full
    coverage.  Pins the CLAMP only; the polar gap itself is handled in the WEIGHTS
    (fracarea + polar_fill), see regrid_polar_coverage_2026-07-24.md."""
    pytest.importorskip("dask")
    src_path = tmp_path / "gaussianish_jra55.zarr"
    n_lat, n_lon = 8, 16
    n_t = 8
    time = np.arange(n_t, dtype=np.float64) * (3.0 / 24.0)
    # Outer centre 86 deg; uniform edge inference overshoots the pole so the
    # outermost source cell has sin(edge) < 1 -> a genuine polar gap.
    lat = np.linspace(86.0, -86.0, n_lat)
    half_lon = 180.0 / n_lon
    lon = np.linspace(half_lon, 360.0 - half_lon, n_lon)
    const_values = {
        "uas": 5.0, "vas": -3.0, "tas": 290.0, "huss": 0.01,
        "psl": 1.013e5, "rsds": 250.0, "rlds": 350.0,
        "prra": 1e-5, "prsn": 0.0, "friver": 0.0,
    }
    data_vars = {
        var: (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), val))
        for var, val in const_values.items()
    }
    ds = xr.Dataset(
        data_vars=data_vars, coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["time"].attrs["units"] = "days since 1958-01-01 00:00:00"
    ds.to_zarr(str(src_path), mode="w", consolidated=True)
    cfg = JRA55DoConfig(
        source_path=str(src_path),
        years=(1958, 1958),
        target_lat_edges=np.deg2rad(np.linspace(-90.0, 90.0, 5)),
        target_lon_edges=np.deg2rad(np.linspace(0.0, 360.0, 9)),
        cache_dir=tmp_path / "cache_gauss",
    )
    cache_path = build_jra55_cache(cfg, overwrite=True, progress=False)
    # Constant source must regrid to the same constant on EVERY row,
    # including the two polar rows, once the lat clamp closes the pole gap.
    slc = load_jra55_slice(cache_path, day=0.0)
    np.testing.assert_allclose(
        np.asarray(slc.tas), 290.0, atol=1e-6,
        err_msg="polar tas reduced by uncovered pole cell",
    )


def test_cache_polar_gap_is_treated_not_diluted(tmp_path):
    """Pins ``normalization='fracarea'`` AT THIS CALLER, which nothing else does.

    Every other jra55 fixture builds a latitude axis that tiles [-90, 90] exactly,
    so its inferred edges land on the poles, latitude coverage is 1 everywhere, and
    BOTH polar treatments are no-ops -- deleting ``normalization=`` from
    ``build_jra55_cache`` leaves those green.  Here the source's inferred edges stop
    at +-89.5 deg: a REAL polar gap of the size JRA55-do actually has (0.151 deg;
    CORE-II's is 0.514), inside the 2 deg extrapolation budget, and the lat clamp is
    a no-op so this is not a restatement of ``test_cache_polar_coverage_after_lat_clamp``.

    Onto 1 deg target rows the polar row is 0.75 covered -- partial, never empty --
    so ``fracarea`` alone must carry it and ``polar_fill`` plays no part.  Untreated
    that row returns 0.75 x 290 = 217.5 K, which is the dilution the treatment
    removes; the anti-vacuity leg below computes exactly that.
    """
    pytest.importorskip("dask")
    src_path = tmp_path / "polar_gap_jra55.zarr"
    n_lat, n_lon, n_t = 8, 16, 8
    # Outer centre 78.3125 deg -> uniform edge inference gives exactly +-89.5.
    lat = np.linspace(78.3125, -78.3125, n_lat)
    half_lon = 180.0 / n_lon
    lon = np.linspace(half_lon, 360.0 - half_lon, n_lon)
    const_values = {
        "uas": 5.0, "vas": -3.0, "tas": 290.0, "huss": 0.01,
        "psl": 1.013e5, "rsds": 250.0, "rlds": 350.0,
        "prra": 1e-5, "prsn": 0.0, "friver": 0.0,
    }
    ds = xr.Dataset(
        data_vars={
            var: (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), val))
            for var, val in const_values.items()
        },
        coords={
            "time": np.arange(n_t, dtype=np.float64) * (3.0 / 24.0),
            "lat": lat, "lon": lon,
        },
    )
    ds["time"].attrs["units"] = "days since 1958-01-01 00:00:00"
    ds.to_zarr(str(src_path), mode="w", consolidated=True)

    target_lat_edges = np.deg2rad(np.linspace(-90.0, 90.0, 181))   # 1 deg rows
    target_lon_edges = np.deg2rad(np.linspace(0.0, 360.0, 9))
    cfg = JRA55DoConfig(
        source_path=str(src_path),
        years=(1958, 1958),
        target_lat_edges=target_lat_edges,
        target_lon_edges=target_lon_edges,
        cache_dir=tmp_path / "cache_polar_gap",
    )
    cache_path = build_jra55_cache(cfg, overwrite=True, progress=False)
    slc = load_jra55_slice(cache_path, day=0.0)
    # The constant survives on EVERY row, polar rows included.
    np.testing.assert_allclose(
        np.asarray(slc.tas), 290.0, atol=1e-6,
        err_msg="polar tas diluted -- is normalization='fracarea' still passed?",
    )

    # Anti-vacuity: the same geometry with the DEFAULT normalisation returns the
    # polar row reduced to ~217.5 K, so this fixture genuinely exercises fracarea.
    from legoesm.grids.conservative_regrid import (
        apply_conservative_regrid, compute_overlap_weights,
    )
    src_lat_e = np.deg2rad(np.linspace(-89.5, 89.5, n_lat + 1))
    src_lon_e = np.deg2rad(np.linspace(0.0, 360.0, n_lon + 1))
    untreated = np.asarray(apply_conservative_regrid(
        np.full((n_lat, n_lon), 290.0),
        compute_overlap_weights(src_lat_e, src_lon_e,
                                target_lat_edges, target_lon_edges),
    ))
    np.testing.assert_allclose(untreated[1:-1, :], 290.0, atol=1e-9)
    np.testing.assert_allclose(untreated[[0, -1], :], 290.0 * 0.749995, rtol=1e-4)


def test_ocean_loader_reads_builder_cache(tmp_path):
    """The ocean OMIP-2 loader must read what build_jra55_cache writes
    instead of silently falling back to synthetic forcing."""
    import xarray as xr
    from legoesm.ocean.forcing.jra55_do import load_jra55_do

    src_path = tmp_path / "synthetic_jra55.zarr"
    _make_synthetic_jra55_zarr(src_path, n_lat=8, n_lon=16, cadence_hours=6)
    cfg = JRA55DoConfig(
        source_path=str(src_path), years=(1958, 1958),
        target_lat_edges=np.deg2rad(np.linspace(-90.0, 90.0, 5)),
        target_lon_edges=np.deg2rad(np.linspace(0.0, 360.0, 9)),
        cache_dir=tmp_path / "cache",
    )
    cache_path = build_jra55_cache(cfg, overwrite=True, progress=False)
    for root in (cfg.cache_dir, cache_path):   # directory or the store itself
        f = load_jra55_do(1958, cache_dir=root, allow_synthetic=False)
        ds = xr.open_zarr(cache_path)
        n = 365 * RECORDS_PER_DAY
        assert f.u10.shape == (n, 4, 8)
        np.testing.assert_array_equal(f.T_air, ds.tas.values[:n])
        np.testing.assert_array_equal(
            f.precip, ds.prra.values[:n] + ds.prsn.values[:n])
        np.testing.assert_array_equal(f.slp, ds.psl.values[:n])
        assert np.all((f.lon >= 0.0) & (f.lon < 360.0))
        assert f.time_s[1] - f.time_s[0] == 3 * 3600.0
    with pytest.raises(FileNotFoundError):
        load_jra55_do(1959, cache_dir=cfg.cache_dir, allow_synthetic=False)
