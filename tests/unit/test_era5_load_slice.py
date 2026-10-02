"""Tests for the ERA5 variable resolver + slice loader (the real-ERA5 INPUT path).

``load_era5_slice`` feeds the whole compare-reanalysis pipeline; a missing/
misnamed REQUIRED variable used to SILENTLY load as zeros (corrupting the bias so
the loop "corrects" garbage).  These lock: bidirectional name resolution (long
``"temperature"`` ↔ short ``"t"``) and the fail-fast on a missing required field.
"""

from __future__ import annotations

import jax.numpy as jnp
import legoesm.training.era5_to_state as e2s
import numpy as np
import pytest
from legoesm.training.era5_to_state import (
    TrainingERA5Config,
    load_era5_slice,
    resolve_var,
)
from legoesm.training.era5_to_state import era5_terrain_product

_LEVELS = (1000.0, 500.0, 100.0)   # hPa, descending (the loader sorts to ascending Pa)


def test_resolve_var_bidirectional():
    """A long-name request finds a short store var AND a short request finds a long
    store var (robust to either ERA5 naming convention)."""
    # exact match
    assert resolve_var({"temperature"}, "temperature") == "temperature"
    # long request → short store var
    assert resolve_var({"t", "u"}, "temperature") == "t"
    # short request → long store var (the bidirectional capability)
    assert resolve_var({"temperature", "u_component_of_wind"}, "t") == "temperature"
    # short request → short store var
    assert resolve_var({"t"}, "t") == "t"
    # genuinely absent → None
    assert resolve_var({"u", "v"}, "temperature") is None
    assert resolve_var({"foo"}, "bar") is None


def _synthetic_era5(names="long", *, drop=(), extra2d=None):
    """A tiny in-memory ERA5-like dataset with the requested naming convention.

    ``extra2d``: {long_name: scalar-or-(nlat,nlon)-array} appended as extra
    (time, lat, lon) surface variables — used to exercise the sst fallback
    chain (sea_surface_temperature / 2m_temperature).
    """
    import xarray as xr

    nlat, nlon, nlev = 5, 6, 3
    long_to_short = {
        "temperature": "t", "u_component_of_wind": "u",
        "v_component_of_wind": "v", "specific_humidity": "q",
        "surface_pressure": "sp", "skin_temperature": "skt",
        "sea_surface_temperature": "sst", "2m_temperature": "t2m",
        "geopotential_at_surface": "z_sfc",
    }
    # temperature VARIES by level (300/250/200 K at 1000/500/100 hPa) so the loader's
    # descending→ascending-pressure reversal is genuinely exercised (Codex iter 106);
    # u VARIES by LATITUDE (10 + 0.5·lat/90, linear) so the horizontal regrid's lat
    # interpolation is genuinely exercised (Codex iter 107 — a lat/lon axis swap would
    # otherwise pass on a uniform field); v/q level+space-constant for simple checks.
    t_profile = np.array([300.0, 250.0, 200.0], dtype=np.float32)
    # GLOBAL coverage (lat 90→-90 descending, the ERA5 convention; lon 0..300) so a
    # model grid (cell-centred, ⊂ this range) regrids by INTERPOLATION, not edge
    # extrapolation — exercised by the load→regrid→reference integration test below.
    lat_deg = np.linspace(90.0, -90.0, nlat)
    u_lat = (10.0 + 0.5 * (lat_deg / 90.0)).astype(np.float32)   # (nlat,), linear in lat
    vals3 = {"temperature": None, "u_component_of_wind": None,
             "v_component_of_wind": 2.0, "specific_humidity": 5e-3}
    vals2 = {"surface_pressure": 1.0e5, "skin_temperature": 290.0,
             "geopotential_at_surface": 0.0}
    coords = {"time": [0], "level": list(_LEVELS),
              "lat": lat_deg, "lon": np.linspace(0.0, 300.0, nlon)}

    def _key(long):
        return long if names == "long" else long_to_short[long]

    data = {}
    for long, val in vals3.items():
        if long in drop:
            continue
        if long == "temperature":
            arr = np.broadcast_to(
                t_profile[None, :, None, None], (1, nlev, nlat, nlon)).astype(np.float32)
        elif long == "u_component_of_wind":
            arr = np.broadcast_to(
                u_lat[None, None, :, None], (1, nlev, nlat, nlon)).astype(np.float32)
        else:
            arr = np.full((1, nlev, nlat, nlon), val, dtype=np.float32)
        data[_key(long)] = (("time", "level", "lat", "lon"), arr)
    for long, val in vals2.items():
        if long in drop:
            continue
        data[_key(long)] = (("time", "lat", "lon"),
                            np.full((1, nlat, nlon), val, dtype=np.float32))
    for long, val in (extra2d or {}).items():
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((nlat, nlon), float(arr), dtype=np.float32)
        data[_key(long)] = (("time", "lat", "lon"), arr[None, :, :])
    return xr.Dataset(data, coords=coords)


def _config():
    return TrainingERA5Config(zarr_store="dummy", levels=_LEVELS)


def test_load_era5_slice_long_names(monkeypatch):
    """A long-name store loads all required fields with the right shapes/values."""
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _synthetic_era5("long"))
    sl = load_era5_slice(_config(), 0)
    assert sl.T.shape == (5, 6, 3) and sl.p_s.shape == (5, 6)
    # levels are reversed descending(1000,500,100 hPa) → ascending Pa, and T rides
    # with them: 300/250/200 K at 1000/500/100 hPa → [200,250,300] ascending-P.
    np.testing.assert_allclose(sl.plev_Pa, [10000.0, 50000.0, 100000.0])
    np.testing.assert_allclose(sl.T[0, 0, :], [200.0, 250.0, 300.0])
    np.testing.assert_allclose(sl.u[0, 0, :], 10.5)   # lat=+90 row: 10 + 0.5·(90/90)
    np.testing.assert_allclose(sl.u[2, 0, :], 10.0)   # lat=0 row
    np.testing.assert_allclose(sl.q, 5e-3, rtol=1e-5)
    np.testing.assert_allclose(sl.p_s, 1.0e5)


def test_load_era5_slice_out_of_range_time_idx_gives_clear_error(monkeypatch):
    """A typo'd ``--era5-time-idx`` / held-out index (the runbook directs the operator
    to pick one) raises a CAMPAIGN-specific ``IndexError`` naming the store's actual
    time count — not xarray's generic 'index N is out of bounds for axis 0'.  Negative
    indices keep xarray semantics (in range ⇒ accepted); only genuinely out-of-range
    indices raise.  The synthetic store has 1 time, so 0 / −1 are valid; 1 / −2 are not."""
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _synthetic_era5("long"))

    # In range (including the negative wrap to the single time) → loads, no IndexError.
    assert load_era5_slice(_config(), 0).T.shape == (5, 6, 3)
    assert load_era5_slice(_config(), -1).T.shape == (5, 6, 3)
    # Out of range → clear IndexError naming the store's time count (here 1).
    with pytest.raises(IndexError, match="out of range.*ERA5 store has 1 time"):
        load_era5_slice(_config(), 1)
    with pytest.raises(IndexError, match="out of range"):
        load_era5_slice(_config(), -2)


def test_load_era5_slice_nonmonotonic_levels_pair_data_with_correct_pressure(monkeypatch):
    """A NON-monotonic ``config.levels`` must still pair each level's data with the
    right pressure.  ``plev_Pa = np.sort(plev_hPa)`` robustly sorts the COORDINATE, but
    the old data reorder was a bare ``[::-1]`` flip keyed on ``plev_hPa[0] > [-1]`` —
    correct ONLY for a monotonic list.  config.levels is NOT validated monotonic, so a
    scrambled list (here ``[500, 1000, 100]`` hPa) would flip the data to ``[200, 300,
    250]`` while the sorted plev_Pa is ``[100, 500, 1000]`` hPa → the 500/1000 hPa data
    SWAPPED onto the wrong pressures, silently corrupting the vertical interp.  The
    argsort reorder pairs them correctly: T = [200, 250, 300] at ascending pressure.
    (Non-vacuous: the pre-fix flip fails this; mutation-checked separately.)"""
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _synthetic_era5("long"))
    # store levels are (1000, 500, 100) hPa with T = (300, 250, 200) K; request them
    # SCRAMBLED so the flip-vs-argsort distinction bites.
    cfg = TrainingERA5Config(zarr_store="dummy", levels=(500.0, 1000.0, 100.0))
    sl = load_era5_slice(cfg, 0)
    np.testing.assert_allclose(sl.plev_Pa, [10000.0, 50000.0, 100000.0])   # 100,500,1000 hPa
    # T rides with the SORTED pressure: 200 K @100 hPa, 250 K @500 hPa, 300 K @1000 hPa.
    np.testing.assert_allclose(sl.T[0, 0, :], [200.0, 250.0, 300.0])


def test_load_era5_slice_short_names_via_bidirectional_resolve(monkeypatch):
    """A SHORT-name store (t/u/v/q/sp) loads via the bidirectional resolver even
    though the loader requests the LONG names."""
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _synthetic_era5("short"))
    sl = load_era5_slice(_config(), 0)
    np.testing.assert_allclose(sl.T[0, 0, :], [200.0, 250.0, 300.0])   # ascending-P
    np.testing.assert_allclose(sl.v, 2.0)


def test_load_era5_slice_missing_required_raises(monkeypatch):
    """A missing REQUIRED variable (temperature) RAISES instead of silently loading
    zeros — the catastrophe this fix closes (iter 106)."""
    monkeypatch.setattr(
        e2s, "open_era5_zarr",
        lambda store: _synthetic_era5("long", drop=("temperature",)))
    with pytest.raises(ValueError, match="REQUIRED.*temperature.*not found"):
        load_era5_slice(_config(), 0)


# --- sst fallback chain (#797 bug 7: WB2 6h zarr has NO skin_temperature; the
# ---   old unconditional zero-fill silently forced 0 K SST on the WB trainer) ---

def _no_skt(extra2d):
    return _synthetic_era5("long", drop=("skin_temperature",), extra2d=extra2d)


def test_sst_falls_back_to_sea_surface_gap_filled_with_t2m(monkeypatch):
    """No skin_temperature: sea_surface_temperature (NaN over land) is gap-filled
    with 2m_temperature — the WB2-store shape that hit the 0 K zero-fill."""
    sst = np.full((5, 6), 290.0, dtype=np.float32)
    sst[0, 0] = np.nan                                     # a 'land' cell
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _no_skt(
        {"sea_surface_temperature": sst, "2m_temperature": 280.0}))
    sl = load_era5_slice(_config(), 0)
    np.testing.assert_allclose(sl.sst[0, 0], 280.0)        # land <- t2m
    np.testing.assert_allclose(sl.sst[1, 1], 290.0)        # ocean <- sst
    assert np.all(np.isfinite(sl.sst))


def test_sst_sea_surface_only_fills_nan_with_finite_mean(monkeypatch):
    """sea_surface_temperature without 2m_temperature: NaN cells take the finite
    mean instead of leaking NaN (or 0 K) into the forcing."""
    sst = np.full((5, 6), 290.0, dtype=np.float32)
    sst[0, 0] = np.nan
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _no_skt(
        {"sea_surface_temperature": sst}))
    sl = load_era5_slice(_config(), 0)
    np.testing.assert_allclose(sl.sst[0, 0], 290.0)
    assert np.all(np.isfinite(sl.sst))


def test_sst_falls_back_to_t2m_alone(monkeypatch):
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _no_skt(
        {"2m_temperature": 281.5}))
    sl = load_era5_slice(_config(), 0)
    np.testing.assert_allclose(sl.sst, 281.5)


def test_sst_zero_fill_is_last_resort_and_warns(monkeypatch, caplog):
    """With NO surface-temperature variable at all, the legacy zero-fill remains
    (idealized ICs must still load) but is no longer silent."""
    import logging
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _no_skt(None))
    with caplog.at_level(logging.WARNING, logger=e2s.logger.name):
        sl = load_era5_slice(_config(), 0)
    np.testing.assert_allclose(sl.sst, 0.0)
    assert any("sst zero-filled" in r.message for r in caplog.records)


def test_sst_skin_temperature_still_wins_over_fallbacks(monkeypatch):
    """A store WITH skin_temperature is untouched by the new chain."""
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _synthetic_era5(
        "long", extra2d={"sea_surface_temperature": 250.0, "2m_temperature": 251.0}))
    sl = load_era5_slice(_config(), 0)
    np.testing.assert_allclose(sl.sst, 290.0)              # vals2 skin_temperature


# --- surface geopotential (phis): loud zero-fill + dimension-checked 'z' alias
# --- (T1: silent flat-topography fallback — real ERA5 p_s (~600 hPa over Tibet)
# --- with phis=0 is a grossly non-hydrostatic IC; the zero-fill must warn) ---


def test_phis_missing_warns_loudly_and_zero_fills(monkeypatch, caplog):
    """A store with NO surface geopotential keeps the zero-fill (idealized ICs
    still load) but warns LOUDLY, same mechanism/level as the sst zero-fill."""
    import logging
    monkeypatch.setattr(
        e2s, "open_era5_zarr",
        lambda store: _synthetic_era5("long", drop=("geopotential_at_surface",)))
    with caplog.at_level(logging.WARNING, logger=e2s.logger.name):
        sl = load_era5_slice(_config(), 0)
    np.testing.assert_allclose(sl.phis, 0.0)
    assert any("phis zero-filled" in r.message for r in caplog.records)
    assert any("non-hydrostatic" in r.message for r in caplog.records)


def test_phis_resolves_2d_short_z(monkeypatch, caplog):
    """An ERA5-style invariant store carrying surface geopotential under the
    bare short name 'z' (2-D, no level dim) resolves to phis — no zero-fill,
    no warning."""
    import logging
    ds = _synthetic_era5("long", drop=("geopotential_at_surface",))
    nlat, nlon = 5, 6
    ds = ds.assign(z=(("time", "lat", "lon"),
                      np.full((1, nlat, nlon), 123.0, dtype=np.float32)))
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: ds)
    with caplog.at_level(logging.WARNING, logger=e2s.logger.name):
        sl = load_era5_slice(_config(), 0)
    np.testing.assert_allclose(sl.phis, 123.0)
    assert not any("phis zero-filled" in r.message for r in caplog.records)


def test_phis_does_not_resolve_3d_z(monkeypatch, caplog):
    """A 3-D 'z' (the pressure-level geopotential, same GRIB short name) must
    NOT be mistaken for phis: zero-fill + loud warning instead."""
    import logging
    ds = _synthetic_era5("long", drop=("geopotential_at_surface",))
    nlat, nlon, nlev = 5, 6, 3
    ds = ds.assign(z=(("time", "level", "lat", "lon"),
                      np.full((1, nlev, nlat, nlon), 9.8e4, dtype=np.float32)))
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: ds)
    with caplog.at_level(logging.WARNING, logger=e2s.logger.name):
        sl = load_era5_slice(_config(), 0)
    np.testing.assert_allclose(sl.phis, 0.0)
    assert any("phis zero-filled" in r.message for r in caplog.records)


def test_resolve_var_z_surface_geopotential_dimension_gate():
    """resolve_var treats 'z' as surface geopotential ONLY when it has no
    level dimension; a dims-less container (plain set) safely gives None."""
    import xarray as xr
    ds2 = xr.Dataset({"z": (("lat", "lon"), np.zeros((2, 3)))},
                     coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]})
    assert resolve_var(ds2, "geopotential_at_surface") == "z"
    assert resolve_var(ds2, "z_sfc") == "z"
    ds3 = xr.Dataset(
        {"z": (("level", "lat", "lon"), np.zeros((2, 2, 3)))},
        coords={"level": [1000.0, 500.0], "lat": [0.0, 1.0],
                "lon": [0.0, 1.0, 2.0]})
    assert resolve_var(ds3, "geopotential_at_surface") is None
    # a plain set has no dims metadata → no false resolution
    assert resolve_var({"z"}, "geopotential_at_surface") is None
    # an explicit 'z_sfc' store variable still wins as before
    assert resolve_var({"z_sfc"}, "geopotential_at_surface") == "z_sfc"


def _synthetic_era5_multitime():
    """A 2-time ERA5-like store where p_s + T VARY by time (so the mean is non-trivial),
    sst has a NaN 'land' cell every time, and phis is STATIC."""
    import xarray as xr

    nt, nlat, nlon, nlev = 2, 5, 6, 3
    t_prof = np.array([300.0, 250.0, 200.0], dtype=np.float32)        # per level (hPa order)
    temp = np.stack([                                                # +0 at t0, +20 at t1
        np.broadcast_to((t_prof + off)[:, None, None], (nlev, nlat, nlon))
        for off in (0.0, 20.0)]).astype(np.float32)
    ps = np.stack([np.full((nlat, nlon), v, dtype=np.float32)         # 1e5, 1.1e5
                   for v in (1.0e5, 1.1e5)])
    sst = np.full((nt, nlat, nlon), 290.0, dtype=np.float32)
    sst[:, 0, 0] = np.nan                                            # consistent 'land' NaN
    phis = np.full((nt, nlat, nlon), 100.0, dtype=np.float32)        # STATIC
    u = np.full((nt, nlev, nlat, nlon), 10.0, dtype=np.float32)
    v = np.full((nt, nlev, nlat, nlon), 2.0, dtype=np.float32)
    q = np.full((nt, nlev, nlat, nlon), 5e-3, dtype=np.float32)
    coords = {"time": list(range(nt)), "level": list(_LEVELS),
              "lat": np.linspace(90.0, -90.0, nlat), "lon": np.linspace(0.0, 300.0, nlon)}
    return xr.Dataset(
        {"temperature": (("time", "level", "lat", "lon"), temp),
         "u_component_of_wind": (("time", "level", "lat", "lon"), u),
         "v_component_of_wind": (("time", "level", "lat", "lon"), v),
         "specific_humidity": (("time", "level", "lat", "lon"), q),
         "surface_pressure": (("time", "lat", "lon"), ps),
         "skin_temperature": (("time", "lat", "lon"), sst),
         "geopotential_at_surface": (("time", "lat", "lon"), phis)},
        coords=coords)


def test_load_era5_time_mean(monkeypatch):
    """load_era5_time_mean averages each field over the times (the climatology); coords
    + static phis unchanged; NaN propagates (land sst); dtype preserved; a single index
    is byte-identical to load_era5_slice; empty raises (Codex design conditions)."""
    from legoesm.training.era5_to_state import load_era5_time_mean

    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _synthetic_era5_multitime())
    cfg = _config()
    mean = load_era5_time_mean(cfg, [0, 1])
    s0 = load_era5_slice(cfg, 0)
    # p_s time-mean = (1e5 + 1.1e5)/2; T mean = ascending-P [200,250,300] + (0+20)/2.
    np.testing.assert_allclose(mean.p_s, 1.05e5, rtol=1e-5)
    np.testing.assert_allclose(mean.T[0, 0, :], [210.0, 260.0, 310.0])
    # coords + static phis unchanged.
    np.testing.assert_array_equal(mean.lat, s0.lat)
    np.testing.assert_allclose(mean.plev_Pa, s0.plev_Pa)
    np.testing.assert_allclose(mean.phis, s0.phis)            # static ⇒ mean is a no-op
    # NaN-propagating land sst; ocean cell averaged.
    assert np.isnan(mean.sst[0, 0])
    np.testing.assert_allclose(mean.sst[1, 1], 290.0)
    # dtype PRESERVED (np.mean upcasts float32→float64 without the cast).
    assert mean.T.dtype == np.float32 and mean.p_s.dtype == np.float32
    # SINGLE index ⇒ load_era5_slice unchanged (byte-identical, the old behaviour).
    one = load_era5_time_mean(cfg, [0])
    np.testing.assert_array_equal(one.T, s0.T)
    np.testing.assert_array_equal(one.p_s, s0.p_s)
    # EMPTY ⇒ raise.
    with pytest.raises(ValueError, match="non-empty"):
        load_era5_time_mean(cfg, [])


def test_load_era5_time_mean_out_of_range_fails_loud(monkeypatch):
    """An out-of-range time index (the window exceeds the store's times) FAILS LOUD with
    a clear, actionable error naming the CLI flags — NOT a cryptic xarray IndexError
    mid-load on a multi-day HPC launch."""
    from legoesm.training.era5_to_state import load_era5_time_mean

    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _synthetic_era5_multitime())
    cfg = _config()                                  # the synthetic store has only 2 times
    with pytest.raises(ValueError, match="out of range.*--era5-n-times"):
        load_era5_time_mean(cfg, [0, 5])             # index 5 exceeds the 2-time store
    # The single-index fast path also fails loud (not a cryptic IndexError).
    with pytest.raises(ValueError, match="out of range"):
        load_era5_time_mean(cfg, [9])


def test_load_era5_slice_missing_required_surface_pressure_raises(monkeypatch):
    monkeypatch.setattr(
        e2s, "open_era5_zarr",
        lambda store: _synthetic_era5("long", drop=("surface_pressure",)))
    with pytest.raises(ValueError, match="REQUIRED.*surface_pressure"):
        load_era5_slice(_config(), 0)


def test_load_era5_slice_reports_all_missing_required_at_once(monkeypatch):
    """The upfront preflight (iter 306) lists EVERY missing required variable in ONE error
    (not one-per-failed-load), so an operator preparing a real-ERA5 zarr fixes the whole
    variable-naming pass at once. Two dropped required fields → a single raise naming BOTH
    (with aliases). Non-vacuous: before the preflight, the first _get_3d raised on
    'temperature' alone and 'specific_humidity' never appeared in the message."""
    monkeypatch.setattr(
        e2s, "open_era5_zarr",
        lambda store: _synthetic_era5("long",
                                      drop=("temperature", "specific_humidity")))
    with pytest.raises(ValueError) as exc:
        load_era5_slice(_config(), 0)
    msg = str(exc.value)
    assert "temperature" in msg and "specific_humidity" in msg   # BOTH listed at once
    assert "REQUIRED" in msg and "not found" in msg              # the documented contract


def test_load_era5_slice_missing_optional_zero_fills(monkeypatch):
    """An OPTIONAL surface field (skin_temperature) absent → zero-filled, not raised
    (an IC missing it still loads)."""
    monkeypatch.setattr(
        e2s, "open_era5_zarr",
        lambda store: _synthetic_era5("long", drop=("skin_temperature",)))
    sl = load_era5_slice(_config(), 0)
    np.testing.assert_allclose(sl.sst, 0.0)          # optional → zeros, no raise
    np.testing.assert_allclose(sl.T[0, 0, :], [200.0, 250.0, 300.0])  # required loaded


def test_era5_load_regrid_to_reference_column_state_integration(monkeypatch):
    """END-TO-END input chain (the path the empirical run consumes, currently bypassed
    by the compare test's monkeypatch): REAL load_era5_slice → REAL era5_to_latlon_carry
    (regrid + log-p interp, q as specific humidity) → column_state_from_carry → a PHYSICALLY
    VALID reference ColumnState on the model grid+sigma (iter 107). This is exactly the
    integration gap that hid iter 106's silent-zeros bug."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.compare_reanalysis import (
        column_state_from_carry,
        validate_reference_physical,
    )
    from legoesm.training.era5_to_state import era5_to_latlon_carry

    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _synthetic_era5("long"))
    era5 = load_era5_slice(_config(), 0)          # REAL load

    # model lats [-60, 0, 60] fall BETWEEN the ERA5 lat nodes (±45/±90) so the bilinear
    # lat interpolation is genuinely exercised (lon 0/90/180/270 ⊂ ERA5 [0,300]).
    nlat, nlon, nlev = 3, 4, 5
    grid = create_latlon_grid(nlat, nlon)
    sigma = create_sigma_coordinate(nlev)
    carry = era5_to_latlon_carry(era5, grid, sigma,   # REAL regrid + interp + carry
                                 target_phis=era5_terrain_product(era5, grid))
    ref = column_state_from_carry(carry)

    # shapes land on the MODEL grid + sigma (not the ERA5 grid/levels).
    assert ref.T.shape == (nlat, nlon, nlev)
    assert ref.q_v.shape == (nlat, nlon, nlev)
    assert ref.p_s.shape == (nlat, nlon)
    # the chain produced a PHYSICALLY-VALID reference (T∈[150,350] K, q≥0, p_s in Pa,
    # |wind|<200): the iter-99 guard passes, proving regrid+interp+q-conversion are sane.
    validate_reference_physical(ref, name="regridded ERA5")
    assert 150.0 < float(jnp.min(ref.T)) and float(jnp.max(ref.T)) < 350.0
    # the LAT interpolation is exact for the linear u field: u(model_lat) = 10 +
    # 0.5·(lat_deg/90) at EACH model latitude (an axis swap or mis-aligned regrid fails).
    model_lat_deg = np.rad2deg(np.asarray(grid.lat))           # [-60, 0, 60]
    u = np.asarray(ref.u)
    for k, ld in enumerate(model_lat_deg):
        np.testing.assert_allclose(u[k], 10.0 + 0.5 * (ld / 90.0), atol=2e-3)
    # a non-flat u profile across latitude (proves it is NOT a constant-fill).
    assert float(u[0].mean()) < float(u[1].mean()) < float(u[2].mean())
    # q is loaded AS IS -- SPECIFIC humidity, the tracer convention on every
    # lane (2026-09-28); the former r = q/(1-q) = 5.025e-3 fails at 1e-4.
    np.testing.assert_allclose(np.asarray(ref.q_v), 5e-3, rtol=1e-4)


def test_era5_to_latlon_carry_hybrid_over_terrain_is_physical(monkeypatch):
    """END-TO-END smoke (iters 339/345): the full ERA5→carry chain runs with a HYBRID
    coordinate over a TERRAIN column (p_s != p_ref) and produces a PHYSICALLY VALID reference
    — the @slow campaign tests use a uniform p_s = 1e5 = p_ref (where hybrid == pure-sigma), so
    this is the only coverage of the hybrid reference regrid where it actually differs from
    pure-sigma.  (The iter-339 `p_full` fix itself is locked NON-vacuously by the
    interp_pressure_to_sigma unit test; this confirms the carry wiring stays finite + in-range
    when the hybrid pressures genuinely diverge from σ·p_s.)"""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import make_hybrid_levels
    from legoesm.training.compare_reanalysis import (
        column_state_from_carry,
        validate_reference_physical,
    )
    from legoesm.training.era5_to_state import era5_to_latlon_carry

    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _synthetic_era5("long"))
    era5 = load_era5_slice(_config(), 0)
    era5 = era5._replace(p_s=np.full_like(np.asarray(era5.p_s), 7.0e4))   # 700-hPa terrain
    grid = create_latlon_grid(3, 4)
    ref = column_state_from_carry(
        era5_to_latlon_carry(era5, grid, make_hybrid_levels(5, p_top_Pa=100.0),
                             target_phis=era5_terrain_product(era5, grid)))
    validate_reference_physical(ref, name="hybrid-over-terrain ERA5 reference")  # T/q/p_s in range
    assert bool(np.all(np.isfinite(np.asarray(ref.T))))
    assert 150.0 < float(np.min(ref.T)) and float(np.max(ref.T)) < 350.0


def test_reference_columnstate_invariant_to_config_levels_order(monkeypatch):
    """COMPOSITION-level lock on the iter-327 fix: the regridded reference ColumnState must
    be INVARIANT to the ORDER of ``config.levels``.  Both a monotonic and a SCRAMBLED levels
    list sort to the same ascending ``plev_Pa``; the loader pairs the data with it (iter 327
    argsort, not a monotonic-only ``[::-1]`` flip), so the FULL chain (load → regrid → log-p
    vertical interp → ColumnState) must yield the IDENTICAL reference.  The log-p interp
    depends on data↔pressure pairing being correct, so this catches a mispairing regression
    that the unit test alone could miss.  Non-vacuous: the pre-iter-327 flip would mispair the
    500/1000 hPa data for the scrambled list → a DIFFERENT (corrupted) reference."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.compare_reanalysis import column_state_from_carry
    from legoesm.training.era5_to_state import era5_to_latlon_carry

    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _synthetic_era5("long"))
    grid = create_latlon_grid(3, 4)
    sigma = create_sigma_coordinate(5)

    def _ref(levels):
        era5 = load_era5_slice(TrainingERA5Config(zarr_store="dummy", levels=levels), 0)
        return column_state_from_carry(era5_to_latlon_carry(
            era5, grid, sigma, target_phis=era5_terrain_product(era5, grid)))

    ref_mono = _ref((1000.0, 500.0, 100.0))         # monotonic descending (store order)
    ref_scrambled = _ref((500.0, 1000.0, 100.0))    # non-monotonic permutation
    for name in ("T", "q_v", "u", "v", "p_s"):
        np.testing.assert_allclose(
            np.asarray(getattr(ref_scrambled, name)),
            np.asarray(getattr(ref_mono, name)), rtol=1e-6,
            err_msg=f"reference {name} changed with config.levels ORDER — a data↔pressure "
                    "mispairing in the ingest/interp chain (cf. iter 327).")
