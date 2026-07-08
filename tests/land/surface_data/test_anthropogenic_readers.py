"""Tests for the HYDE / Pongratz / KK10 anthropogenic readers + generic reader."""

from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from legoesm.land.surface_data.sources.anthropogenic import (
    AnthropogenicSourceConfig,
    read_anthropogenic_states,
)
from legoesm.land.surface_data.sources.hyde import read_hyde
from legoesm.land.surface_data.sources.kk10 import read_kk10
from legoesm.land.surface_data.sources.pongratz import read_pongratz

_LAT = np.array([-45.0, 45.0])
_LON = np.array([0.0, 180.0])
_YEARS = np.array([1900, 1950, 2000])


def _da(vals, *, time=True):
    dims = ("time", "lat", "lon") if time else ("lat", "lon")
    coords = {"lat": _LAT, "lon": _LON}
    if time:
        coords = {"time": _YEARS.astype(float), **coords}
    return xr.DataArray(np.asarray(vals, dtype=float), dims=dims, coords=coords)


# --- generic reader -------------------------------------------------------

def test_generic_reader_groups_and_windows():
    ds = xr.Dataset({
        "cropland": _da(np.full((3, 2, 2), 0.3)),
        "pasture": _da(np.full((3, 2, 2), 0.1)),
        "rangeland": _da(np.full((3, 2, 2), 0.05)),
        "built_up": _da(np.full((3, 2, 2), 0.02)),
    })
    cfg = AnthropogenicSourceConfig(
        crop_vars=("cropland",), pasture_vars=("pasture", "rangeland"),
        urban_vars=("built_up",))
    out = read_anthropogenic_states(dataset=ds, config=cfg, years=(1940, 1960))
    assert out["years"].tolist() == [1950]                 # window selects 1 year
    np.testing.assert_allclose(out["crop"], 0.3)
    np.testing.assert_allclose(out["pasture"], 0.15)        # 0.1 + 0.05 summed
    np.testing.assert_allclose(out["urban"], 0.02)


def test_generic_reader_area_normalizes_km2():
    # cropland stored as km²; garea is per-cell km² -> fraction = 50/200 = 0.25.
    ds = xr.Dataset({
        "cropland": _da(np.full((3, 2, 2), 50.0)),
        "garea": _da(np.full((2, 2), 200.0), time=False),
    })
    cfg = AnthropogenicSourceConfig(crop_vars=("cropland",), area_var="garea")
    out = read_anthropogenic_states(dataset=ds, config=cfg)
    np.testing.assert_allclose(out["crop"], 0.25)
    np.testing.assert_allclose(out["pasture"], 0.0)         # empty group -> zeros


def test_generic_reader_missing_var_raises():
    ds = xr.Dataset({"cropland": _da(np.zeros((3, 2, 2)))})
    cfg = AnthropogenicSourceConfig(crop_vars=("cropland",), pasture_vars=("nope",))
    with pytest.raises(ValueError, match="missing variables"):
        read_anthropogenic_states(dataset=ds, config=cfg)


def test_generic_reader_window_outside_raises():
    ds = xr.Dataset({"cropland": _da(np.zeros((3, 2, 2)))})
    cfg = AnthropogenicSourceConfig(crop_vars=("cropland",))
    with pytest.raises(ValueError, match="no years in requested window"):
        read_anthropogenic_states(dataset=ds, config=cfg, years=(1800, 1850))


def test_generic_reader_transposes_by_name():
    # File stored (time, lon, lat): reader must transpose to (time, lat, lon).
    swapped = xr.DataArray(
        np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2),
        dims=("time", "lon", "lat"),
        coords={"time": _YEARS.astype(float), "lon": _LON, "lat": _LAT})
    ds = xr.Dataset({"cropland": swapped})
    cfg = AnthropogenicSourceConfig(crop_vars=("cropland",))
    out = read_anthropogenic_states(dataset=ds, config=cfg)
    assert out["crop"].shape == (3, _LAT.size, _LON.size)


# --- HYDE -----------------------------------------------------------------

def test_read_hyde_sums_rangeland_into_pasture_and_normalizes():
    ds = xr.Dataset({
        "cropland": _da(np.full((3, 2, 2), 40.0)),
        "pasture": _da(np.full((3, 2, 2), 20.0)),
        "rangeland": _da(np.full((3, 2, 2), 10.0)),
        "built_up": _da(np.full((3, 2, 2), 4.0)),
        "garea": _da(np.full((2, 2), 100.0), time=False),
    })
    out = read_hyde(dataset=ds)
    np.testing.assert_allclose(out["crop"], 0.40)
    np.testing.assert_allclose(out["pasture"], 0.30)        # (20+10)/100
    np.testing.assert_allclose(out["urban"], 0.04)


# --- Pongratz -------------------------------------------------------------

def test_read_pongratz_fractions_no_urban():
    ds = xr.Dataset({
        "crop": _da(np.full((3, 2, 2), 0.2)),
        "pasture": _da(np.full((3, 2, 2), 0.15)),
    })
    out = read_pongratz(dataset=ds)
    np.testing.assert_allclose(out["crop"], 0.2)
    np.testing.assert_allclose(out["pasture"], 0.15)
    np.testing.assert_allclose(out["urban"], 0.0)


# --- KK10 -----------------------------------------------------------------

def test_read_kk10_splits_total_by_crop_share():
    ds = xr.Dataset({"land_use": _da(np.full((3, 2, 2), 0.4))})
    out = read_kk10(dataset=ds, crop_share=0.75)
    np.testing.assert_allclose(out["crop"], 0.3)            # 0.4 * 0.75
    np.testing.assert_allclose(out["pasture"], 0.1)         # 0.4 * 0.25
    np.testing.assert_allclose(out["urban"], 0.0)


def test_read_kk10_default_half_split():
    ds = xr.Dataset({"land_use": _da(np.full((3, 2, 2), 0.4))})
    out = read_kk10(dataset=ds)
    np.testing.assert_allclose(out["crop"], 0.2)
    np.testing.assert_allclose(out["pasture"], 0.2)


def test_read_kk10_bad_crop_share_raises():
    ds = xr.Dataset({"land_use": _da(np.zeros((3, 2, 2)))})
    with pytest.raises(ValueError, match="crop_share"):
        read_kk10(dataset=ds, crop_share=1.5)


def test_generic_reader_sorts_nonmonotonic_years():
    # A file whose time axis is out of order must come back sorted ascending, so
    # the runtime interp_annual (jnp.interp) blends the right slices.
    times = np.array([2000.0, 1900.0, 1950.0])
    vals = np.stack([np.full((2, 2), 0.3),   # tagged to year 2000
                     np.full((2, 2), 0.1),   # 1900
                     np.full((2, 2), 0.2)])   # 1950
    da = xr.DataArray(vals, dims=("time", "lat", "lon"),
                      coords={"time": times, "lat": _LAT, "lon": _LON})
    ds = xr.Dataset({"cropland": da})
    cfg = AnthropogenicSourceConfig(crop_vars=("cropland",))
    out = read_anthropogenic_states(dataset=ds, config=cfg)
    assert out["years"].tolist() == [1900, 1950, 2000]     # ascending
    np.testing.assert_allclose(out["crop"][:, 0, 0], [0.1, 0.2, 0.3])   # slices follow
