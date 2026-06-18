"""Tests for the ERA5 variable resolver + slice loader (the real-ERA5 INPUT path).

``load_era5_slice`` feeds the whole compare-reanalysis pipeline; a missing/
misnamed REQUIRED variable used to SILENTLY load as zeros (corrupting the bias so
the loop "corrects" garbage).  These lock: bidirectional name resolution (long
``"temperature"`` ↔ short ``"t"``) and the fail-fast on a missing required field.
"""

from __future__ import annotations

import legoesm.training.era5_to_state as e2s
import numpy as np
import pytest
from legoesm.training.era5_to_state import (
    TrainingERA5Config,
    load_era5_slice,
    resolve_var,
)

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


def _synthetic_era5(names="long", *, drop=()):
    """A tiny in-memory ERA5-like dataset with the requested naming convention."""
    import xarray as xr

    nlat, nlon, nlev = 4, 5, 3
    long_to_short = {
        "temperature": "t", "u_component_of_wind": "u",
        "v_component_of_wind": "v", "specific_humidity": "q",
        "surface_pressure": "sp", "skin_temperature": "skt",
        "geopotential_at_surface": "z_sfc",
    }
    # temperature VARIES by level (300/250/200 K at 1000/500/100 hPa) so the loader's
    # descending→ascending-pressure reversal is genuinely exercised (Codex iter 106);
    # the others are level-constant for simple value checks.
    t_profile = np.array([300.0, 250.0, 200.0], dtype=np.float32)
    vals3 = {"temperature": None, "u_component_of_wind": 10.0,
             "v_component_of_wind": 2.0, "specific_humidity": 5e-3}
    vals2 = {"surface_pressure": 1.0e5, "skin_temperature": 290.0,
             "geopotential_at_surface": 0.0}
    coords = {"time": [0], "level": list(_LEVELS),
              "lat": np.linspace(-60.0, 60.0, nlat),
              "lon": np.linspace(0.0, 300.0, nlon)}

    def _key(long):
        return long if names == "long" else long_to_short[long]

    data = {}
    for long, val in vals3.items():
        if long in drop:
            continue
        if long == "temperature":
            arr = np.broadcast_to(
                t_profile[None, :, None, None], (1, nlev, nlat, nlon)).astype(np.float32)
        else:
            arr = np.full((1, nlev, nlat, nlon), val, dtype=np.float32)
        data[_key(long)] = (("time", "level", "lat", "lon"), arr)
    for long, val in vals2.items():
        if long in drop:
            continue
        data[_key(long)] = (("time", "lat", "lon"),
                            np.full((1, nlat, nlon), val, dtype=np.float32))
    return xr.Dataset(data, coords=coords)


def _config():
    return TrainingERA5Config(zarr_store="dummy", levels=_LEVELS)


def test_load_era5_slice_long_names(monkeypatch):
    """A long-name store loads all required fields with the right shapes/values."""
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _synthetic_era5("long"))
    sl = load_era5_slice(_config(), 0)
    assert sl.T.shape == (4, 5, 3) and sl.p_s.shape == (4, 5)
    # levels are reversed descending(1000,500,100 hPa) → ascending Pa, and T rides
    # with them: 300/250/200 K at 1000/500/100 hPa → [200,250,300] ascending-P.
    np.testing.assert_allclose(sl.plev_Pa, [10000.0, 50000.0, 100000.0])
    np.testing.assert_allclose(sl.T[0, 0, :], [200.0, 250.0, 300.0])
    np.testing.assert_allclose(sl.u, 10.0)
    np.testing.assert_allclose(sl.q, 5e-3, rtol=1e-5)
    np.testing.assert_allclose(sl.p_s, 1.0e5)


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


def test_load_era5_slice_missing_required_surface_pressure_raises(monkeypatch):
    monkeypatch.setattr(
        e2s, "open_era5_zarr",
        lambda store: _synthetic_era5("long", drop=("surface_pressure",)))
    with pytest.raises(ValueError, match="REQUIRED.*surface_pressure"):
        load_era5_slice(_config(), 0)


def test_load_era5_slice_missing_optional_zero_fills(monkeypatch):
    """An OPTIONAL surface field (skin_temperature) absent → zero-filled, not raised
    (an IC missing it still loads)."""
    monkeypatch.setattr(
        e2s, "open_era5_zarr",
        lambda store: _synthetic_era5("long", drop=("skin_temperature",)))
    sl = load_era5_slice(_config(), 0)
    np.testing.assert_allclose(sl.sst, 0.0)          # optional → zeros, no raise
    np.testing.assert_allclose(sl.T[0, 0, :], [200.0, 250.0, 300.0])  # required loaded
