"""#895: the local ERA5 cache is scoped to the span of windows THIS call loads.
The scope must (a) prefer the explicit ``windows`` arg over ``config.windows``
(they can differ — else a caller's window is read against a cache built for the
wrong years), and (b) add +1 year of headroom so target snapshots that spill
into the following year are covered.

#985: the cache is also (c) ATOMIC + completeness-marked (an interrupted build
is rebuilt, never read as fill-value NaNs) and (d) WINDOW-scopeable (materialise
only the touched snapshots, addressable back by timestamp).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from legoesm.training import era5_to_state as e2s
from legoesm.training.era5_to_state import (
    TrainingERA5Config,
    _cache_is_complete,
    _read_cache_marker,
    ensure_local_cache,
)
from legoesm.training.neural_gcm_spectral import _training_year_range


def test_year_range_spans_windows_arg_with_target_headroom():
    wins = [(1979, 0, 5), (2000, 10, 5), (2014, 3, 5)]
    # min start year .. max start year + 1 (targets spill into the next year)
    assert _training_year_range(wins, SimpleNamespace(windows=None)) == (1979, 2015)


def test_windows_arg_overrides_config_windows():
    # The data actually read is the ARG, not config.windows -> cache the arg's span.
    cfg = SimpleNamespace(windows=[(1979, 0, 5)])
    assert _training_year_range([(2000, 0, 5)], cfg) == (2000, 2001)


def test_year_range_fallback_without_windows():
    cfg = SimpleNamespace(windows=None, start_year=2015, n_train_days=400)
    # 400 days spans 2 calendar years, +1 headroom for the target lead
    assert _training_year_range(None, cfg) == (2015, 2017)


# --------------------------------------------------------------------------- #
# #985: atomic + completeness-marked + window-scoped cache
# --------------------------------------------------------------------------- #

def _synth_era5(n_time: int) -> xr.Dataset:
    """A tiny WB2-shaped ERA5 store. Each snapshot j carries the SCALAR j in
    every field, so a cached snapshot can be checked against its source index."""
    time = np.datetime64("2015-01-01") + np.arange(n_time) * np.timedelta64(6, "h")
    lat = np.array([-45.0, 0.0, 45.0])
    lon = np.array([0.0, 120.0, 240.0])
    level = np.array([850.0, 500.0])
    j = np.arange(n_time, dtype=np.float32)
    d3 = j[:, None, None, None] * np.ones((n_time, len(level), len(lat), len(lon)), np.float32)
    d2 = j[:, None, None] * np.ones((n_time, len(lat), len(lon)), np.float32)
    return xr.Dataset(
        {
            "t": (("time", "level", "lat", "lon"), d3),
            "u": (("time", "level", "lat", "lon"), d3),
            "v": (("time", "level", "lat", "lon"), d3),
            "q": (("time", "level", "lat", "lon"), d3),
            "sp": (("time", "lat", "lon"), d2),
            "skt": (("time", "lat", "lon"), d2),
            "z_sfc": (("lat", "lon"), np.ones((len(lat), len(lon)), np.float32)),
        },
        coords={"time": time, "level": level, "lat": lat, "lon": lon},
    )


@pytest.fixture
def _patched_remote(monkeypatch):
    """Serve the synthetic store from both loader entry points, counting builds
    so a test can prove the cache is (or is not) rebuilt."""
    full = _synth_era5(40)
    calls = {"open": 0, "create": 0}

    def _open(_store):
        calls["open"] += 1
        return full

    def _create(cfg):
        calls["create"] += 1
        # full-year path caches only the pressure vars from create_era5_dataset
        return full[["t", "u", "v", "q"]]

    monkeypatch.setattr(e2s, "open_era5_zarr", _open)
    monkeypatch.setattr(e2s, "create_era5_dataset", _create)
    return full, calls


def test_marker_gates_completeness(tmp_path):
    store = tmp_path / "era5_training_cache.zarr"
    store.mkdir()
    # A bare .zarr dir with NO marker = an interrupted build -> NOT complete
    # (the old existence-only check read this as fill-value NaNs, #942/#985).
    assert _read_cache_marker(store) is None
    assert not _cache_is_complete(store, expected_n_time=10)
    (store / ".cache_complete.json").write_text('{"n_time": 10}')
    assert _cache_is_complete(store, expected_n_time=10)
    assert _cache_is_complete(store, expected_n_time=None)   # count unknown -> trust marker
    assert not _cache_is_complete(store, expected_n_time=7)  # count mismatch -> rebuild


def test_full_year_cache_atomic_and_idempotent(tmp_path, _patched_remote):
    full, calls = _patched_remote
    cfg = TrainingERA5Config()
    path = ensure_local_cache(cfg, tmp_path / "c", years=(2015, 2015))
    # Built, marked complete, and readable back.
    assert (path / ".cache_complete.json").is_file()
    assert _cache_is_complete(path, expected_n_time=None)
    assert xr.open_zarr(path).sizes["time"] == full.sizes["time"]
    # No leftover tmp build dir.
    assert not (tmp_path / "c" / "era5_training_cache.zarr.building").exists()
    built = calls["create"]
    # Second call is a no-op: it must NOT re-open/re-build the remote store.
    ensure_local_cache(cfg, tmp_path / "c", years=(2015, 2015))
    assert calls["create"] == built


def test_interrupted_build_is_rebuilt_not_read(tmp_path, _patched_remote):
    full, calls = _patched_remote
    cfg = TrainingERA5Config()
    cdir = tmp_path / "c"
    path = ensure_local_cache(cfg, cdir, years=(2015, 2015))
    built = calls["create"]
    # Simulate a torn write: the store dir survives but the marker is gone.
    (path / ".cache_complete.json").unlink()
    ensure_local_cache(cfg, cdir, years=(2015, 2015))
    assert calls["create"] == built + 1     # rebuilt, not silently reused


def test_window_scoped_cache_subsets_and_preserves_timestamps(tmp_path, _patched_remote):
    full, _calls = _patched_remote
    cfg = TrainingERA5Config()
    sel = [3, 4, 5, 17, 18, 30]     # scattered absolute indices (the touched snapshots)
    path = ensure_local_cache(
        cfg, tmp_path / "w", years=(2015, 2015), time_selection=sel,
    )
    cached = xr.open_zarr(path)
    # Exactly the requested snapshots, in order, with the REAL timestamps kept.
    assert cached.sizes["time"] == len(sel)
    np.testing.assert_array_equal(
        cached.time.values, full.time.values[sel]
    )
    # Each cached snapshot is byte-identical to the full store's at that index,
    # and is addressable BY TIMESTAMP (the reader's access pattern).
    for abs_idx in sel:
        t = full.time.values[abs_idx]
        assert float(cached["t"].sel(time=t).values.flat[0]) == float(abs_idx)
        assert float(cached["sp"].sel(time=t).values.flat[0]) == float(abs_idx)
    # Static field carried through; marker records the scoped count.
    assert "z_sfc" in cached
    assert _read_cache_marker(path)["n_time"] == len(sel)
