"""WB training: window-scoped local ERA5 cache under the deck's ``cache_dir``."""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from legoesm.training import scale_build as sb
from legoesm.training.era5_to_state import TrainingERA5Config


def _write_store(path, n_time=24, var_3d=("temperature",), var_2d=("surface_pressure",)):
    """Tiny 6-hourly store with random float32 values (and a NaN), on disk."""
    rng = np.random.default_rng(0)
    time = np.datetime64("2015-01-01") + np.arange(n_time) * np.timedelta64(6, "h")
    lat, lon = np.array([-45.0, 0.0, 45.0]), np.array([0.0, 120.0, 240.0])
    level = np.array([500.0, 850.0, 999.0])
    data = {}
    for v in var_3d:
        a = rng.standard_normal((n_time, 3, 3, 3)).astype(np.float32)
        a[0, 0, 0, 0] = np.nan
        data[v] = (("time", "level", "lat", "lon"), a, {"units": "K"})
    for v in var_2d:
        data[v] = (("time", "lat", "lon"),
                   rng.standard_normal((n_time, 3, 3)).astype(np.float32))
    data["geopotential_at_surface"] = (("lat", "lon"), np.ones((3, 3), np.float32))
    ds = xr.Dataset(data, coords={"time": time, "level": level, "lat": lat, "lon": lon})
    ds.to_zarr(str(path), mode="w")
    return ds


def _cfg(store, **kw):
    return TrainingERA5Config(
        zarr_store=str(store), pressure_variables=("temperature",),
        surface_variables=("surface_pressure",), levels=(500, 850), **kw)


# --- cache location ---------------------------------------------------------

def test_cache_root_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("LEGOESM_ERA5_LOCAL_CACHE", raising=False)
    assert sb.wb_era5_cache_root({}) is None
    assert sb.wb_era5_cache_root({"cache_dir": ".cache/x"}) == sb._REPO / ".cache/x"
    assert sb.wb_era5_cache_root({"cache_dir": str(tmp_path)}) == tmp_path
    monkeypatch.setenv("LEGOESM_ERA5_LOCAL_CACHE", "0")
    assert sb.wb_era5_cache_root({"cache_dir": str(tmp_path)}) is None


def test_cache_dir_is_now_consumed_not_metadata():
    assert "cache_dir" in sb._WB_CONSUMED_TOP_KEYS
    assert "cache_dir" not in sb._WB_METADATA_TOP_KEYS


# --- which store serves which variables ------------------------------------

def test_store_groups_follow_the_loader_store_choice():
    base = TrainingERA5Config(zarr_store="S", cloud_zarr="A", flux_zarr="A",
                              load_cloud_condensate=True)
    g = sb._era5_store_groups(base, base._replace(load_surface_fluxes=True))
    assert set(g) == {"S", "A"}
    assert "land_sea_mask" in g["S"][1]
    assert g["A"][0] == sb._ERA5_CACHE_CLOUD_VARS
    assert g["A"][1] == sb._ERA5_CACHE_FLUX_VARS + ("land_sea_mask",)
    # land fraction alone still caches the flux store's mask (codex review)
    g3 = sb._era5_store_groups(base, base._replace(load_land_frac=True))
    assert g3["A"][1] == ("land_sea_mask",)
    # flux_zarr == "" means "fluxes come from the state store" (codex review)
    g2 = sb._era5_store_groups(base, base._replace(load_surface_fluxes=True, flux_zarr=""))
    assert set(sb._ERA5_CACHE_FLUX_VARS) <= set(g2["S"][1])
    assert sb._ERA5_CACHE_FLUX_VARS[0] not in g2["A"][1]
    # nothing asked for -> state store only
    assert set(sb._era5_store_groups(base._replace(load_cloud_condensate=False), base)) == {"S"}


# --- building one store cache -----------------------------------------------

def test_store_cache_is_bit_identical_at_the_requested_times(tmp_path):
    src = _write_store(tmp_path / "src.zarr")
    remote = xr.open_zarr(str(tmp_path / "src.zarr"))
    stamps = remote.time.values[[1, 5, 6, 17]].astype("datetime64[ns]")
    path = sb._ensure_era5_store_cache(
        _cfg(tmp_path / "src.zarr"), str(tmp_path / "src.zarr"),
        ("temperature",), ("surface_pressure",), stamps, tmp_path / "cache")
    got = xr.open_zarr(str(path))
    np.testing.assert_array_equal(got.time.values, stamps)
    want = src.sel(time=stamps, level=[500.0, 850.0])
    for v in ("temperature", "surface_pressure"):
        assert got[v].dtype == np.float32
        assert np.array_equal(got[v].values, want[v].values, equal_nan=True)
    assert got["temperature"].attrs["units"] == "K"
    assert np.isnan(got["temperature"].values).any() == np.isnan(
        want["temperature"].values).any()


def test_store_cache_refuses_a_missing_time(tmp_path):
    _write_store(tmp_path / "src.zarr")
    bad = np.array(["2030-01-01T00"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="lacks"):
        sb._ensure_era5_store_cache(
            _cfg(tmp_path / "src.zarr"), str(tmp_path / "src.zarr"),
            ("temperature",), (), bad, tmp_path / "cache")


# --- index remap over the whole sample list ---------------------------------

def test_local_stores_remap_every_index_by_timestamp(tmp_path, monkeypatch):
    _write_store(tmp_path / "src.zarr")
    times = xr.open_zarr(str(tmp_path / "src.zarr")).time.values.astype("datetime64[ns]")
    pairs = [(2015, 3, 4), (2015, 4, 5), (2015, 11, 12), (2015, 20, 21)]
    monkeypatch.setattr(sb, "_training_sample_indices", lambda *a: iter(pairs))
    era5 = _cfg(tmp_path / "src.zarr")
    out = sb._wb_local_stores(
        SimpleNamespace(), {"cache_dir": str(tmp_path / "cache")}, era5, era5,
        times, 4, 1, rank=0, nproc=1)
    ds_l, cloud_l, flux_l, at = out
    assert cloud_l is None and flux_l is None
    assert ds_l.sizes["time"] == 7          # unique ICs + targets only
    for _, a, b in pairs:
        for i in (a, b):
            assert ds_l.time.values[at(i)] == times[i]


def test_no_cache_dir_means_no_local_stores(tmp_path):
    era5 = _cfg(tmp_path / "unused.zarr")
    assert sb._wb_local_stores(SimpleNamespace(), {}, era5, era5,
                               np.array([], "datetime64[ns]"), 4, 1, 0, 1) is None


# --- one rank builds, the others wait ---------------------------------------

def test_single_process_just_builds(tmp_path):
    assert sb._wb_build_or_wait(lambda: 7, tmp_path / ".b", 0, 1) == 7


def test_multi_rank_needs_slurm(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="SLURM_JOB_ID"):
        sb._wb_build_or_wait(lambda: 1, tmp_path / ".b", 0, 2)


def test_builder_marks_ready_and_waiter_proceeds(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setattr(sb, "_CACHE_POLL_S", 0)
    assert sb._wb_build_or_wait(lambda: "built", tmp_path / ".b", 0, 2) == "built"
    assert (tmp_path / ".b.ready.42.0.0").exists()
    assert sb._wb_build_or_wait(lambda: "read", tmp_path / ".b", 1, 2) == "read"


def test_builder_failure_reaches_waiters(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "43")

    def boom():
        raise OSError("disk full")

    with pytest.raises(OSError):
        sb._wb_build_or_wait(boom, tmp_path / ".b", 0, 2)
    with pytest.raises(RuntimeError, match="disk full"):
        sb._wb_build_or_wait(lambda: None, tmp_path / ".b", 1, 2)


def test_silent_builder_times_out(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "44")
    monkeypatch.setattr(sb, "_CACHE_POLL_S", 0)
    monkeypatch.setattr(sb, "_CACHE_STALE_S", 5)
    beat = tmp_path / ".heartbeat.44.0.0"
    beat.touch()
    old = beat.stat().st_mtime - 60
    os.utime(beat, (old, old))
    with pytest.raises(RuntimeError, match="went silent"):
        sb._wb_build_or_wait(lambda: None, tmp_path / ".b", 1, 2)


# --- the first-sample gate --------------------------------------------------

def test_gate_accepts_identical_samples_including_nan():
    a = (np.array([1.0, np.nan], np.float32), {"x": np.arange(3)})
    sb._assert_same_sample(a, (a[0].copy(), {"x": np.arange(3)}))


@pytest.mark.parametrize("other", [
    (np.nextafter(np.float32(1.0), np.float32(2.0)), np.float32(np.nan)),
])
def test_gate_rejects_one_ulp(other):
    a = (np.array([1.0, np.nan], np.float32),)
    with pytest.raises(RuntimeError, match="leaf 0"):
        sb._assert_same_sample(a, (np.array(other, np.float32),))


def test_gate_rejects_dtype_change():
    with pytest.raises(RuntimeError, match="dtype"):
        sb._assert_same_sample((np.zeros(2, np.float32),), (np.zeros(2, np.float64),))


def test_requeue_ignores_the_previous_attempts_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "45")
    monkeypatch.setattr(sb, "_CACHE_POLL_S", 0)
    (tmp_path / ".b.failed.45.0.0").write_text("old attempt")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    assert sb._wb_build_or_wait(lambda: "ok", tmp_path / ".b", 0, 2) == "ok"
    assert sb._wb_build_or_wait(lambda: "ok", tmp_path / ".b", 1, 2) == "ok"


def test_waiter_on_a_later_load_trusts_the_job_heartbeat(tmp_path, monkeypatch):
    # rank 0 is alive (job heartbeat fresh) but reaches this load only after
    # the waiter has waited longer than the stale limit (GLM review)
    monkeypatch.setenv("SLURM_JOB_ID", "46")
    monkeypatch.setattr(sb, "_CACHE_STALE_S", 1)
    real_sleep, polls = time.sleep, []

    def poll(_):
        polls.append(1)
        (tmp_path / ".heartbeat.46.0.0").touch()      # rank 0 still alive
        real_sleep(0.6)
        if len(polls) == 4:                            # > 2 s after start
            (tmp_path / ".val.ready.46.0.0").touch()

    monkeypatch.setattr(sb.time, "sleep", poll)
    assert sb._wb_build_or_wait(lambda: "val", tmp_path / ".val", 1, 2) == "val"


def test_retry_in_the_same_step_clears_the_old_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "47")
    monkeypatch.setattr(sb, "_CACHE_POLL_S", 0)

    def boom():
        raise OSError("transient")

    with pytest.raises(OSError):
        sb._wb_build_or_wait(boom, tmp_path / ".b", 0, 2)
    assert sb._wb_build_or_wait(lambda: "ok", tmp_path / ".b", 0, 2) == "ok"
    assert sb._wb_build_or_wait(lambda: "ok", tmp_path / ".b", 1, 2) == "ok"


def test_land_fraction_alone_gets_the_local_flux_store(tmp_path, monkeypatch):
    _write_store(tmp_path / "src.zarr", var_2d=("surface_pressure", "land_sea_mask"))
    _write_store(tmp_path / "flux.zarr", var_3d=(), var_2d=("land_sea_mask",))
    times = xr.open_zarr(str(tmp_path / "src.zarr")).time.values.astype("datetime64[ns]")
    monkeypatch.setattr(sb, "_training_sample_indices", lambda *a: iter([(2015, 2, 3)]))
    era5 = _cfg(tmp_path / "src.zarr", flux_zarr=str(tmp_path / "flux.zarr"))
    _, _, flux_l, _ = sb._wb_local_stores(
        SimpleNamespace(), {"cache_dir": str(tmp_path / "cache")}, era5,
        era5._replace(load_land_frac=True), times, 4, 1, 0, 1)
    assert flux_l is not None and "land_sea_mask" in flux_l
    assert str(tmp_path / "cache") in flux_l.encoding.get("source", str(tmp_path / "cache"))
