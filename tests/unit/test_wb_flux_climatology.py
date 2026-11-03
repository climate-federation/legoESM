"""The CAM6 arm's ERA5 flux climatology: calendar, smoothing, frames, the
five-year build, and the loader hand-off (replaces the held-at-init flux
no-leak test for that arm)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from legoesm.training import wb_flux_climatology as wfc


def test_noleap_day_maps_feb29_to_feb28():
    assert wfc.noleap_day("2015-02-28") == 58
    assert wfc.noleap_day("2016-02-28") == 58
    assert wfc.noleap_day("2016-02-29") == 58          # owner 2026-11-03
    assert wfc.noleap_day("2016-03-01") == wfc.noleap_day("2015-03-01") == 59
    assert wfc.noleap_day("2016-12-31") == wfc.noleap_day("2015-12-31") == 364


def test_climatology_hours_are_five_years_without_feb29():
    for day in (0, 58, 59, 364):
        hrs = wfc.climatology_hours(day)
        assert len(hrs) == 24 and all(len(h) == 5 for h in hrs)
        flat = [t for h in hrs for t in h]
        assert {int(str(t)[:4]) for t in flat} == set(range(2015, 2020))
        assert not any(str(t)[5:10] == "02-29" for t in flat)
        assert all(wfc.noleap_day(t) == day for t in flat)
        assert [int(str(h[0])[11:13]) for h in hrs] == list(range(24))


def test_smooth_days_is_a_wrapping_centred_mean():
    raw = np.random.default_rng(0).normal(size=(365, 2, 3))
    got = wfc.smooth_days(raw, 61)
    for d in (0, 30, 200, 364):
        idx = [(d + k) % 365 for k in range(-30, 31)]
        np.testing.assert_allclose(got[d], raw[idx].mean(axis=0), rtol=1e-12)
    with pytest.raises(ValueError):
        wfc.smooth_days(raw, 60)


def _clim():
    """clim[d, h] = 1000 d + h on every plane and column."""
    c = (1000.0 * np.arange(365)[:, None] + np.arange(24)[None, :])
    return np.broadcast_to(c[:, :, None, None], (365, 24, 7, 5)).copy()


def test_window_frames_cross_feb29_and_new_year():
    f = wfc.window_frames(_clim(), "2016-02-28T22", 4)
    assert f[:, 0, 0].tolist() == [58022, 58023, 58000, 58001, 58002]  # 29 Feb = 28 Feb
    g = wfc.window_frames(_clim(), "2019-12-31T23", 2)
    assert g[:, 0, 0].tolist() == [364023, 0, 1]
    with pytest.raises(ValueError, match="whole hour"):
        wfc.window_frames(_clim(), "2019-12-31T23:30", 2)


def test_frames_at_is_linear_in_hour():
    fr = jnp.asarray(np.arange(3.0)[:, None, None] * np.ones((3, 4, 5)) * 10.0)
    np.testing.assert_allclose(wfc.frames_at(fr, 0.0)[0, 0], 0.0)
    np.testing.assert_allclose(wfc.frames_at(fr, 5400.0)[0, 0], 15.0)
    np.testing.assert_allclose(wfc.frames_at(fr, 7200.0)[0, 0], 20.0)


def _builder():
    p = Path(__file__).resolve().parents[2] / "scripts/data/build_wb_flux_climatology.py"
    spec = importlib.util.spec_from_file_location("build_wb_flux_climatology", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_day_means_use_2015_to_2019_only_with_the_loader_signs():
    """A store spanning 2014-2020 whose sensible flux is (year - 2000)^2: the
    day-0 hour-6 mean is minus its 2015-2019 mean (291; 2014-2020 would give
    293); the other planes carry the loader's signs."""
    from legoesm.grids.gaussian import create_gaussian_grid

    grid = create_gaussian_grid(n_max=10)
    times = np.concatenate([np.arange(np.datetime64(f"{y}-01-01T00"),
                                      np.datetime64(f"{y}-01-02T00"),
                                      np.timedelta64(1, "h"))
                            for y in range(2014, 2021)]).astype("datetime64[ns]")
    year = np.array([int(str(t)[:4]) for t in times], float)
    lat, lon = np.linspace(90, -90, 19), np.arange(0, 360, 10.0)
    shape = (len(times), len(lat), len(lon))
    val = {"mean_surface_sensible_heat_flux": (year - 2000.0) ** 2,
           "mean_surface_latent_heat_flux": np.full(len(times), 1.0),
           "mean_eastward_turbulent_surface_stress": np.full(len(times), 2.0),
           "mean_northward_turbulent_surface_stress": np.full(len(times), 3.0),
           "mean_surface_downward_short_wave_radiation_flux": np.full(len(times), 300.0),
           "mean_surface_net_short_wave_radiation_flux": np.full(len(times), 270.0),
           "mean_surface_downward_long_wave_radiation_flux": np.full(len(times), 350.0),
           "mean_surface_net_long_wave_radiation_flux": np.full(len(times), -50.0)}
    data = {v: (("time", "lat", "lon"), np.broadcast_to(x[:, None, None], shape).copy())
            for v, x in val.items()}
    fds = xr.Dataset(data, coords={"time": times, "lat": lat, "lon": lon})
    m = _builder()
    out = m.day_means(fds, 0, grid, 1.0)
    assert out.shape == (24, 7, int(grid.n_lat) * int(grid.n_lon))
    np.testing.assert_allclose(out[6, 0], -291.0, rtol=1e-6)    # 2014/2020 excluded
    want = np.array([-1.0, -2.0, -3.0, 30.0, 300.0, 400.0])[:, None]
    np.testing.assert_allclose(out[6, 1:7], np.broadcast_to(want, out[6, 1:7].shape),
                               rtol=1e-6)


def test_cached_sample_reads_the_climatology_not_its_own_fluxes(tmp_path, monkeypatch):
    """CAM6 with era5_flux_climatology: the sample's own flux planes are
    dropped and the frames are the climatology at its valid times, the same
    whatever the sample's own ERA5 flux record holds."""
    from legoesm.training import scale_build as sb
    from legoesm.training import wb_state_cache as wsc
    from tests.ml.test_wb_state_cache import _rec, _times

    class _Clim:
        def frames(self, t0, n_hours):
            return wfc.window_frames(_clim(), t0, n_hours)

    monkeypatch.setattr(sb, "wb_state_cache_configs", lambda cfg, yml: (None, None, "none"))
    monkeypatch.setattr(sb, "validate_carry_holds_scheme", lambda *a, **k: 3)
    outs = []
    for shift in (0.0, 1e4):
        def rec(i, *a, _s=shift):
            c, s, f = _rec(i)
            return c, s, {**f, "sfc_shf": f["sfc_shf"] + _s}
        monkeypatch.setattr(sb, "wb_state_record", rec)
        d = tmp_path / str(int(shift))
        d.mkdir()
        p = d / "s.zarr"
        arrays, structure = wsc.record_arrays(*rec(0))
        wsc.create_store(p, [0, 1, 2, 3, 4], arrays, structure, {})
        for i in range(5):
            wsc.write_time(p, i, wsc.record_arrays(*rec(i))[0])
        monkeypatch.setattr(sb, "wb_state_cache_path", lambda *a, _p=p: _p)
        s = sb._state_cached_samples(SimpleNamespace(mode="physics"), {}, None, None,
                                     [(2015, 2, 4)], (6, 12), (1, 2), _times(),
                                     None, None, None, "none", None, clim=_Clim())
        outs.append(s[0][2])
    t = _times()
    hours = int((t[4] - t[2]) / np.timedelta64(1, "h"))
    want = wfc.window_frames(_clim(), t[2], hours)
    for f in outs:
        assert not any(k in f for k in sb._ERA5_FLUX_PLANE_KEYS)
        np.testing.assert_array_equal(np.asarray(f["sfc_flux_frames"]), want)


def test_climatology_is_physics_only_and_needs_a_cache_dir():
    from legoesm.training import scale_build as sb

    assert sb.wb_flux_climatology(SimpleNamespace(mode="physics"), {}, None) is None
    with pytest.raises(ValueError, match="CAM6"):
        sb.wb_flux_climatology(SimpleNamespace(mode="sfno"),
                               {"era5_flux_climatology": True}, None)
    with pytest.raises(ValueError, match="needs era5_surface_fluxes"):
        sb.wb_flux_climatology(SimpleNamespace(mode="physics"),
                               {"era5_flux_climatology": True}, None)
    with pytest.raises(ValueError, match="state_cache_dir"):
        sb.wb_flux_climatology(SimpleNamespace(mode="physics"),
                               {"era5_flux_climatology": True,
                                "era5_surface_fluxes": True}, None)


def test_store_built_from_another_flux_source_is_refused(tmp_path):
    import zarr

    p = tmp_path / "clim.zarr"
    g = zarr.open_group(str(p), mode="w")
    g.create_array("clim", data=np.zeros((wfc.N_DAYS, 1, 1, 1), np.float32))
    g.attrs.update({"complete": True, "flux_zarr": "gs://a"})
    wfc.FluxClimatology(p, "gs://a")
    with pytest.raises(RuntimeError, match="built from 'gs://a'"):
        wfc.FluxClimatology(p, "gs://b")
