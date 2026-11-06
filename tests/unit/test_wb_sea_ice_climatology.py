"""ERA5 sea-ice climatology of the learned WB arms (training years only)."""
from __future__ import annotations

import importlib.util
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training import wb_flux_climatology as wfc

_BUILDER = pathlib.Path(__file__).parents[2] / "scripts/data/build_wb_flux_climatology.py"


def _builder():
    spec = importlib.util.spec_from_file_location("_bwfc", _BUILDER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_day_averages_four_snapshots_of_each_training_year():
    ts = wfc.sic_day_times(100)
    years = set(range(wfc.YEARS[0], wfc.YEARS[1] + 1))   # (first, last)
    assert years == {2015, 2016, 2017, 2018, 2019}
    assert len(ts) == 4 * len(years)
    assert {int(str(t)[:4]) for t in ts} == years
    assert {int(str(t)[11:13]) for t in ts} == {0, 6, 12, 18}
    # 28 February (noleap day 58) never reads 29 February.
    assert not any(str(t)[5:10] == "02-29" for t in wfc.sic_day_times(58))


def _grid():
    from legoesm.grids.gaussian import create_gaussian_grid
    return create_gaussian_grid(10)


def _dataset(day):
    import xarray as xr

    lat = np.linspace(90.0, -90.0, 73)          # ERA5: descending
    lon = np.arange(0.0, 360.0, 5.0)
    f = np.where(lat[:, None] > 40.0, 1.0, 0.0) * np.ones((1, lon.size))
    # Land (NaN) in the north beside the ice.
    f = np.where((lat[:, None] > 60.0) & (lon[None, :] < 90.0), np.nan, f)
    ts = [np.datetime64(t, "ns") for t in wfc.sic_day_times(day)]
    data = np.broadcast_to(f[:, :, None], (lat.size, lon.size, len(ts)))
    return xr.Dataset({wfc.SIC_VAR: (("lat", "lon", "time"), data.copy())},
                      coords={"lat": lat, "lon": lon, "time": ts}), f, lat, lon


def test_the_day_mean_is_the_aimip_regridder_clipped():
    from legoesm.training.aimip_amip_forcing import regrid_era5_field_2d

    grid = _grid()
    ds, f, lat, lon = _dataset(100)
    got = _builder().sic_day_mean(ds, 100, grid)
    want = np.clip(regrid_era5_field_2d(f, lat, lon, grid).reshape(-1), 0.0, 1.0)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    assert np.all(np.isfinite(got)) and got.min() >= 0.0 and got.max() <= 1.0


def test_coastal_ice_is_not_diluted_by_land():
    """Land is filled from the nearest ocean, so a cell between ice-covered
    ocean and land keeps full ice; zero-filled land would halve it."""
    grid = _grid()
    ds, *_ = _dataset(100)
    got = _builder().sic_day_mean(ds, 100, grid).reshape(int(grid.n_lat),
                                                          int(grid.n_lon))
    north = np.degrees(np.asarray(grid.lat)) > 65.0
    assert np.all(got[north] == pytest.approx(1.0, abs=1e-12))


class _Clim:
    def __init__(self, plane):
        self._p = plane

    def plane(self, t0):
        return self._p


def test_the_plane_replaces_sic_and_is_zero_on_land():
    from legoesm.training.scale_build import with_sea_ice_climatology

    plane = np.array([0.9, 0.8, 0.7, 0.6])
    f = {"sic": jnp.zeros(4), "land_frac": jnp.array([0.0, 0.49, 0.5, 1.0]),
         "T_sfc": jnp.ones(4)}
    out = with_sea_ice_climatology(f, _Clim(plane), np.datetime64("2016-03-01"))
    np.testing.assert_array_equal(np.asarray(out["sic"]), [0.9, 0.8, 0.0, 0.0])
    assert out["T_sfc"] is f["T_sfc"]
    with pytest.raises(ValueError):
        with_sea_ice_climatology(f, _Clim(np.zeros(5)), np.datetime64("2016-03-01"))


def test_the_store_tag_names_the_regridding():
    grid = _grid()
    a = wfc.sic_store_tag(grid, "src")
    orig = wfc.SIC_REGRID
    try:
        wfc.SIC_REGRID = "other"
        assert wfc.sic_store_tag(grid, "src") != a
    finally:
        wfc.SIC_REGRID = orig


def test_the_deck_switch_is_learned_arms_only(tmp_path):
    from types import SimpleNamespace

    from legoesm.training import scale_build as sb

    yml = {"era5_sea_ice_climatology": True, "state_cache_dir": str(tmp_path),
           "era5_zarr": "src"}
    assert sb.wb_sea_ice_climatology(SimpleNamespace(mode="sfno"),
                                     {"era5_sea_ice_climatology": False}, None) is None
    with pytest.raises(ValueError):
        sb.wb_sea_ice_climatology(SimpleNamespace(mode="physics"), yml, _grid())
    with pytest.raises(FileNotFoundError):     # selected but never built
        sb.wb_sea_ice_climatology(SimpleNamespace(mode="neural_gcm"), yml, _grid())


def test_selecting_the_climatology_loads_the_land_fraction():
    from legoesm.training.scale_build import wb_needs_land_frac

    assert not wb_needs_land_frac("sfno", {"sfno": {}})
    assert wb_needs_land_frac("sfno", {"sfno": {}, "era5_sea_ice_climatology": True})
