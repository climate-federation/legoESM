"""Tests for the AMIP forcing loader.

Uses synthetic NetCDF datasets to verify unit conversion, SIC clamping,
time interpolation, and missing-value fill.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.forcing.amip import (
    AMIPForcingConfig,
    load_amip_forcing,
    get_forcing_at_time,
    get_amip_preset,
    _fill_nan_nearest,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere


def _write_synthetic_forcing(path, n_time=4, sst_celsius=True, sic_percent=True,
                              add_nans=False):
    """Create a minimal synthetic AMIP forcing NetCDF file."""
    import xarray as xr

    n_lat, n_lon = 18, 36
    lat = np.linspace(-85, 85, n_lat)
    lon = np.linspace(5, 355, n_lon)
    times = np.arange(n_time) * 30.0  # days 0, 30, 60, 90

    sst = np.full((n_time, n_lat, n_lon), 20.0)  # 20 C or K (depends on offset)
    sic = np.full((n_time, n_lat, n_lon), 10.0 if sic_percent else 0.1)

    # Add latitude gradient to SST
    sst += lat[None, :, None] * 0.1

    # Add time variation
    sst += np.arange(n_time)[:, None, None] * 0.5

    # Polar ice only
    sic[:, :2, :] = 80.0 if sic_percent else 0.8  # high-lat ice
    sic[:, -2:, :] = 60.0 if sic_percent else 0.6

    if add_nans:
        # Simulate land mask
        sst[:, 5:8, 10:15] = np.nan
        sic[:, 5:8, 10:15] = np.nan

    time_coord = np.array([np.datetime64("2000-01-01") + np.timedelta64(int(t), "D")
                           for t in times])

    ds = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), sst.astype(np.float32)),
            "sic": (("time", "lat", "lon"), sic.astype(np.float32)),
        },
        coords={
            "time": time_coord,
            "lat": lat,
            "lon": lon,
        },
    )
    ds.to_netcdf(path)
    return ds


def _write_split_custom_forcing(sst_path, sic_path):
    """Create separate SST/SIC files with custom coord names and cftime."""
    import cftime
    import xarray as xr

    n_lat, n_lon, n_time = 18, 36, 4
    lat = np.linspace(-85, 85, n_lat)
    lon = np.linspace(5, 355, n_lon)
    times = [cftime.DatetimeNoLeap(2000, month, 1) for month in range(1, n_time + 1)]

    sst = np.full((n_time, n_lat, n_lon), 20.0, dtype=np.float32)
    sic = np.full((n_time, n_lat, n_lon), 0.1, dtype=np.float32)
    sst += lat[None, :, None] * 0.1
    sst += np.arange(n_time, dtype=np.float32)[:, None, None] * 0.5
    sic[:, :2, :] = 0.8
    sic[:, -2:, :] = 0.6

    coords = {
        "time": times,
        "ylat": lat,
        "xlon": lon,
    }
    xr.Dataset(
        {"SST": (("time", "ylat", "xlon"), sst)},
        coords=coords,
    ).to_netcdf(sst_path)
    xr.Dataset(
        {"iceFrac": (("time", "ylat", "xlon"), sic)},
        coords=coords,
    ).to_netcdf(sic_path)


@pytest.fixture(scope="module")
def grid():
    return create_cubed_sphere(4)


@pytest.fixture(scope="module")
def forcing_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("amip") / "forcing.nc"
    _write_synthetic_forcing(str(path), sst_celsius=True, sic_percent=True)
    return str(path)


@pytest.fixture(scope="module")
def forcing_path_nans(tmp_path_factory):
    path = tmp_path_factory.mktemp("amip_nans") / "forcing_nans.nc"
    _write_synthetic_forcing(str(path), sst_celsius=True, sic_percent=True, add_nans=True)
    return str(path)


@pytest.fixture(scope="module")
def split_custom_forcing_paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("amip_split")
    sst_path = root / "sst.nc"
    sic_path = root / "sic.nc"
    _write_split_custom_forcing(str(sst_path), str(sic_path))
    return str(sst_path), str(sic_path)


class TestAMIPPresets:
    def test_cobe_preset(self):
        cfg = get_amip_preset("cobe")
        assert cfg.sst_var == "SST_cpl"
        assert cfg.sic_var == "ice_cov"
        assert cfg.sst_offset == 0.0
        assert cfg.sic_scale == 0.01

    def test_hadisst_preset(self):
        cfg = get_amip_preset("hadisst")
        assert cfg.sst_var == "sst"
        assert cfg.sst_offset == constants.T_freeze

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_amip_preset("nonexistent")


class TestLoadForcing:
    def test_load_basic(self, grid, forcing_path):
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,  # C -> K
            sic_scale=0.01,     # percent -> fraction
        )
        forcing = load_amip_forcing(config, grid)
        assert forcing.sst.shape[0] == 4  # n_time
        assert forcing.sst.shape[1:] == (6, 4, 4)
        assert forcing.sic.shape == forcing.sst.shape
        assert forcing.times.shape == (4,)

    def test_sst_in_kelvin(self, grid, forcing_path):
        """SST should be converted to Kelvin (offset applied)."""
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,
            sic_scale=0.01,
        )
        forcing = load_amip_forcing(config, grid)
        # Raw data is ~20 C, so after +273.15 offset should be ~293 K
        mean_sst = float(jnp.mean(forcing.sst[0]))
        assert 280 < mean_sst < 310, f"SST mean {mean_sst} not in expected Kelvin range"

    def test_sic_clamped(self, grid, forcing_path):
        """SIC should be in [0, 1] after scaling."""
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,
            sic_scale=0.01,
        )
        forcing = load_amip_forcing(config, grid)
        assert float(jnp.min(forcing.sic)) >= 0.0
        assert float(jnp.max(forcing.sic)) <= 1.0

    def test_nan_fill(self, grid, forcing_path_nans):
        """NaN values over land should be filled."""
        config = AMIPForcingConfig(
            path=forcing_path_nans,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,
            sic_scale=0.01,
        )
        forcing = load_amip_forcing(config, grid)
        assert jnp.all(jnp.isfinite(forcing.sst))
        assert jnp.all(jnp.isfinite(forcing.sic))

    def test_missing_variable_raises(self, grid, forcing_path):
        """Missing variable should give a clear error."""
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="nonexistent_sst",
            sic_var="sic",
        )
        with pytest.raises(KeyError, match="Missing variables"):
            load_amip_forcing(config, grid)

    def test_load_split_custom_with_cftime(self, grid, split_custom_forcing_paths):
        """Custom forcing should support separate SIC files and cftime axes."""
        sst_path, sic_path = split_custom_forcing_paths
        config = AMIPForcingConfig(
            dataset="custom",
            path=sst_path,
            sic_path=sic_path,
            sst_var="SST",
            sic_var="iceFrac",
            time_var="time",
            lat_var="ylat",
            lon_var="xlon",
            sst_offset=constants.T_freeze,
            sic_scale=1.0,
        )
        forcing = load_amip_forcing(config, grid)
        assert forcing.sst.shape[0] == 4
        assert forcing.sic.shape == forcing.sst.shape
        assert float(forcing.times[0]) == 0.0
        assert float(forcing.times[1]) > 0.0
        assert jnp.all(jnp.isfinite(forcing.sst))
        assert jnp.all(jnp.isfinite(forcing.sic))


class TestTimeInterpolation:
    def test_at_record_time(self, grid, forcing_path):
        """Interpolation at exact record time should return that record."""
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,
            sic_scale=0.01,
        )
        forcing = load_amip_forcing(config, grid)
        sst_0, sic_0 = get_forcing_at_time(forcing, 0.0)
        assert jnp.allclose(sst_0, forcing.sst[0], atol=1e-5)

    def test_midpoint_interpolation(self, grid, forcing_path):
        """Midpoint between two records should be the average."""
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,
            sic_scale=0.01,
        )
        forcing = load_amip_forcing(config, grid)
        sst_mid, _ = get_forcing_at_time(forcing, 15.0)  # midpoint of 0 and 30
        expected = 0.5 * (forcing.sst[0] + forcing.sst[1])
        assert jnp.allclose(sst_mid, expected, atol=1e-4)

    def test_cyclic_outside_range(self, grid, forcing_path):
        """Times outside forcing range should wrap cyclically."""
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,
            sic_scale=0.01,
        )
        forcing = load_amip_forcing(config, grid)
        period = float(forcing.times[-1] - forcing.times[0])
        # A time one full period later should give the same SST as the start.
        sst_wrap, _ = get_forcing_at_time(forcing, float(forcing.times[0]) + period)
        sst_first, _ = get_forcing_at_time(forcing, float(forcing.times[0]))
        assert jnp.allclose(sst_wrap, sst_first, atol=1e-5)


class TestFillNanNearest:
    def test_no_nans_passthrough(self):
        data = np.ones((2, 5, 10))
        lat = np.linspace(-90, 90, 5)
        lon = np.linspace(0, 350, 10)
        result = _fill_nan_nearest(data, lat, lon)
        np.testing.assert_array_equal(result, data)

    def test_fills_nans(self):
        data = np.ones((1, 5, 10))
        data[0, 2, 5] = np.nan
        lat = np.linspace(-90, 90, 5)
        lon = np.linspace(0, 350, 10)
        result = _fill_nan_nearest(data, lat, lon)
        assert not np.any(np.isnan(result))
        assert result[0, 2, 5] == 1.0  # nearest neighbor should be 1.0


def test_get_forcing_at_time_single_record():
    """PR D: a single-record (ntime==1) forcing — e.g. a climatological mean or
    a 2D file promoted to shape (1, ...) — must return the single SST/SIC field
    for ANY day. The cyclic-wrap + searchsorted clip(0, ntime-2)=clip(0, -1)
    path produced idx=-1, dt=0, a blown-up weight, and ~0 K SST for day !=
    times[0]."""
    from legoesm.forcing.amip import AMIPForcing

    sst = jnp.full((1, 4, 8), 290.0)
    sic = jnp.full((1, 4, 8), 0.3)
    forcing = AMIPForcing(
        times=jnp.array([0.0]), sst=sst, sic=sic, config=AMIPForcingConfig(),
    )
    for day in (0.0, 100.0, 365.0):
        s, i = get_forcing_at_time(forcing, float(day))
        np.testing.assert_allclose(np.asarray(s), 290.0, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(i), 0.3, rtol=1e-6)
