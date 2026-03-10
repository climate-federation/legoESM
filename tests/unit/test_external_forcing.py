"""Tests for external forcing interfaces (GHG, ozone, aerosol, solar).

Creates temporary NetCDF files for file-based interpolation tests.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from legoesm.forcing.external import (
    GHGConfig,
    OzoneConfig,
    AerosolConfig,
    SolarConfig,
    ExternalForcingConfig,
    get_ghg_at_time,
    get_ozone_at_time,
    get_aerosol_at_time,
    get_tsi_at_time,
)


# ==============================================================================
# Helpers: create temporary NetCDF files
# ==============================================================================

def _make_ghg_nc(path, times, co2, ch4, n2o):
    """Write a minimal GHG NetCDF file."""
    import netCDF4
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", len(times))
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = times
        t.units = "days"
        for name, vals in [("co2_ppmv", co2), ("ch4_ppbv", ch4), ("n2o_ppbv", n2o)]:
            v = ds.createVariable(name, "f8", ("time",))
            v[:] = vals


def _make_tsi_nc(path, times, tsi):
    """Write a minimal TSI NetCDF file."""
    import netCDF4
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", len(times))
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = times
        v = ds.createVariable("tsi", "f8", ("time",))
        v[:] = tsi


def _make_monthly_zonal_nc(path, varname, lat, data_12):
    """Write a monthly zonal-mean NetCDF file (12 months x nlat)."""
    import netCDF4
    n_months, n_lat = data_12.shape
    mid_days = np.array([15.5 + 30.4375 * m for m in range(n_months)])
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", n_months)
        ds.createDimension("lat", n_lat)
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = mid_days
        la = ds.createVariable("lat", "f8", ("lat",))
        la[:] = lat
        v = ds.createVariable(varname, "f8", ("time", "lat"))
        v[:] = data_12


# ==============================================================================
# GHG tests
# ==============================================================================

class TestGHG:
    def test_constant_mode(self):
        cfg = GHGConfig(co2_ppmv=400.0, ch4_ppbv=1800.0, n2o_ppbv=320.0)
        result = get_ghg_at_time(cfg, day=100.0)
        assert result["co2_ppmv"] == 400.0
        assert result["ch4_ppbv"] == 1800.0
        assert result["n2o_ppbv"] == 320.0

    def test_file_interpolation(self, tmp_path):
        nc_path = str(tmp_path / "ghg.nc")
        times = np.array([0.0, 365.0, 730.0])
        co2 = np.array([300.0, 350.0, 400.0])
        ch4 = np.array([1500.0, 1600.0, 1700.0])
        n2o = np.array([280.0, 290.0, 300.0])
        _make_ghg_nc(nc_path, times, co2, ch4, n2o)

        cfg = GHGConfig(source="file", path=nc_path)
        # At day 0
        r0 = get_ghg_at_time(cfg, day=0.0)
        assert abs(r0["co2_ppmv"] - 300.0) < 1e-6
        # At day 365 (midpoint)
        r1 = get_ghg_at_time(cfg, day=365.0)
        assert abs(r1["co2_ppmv"] - 350.0) < 1e-6
        # Interpolated at day 182.5
        r_mid = get_ghg_at_time(cfg, day=182.5)
        assert abs(r_mid["co2_ppmv"] - 325.0) < 1e-6

    def test_file_missing_path_raises(self):
        cfg = GHGConfig(source="file", path="")
        with pytest.raises(ValueError, match="path must be set"):
            get_ghg_at_time(cfg, day=0.0)

    def test_unknown_source_raises(self):
        cfg = GHGConfig(source="magic")
        with pytest.raises(ValueError, match="Unknown GHG source"):
            get_ghg_at_time(cfg, day=0.0)


# ==============================================================================
# TSI / Solar tests
# ==============================================================================

class TestSolar:
    def test_constant_mode(self):
        cfg = SolarConfig(S_0=1361.0)
        assert get_tsi_at_time(cfg, day=100.0) == 1361.0

    def test_file_interpolation(self, tmp_path):
        nc_path = str(tmp_path / "tsi.nc")
        times = np.array([0.0, 365.0])
        tsi = np.array([1360.0, 1362.0])
        _make_tsi_nc(nc_path, times, tsi)

        cfg = SolarConfig(source="file", path=nc_path)
        assert abs(get_tsi_at_time(cfg, day=0.0) - 1360.0) < 1e-6
        assert abs(get_tsi_at_time(cfg, day=365.0) - 1362.0) < 1e-6
        assert abs(get_tsi_at_time(cfg, day=182.5) - 1361.0) < 1e-6

    def test_file_missing_path_raises(self):
        cfg = SolarConfig(source="file", path="")
        with pytest.raises(ValueError, match="path must be set"):
            get_tsi_at_time(cfg, day=0.0)

    def test_unknown_source_raises(self):
        cfg = SolarConfig(source="magic")
        with pytest.raises(ValueError, match="Unknown solar source"):
            get_tsi_at_time(cfg, day=0.0)


# ==============================================================================
# Ozone tests
# ==============================================================================

class TestOzone:
    def test_disabled_returns_none(self):
        cfg = OzoneConfig(enabled=False)
        assert get_ozone_at_time(cfg, day=100.0) is None

    def test_enabled_file_interpolation(self, tmp_path):
        nc_path = str(tmp_path / "ozone.nc")
        lat = np.linspace(-90, 90, 19)
        # Linear ramp across months: month i has ozone = i everywhere
        data_12 = np.stack([np.full(19, float(m)) for m in range(12)])
        _make_monthly_zonal_nc(nc_path, "ozone", lat, data_12)

        cfg = OzoneConfig(enabled=True, path=nc_path)
        # Mid-January (day ~15.5) should give ~month 0 = 0.0
        result = get_ozone_at_time(cfg, day=15.5)
        assert "lat" in result
        assert "ozone" in result
        assert result["ozone"].shape == (19,)
        assert abs(result["ozone"][0] - 0.0) < 0.5

    def test_enabled_missing_path_raises(self):
        cfg = OzoneConfig(enabled=True, path="")
        with pytest.raises(ValueError, match="path must be set"):
            get_ozone_at_time(cfg, day=0.0)


# ==============================================================================
# Aerosol tests
# ==============================================================================

class TestAerosol:
    def test_disabled_returns_none(self):
        cfg = AerosolConfig(enabled=False)
        assert get_aerosol_at_time(cfg, day=100.0) is None

    def test_enabled_file_interpolation(self, tmp_path):
        nc_path = str(tmp_path / "aod.nc")
        lat = np.linspace(-90, 90, 19)
        # Uniform AOD = 0.1 for all months
        data_12 = np.full((12, 19), 0.1)
        _make_monthly_zonal_nc(nc_path, "aod", lat, data_12)

        cfg = AerosolConfig(enabled=True, path=nc_path)
        result = get_aerosol_at_time(cfg, day=100.0)
        assert "lat" in result
        assert "aod" in result
        np.testing.assert_allclose(result["aod"], 0.1, atol=1e-6)

    def test_enabled_missing_path_raises(self):
        cfg = AerosolConfig(enabled=True, path="")
        with pytest.raises(ValueError, match="path must be set"):
            get_aerosol_at_time(cfg, day=0.0)


# ==============================================================================
# ExternalForcingConfig integration
# ==============================================================================

class TestExternalForcingConfig:
    def test_default_config_all_constant(self):
        """Default config works without any files."""
        cfg = ExternalForcingConfig()
        ghg = get_ghg_at_time(cfg.ghg, day=0.0)
        assert ghg["co2_ppmv"] == 348.0

        tsi = get_tsi_at_time(cfg.solar, day=0.0)
        assert tsi == 1360.0

        assert get_ozone_at_time(cfg.ozone, day=0.0) is None
        assert get_aerosol_at_time(cfg.aerosol, day=0.0) is None
