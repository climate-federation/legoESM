"""Tests for external forcing interfaces (GHG, ozone, aerosol, solar).

Creates temporary NetCDF files for file-based interpolation tests.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.forcing.external import (
    GHGConfig,
    OzoneConfig,
    AerosolConfig,
    SolarConfig,
    ExternalForcingConfig,
    get_ghg_at_time,
    ghg_concentrations_to_vmr,
    get_ozone_at_time,
    get_aerosol_at_time,
    get_solar_forcing_at_time,
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


def _make_solar_spectral_nc(path, times, tsi, spectral):
    """Write a solar forcing file with broadband + per-g-point spectra."""
    import netCDF4
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("gpt", spectral.shape[1])
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = times
        tsi_v = ds.createVariable("tsi", "f8", ("time",))
        tsi_v[:] = tsi
        spec_v = ds.createVariable("solar_fraction_by_gpt", "f8", ("time", "gpt"))
        spec_v[:] = spectral


def _make_ghg_annual_nc(path, years, co2, ch4, n2o, cfc11, cfc12):
    """Write a CMIP6-style annual global-mean GHG file (time, lat=1, lon=1)."""
    import netCDF4
    n = len(years)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", n)
        ds.createDimension("lat", 1)
        ds.createDimension("lon", 1)
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = years
        t.units = "year as %Y.%f"
        for name, vals in [("CO2", co2), ("CH4", ch4), ("N2O", n2o),
                           ("CFC_11", cfc11), ("CFC_12", cfc12)]:
            v = ds.createVariable(name, "f8", ("time", "lat", "lon"))
            v[:] = vals[:, None, None]


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

    def test_constant_mode_returns_cfcs(self):
        cfg = GHGConfig(cfc11_pptv=250.0, cfc12_pptv=540.0)
        result = get_ghg_at_time(cfg, day=0.0)
        assert result["cfc11_pptv"] == 250.0
        assert result["cfc12_pptv"] == 540.0

    def test_annual_file_interpolation(self, tmp_path):
        nc_path = str(tmp_path / "ghg_annual.nc")
        years = np.array([1979.0, 1980.0, 1981.0])
        co2 = np.array([336.8, 338.7, 340.1])
        ch4 = np.array([1547.0, 1578.0, 1610.0])
        n2o = np.array([301.0, 302.0, 303.0])
        cfc11 = np.array([160.0, 165.0, 170.0])
        cfc12 = np.array([300.0, 320.0, 340.0])
        _make_ghg_annual_nc(nc_path, years, co2, ch4, n2o, cfc11, cfc12)

        cfg = GHGConfig(source="annual_file", path=nc_path, start_year=1979)
        # At day 0 → year 1979.0
        r0 = get_ghg_at_time(cfg, day=0.0)
        assert abs(r0["co2_ppmv"] - 336.8) < 1e-4
        assert abs(r0["ch4_ppbv"] - 1547.0) < 1e-2
        assert abs(r0["n2o_ppbv"] - 301.0) < 1e-4
        assert abs(r0["cfc11_pptv"] - 160.0) < 1e-4
        assert abs(r0["cfc12_pptv"] - 300.0) < 1e-4

        # At day 365.25 → year 1980.0
        r1 = get_ghg_at_time(cfg, day=365.25)
        assert abs(r1["co2_ppmv"] - 338.7) < 1e-4

        # Interpolated at day 365.25/2 → year 1979.5
        r_mid = get_ghg_at_time(cfg, day=365.25 / 2)
        expected_co2_mid = (336.8 + 338.7) / 2
        assert abs(r_mid["co2_ppmv"] - expected_co2_mid) < 0.1

    def test_annual_file_missing_path_raises(self):
        cfg = GHGConfig(source="annual_file", path="")
        with pytest.raises(ValueError, match="path must be set"):
            get_ghg_at_time(cfg, day=0.0)

    def test_ghg_concentrations_to_vmr(self):
        ghg = {
            "co2_ppmv": 400.0,
            "ch4_ppbv": 1800.0,
            "n2o_ppbv": 320.0,
            "cfc11_pptv": 240.0,
            "cfc12_pptv": 530.0,
        }
        vmr = ghg_concentrations_to_vmr(ghg)
        assert abs(vmr["co2"] - 400.0e-6) < 1e-12
        assert abs(vmr["ch4"] - 1800.0e-9) < 1e-15
        assert abs(vmr["n2o"] - 320.0e-9) < 1e-15
        assert abs(vmr["cfc11"] - 240.0e-12) < 1e-18
        assert abs(vmr["cfc12"] - 530.0e-12) < 1e-18

    def test_file_mode_returns_default_cfcs(self, tmp_path):
        """File-based GHG (old format) returns config default CFCs."""
        nc_path = str(tmp_path / "ghg_old.nc")
        times = np.array([0.0, 365.0])
        co2 = np.array([300.0, 350.0])
        ch4 = np.array([1500.0, 1600.0])
        n2o = np.array([280.0, 290.0])
        _make_ghg_nc(nc_path, times, co2, ch4, n2o)

        cfg = GHGConfig(source="file", path=nc_path, cfc11_pptv=222.0, cfc12_pptv=555.0)
        r = get_ghg_at_time(cfg, day=0.0)
        assert r["cfc11_pptv"] == 222.0
        assert r["cfc12_pptv"] == 555.0


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

    def test_spectral_file_interpolation(self, tmp_path):
        nc_path = str(tmp_path / "solar_spectral.nc")
        times = np.array([0.0, 365.0], dtype=np.float64)
        tsi = np.array([1360.0, 1362.0], dtype=np.float64)
        spectral = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=np.float64)
        _make_solar_spectral_nc(nc_path, times, tsi, spectral)

        cfg = SolarConfig(source="spectral_file", path=nc_path)
        forcing = get_solar_forcing_at_time(cfg, day=182.5)
        assert abs(forcing["tsi"] - 1361.0) < 1.0e-6
        weights = np.asarray(forcing["solar_fraction_by_gpt"])
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(weights, np.array([0.45, 0.55]), atol=1.0e-6)


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

    def test_reference_fallback(self):
        lat_grid = jnp.array([0.0, np.deg2rad(45.0)])
        p_grid = jnp.array([[90000.0, 30000.0], [90000.0, 30000.0]])
        cfg = OzoneConfig(enabled=True, use_reference_if_missing=True)
        out = get_ozone_at_time(cfg, day=100.0, lat_grid=lat_grid, p_grid=p_grid)
        assert out.shape == p_grid.shape
        assert np.all(np.isfinite(np.asarray(out)))
        assert float(np.max(out)) > float(np.min(out))


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

    def test_reference_fallback(self):
        lat_grid = jnp.linspace(-np.pi / 2, np.pi / 2, 7)
        cfg = AerosolConfig(enabled=True, use_reference_if_missing=True, reference_aod_550=0.04)
        out = get_aerosol_at_time(cfg, day=12.0, lat_grid=lat_grid)
        assert out.shape == lat_grid.shape
        assert np.all(np.asarray(out) >= 0.0)

    def test_volcanic_addition(self, tmp_path):
        nc_base = str(tmp_path / "aod_base.nc")
        nc_volc = str(tmp_path / "aod_volc.nc")
        lat = np.linspace(-90, 90, 19)
        base = np.full((12, 19), 0.02)
        volc = np.full((12, 19), 0.04)
        _make_monthly_zonal_nc(nc_base, "aod", lat, base)
        _make_monthly_zonal_nc(nc_volc, "aod", lat, volc)

        cfg = AerosolConfig(
            enabled=True,
            path=nc_base,
            volcanic_enabled=True,
            volcanic_path=nc_volc,
            volcanic_scale=0.5,
        )
        out = get_aerosol_at_time(cfg, day=40.0)
        np.testing.assert_allclose(out["aod"], 0.02 + 0.5 * 0.04, atol=1.0e-6)

    def test_3d_aerosol_file_averaged_to_zonal(self, tmp_path):
        """3D Kinne-style (time, band, lat) AOD is averaged to zonal mean (#178).

        The loader averages over non-(time, lat) dimensions before
        interpolation. Uses off-grid latitudes for real interpolation.
        """
        import netCDF4
        nc_path = str(tmp_path / "aod_3d.nc")
        lat_src = np.linspace(-90, 90, 19)
        nband = 14
        # AOD varies linearly with latitude: 0.0 at south pole, 0.1 at north
        aod_lat = np.linspace(0.0, 0.1, 19)  # (19,)
        # Shape (12, nband, 19) — the loader averages over band
        data_3d = np.broadcast_to(
            aod_lat[None, None, :], (12, nband, 19),
        ).copy()
        mid_days = np.array([15.5 + 30.4375 * m for m in range(12)])
        with netCDF4.Dataset(nc_path, "w") as ds:
            ds.createDimension("time", 12)
            ds.createDimension("lat", 19)
            ds.createDimension("lnwl", nband)
            ds.createVariable("time", "f8", ("time",))[:] = mid_days
            ds.createVariable("lat", "f8", ("lat",))[:] = lat_src
            ds.createVariable("aod", "f8", ("time", "lnwl", "lat"))[:] = data_3d

        cfg = AerosolConfig(enabled=True, path=nc_path)
        # Off-grid latitudes to force actual interpolation
        lat_grid = jnp.radians(jnp.array([-45.0, 25.0]))
        result = get_aerosol_at_time(cfg, day=100.0, lat_grid=lat_grid)
        assert result.shape == lat_grid.shape, f"Expected {lat_grid.shape}, got {result.shape}"
        # Loader averages over band → same as per-cell AOD
        expected = np.array([0.025, 0.1 * 115 / 180])
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_3d_volcanic_file_averaged_to_zonal(self, tmp_path):
        """3D volcanic file is also averaged to zonal mean (#178)."""
        import netCDF4
        lat_src = np.linspace(-90, 90, 19)
        nband = 3

        # Base: simple 2D
        nc_base = str(tmp_path / "aod_base.nc")
        base_lat = np.linspace(0.01, 0.03, 19)
        _make_monthly_zonal_nc(
            nc_base, "aod", lat_src,
            np.broadcast_to(base_lat[None, :], (12, 19)).copy(),
        )

        # Volcanic: 3D (time, band, lat)
        nc_volc = str(tmp_path / "aod_volc.nc")
        volc_lat = np.linspace(0.0, 0.06, 19)
        data_volc = np.broadcast_to(
            volc_lat[None, None, :], (12, nband, 19),
        ).copy()
        mid_days = np.array([15.5 + 30.4375 * m for m in range(12)])
        with netCDF4.Dataset(nc_volc, "w") as ds:
            ds.createDimension("time", 12)
            ds.createDimension("lat", 19)
            ds.createDimension("lnwl", nband)
            ds.createVariable("time", "f8", ("time",))[:] = mid_days
            ds.createVariable("lat", "f8", ("lat",))[:] = lat_src
            ds.createVariable("aod", "f8", ("time", "lnwl", "lat"))[:] = data_volc

        cfg = AerosolConfig(
            enabled=True, path=nc_base,
            volcanic_enabled=True, volcanic_path=nc_volc, volcanic_scale=1.0,
        )
        lat_grid = jnp.radians(jnp.array([35.0]))
        result = get_aerosol_at_time(cfg, day=100.0, lat_grid=lat_grid)
        assert result.shape == lat_grid.shape
        # base at 35°: interp ≈ 0.01 + 0.02*(125/180)
        # volcanic at 35°: averaged over band → 0.06*(125/180)
        expected_base = 0.01 + 0.02 * (125.0 / 180.0)
        expected_volc = 0.06 * (125.0 / 180.0)
        np.testing.assert_allclose(result, expected_base + expected_volc, atol=1e-2)


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


# ==============================================================================
# Regression tests for CMIP6 / input4MIPs ingestion (see issue #207)
# ==============================================================================

class TestCFTimeUnits:
    """_to_days_float must honor CF ``units`` (months/hours/seconds)."""

    def test_months_since_scales_by_month_length(self):
        from legoesm.forcing.external import _to_days_float
        days = _to_days_float(np.array([0.0, 12.0]), "months since 1850-01-01")
        assert np.isclose(days[1], 12 * 30.4375)

    def test_hours_since_scales_by_24(self):
        from legoesm.forcing.external import _to_days_float
        days = _to_days_float(np.array([0.0, 48.0]), "hours since 2000-01-01")
        assert np.isclose(days[1], 2.0)

    def test_seconds_since(self):
        from legoesm.forcing.external import _to_days_float
        days = _to_days_float(np.array([0.0, 86400.0]), "seconds since 1970-01-01")
        assert np.isclose(days[1], 1.0)

    def test_parse_ref_year_case_insensitive(self):
        # Regex is case-insensitive so a capitalized CF units string still
        # yields a valid reference year for the non-cyclic ozone alignment.
        from legoesm.forcing.external import _parse_time_ref_year
        assert _parse_time_ref_year("Months since 1850-01-01") == 1850
        assert _parse_time_ref_year("DAYS SINCE 1979-01-01") == 1979
        assert _parse_time_ref_year("") is None

    def test_months_since_nc_roundtrip(self, tmp_path):
        """End-to-end: write a tiny NetCDF with numeric time+units, load it."""
        import netCDF4
        from legoesm.forcing.external import _load_monthly_zonal_with_levels
        path = str(tmp_path / "months.nc")
        with netCDF4.Dataset(path, "w") as ds:
            ds.createDimension("time", 24)
            ds.createDimension("lat", 3)
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = np.arange(24.0)
            t.units = "months since 1850-01-01"
            la = ds.createVariable("lat", "f8", ("lat",))
            la[:] = [-60.0, 0.0, 60.0]
            v = ds.createVariable("vmro3", "f8", ("time", "lat"))
            v[:] = np.zeros((24, 3))
        mid, lat, plev, data, ref = _load_monthly_zonal_with_levels(path, "vmro3")
        assert ref == 1850
        assert mid.shape == (24,)
        # 24 months → roughly two years in days
        assert np.isclose(mid[-1], 23 * 30.4375)


class TestSolarSpectralCaseInsensitive:
    """_load_time_gpt resolves ``TSI``/``SSI_frac`` when defaults are lowercase."""

    def test_uppercase_variable_names(self, tmp_path):
        import netCDF4
        from legoesm.forcing.external import _load_time_gpt
        path = str(tmp_path / "solar.nc")
        ntime, ngpt = 6, 14
        with netCDF4.Dataset(path, "w") as ds:
            ds.createDimension("time", ntime)
            ds.createDimension("band", ngpt)
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = np.arange(ntime, dtype=float)
            # Capitalized per MPI-M CMIP6 convention.
            tsi = ds.createVariable("TSI", "f8", ("time",))
            tsi[:] = 1360.0 + np.arange(ntime)
            ssi = ds.createVariable("SSI_frac", "f8", ("time", "band"))
            ssi[:] = np.full((ntime, ngpt), 1.0 / ngpt)
        # Pass lowercase defaults; case-insensitive lookup must resolve both.
        times, tsi_vals, spec = _load_time_gpt(path, "tsi", "ssi_frac")
        assert tsi_vals is not None
        assert tsi_vals[0] == 1360.0
        assert spec.shape == (ntime, ngpt)


class TestDescendingLatInterp:
    """_interp_zonal_to_grid must handle descending source latitudes (Kinne)."""

    def test_descending_source_preserves_signal(self):
        from legoesm.forcing.external import _interp_zonal_to_grid
        lat_src = np.array([89.5, 30.0, 0.0, -30.0, -89.5])
        field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # monotonic S→N reversed
        # Target in radians, covering N→S so we can compare to source pattern
        lat_tgt_deg = np.array([89.5, 0.0, -89.5])
        lat_tgt = jnp.radians(jnp.asarray(lat_tgt_deg))
        out = np.asarray(_interp_zonal_to_grid(lat_src, field, lat_tgt))
        # After internal reversal, N pole → 1.0, equator → 3.0, S pole → 5.0
        assert np.isclose(out[0], 1.0)
        assert np.isclose(out[1], 3.0)
        assert np.isclose(out[-1], 5.0)

    def test_ascending_source_unchanged(self):
        from legoesm.forcing.external import _interp_zonal_to_grid
        lat_src = np.array([-89.5, 0.0, 89.5])
        field = np.array([10.0, 50.0, 90.0])
        lat_tgt = jnp.radians(jnp.asarray([-89.5, 0.0, 89.5]))
        out = np.asarray(_interp_zonal_to_grid(lat_src, field, lat_tgt))
        assert np.allclose(out, [10.0, 50.0, 90.0])


class TestCMIP6VolcanicIngestion:
    """_load_volcanic_cmip6 integrates ext_sun on altitude → column AOD."""

    def _write_minimal_cmip6_volcanic(self, path):
        """Write a tiny CMIP6-style volcanic file with a known vertical integral."""
        import netCDF4
        nbands, nlat, nalt, nmon = 2, 3, 4, 12
        with netCDF4.Dataset(path, "w") as ds:
            ds.createDimension("solar_bands", nbands)
            ds.createDimension("latitude", nlat)
            ds.createDimension("altitude", nalt)
            ds.createDimension("month", nmon)
            la = ds.createVariable("latitude", "f8", ("latitude",))
            la[:] = [-60.0, 0.0, 60.0]
            alt = ds.createVariable("altitude", "f8", ("altitude",))
            alt[:] = [10.0, 15.0, 20.0, 25.0]  # km
            alt.units = "km"
            ext = ds.createVariable(
                "ext_sun", "f8",
                ("solar_bands", "latitude", "altitude", "month"),
            )
            # Uniform extinction so the expected column is easy to predict.
            # 0.01 [1/km] × total Δz spanning 10→25 km.
            ext[:] = np.full((nbands, nlat, nalt, nmon), 0.01)
        return nbands, nlat, nalt, nmon

    def test_dispatcher_detects_cmip6_schema(self, tmp_path):
        from legoesm.forcing.external import _is_cmip6_volcanic_file
        path = str(tmp_path / "volc.nc")
        self._write_minimal_cmip6_volcanic(path)
        assert _is_cmip6_volcanic_file(path) is True

    def test_column_aod_matches_vertical_integral(self, tmp_path):
        from legoesm.forcing.external import _load_volcanic_cmip6
        path = str(tmp_path / "volc.nc")
        self._write_minimal_cmip6_volcanic(path)
        mid_days, lat, aod = _load_volcanic_cmip6(path)
        # altitude 10→25 km, ext = 0.01 [1/km] ⇒ integral ≈ 0.01 × 15 km = 0.15.
        # The centered Δz over [10,15,20,25] spans ~15 km total via
        # dz = [5, 5, 5, 5] km for end/center rule, so expected ≈ 0.01 × 20 = 0.20.
        assert aod.shape == (12, 3)
        assert 0.1 < aod.mean() < 0.3
        # All months uniform because we used a constant ext field.
        assert np.std(aod) < 1e-10

    def test_auto_dispatcher_to_cmip6_branch(self, tmp_path):
        from legoesm.forcing.external import _load_volcanic_aerosol
        path = str(tmp_path / "volc.nc")
        self._write_minimal_cmip6_volcanic(path)
        mid_days, lat, aod = _load_volcanic_aerosol(path)
        assert aod.shape == (12, 3)


class TestNonCyclicOzone:
    """Multi-year ozone files must preserve interannual evolution."""

    def test_longer_than_12_months_uses_linear_branch(self, tmp_path):
        import netCDF4
        from legoesm.forcing.external import get_ozone_at_time, OzoneConfig
        path = str(tmp_path / "ozone_multi.nc")
        nmon = 36  # 3 years
        nlat, nlev = 4, 5
        # Construct a field that grows linearly in time so interannual
        # evolution is distinguishable across successive Januaries.
        data = (np.arange(nmon)[:, None, None] *
                np.ones((nmon, nlat, nlev))).astype(np.float64) * 1e-7
        with netCDF4.Dataset(path, "w") as ds:
            ds.createDimension("time", nmon)
            ds.createDimension("lat", nlat)
            ds.createDimension("plev", nlev)
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = np.arange(nmon, dtype=float)
            t.units = "months since 1990-01-01"
            la = ds.createVariable("lat", "f8", ("lat",))
            la[:] = np.linspace(-60.0, 60.0, nlat)
            p = ds.createVariable("plev", "f8", ("plev",))
            p[:] = np.linspace(100.0, 100000.0, nlev)
            v = ds.createVariable("vmro3", "f8", ("time", "lat", "plev"))
            v[:] = data

        cfg_y1 = OzoneConfig(enabled=True, source="climatology",
                             path=path, start_year=1990)
        cfg_y3 = OzoneConfig(enabled=True, source="climatology",
                             path=path, start_year=1992)
        out_y1 = get_ozone_at_time(cfg_y1, day=15.0)
        out_y3 = get_ozone_at_time(cfg_y3, day=15.0)
        # Field grows with time, so year-3 January must exceed year-1 January.
        assert out_y3["ozone"].mean() > out_y1["ozone"].mean()
