"""Tests for external forcing interfaces (GHG, ozone, aerosol, solar).

Creates temporary NetCDF files for file-based interpolation tests.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
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
        cfg = SolarConfig(S_0=constants.S_0)
        assert get_tsi_at_time(cfg, day=100.0) == constants.S_0

    def test_file_interpolation(self, tmp_path):
        nc_path = str(tmp_path / "tsi.nc")
        times = np.array([0.0, 365.0])
        tsi = np.array([1360.0, 1362.0])
        _make_tsi_nc(nc_path, times, tsi)

        cfg = SolarConfig(source="file", path=nc_path)
        assert abs(get_tsi_at_time(cfg, day=0.0) - 1360.0) < 1e-6
        assert abs(get_tsi_at_time(cfg, day=365.0) - 1362.0) < 1e-6
        assert abs(get_tsi_at_time(cfg, day=182.5) - constants.S_0) < 1e-6

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
        assert abs(forcing["tsi"] - constants.S_0) < 1.0e-6
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
        assert tsi == constants.S_0  # SolarConfig.S_0 default (constants.S_0)

        assert get_ozone_at_time(cfg.ozone, day=0.0) is None
        assert get_aerosol_at_time(cfg.aerosol, day=0.0) is None


# ==============================================================================
# Issue #207 regression tests — CMIP6 real-file ingestion bug fixes
# ==============================================================================

class TestCFTimeUnitsBug1:
    """Issue #207 bug 1: ``_to_days_float`` must decode CF 'X since Y' units.

    Without this, a CMIP6 ozone axis ``"months since 1850-01-01"`` with
    1980 values [0.5, 1.5, ...] collapses from 165 years of data into a
    2-year window (days [0..1980]) and the interpolator draws nonsense.
    """

    def test_months_since_scales_to_days(self):
        from legoesm.forcing.external import _to_days_float
        import numpy as np
        vals = np.array([0.5, 1.5, 600.5], dtype=np.float64)
        days = _to_days_float(vals, units="months since 1850-01-01",
                              calendar="standard")
        # 30.4375 = 365.25 / 12 (the CMIP6 month-midpoint convention).
        assert np.allclose(days[0], 0.0)
        assert np.allclose(days[1], 30.4375)
        assert np.allclose(days[-1], 600.0 * 30.4375)  # ≈ 18262.5 ≈ 50 yr

    def test_hours_since_scales_to_days(self):
        from legoesm.forcing.external import _to_days_float
        import numpy as np
        vals = np.array([0.0, 24.0, 720.0], dtype=np.float64)
        days = _to_days_float(vals, units="hours since 2000-01-01",
                              calendar="standard")
        assert np.allclose(days, [0.0, 1.0, 30.0])

    def test_unitless_numeric_passthrough(self):
        """Files without CF ``units`` are treated as raw days (legacy)."""
        from legoesm.forcing.external import _to_days_float
        import numpy as np
        vals = np.array([0.0, 15.5, 365.0])
        assert np.allclose(_to_days_float(vals), vals)

    def test_malformed_units_falls_back(self):
        """Unparseable CF units must not crash; fall back to raw numeric."""
        from legoesm.forcing.external import _to_days_float
        import numpy as np
        vals = np.array([1.0, 2.0])
        assert np.allclose(
            _to_days_float(vals, units="not-a-cf-units-string"), vals,
        )

    def test_360_day_calendar_uses_exact_month(self):
        """On the ``360_day`` calendar months are exactly 30 days."""
        from legoesm.forcing.external import _to_days_float
        import numpy as np
        vals = np.array([0.0, 1.0, 12.0, 24.0])
        days = _to_days_float(vals, units="months since 2000-01-01",
                              calendar="360_day")
        np.testing.assert_allclose(days, [0.0, 30.0, 360.0, 720.0])

    def test_first_record_includes_fractional_offset(self):
        """``_extract_first_date`` must capture the fractional-month
        offset, so mid-month sample files anchor to the mid-month date
        rather than the units reference (fix per Codex 2026-04-24)."""
        from legoesm.forcing.external import _extract_first_date
        import numpy as np
        vals = np.array([0.5, 1.5, 2.5])
        first = _extract_first_date(
            vals, units="months since 1850-01-01", calendar="standard",
        )
        # First record = anchor + 0.5 months ≈ mid-January 1850.
        assert first.year == 1850
        assert first.month == 1
        # ``anchor + 0.5 * 30.4375 d`` = Jan 16 @ 05:15 UTC.
        assert first.day == 16


class TestInterannualOzoneBug2:
    """Issue #207 bug 2: ozone files with >12 months must use the non-
    cyclic interannual branch instead of the 365.25-day cyclic interp.
    """

    def test_twelve_months_uses_cyclic_path(self, tmp_path):
        """A 12-month file is a climatology — expect annual periodicity."""
        import netCDF4, numpy as np
        nc_path = str(tmp_path / "ozone_clim.nc")
        lat = np.linspace(-90, 90, 9)
        months = np.arange(12).astype(float)
        data = np.stack([np.full_like(lat, 1.0e-6 * (m + 1)) for m in range(12)])
        with netCDF4.Dataset(nc_path, "w") as ds:
            ds.createDimension("time", 12)
            ds.createDimension("lat", len(lat))
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = np.array([15.5 + 30.4375 * m for m in range(12)])
            la = ds.createVariable("lat", "f8", ("lat",))
            la[:] = lat
            v = ds.createVariable("ozone", "f8", ("time", "lat"))
            v[:] = data
        cfg = OzoneConfig(enabled=True, path=nc_path, start_year=2000)
        # day=0.0 and day=365.0 should give the same value: the cyclic wrap
        # period is 365.0 d, matching the model's noleap clock and the AMIP
        # SST climatology wrap (audit F5; the old pin at 365.25 encoded the
        # inconsistent-period bug).
        lat_rad = jnp.radians(jnp.array(lat))
        v0 = np.asarray(get_ozone_at_time(cfg, day=0.0, lat_grid=lat_rad))
        v365 = np.asarray(get_ozone_at_time(cfg, day=365.0, lat_grid=lat_rad))
        np.testing.assert_allclose(v0, v365, atol=1e-12)

    def test_multiyear_file_interannual_path(self, tmp_path):
        """A >12-month file must expose interannual variation."""
        import netCDF4, numpy as np
        # Build a 2-year ozone file where year 1 is half the value of year 2.
        nc_path = str(tmp_path / "ozone_interannual.nc")
        lat = np.linspace(-90, 90, 9)
        n_months = 24
        months_axis = np.arange(n_months).astype(float) + 0.5
        data = np.zeros((n_months, len(lat)))
        data[:12] = 1.0e-6  # year 1
        data[12:] = 2.0e-6  # year 2
        with netCDF4.Dataset(nc_path, "w") as ds:
            ds.createDimension("time", n_months)
            ds.createDimension("lat", len(lat))
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = months_axis
            t.units = "months since 1850-01-01"
            t.calendar = "standard"
            la = ds.createVariable("lat", "f8", ("lat",))
            la[:] = lat
            v = ds.createVariable("ozone", "f8", ("time", "lat"))
            v[:] = data
        lat_rad = jnp.radians(jnp.array(lat))
        # start_year=1850: sim day 0 → file day 0 → year 1 (ozone ≈ 1e-6).
        cfg_y1 = OzoneConfig(enabled=True, path=nc_path, start_year=1850)
        v_y1 = np.asarray(get_ozone_at_time(cfg_y1, day=60.0, lat_grid=lat_rad))
        assert np.allclose(v_y1, 1.0e-6, atol=1e-10)
        # start_year=1851: sim day 60 → file day 365.25+60 → year 2 (≈ 2e-6).
        cfg_y2 = OzoneConfig(enabled=True, path=nc_path, start_year=1851)
        v_y2 = np.asarray(get_ozone_at_time(cfg_y2, day=60.0, lat_grid=lat_rad))
        assert np.allclose(v_y2, 2.0e-6, atol=1e-10)
        # Without the fix, cyclic interp would return the same value for
        # both start years (year-1 climatology always sampled).
        assert not np.allclose(v_y1, v_y2)


class TestCaseInsensitiveTSIBug4:
    """Issue #207 bug 4: ``_load_time_gpt`` must do case-insensitive
    variable lookup so that the MPI-M CMIP6 solar file (which stores
    ``TSI`` / ``SSI_frac``) is readable with the in-tree default
    config using lowercase names.
    """

    def test_uppercase_variables_resolved(self, tmp_path):
        import netCDF4, numpy as np
        nc_path = str(tmp_path / "solar_upper.nc")
        times = np.array([0.0, 365.0])
        tsi_vals = np.array([1360.0, 1362.0])
        spec = np.array([[0.6, 0.4], [0.3, 0.7]])
        with netCDF4.Dataset(nc_path, "w") as ds:
            ds.createDimension("time", 2)
            ds.createDimension("band", 2)
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = times
            TSI = ds.createVariable("TSI", "f8", ("time",))
            TSI[:] = tsi_vals
            sfrac = ds.createVariable("SSI_frac", "f8", ("time", "band"))
            sfrac[:] = spec
        from legoesm.forcing.external import _load_time_gpt
        # Request lowercase names — must still succeed via case-insensitive lookup.
        _load_time_gpt.cache_clear()  # lru_cache
        out_times, out_tsi, out_spec = _load_time_gpt(nc_path, "tsi", "ssi_frac")
        assert out_tsi is not None
        np.testing.assert_allclose(out_tsi, tsi_vals)
        np.testing.assert_allclose(out_spec, spec)


class TestDescendingLatBug6:
    """Issue #207 bug 6: ``_interp_zonal_to_grid`` must handle descending
    source latitudes (Kinne aerosol convention) instead of silently
    producing a constant at the endpoint.
    """

    def test_descending_lat_matches_ascending_flip(self):
        from legoesm.forcing.external import _interp_zonal_to_grid
        import numpy as np
        lat_asc = np.linspace(-90.0, 90.0, 181)
        lat_desc = lat_asc[::-1]
        # Linear ramp field so interpolation is trivially exact.
        field_asc = lat_asc.astype(np.float64)
        field_desc = lat_desc.astype(np.float64)
        lat_grid = jnp.radians(jnp.array([-45.0, 0.0, 60.0]))
        out_asc = np.asarray(_interp_zonal_to_grid(lat_asc, field_asc, lat_grid))
        out_desc = np.asarray(_interp_zonal_to_grid(lat_desc, field_desc, lat_grid))
        np.testing.assert_allclose(out_desc, out_asc, atol=1e-12)
        np.testing.assert_allclose(out_desc, [-45.0, 0.0, 60.0], atol=1e-12)

    def test_descending_lat_2d_field(self):
        """Same for the 2D/3D path (flattens trailing dims)."""
        from legoesm.forcing.external import _interp_zonal_to_grid
        import numpy as np
        lat_desc = np.linspace(90.0, -90.0, 19)
        field = np.tile(lat_desc[:, None], (1, 4))  # (nlat, nband)
        lat_grid = jnp.radians(jnp.array([-30.0, 30.0]))
        out = np.asarray(_interp_zonal_to_grid(lat_desc, field, lat_grid))
        assert out.shape == (2, 4)
        np.testing.assert_allclose(out[:, 0], [-30.0, 30.0], atol=1e-12)


class TestCMIP6VolcanicBug3:
    """Issue #207 bug 3: CMIP6 MPI-M volcanic files carry
    ``ext_sun(solar_bands, lat, altitude, month)`` in [1/km], not a
    flat ``aod(time, lat)``.  The auto-dispatcher must integrate over
    altitude and reshape to ``(time, lat, [band])`` transparently.
    """

    def test_cmip6_volcanic_dispatch(self, tmp_path):
        import netCDF4, numpy as np
        nc_path = str(tmp_path / "cmip6_volc.nc")
        nbands, nlat, nalt, nm = 3, 5, 4, 12
        lat = np.linspace(-80, 80, nlat)
        alt = np.linspace(10.0, 25.0, nalt)  # km
        # Constant 0.02 /km extinction across all dims → AOD per band =
        # 0.02 * (25-10) = 0.3 after altitude integration.  After the
        # mean-over-bands collapse, the returned AOD is also 0.3 (same
        # value for every band → mean == any band value).
        ext = np.full((nbands, nlat, nalt, nm), 0.02, dtype=np.float64)
        with netCDF4.Dataset(nc_path, "w") as ds:
            ds.createDimension("solar_bands", nbands)
            ds.createDimension("lat", nlat)
            ds.createDimension("altitude", nalt)
            ds.createDimension("month", nm)
            la = ds.createVariable("lat", "f8", ("lat",))
            la[:] = lat
            a = ds.createVariable("altitude", "f8", ("altitude",))
            a[:] = alt
            v = ds.createVariable(
                "ext_sun", "f8", ("solar_bands", "lat", "altitude", "month"),
            )
            v[:] = ext
        from legoesm.forcing.external import _load_volcanic_auto
        _load_volcanic_auto.cache_clear()
        mid_days, out_lat, out_aod = _load_volcanic_auto(nc_path)
        assert mid_days.shape == (nm,)
        np.testing.assert_allclose(out_lat, lat)
        # Shape: (time, lat) after mean-over-bands collapse.
        assert out_aod.shape == (nm, nlat)
        np.testing.assert_allclose(out_aod, 0.3, atol=1e-12)

    def test_cmip6_volcanic_mean_not_sum(self, tmp_path):
        """Ensure the band collapse is MEAN, not SUM (Codex review).

        14 SW bands each with the same per-band AOD must yield that
        per-band AOD as the returned broadband value — not 14x that.
        """
        import netCDF4, numpy as np
        nc_path = str(tmp_path / "cmip6_volc_mean.nc")
        nbands, nlat, nalt, nm = 14, 4, 5, 12
        lat = np.linspace(-60, 60, nlat)
        alt = np.linspace(15.0, 30.0, nalt)
        ext = np.full((nbands, nlat, nalt, nm), 0.01, dtype=np.float64)
        with netCDF4.Dataset(nc_path, "w") as ds:
            ds.createDimension("solar_bands", nbands)
            ds.createDimension("lat", nlat)
            ds.createDimension("altitude", nalt)
            ds.createDimension("month", nm)
            la = ds.createVariable("lat", "f8", ("lat",))
            la[:] = lat
            a = ds.createVariable("altitude", "f8", ("altitude",))
            a[:] = alt
            v = ds.createVariable(
                "ext_sun", "f8", ("solar_bands", "lat", "altitude", "month"),
            )
            v[:] = ext
        from legoesm.forcing.external import _load_volcanic_auto
        _load_volcanic_auto.cache_clear()
        _, _, out_aod = _load_volcanic_auto(nc_path)
        # Per-band AOD = 0.01 /km * (30-15) km = 0.15.  MEAN over 14
        # identical bands is still 0.15.  SUM would give 0.15 * 14 = 2.1.
        np.testing.assert_allclose(out_aod, 0.15, atol=1e-12)

    def test_ext_earth_only_rejected(self, tmp_path):
        """CMIP6 files with only ``ext_earth`` (LW) must not be accepted
        as a SW AOD source."""
        import netCDF4, numpy as np, pytest
        nc_path = str(tmp_path / "cmip6_lw_only.nc")
        with netCDF4.Dataset(nc_path, "w") as ds:
            ds.createDimension("terrestrial_bands", 2)
            ds.createDimension("lat", 3)
            ds.createDimension("altitude", 2)
            ds.createDimension("month", 12)
            la = ds.createVariable("lat", "f8", ("lat",))
            la[:] = np.array([-60.0, 0.0, 60.0])
            a = ds.createVariable("altitude", "f8", ("altitude",))
            a[:] = np.array([15.0, 30.0])
            v = ds.createVariable(
                "ext_earth", "f8",
                ("terrestrial_bands", "lat", "altitude", "month"),
            )
            v[:] = 0.01
        from legoesm.forcing.external import _load_volcanic_auto
        _load_volcanic_auto.cache_clear()
        with pytest.raises(ValueError, match="ext_sun"):
            _load_volcanic_auto(nc_path)

    def test_auto_dispatch_legacy_aod_still_works(self, tmp_path):
        """Files that carry the legacy ``aod`` variable still route through
        :func:`_load_monthly_zonal_anchored` unchanged."""
        import netCDF4, numpy as np
        nc_path = str(tmp_path / "legacy_aod.nc")
        lat = np.linspace(-90, 90, 7)
        data = np.stack([np.full_like(lat, 0.05 + 0.01 * m) for m in range(12)])
        with netCDF4.Dataset(nc_path, "w") as ds:
            ds.createDimension("time", 12)
            ds.createDimension("lat", len(lat))
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = np.array([15.5 + 30.4375 * m for m in range(12)])
            la = ds.createVariable("lat", "f8", ("lat",))
            la[:] = lat
            v = ds.createVariable("aod", "f8", ("time", "lat"))
            v[:] = data
        from legoesm.forcing.external import _load_volcanic_auto
        mid_days, out_lat, out_data = _load_volcanic_auto(nc_path)
        assert mid_days.shape == (12,)
        np.testing.assert_allclose(out_lat, lat)
        np.testing.assert_allclose(out_data, data)


# ==============================================================================
# Audit F1 (external) — noleap sim day onto a Gregorian-dated file axis
# ==============================================================================

class TestSimDayGregorianByCalendarDate:
    """``_simday_to_file_day`` must map a noleap model day onto a
    Gregorian-dated file BY CALENDAR DATE.  The old form added the noleap
    ``sim_day`` linearly onto the Gregorian epoch offset, drifting ~1 day
    per 4 years (leap days the model clock never lives through)."""

    def test_datetime64_anchor_maps_by_calendar_date(self):
        from legoesm.forcing.external import _simday_to_file_day
        anchor = np.datetime64("1979-01-16")
        sim_day = 8 * 365 + 196.0          # model 1987-07-16 00:00 (noleap)
        expected = float(
            (np.datetime64("1987-07-16") - anchor) / np.timedelta64(1, "D")
        )
        got = _simday_to_file_day(sim_day, 1979, anchor)
        assert got == expected
        # Old linear mapping: (1979-01-01 - 1979-01-16) + sim_day — 2 days
        # short of the file's real 1987-07-16 location (leap 1980, 1984).
        old_linear = -15.0 + sim_day
        assert expected - old_linear == 2.0
        assert got != old_linear

    def test_cftime_gregorian_anchor_maps_by_calendar_date(self):
        import cftime
        from legoesm.forcing.external import _simday_to_file_day
        anchor = cftime.DatetimeGregorian(1979, 1, 16)
        sim_day = 8 * 365 + 196.25          # 06:00 model time carries through
        got = _simday_to_file_day(sim_day, 1979, anchor)
        expected = (cftime.DatetimeGregorian(1987, 7, 16) - anchor).days + 0.25
        assert got == float(expected)

    def test_noleap_anchor_unchanged_for_nonzero_sim_day(self):
        # Noleap-dated files were already exact — must NOT change.
        import cftime
        from legoesm.forcing.external import _simday_to_file_day
        anchor = cftime.DatetimeNoLeap(1850, 1, 1)
        sim_day = 8 * 365 + 196.5
        got = _simday_to_file_day(sim_day, 1979, anchor)
        assert got == (1979 - 1850) * 365 + sim_day

    def test_360day_anchor_unchanged_for_nonzero_sim_day(self):
        # 360_day keeps the legacy linear form (out of F1 scope).
        import cftime
        from legoesm.forcing.external import _simday_to_file_day
        anchor = cftime.Datetime360Day(1850, 1, 1)
        sim_day = 100.5
        got = _simday_to_file_day(sim_day, 1979, anchor)
        assert got == (1979 - 1850) * 360 + sim_day

    def test_negative_sim_day_maps_to_prior_year(self):
        from legoesm.forcing.external import _simday_to_file_day
        anchor = np.datetime64("1979-01-01")
        # sim_day -1 = model 1978-12-31 (noleap) -> Gregorian 1978-12-31.
        got = _simday_to_file_day(-1.0, 1979, anchor)
        assert got == -1.0

    def test_model_noleap_date_roundtrip(self):
        from legoesm.forcing.external import _model_noleap_date
        assert _model_noleap_date(0.0, 1979) == (1979, 1, 1, 0.0)
        y, m, d, frac = _model_noleap_date(8 * 365 + 196.5, 1979)
        assert (y, m, d) == (1987, 7, 16) and abs(frac - 0.5) < 1e-12
        # Day 364 of a year = Dec 31; day 365 rolls into the next year.
        assert _model_noleap_date(364.0, 1979)[:3] == (1979, 12, 31)
        assert _model_noleap_date(365.0, 1979)[:3] == (1980, 1, 1)


# ==============================================================================
# Audits F4/F5 — cyclic climatology phase + wrap period
# ==============================================================================

def _write_clim_zonal_nc(path, varname, lat, data_12, dated, calendar="noleap"):
    """12-record monthly zonal climatology; ``dated`` adds CF units so the
    axis decodes to real mid-month dates (vs the units-less legacy axis)."""
    import netCDF4
    mid_days = np.array([15.5 + 30.4375 * m for m in range(12)])
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lat", len(lat))
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = mid_days
        if dated:
            t.units = "days since 2000-01-01"
            t.calendar = calendar
        la = ds.createVariable("lat", "f8", ("lat",))
        la[:] = lat
        v = ds.createVariable(varname, "f8", ("time", "lat"))
        v[:] = data_12


class TestCyclicClimatologyPhase:
    """Audit F4: a CF-dated monthly climatology previously lost its
    mid-month phase — ``_read_time_axis`` returns days since the FIRST
    RECORD (mid_days[0]==0), so a mid-January-stamped record was placed at
    Jan 1 (~15-day forward shift).  The cyclic branch now offsets mid_days
    by the first record's noleap day-of-year."""

    @staticmethod
    def _make_cfgs(tmp_path, calendar="noleap"):
        lat = np.linspace(-90, 90, 19)
        data = np.stack([np.full(19, float(m + 1)) for m in range(12)])
        p_undated = str(tmp_path / "clim_undated.nc")
        p_dated = str(tmp_path / f"clim_dated_{calendar}.nc")
        _write_clim_zonal_nc(p_undated, "ozone", lat, data, dated=False)
        _write_clim_zonal_nc(p_dated, "ozone", lat, data, dated=True,
                             calendar=calendar)
        return (OzoneConfig(enabled=True, path=p_undated),
                OzoneConfig(enabled=True, path=p_dated))

    def test_dated_and_undated_climatology_agree(self, tmp_path):
        # The same climatology, once with a CF-dated axis and once with the
        # legacy units-less day-of-year axis, must interpolate identically.
        cfg_u, cfg_d = self._make_cfgs(tmp_path)
        for day in (0.0, 15.5, 100.0, 200.25, 364.9):
            v_u = np.asarray(get_ozone_at_time(cfg_u, day=day)["ozone"])
            v_d = np.asarray(get_ozone_at_time(cfg_d, day=day)["ozone"])
            np.testing.assert_allclose(v_d, v_u, atol=1e-6,
                                       err_msg=f"day={day}")

    def test_midmonth_record_sampled_at_midmonth(self, tmp_path):
        # Phase check on the dated file: day 15.5 lands EXACTLY on the
        # mid-Jan record; day 0 interpolates Dec->Jan (pre-fix it returned
        # the January record exactly — the 15-day shift).
        _, cfg_d = self._make_cfgs(tmp_path)
        v_jan = np.asarray(get_ozone_at_time(cfg_d, day=15.5)["ozone"])
        np.testing.assert_allclose(v_jan, 1.0, atol=1e-12)
        v_0 = np.asarray(get_ozone_at_time(cfg_d, day=0.0)["ozone"])
        assert float(np.max(np.abs(v_0 - 1.0))) > 1.0   # Dec(12)->Jan(1) mix

    def test_wrap_period_is_365(self, tmp_path):
        # Audit F5: cyclic wrap at 365.0 (the noleap model clock), matching
        # amip.get_forcing_at_time — not 365.25.
        cfg_u, cfg_d = self._make_cfgs(tmp_path)
        for cfg in (cfg_u, cfg_d):
            v_a = np.asarray(get_ozone_at_time(cfg, day=10.0)["ozone"])
            v_b = np.asarray(get_ozone_at_time(cfg, day=10.0 + 365.0)["ozone"])
            np.testing.assert_allclose(v_b, v_a, atol=1e-12)

    def test_dated_aerosol_climatology_phase(self, tmp_path):
        # Same F4 anchoring on the aerosol cyclic branch.
        lat = np.linspace(-90, 90, 19)
        data = np.stack([np.full(19, 0.01 * (m + 1)) for m in range(12)])
        p = str(tmp_path / "aod_dated.nc")
        _write_clim_zonal_nc(p, "aod", lat, data, dated=True)
        cfg = AerosolConfig(enabled=True, path=p)
        v_jan = np.asarray(get_aerosol_at_time(cfg, day=15.5)["aod"])
        np.testing.assert_allclose(v_jan, 0.01, atol=1e-12)

    def test_midyear_start_climatology_sorted_and_bounded(self, tmp_path):
        # Codex review: a climatology whose FIRST record is mid-year pushes
        # the anchored axis past 365; it must be wrapped and co-sorted with
        # the data (an unsorted mod-365 axis broke searchsorted and could
        # extrapolate wildly).
        import netCDF4
        lat = np.linspace(-90, 90, 19)
        data = np.stack([np.full(19, float(m + 1)) for m in range(12)])
        p = str(tmp_path / "clim_july_start.nc")
        mid_days = np.array([15.5 + 30.4375 * m for m in range(12)])
        with netCDF4.Dataset(p, "w") as ds:
            ds.createDimension("time", 12)
            ds.createDimension("lat", 19)
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = mid_days
            t.units = "days since 2000-07-01"   # first record = mid-July
            t.calendar = "noleap"
            la = ds.createVariable("lat", "f8", ("lat",))
            la[:] = lat
            v = ds.createVariable("ozone", "f8", ("time", "lat"))
            v[:] = data
        cfg = OzoneConfig(enabled=True, path=p)
        # Mid-July (noleap doy 197 -> day 196.5) is the FIRST record exactly.
        v_jul = np.asarray(get_ozone_at_time(cfg, day=196.5)["ozone"])
        np.testing.assert_allclose(v_jul, 1.0, atol=1e-9)
        # The 7th record (dated mid-Jan 2001) lands at day ~14.125.
        v_jan = np.asarray(
            get_ozone_at_time(cfg, day=(196.5 + 6 * 30.4375) % 365.0)["ozone"]
        )
        np.testing.assert_allclose(v_jan, 7.0, atol=1e-9)
        # No extrapolation blow-ups anywhere in the year: values stay
        # within the record range [1, 12].
        for day in np.linspace(0.0, 365.0, 74):
            v = np.asarray(get_ozone_at_time(cfg, day=float(day))["ozone"])
            assert 1.0 - 1e-9 <= float(v.min()) and float(v.max()) <= 12.0 + 1e-9

    def test_noleap_dated_climatology_phase_is_exact(self, tmp_path):
        # Audit FL4 (residual ACCEPTED): the first-record-offset anchoring is
        # EXACT for the noleap-dated CMIP6 convention — a mid-month record on a
        # noleap axis samples at its noleap day-of-year with no residual.
        _, cfg_d = self._make_cfgs(tmp_path, calendar="noleap")
        # Record 2 (mid-March, noleap doy 15.5 + 2*30.4375 ~ Mar 16) samples at
        # its own value with no leap-day drift.
        for rec, day in ((0, 15.5), (1, 45.9), (2, 76.4)):
            v = np.asarray(get_ozone_at_time(cfg_d, day=day)["ozone"])
            np.testing.assert_allclose(v, float(rec + 1), atol=0.05,
                                       err_msg=f"record {rec}")


class TestSimDayAllLeapByCalendarDate:
    """Codex review: an ``all_leap`` (366-day) file axis drifts +1 d/yr
    under the linear sim-day add — it must use the by-calendar-date map."""

    def test_all_leap_anchor_maps_by_calendar_date(self):
        import cftime
        from legoesm.forcing.external import _simday_to_file_day
        anchor = cftime.DatetimeAllLeap(1850, 1, 1)
        # Model day 365 = 1851-01-01 (noleap); on the all-leap axis that
        # date sits 366 days after the anchor.
        got = _simday_to_file_day(365.0, 1850, anchor)
        assert got == 366.0
        # Linear add (the old behavior) would have returned 365.0.
