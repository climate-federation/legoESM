"""Phase 6: CMIP experiment end-to-end stress tests.

Runs actual CMIP experiment templates end-to-end with prescribed GHG
concentrations, validating the complete pipeline from config through
simulation to CMOR output.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_driver_from_experiment(
    experiment_name, days=30, resolution=8, nlev=5, dt=600.0,
    cmip_output=False, output_dir=None,
):
    """Create a ModelDriver from a CMIP experiment template."""
    from legoesm.forcing.experiments import create_experiment_config
    from legoesm.driver.model_driver import ModelDriver

    cfg = create_experiment_config(
        experiment_name,
        resolution=resolution,
        nlev=nlev,
        dt=dt,
        days=days,
    )
    # Override CMOR output if requested
    if cmip_output:
        cfg = cfg._replace(
            output=cfg.output._replace(cmip_output=True),
        )
    driver = ModelDriver(cfg, output_dir=output_dir)
    driver.setup()
    return driver


# ---------------------------------------------------------------------------
# 6.1  piControl: 30-day integration, fixed GHG
# ---------------------------------------------------------------------------

class TestPiControl:
    """piControl: fixed pre-industrial GHG, stable integration."""

    def test_picontrol_30day(self, tmp_path):
        """piControl runs 30 days with fixed CO2 = 284.3 ppmv."""
        driver = _make_driver_from_experiment(
            "piControl", days=30, output_dir=str(tmp_path),
        )
        status = driver.run()

        # Completed
        assert jnp.all(jnp.isfinite(driver.state.T.data))
        assert jnp.all(jnp.isfinite(driver.state.p_s.data))

        # CO2 should be at pre-industrial
        assert driver.config.co2_ppmv == pytest.approx(284.3, abs=0.1)

    def test_picontrol_cmor_writer_initialized(self, tmp_path):
        """piControl with cmip_output=True initializes the CFWriter."""
        driver = _make_driver_from_experiment(
            "piControl", days=30, cmip_output=True, output_dir=str(tmp_path),
        )

        # CFWriter should be created by the DiagnosticCollector
        assert driver.diagnostics.cf_writer is not None, (
            "CFWriter not initialized despite cmip_output=True"
        )

        status = driver.run()
        assert jnp.all(jnp.isfinite(driver.state.T.data))

        # CMOR output must produce actual NetCDF files
        cmor_dir = tmp_path / "cmor"
        assert cmor_dir.exists(), "cmor/ directory not created"
        nc_files = list(cmor_dir.rglob("*.nc"))
        assert len(nc_files) > 0, (
            "cmip_output=True produced zero NetCDF files"
        )

        # Files must have correct CMIP metadata
        import xarray as xr
        ds = xr.open_dataset(nc_files[0])
        # "CF-1.8" is REJECTED by the CMIP6 CV, whose Conventions regex is
        # ^CF-1.7 CMIP-6.[0-2]( UGRID-1.0){0,}$ -- the value now comes from
        # the vendored table Header instead of a hand-typed string.
        assert ds.attrs.get("Conventions") == "CF-1.7 CMIP-6.2"
        assert ds.attrs.get("experiment_id") == "piControl", (
            f"Wrong experiment_id: {ds.attrs.get('experiment_id')}"
        )
        ds.close()

    def test_picontrol_cmor_spatial_fields_not_zonal(self, tmp_path):
        """CMIP output fields must vary in longitude (not zonal broadcast)."""
        driver = _make_driver_from_experiment(
            "piControl", days=30, cmip_output=True, output_dir=str(tmp_path),
        )
        driver.run()

        cmor_dir = tmp_path / "cmor"
        nc_files = list(cmor_dir.rglob("*.nc"))
        assert len(nc_files) > 0, "No NC files written"

        import xarray as xr
        for nc_path in nc_files[:3]:
            ds = xr.open_dataset(nc_path)
            for var in ds.data_vars:
                if var in ("time_bnds",):
                    continue
                arr = ds[var].values
                if arr.ndim >= 3 and "lon" in ds[var].dims:
                    # Check that at least one time slice has longitude variation
                    lon_idx = list(ds[var].dims).index("lon")
                    std_along_lon = np.std(arr, axis=lon_idx)
                    has_variation = np.any(std_along_lon > 1e-10)
                    assert has_variation, (
                        f"Variable {var} in {nc_path.name} is zonally "
                        f"uniform — likely a zonal-broadcast artifact"
                    )
            ds.close()


# ---------------------------------------------------------------------------
# 6.2  1pctCO2: CO2 increases over time
# ---------------------------------------------------------------------------

class Test1pctCO2:
    """1pctCO2: CO2 grows at 1%/year from 284.3 ppmv."""

    def test_1pctco2_config(self):
        """1pctCO2 config has correct base CO2."""
        from legoesm.forcing.experiments import (
            create_experiment_config, EXPERIMENT_TEMPLATES,
        )
        cfg = create_experiment_config("1pctCO2")
        tmpl = EXPERIMENT_TEMPLATES["1pctCO2"]
        assert tmpl.base_co2_ppmv == pytest.approx(284.3, abs=0.1)
        assert cfg.co2_ppmv == pytest.approx(284.3, abs=0.1)

    def test_1pctco2_ghg_increases(self):
        """GHG concentrations increase over 70 years."""
        from legoesm.forcing.experiments import ghg_at_year
        co2_0, _, _ = ghg_at_year("1pctCO2", 1850)
        co2_70, _, _ = ghg_at_year("1pctCO2", 1920)

        assert co2_70 > co2_0
        # After 70 years: 284.3 * 1.01^70 ≈ 571
        expected = 284.3 * (1.01 ** 70)
        assert co2_70 == pytest.approx(expected, rel=0.01)

    def test_1pctco2_30day_run(self, tmp_path):
        """1pctCO2 runs 30 days, all fields bounded."""
        driver = _make_driver_from_experiment(
            "1pctCO2", days=30, output_dir=str(tmp_path),
        )
        status = driver.run()
        assert jnp.all(jnp.isfinite(driver.state.T.data))


# ---------------------------------------------------------------------------
# 6.3  AMIP: atmosphere-only with prescribed SST
# ---------------------------------------------------------------------------

class TestAMIP:
    """AMIP: atmosphere-only, prescribed SST/SIC."""

    def test_amip_30day(self, tmp_path):
        """AMIP runs 30 days with analytical SST."""
        driver = _make_driver_from_experiment(
            "amip", days=30, output_dir=str(tmp_path),
        )
        status = driver.run()
        assert jnp.all(jnp.isfinite(driver.state.T.data))

    def test_amip_baseline_ghg(self):
        """AMIP starts in 1979 with the CMIP6 historical GHG baseline
        (CO2 ≈ 336.78 ppmv, NOAA Mauna Loa + AGGI for the AMIP start
        year)."""
        from legoesm.forcing.experiments import create_experiment_config
        cfg = create_experiment_config("amip")
        assert cfg.co2_ppmv == pytest.approx(336.78, abs=1.0)

    def test_amip_warns_analytical_sst(self):
        """AMIP with analytical SST emits a warning."""
        import warnings
        from legoesm.forcing.experiments import create_experiment_config
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = create_experiment_config("amip")
            sst_warns = [x for x in w if "analytical" in str(x.message).lower()]
            assert len(sst_warns) > 0, (
                "AMIP with analytical SST should emit a warning"
            )

    def test_amip_default_radiation_rrtmgp(self):
        """AMIP experiment defaults to rrtmgp radiation."""
        from legoesm.forcing.experiments import create_experiment_config
        cfg = create_experiment_config("amip")
        assert cfg.radiation == "rrtmgp"


# ---------------------------------------------------------------------------
# 6.4  historical config verification
# ---------------------------------------------------------------------------

class TestHistorical:
    """historical experiment config verification (no simulation)."""

    def test_historical_config(self):
        from legoesm.forcing.experiments import create_experiment_config
        cfg = create_experiment_config("historical")
        # 1850 start
        assert cfg.co2_ppmv == pytest.approx(284.3, abs=0.1)

    def test_historical_duration(self):
        from legoesm.forcing.experiments import EXPERIMENT_TEMPLATES
        tmpl = EXPERIMENT_TEMPLATES["historical"]
        expected_years = tmpl.end_year - tmpl.start_year
        assert expected_years >= 100  # 1850-2014 = 164 years


# ---------------------------------------------------------------------------
# 6.4b  Transient GHG applied at runtime
# ---------------------------------------------------------------------------

class TestTransientGHG:
    """Verify that transient experiments produce time-varying GHG VMR."""

    def test_precompute_returns_transient_ghg(self):
        """_precompute_external_forcing returns non-None ghg_vmr for transient experiments."""
        driver = _make_driver_from_experiment(
            "historical", days=30, resolution=8, nlev=5, dt=600.0,
        )
        # At day 0 (year 1850), GHG should be at pre-industrial
        p_s, lat = driver._owned_p_s_and_lat()
        _, _, ghg_vmr_0 = driver._precompute_external_forcing(0.0, p_s, lat)

        # historical is transient + rrtmgp → should get VMR override
        assert ghg_vmr_0 is not None, (
            "Transient CMIP experiment did not produce GHG VMR override"
        )
        assert "co2" in ghg_vmr_0

    def test_transient_ghg_evolves_over_time(self):
        """GHG VMR changes between different simulation days for transient experiments."""
        driver = _make_driver_from_experiment(
            "1pctCO2", days=30, resolution=8, nlev=5, dt=600.0,
        )
        p_s, lat = driver._owned_p_s_and_lat()
        _, _, ghg_vmr_0 = driver._precompute_external_forcing(0.0, p_s, lat)
        # 50 years later (day = 50*365)
        _, _, ghg_vmr_50y = driver._precompute_external_forcing(50 * 365.0, p_s, lat)

        assert ghg_vmr_0 is not None
        assert ghg_vmr_50y is not None
        assert ghg_vmr_50y["co2"] > ghg_vmr_0["co2"], (
            f"1pctCO2 GHG did not increase: day 0 CO2={ghg_vmr_0['co2']:.6e}, "
            f"day 18250 CO2={ghg_vmr_50y['co2']:.6e}"
        )

    def test_picontrol_ghg_is_none_or_constant(self):
        """piControl (fixed forcing) should NOT get transient GHG override."""
        driver = _make_driver_from_experiment(
            "piControl", days=30, resolution=8, nlev=5, dt=600.0,
        )
        p_s, lat = driver._owned_p_s_and_lat()
        _, _, ghg_vmr = driver._precompute_external_forcing(0.0, p_s, lat)
        # piControl is fixed forcing — ghg_vmr should be None (uses config defaults)
        assert ghg_vmr is None, (
            "piControl should not produce transient GHG override"
        )


# ---------------------------------------------------------------------------
# 6.5  SSP585 GHG trajectory at key years
# ---------------------------------------------------------------------------

class TestSSP585GHG:
    """SSP5-8.5 GHG concentrations at benchmark years."""

    @pytest.mark.parametrize("year,expected_co2", [
        (2015, 401.0),
        (2030, 472.0),
        (2050, 601.0),
        (2070, 798.0),
        (2100, 1135.0),
    ])
    def test_ssp585_benchmark_years(self, year, expected_co2):
        from legoesm.forcing.experiments import ghg_at_year
        co2, _, _ = ghg_at_year("ssp585", year)
        assert co2 == pytest.approx(expected_co2, abs=0.1)

    def test_ssp585_monotonic(self):
        """CO2 is monotonically increasing from 2015 to 2100."""
        from legoesm.forcing.experiments import ghg_at_year
        co2_prev = 0.0
        for year in range(2015, 2101):
            co2, _, _ = ghg_at_year("ssp585", year)
            assert co2 >= co2_prev, f"CO2 decreased at year {year}"
            co2_prev = co2
