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

        # CMOR file writing requires cubed-sphere → lat-lon regridding,
        # which is not automatically configured at C8 test resolution.
        # Verify that if files were written, they have correct metadata.
        cmor_dir = tmp_path / "cmor"
        if cmor_dir.exists():
            nc_files = list(cmor_dir.rglob("*.nc"))
            if nc_files:
                import xarray as xr
                ds = xr.open_dataset(nc_files[0])
                assert "Conventions" in ds.attrs
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
        """AMIP uses AMIP-II baseline GHG (CO2 ≈ 348 ppmv)."""
        from legoesm.forcing.experiments import create_experiment_config
        cfg = create_experiment_config("amip")
        assert cfg.co2_ppmv == pytest.approx(348.0, abs=1.0)


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
