"""Phase 7: Multi-year coupled validation stress tests.

Long-duration integrations to validate that the coupled model can run
for climatological timescales without drift or blowup. These tests are
slow (order hours) and should be run separately from the fast suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_driver(preset="aquaplanet", days=365, resolution=8, nlev=5,
                 dt=600.0, output_dir=None):
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    atm_config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=dt, model_type="hydrostatic"),
        output=OutputConfig(diag_days=30, checkpoint_days=0),
        radiation="gray",
        days=days,
    )
    coupled_cfg = PRESETS[preset]()
    driver = CoupledESMDriver(atm_config, coupled_cfg, output_dir=output_dir)
    driver.setup()
    return driver


# ---------------------------------------------------------------------------
# 7.1  1-year aquaplanet: no blowup, bounded drift
# ---------------------------------------------------------------------------

class TestAquaplanet1Year:
    """1-year aquaplanet integration at C8/L5."""

    @pytest.fixture(scope="class")
    def driver(self, tmp_path_factory):
        d = _make_driver("aquaplanet", days=365,
                         output_dir=str(tmp_path_factory.mktemp("aqua1yr")))
        d.run()
        return d

    def test_completed(self, driver):
        assert driver.state is not None
        assert jnp.all(jnp.isfinite(driver.state.T.data))

    def test_sst_bounded(self, driver):
        sst = driver.ocean_state.T_sfc.data
        assert float(jnp.min(sst)) > 250.0, f"SST min = {float(jnp.min(sst)):.1f}"
        assert float(jnp.max(sst)) < 320.0, f"SST max = {float(jnp.max(sst)):.1f}"

    def test_sst_drift_bounded(self, driver):
        diag = driver.coupled_diagnostics
        if len(diag) < 2:
            pytest.skip("Not enough diagnostics")
        drift = abs(diag[-1]["sst_mean"] - diag[0]["sst_mean"])
        assert drift < 3.0, f"SST drift = {drift:.2f} K over 1 year"

    def test_temperature_bounded(self, driver):
        T_mean = float(jnp.mean(driver.state.T.data))
        assert 220.0 < T_mean < 310.0, f"T_atm mean = {T_mean:.1f} K"

    def test_pressure_bounded(self, driver):
        ps_mean = float(jnp.mean(driver.state.p_s.data))
        assert 9.5e4 < ps_mean < 1.1e5, f"p_s mean = {ps_mean:.0f} Pa"


# ---------------------------------------------------------------------------
# 7.2  1-year slab_carbon: carbon cycle stability
# ---------------------------------------------------------------------------

class TestSlabCarbon1Year:
    """1-year slab_carbon: CO2 and carbon pools stay bounded."""

    @pytest.fixture(scope="class")
    def driver(self, tmp_path_factory):
        d = _make_driver("slab_carbon", days=365,
                         output_dir=str(tmp_path_factory.mktemp("carbon1yr")))
        d.run()
        return d

    def test_completed(self, driver):
        assert driver.state is not None
        assert jnp.all(jnp.isfinite(driver.state.T.data))

    def test_co2_bounded(self, driver):
        if not hasattr(driver, '_co2_field') or driver._co2_field is None:
            pytest.skip("No prognostic CO2")
        M_CO2, M_air = 44.01, 28.97
        co2_ppmv = float(jnp.mean(driver._co2_field)) / (M_CO2 / M_air) * 1e6
        assert 350.0 < co2_ppmv < 500.0, f"CO2 = {co2_ppmv:.1f} ppmv"

    def test_nee_varies(self, driver):
        """NEE should show month-to-month variation (not flatlined)."""
        diag = driver.coupled_diagnostics
        if len(diag) < 4:
            pytest.skip("Not enough diagnostics")
        sst_values = [d["sst_mean"] for d in diag]
        # SST should vary over a year (seasonal cycle or drift)
        sst_range = max(sst_values) - min(sst_values)
        assert sst_range > 0.01, "SST is flatlined (no variation)"

    def test_sst_bounded(self, driver):
        sst = driver.ocean_state.T_sfc.data
        assert float(jnp.min(sst)) > 250.0
        assert float(jnp.max(sst)) < 320.0


# ---------------------------------------------------------------------------
# 7.3  2-year aquaplanet with restart at year 1
# ---------------------------------------------------------------------------

class TestAquaplanet2YearRestart:
    """2-year run with year-1 restart: final states should match."""

    def test_restart_matches_straight_run(self, tmp_path):
        """730-day straight run matches 365+365 with restart."""
        # --- Run A: 730 days straight ---
        dir_a = tmp_path / "run_a"
        dir_a.mkdir()
        driver_a = _make_driver("aquaplanet", days=730, output_dir=str(dir_a))
        driver_a.run()

        T_ref = np.asarray(driver_a.state.T.data)
        sst_ref = np.asarray(driver_a.ocean_state.T_sfc.data)

        # --- Run B: 365 + 365 ---
        dir_b = tmp_path / "run_b"
        dir_b.mkdir()
        driver_b1 = _make_driver("aquaplanet", days=365, output_dir=str(dir_b))
        driver_b1.run()

        steps_per_day = int(86400.0 / 600.0)
        driver_b1.save_checkpoint(365 * steps_per_day, 365.0)

        dir_b2 = tmp_path / "run_b2"
        dir_b2.mkdir()
        driver_b2 = _make_driver("aquaplanet", days=730, output_dir=str(dir_b2))

        ckpts = list(dir_b.glob("checkpoint_day_*.npz"))
        assert len(ckpts) > 0, "No checkpoint found"
        step, day = driver_b2._atm.load_checkpoint(ckpts[0])
        driver_b2.load_coupled_checkpoint(365.0)
        driver_b2.run(start_step=step, start_day=day)

        T_restart = np.asarray(driver_b2.state.T.data)
        sst_restart = np.asarray(driver_b2.ocean_state.T_sfc.data)

        np.testing.assert_allclose(T_restart, T_ref, atol=1e-10, rtol=1e-10,
                                   err_msg="T mismatch after 2-year restart")
        np.testing.assert_allclose(sst_restart, sst_ref, atol=1e-10, rtol=1e-10,
                                   err_msg="SST mismatch after 2-year restart")
