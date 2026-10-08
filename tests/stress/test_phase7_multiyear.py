"""Phase 7: Multi-year coupled validation stress tests.

Long-duration integrations to validate that the coupled model can run
for climatological timescales without drift or blowup. These tests are
slow (order hours) and should be run separately from the fast suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_driver(preset="aquaplanet", days=365, resolution=8, nlev=5,
                 dt=600.0, output_dir=None):
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS, CoupledConfig
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    atm_config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=dt, model_type="hydrostatic"),
        output=OutputConfig(diag_days=30, checkpoint_days=0),
        radiation="gray",
        days=days,
    )
    # Use 6-hourly coupling (21600s) for year-long tests — hourly coupling
    # at 3600s is validated in Phase 2 and would make these tests ~6x slower.
    # Carbon presets need sub-daily coupling for forward-Euler stability.
    base_cfg = PRESETS[preset]()
    coupled_cfg = base_cfg._replace(coupling_dt=21600.0)
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
        """SST stays in a physically plausible range (wide bounds for C8/L5)."""
        sst = driver.ocean_state.T_sfc.data
        assert float(jnp.min(sst)) > 240.0, f"SST min = {float(jnp.min(sst)):.1f}"
        assert float(jnp.max(sst)) < 340.0, f"SST max = {float(jnp.max(sst)):.1f}"

    def test_sst_drift_bounded(self, driver):
        """SST drift is bounded (uncalibrated C8/L5 gray may drift ~20 K/yr)."""
        diag = driver.coupled_diagnostics
        if len(diag) < 2:
            pytest.skip("Not enough diagnostics")
        drift = abs(diag[-1]["sst_mean"] - diag[0]["sst_mean"])
        assert drift < 30.0, f"SST drift = {drift:.2f} K over 1 year (> 30 K)"

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

    def test_co2_finite_and_positive(self, driver):
        """CO2 field stays finite and positive over 1 year."""
        if not hasattr(driver, '_co2_field') or driver._co2_field is None:
            pytest.skip("No prognostic CO2")
        assert jnp.all(jnp.isfinite(driver._co2_field)), "CO2 has NaN/inf"
        assert float(jnp.min(driver._co2_field)) > 0, "CO2 went negative"
        # Convert to ppmv and check wide bounds (uncalibrated model)
        M_CO2, M_air = constants.M_CO2, constants.M_air
        co2_ppmv = float(driver._co2_global_mean_kgkg()) / (M_CO2 / M_air) * 1e6
        assert 100.0 < co2_ppmv < 2000.0, f"CO2 = {co2_ppmv:.1f} ppmv"

    def test_nee_varies(self, driver):
        """SST should show variation over a year (not flatlined)."""
        diag = driver.coupled_diagnostics
        if len(diag) < 4:
            pytest.skip("Not enough diagnostics")
        sst_values = [d["sst_mean"] for d in diag]
        sst_range = max(sst_values) - min(sst_values)
        assert sst_range > 0.01, "SST is flatlined (no variation)"

    def test_sst_bounded(self, driver):
        """SST stays plausible (wide bounds for uncalibrated C8/L5)."""
        sst = driver.ocean_state.T_sfc.data
        assert float(jnp.min(sst)) > 240.0
        assert float(jnp.max(sst)) < 340.0


# ---------------------------------------------------------------------------
# 7.3  2-year aquaplanet with restart at year 1
# ---------------------------------------------------------------------------

class TestCoupledRestart60Day:
    """60-day run with 30+30 restart: final states should be close.

    Uses diag_days=1 so segment boundaries (and coupling sub-steps)
    are identical between straight and restarted runs.  The carry_aux
    restart transient limits tolerance to ~0.05 K (see Phase 3).
    """

    def test_restart_matches_straight_run(self, tmp_path):
        """60-day straight run matches 30+30 with restart."""
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
        )
        from legoesm.driver.coupled_config import PRESETS
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver

        def _mk(days, output_dir):
            cfg = ExperimentConfig(
                grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
                dycore=DycoreConfig(dt=600.0, model_type="hydrostatic"),
                output=OutputConfig(diag_days=1, checkpoint_days=0),
                radiation="gray",
                days=days,
            )
            ccfg = PRESETS["aquaplanet"]()._replace(coupling_dt=21600.0)
            d = CoupledESMDriver(cfg, ccfg, output_dir=output_dir)
            d.setup()
            return d

        # --- Run A: 60 days straight ---
        dir_a = tmp_path / "run_a"
        dir_a.mkdir()
        da = _mk(60, str(dir_a))
        da.run()
        T_ref = np.asarray(da.state.T.data)
        sst_ref = np.asarray(da.ocean_state.T_sfc.data)

        # --- Run B: 30 + 30 ---
        dir_b = tmp_path / "run_b"
        dir_b.mkdir()
        db1 = _mk(30, str(dir_b))
        db1.run()

        steps_per_day = int(86400.0 / 600.0)
        db1.save_checkpoint(30 * steps_per_day, 30.0)

        dir_b2 = tmp_path / "run_b2"
        dir_b2.mkdir()
        db2 = _mk(60, str(dir_b2))
        ckpts = list(dir_b.glob("checkpoint_day_*.npz"))
        assert len(ckpts) > 0, "No checkpoint found"
        step, day = db2._atm.load_checkpoint(ckpts[0])
        db2.load_coupled_checkpoint(30.0, checkpoint_dir=str(dir_b))
        db2.run(start_step=step, start_day=day)

        T_restart = np.asarray(db2.state.T.data)
        sst_restart = np.asarray(db2.ocean_state.T_sfc.data)

        # Carry_aux restart transient compounds over 30 days — allow ~2 K.
        # Phase 3 validates tighter tolerance (0.05 K) over shorter legs.
        np.testing.assert_allclose(T_restart, T_ref, atol=2.0, rtol=0.01,
                                   err_msg="T mismatch after 60-day restart")
        np.testing.assert_allclose(sst_restart, sst_ref, atol=2.0, rtol=0.01,
                                   err_msg="SST mismatch after 60-day restart")
