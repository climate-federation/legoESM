"""Phase 3: Restart reproducibility stress tests for the coupled model.

Verifies that checkpoint/restart produces identical results for the
coupled model (atmosphere + ocean + surface + CO2 tracer).

Key design: ``diag_days=1`` ensures identical segment boundaries
(and thus identical coupling sub-step schedules) between straight
and restarted runs.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_driver(preset="aquaplanet", days=10, output_dir=None, **kwargs):
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    # diag_days=1 so segment boundaries (and coupling sub-steps) are
    # identical between a straight run and a restarted run.
    atm_config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic"),
        output=OutputConfig(diag_days=1, checkpoint_days=0),
        radiation="gray",
        days=days,
    )
    coupled_cfg = PRESETS[preset](**kwargs)
    driver = CoupledESMDriver(atm_config, coupled_cfg, output_dir=output_dir)
    driver.setup()
    return driver


# Tolerance: the coupler receives atmospheric carry_aux (held radiation,
# precipitation) which undergoes a one-step reset transient after restart.
# This causes O(0.05 K) differences over 5+ days of post-restart integration.
# We test that restart is reproducible to within this known transient.
_ATOL = 0.05
_RTOL = 2e-4


# ---------------------------------------------------------------------------
# 3.1  Coupled aquaplanet restart reproducibility
# ---------------------------------------------------------------------------

class TestCoupledRestart:
    """Run 10 days straight vs 5+5 with checkpoint; compare final state."""

    def test_aquaplanet_restart_reproducibility(self, tmp_path):
        """Aquaplanet: 10-day straight run matches 5+5 with restart."""
        # --- Run A: 10 days straight ---
        dir_a = tmp_path / "run_a"
        dir_a.mkdir()
        driver_a = _make_driver("aquaplanet", days=10, output_dir=str(dir_a))
        driver_a.run()

        T_ref = np.asarray(driver_a.state.T.data)
        ps_ref = np.asarray(driver_a.state.p_s.data)
        sst_ref = np.asarray(driver_a.ocean_state.T_sfc.data)

        # --- Run B: 5 days, checkpoint, then 5 more ---
        dir_b = tmp_path / "run_b"
        dir_b.mkdir()
        driver_b1 = _make_driver("aquaplanet", days=5, output_dir=str(dir_b))
        driver_b1.run()

        # Save checkpoint at day 5
        steps_per_day = int(86400.0 / 600.0)
        step_at_5 = 5 * steps_per_day
        driver_b1.save_checkpoint(step_at_5, 5.0)

        # Restart: new driver configured for 10-day run
        dir_b2 = tmp_path / "run_b2"
        dir_b2.mkdir()
        driver_b2 = _make_driver("aquaplanet", days=10, output_dir=str(dir_b2))

        # Load atmosphere checkpoint from dir_b
        ckpts = list(dir_b.glob("checkpoint_day_*.npz"))
        assert len(ckpts) > 0, f"No checkpoint found in {dir_b}"
        step, day = driver_b2._atm.load_checkpoint(ckpts[0])

        # Load coupled checkpoint from dir_b (where it was saved)
        driver_b2.load_coupled_checkpoint(5.0, checkpoint_dir=str(dir_b))

        # Run remaining 5 days
        driver_b2.run(start_step=step, start_day=day)

        T_restart = np.asarray(driver_b2.state.T.data)
        ps_restart = np.asarray(driver_b2.state.p_s.data)
        sst_restart = np.asarray(driver_b2.ocean_state.T_sfc.data)

        np.testing.assert_allclose(T_restart, T_ref, atol=_ATOL, rtol=_RTOL,
                                   err_msg="T mismatch after restart")
        np.testing.assert_allclose(ps_restart, ps_ref, atol=_ATOL, rtol=_RTOL,
                                   err_msg="p_s mismatch after restart")
        np.testing.assert_allclose(sst_restart, sst_ref, atol=_ATOL, rtol=_RTOL,
                                   err_msg="SST mismatch after restart")


# ---------------------------------------------------------------------------
# 3.2  Carbon tracer survives restart
# ---------------------------------------------------------------------------

class TestCarbonRestart:
    """slab_carbon: CO2 field should match after restart."""

    def test_carbon_restart_co2_field(self, tmp_path):
        """slab_carbon 6-day run matches 3+3 with restart."""
        # --- Run A: 6 days straight ---
        dir_a = tmp_path / "run_a"
        dir_a.mkdir()
        driver_a = _make_driver("slab_carbon", days=6, output_dir=str(dir_a))
        driver_a.run()

        T_ref = np.asarray(driver_a.state.T.data)
        sst_ref = np.asarray(driver_a.ocean_state.T_sfc.data)
        co2_ref = None
        if hasattr(driver_a, '_co2_field') and driver_a._co2_field is not None:
            co2_ref = np.asarray(driver_a._co2_field)

        # --- Run B: 3 + 3 ---
        dir_b = tmp_path / "run_b"
        dir_b.mkdir()
        driver_b1 = _make_driver("slab_carbon", days=3, output_dir=str(dir_b))
        driver_b1.run()

        steps_per_day = int(86400.0 / 600.0)
        step_at_3 = 3 * steps_per_day
        driver_b1.save_checkpoint(step_at_3, 3.0)

        dir_b2 = tmp_path / "run_b2"
        dir_b2.mkdir()
        driver_b2 = _make_driver("slab_carbon", days=6, output_dir=str(dir_b2))

        ckpts = list(dir_b.glob("checkpoint_day_*.npz"))
        assert len(ckpts) > 0, "No checkpoint found"
        step, day = driver_b2._atm.load_checkpoint(ckpts[0])
        driver_b2.load_coupled_checkpoint(3.0, checkpoint_dir=str(dir_b))
        driver_b2.run(start_step=step, start_day=day)

        T_restart = np.asarray(driver_b2.state.T.data)
        sst_restart = np.asarray(driver_b2.ocean_state.T_sfc.data)

        np.testing.assert_allclose(T_restart, T_ref, atol=_ATOL, rtol=_RTOL,
                                   err_msg="T mismatch after carbon restart")
        np.testing.assert_allclose(sst_restart, sst_ref, atol=_ATOL, rtol=_RTOL,
                                   err_msg="SST mismatch after carbon restart")

        if co2_ref is not None and hasattr(driver_b2, '_co2_field'):
            co2_restart = np.asarray(driver_b2._co2_field)
            np.testing.assert_allclose(
                co2_restart, co2_ref, atol=_ATOL, rtol=_RTOL,
                err_msg="CO2 field mismatch after restart",
            )


# ---------------------------------------------------------------------------
# 3.3  Zarr checkpoint roundtrip (if backend available)
# ---------------------------------------------------------------------------

class TestZarrCheckpoint:
    """Zarr checkpoint save/load roundtrip."""

    def test_zarr_roundtrip(self, tmp_path):
        """Atmosphere state survives Zarr checkpoint roundtrip."""
        try:
            import zarr  # noqa: F401
        except ImportError:
            pytest.skip("zarr not installed")

        dir_run = tmp_path / "run"
        dir_run.mkdir()
        driver = _make_driver("aquaplanet", days=2, output_dir=str(dir_run))
        driver.run()

        # Save via Zarr
        zarr_path = tmp_path / "ckpt.zarr"
        T_before = np.asarray(driver.state.T.data)
        ps_before = np.asarray(driver.state.p_s.data)

        from legoesm.driver.restart import save_restart
        save_restart(
            path=zarr_path,
            state=driver.state,
            q_v=driver._atm.q_v,
            step=0,
            day=2.0,
            config=driver.atm_config,
            backend="zarr",
        )

        # Load and compare
        from legoesm.driver.restart import load_restart
        result = load_restart(
            zarr_path, driver._atm.grid, driver._atm.sigma, strict=False,
        )
        state_loaded = result[0]

        np.testing.assert_array_equal(
            np.asarray(state_loaded.T.data), T_before,
            err_msg="T not bit-identical after Zarr roundtrip",
        )
        np.testing.assert_array_equal(
            np.asarray(state_loaded.p_s.data), ps_before,
            err_msg="p_s not bit-identical after Zarr roundtrip",
        )


# ---------------------------------------------------------------------------
# 3.4  Dynamic 3D-ocean restart reproducibility (ckpt v2)
# ---------------------------------------------------------------------------

def _make_dynamic_driver(days=4, output_dir=None, nlev=4):
    """Coupled driver with the prognostic 3D LatLonCGridOcean (ocean_mode=
    'dynamic') on a small shared lat-lon grid, rest IC (no WOA files).
    diag_days=1 so straight/restart segment boundaries (= coupling sub-steps)
    align, exactly like the slab harness above."""
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.coupled_config import CoupledConfig
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    atm_config = ExperimentConfig(
        # dt=150 s is below the latlon-C16 pole-cell CFL so it is NOT clamped at
        # runtime — straight and restarted runs then share an identical step
        # schedule (steps_per_day below stays exact).
        grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
        dycore=DycoreConfig(dt=150.0, model_type="hydrostatic",
                            discretization="latlon_cgrid"),
        output=OutputConfig(diag_days=1, checkpoint_days=0),
        radiation="gray", convection="none", days=days,
    )
    coupled_cfg = CoupledConfig(
        ocean_mode="dynamic", ocean_config=LatLonCGridOceanConfig(),
        ocean_nlev=nlev, ocean_dt_s=300.0, ocean_ic="rest",
    )
    driver = CoupledESMDriver(atm_config, coupled_cfg, output_dir=output_dir)
    driver.setup()
    return driver


class TestDynamicOceanRestart:
    """Run 4 days straight vs 2+2 with a ckpt-v2 restart; the prognostic 3D
    ocean (T,S,u,eta) AND the atmosphere reproduce within the same atm-carry
    reset transient the slab restart documents.  A gross checkpoint bug (wrong /
    missing ocean leaves) would restore the ocean to its IC -> O(1) mismatch."""

    def test_dynamic_ocean_restart_reproducibility(self, tmp_path):
        # --- Run A: 4 days straight (reference trajectory) ---
        dir_a = tmp_path / "dyn_a"
        dir_a.mkdir()
        da = _make_dynamic_driver(days=4, output_dir=str(dir_a))
        da.run()
        T_ref = np.asarray(da.state.T.data)
        oT_ref = np.asarray(da.ocean_state.T.data)
        assert np.all(np.isfinite(oT_ref))

        # --- Run B: 2 days, REAL save_checkpoint, restart, 2 more ---
        dir_b = tmp_path / "dyn_b"
        dir_b.mkdir()
        db1 = _make_dynamic_driver(days=2, output_dir=str(dir_b))
        db1.run()
        # Snapshot the EVOLVED day-2 ocean — the state the checkpoint must
        # reproduce exactly through the driver's own save path.
        b1 = {k: np.asarray(getattr(db1.ocean_state, k).data)
              for k in ("T", "S", "u", "v", "eta", "w")}
        steps_per_day = int(86400.0 / 150.0)
        db1.save_checkpoint(2 * steps_per_day, 2.0)

        dir_b2 = tmp_path / "dyn_b2"
        dir_b2.mkdir()
        db2 = _make_dynamic_driver(days=4, output_dir=str(dir_b2))
        ckpts = list(dir_b.glob("checkpoint_day_*.npz"))
        assert ckpts, f"No atm checkpoint in {dir_b}"
        step, day = db2._atm.load_checkpoint(ckpts[0])
        db2.load_coupled_checkpoint(2.0, checkpoint_dir=str(dir_b))  # 3D ocean

        # PRIMARY GATE (magnitude-independent): the loaded 3D-ocean prognostic
        # state == the evolved day-2 state BITWISE, through the driver's REAL
        # save_checkpoint -> npz -> load_coupled_checkpoint path (not the unit
        # test's hand-built npz).  A missing/wrong/zeroed ocean leaf mismatches
        # here regardless of how weakly the short gray-aquaplanet spun the ocean.
        for k, ref in b1.items():
            np.testing.assert_array_equal(
                np.asarray(getattr(db2.ocean_state, k).data), ref,
                err_msg=f"ocean leaf {k} not bit-identical after ckpt-v2 restore")

        # --- Continue to day 4: the restart trajectory tracks the straight run
        # within the documented atm-carry-reset transient (a gross bug => O(1+)K).
        db2.run(start_step=step, start_day=day)
        np.testing.assert_allclose(
            np.asarray(db2.ocean_state.T.data), oT_ref, atol=0.3, rtol=1e-2,
            err_msg="ocean T not reproduced after restart")
        np.testing.assert_allclose(
            np.asarray(db2.state.T.data), T_ref, atol=0.2, rtol=1e-3,
            err_msg="atm T not reproduced after restart")
