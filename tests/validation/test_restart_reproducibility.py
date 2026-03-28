"""Restart reproducibility test.

Validates that checkpoint/restart produces bit-identical results to a
straight-through run.  The test exercises:

- Full state round-trip through save_restart / load_restart (T, u, v, p_s, q_v).
- Held radiation fields (carry_aux) surviving the checkpoint boundary.
- Radiation sub-cycling restart with rad_update_steps=3.
- C8/L5 cubed-sphere with analytical forcing for speed.
- fp64 precision (mixed has a known digest issue).

Strategy
--------
Driver A: run 2 days straight.
Driver B: run 1 day, checkpoint, load checkpoint, run 1 more day.
Final states must match to float64 rounding tolerance.
"""
from __future__ import annotations

import glob

import numpy as np
import pytest

from legoesm.driver.config import (
    ExperimentConfig,
    GridConfig,
    DycoreConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    days: int,
    output_dir: str,
    checkpoint_days: int = 0,
) -> ExperimentConfig:
    """Build a minimal C8/L5 analytical config for fast restart testing."""
    return ExperimentConfig(
        grid=GridConfig(
            grid_type="cubed_sphere",
            resolution=8,
            nlev=5,
            vertical_coord="hybrid",
            p_top_Pa=200.0,
            stretching=2.0,
        ),
        dycore=DycoreConfig(
            model_type="hydrostatic",
            discretization="cdgrid",
            dt=600.0,
            hyperdiff_scale=1.0,
            div_damp_scale=1.0,
            conservation_fixer=True,
            fix_mass=True,
        ),
        output=OutputConfig(
            output_dir=output_dir,
            diag_days=0,
            checkpoint_days=checkpoint_days,
            monthly_means=False,
        ),
        days=days,
        start_day=0.0,
        dataset="analytical",
        radiation="gray",
        rad_update_steps=3,
        convection="sbm",
        turbulence="none",
        gravity_wave_drag="none",
        cloud_scheme="none",
        microphysics="none",
        topography="flat",
        T_init=300.0,
        RH_init=0.7,
        precision="fp64",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRestartReproducibility:
    """Checkpoint/restart must produce bit-identical results."""

    def test_restart_matches_straight_run(self, tmp_path):
        """Run 2 days straight vs 1+1 day with checkpoint in between.

        All prognostic fields (T, u, v, p_s, q_v) must match to fp64
        rounding tolerance after restart.
        """
        dir_a = str(tmp_path / "straight")
        dir_b1 = str(tmp_path / "restart_leg1")
        dir_b2 = str(tmp_path / "restart_leg2")

        # --- Driver A: straight 2-day run ---
        cfg_a = _make_config(days=2, output_dir=dir_a)
        driver_a = ModelDriver(cfg_a)
        driver_a.setup()
        status_a = driver_a.run(compiled=True)
        assert status_a == "COMPLETED", f"Straight run failed: {status_a}"

        # --- Driver B leg 1: run 1 day, then checkpoint ---
        cfg_b1 = _make_config(days=1, output_dir=dir_b1, checkpoint_days=1)
        driver_b1 = ModelDriver(cfg_b1)
        driver_b1.setup()
        status_b1 = driver_b1.run(compiled=True)
        assert status_b1 == "COMPLETED", f"Leg 1 failed: {status_b1}"

        # Find the checkpoint file written at end of leg 1
        ckpt_files = sorted(glob.glob(str(tmp_path / "restart_leg1" / "checkpoint_day_*.npz")))
        assert len(ckpt_files) >= 1, "No checkpoint file found after leg 1"
        ckpt_path = ckpt_files[-1]

        # --- Driver B leg 2: load checkpoint, run 1 more day ---
        # days=2 so that n_steps_total covers the full 2-day span;
        # start_step from the checkpoint makes the driver resume from
        # step 144 and run the remaining 144 steps (day 1 to day 2).
        cfg_b2 = _make_config(days=2, output_dir=dir_b2)
        driver_b2 = ModelDriver(cfg_b2)
        driver_b2.setup()
        step, day = driver_b2.load_checkpoint(ckpt_path)
        # Pass start_day=0.0 (original epoch), not day=1.0 (current day).
        # The step counter handles the offset within the time loop; start_day
        # is the epoch reference for computing calendar day from step index.
        status_b2 = driver_b2.run(start_step=step, start_day=0.0, compiled=True)
        assert status_b2 == "COMPLETED", f"Leg 2 failed: {status_b2}"

        # --- Compare final states ---
        atol = 1e-12  # fp64 rounding tolerance

        np.testing.assert_allclose(
            np.asarray(driver_a.state.T.data),
            np.asarray(driver_b2.state.T.data),
            atol=atol, rtol=0,
            err_msg="Temperature (T) mismatch after restart",
        )
        np.testing.assert_allclose(
            np.asarray(driver_a.state.u.data),
            np.asarray(driver_b2.state.u.data),
            atol=atol, rtol=0,
            err_msg="Zonal wind (u) mismatch after restart",
        )
        np.testing.assert_allclose(
            np.asarray(driver_a.state.v.data),
            np.asarray(driver_b2.state.v.data),
            atol=atol, rtol=0,
            err_msg="Meridional wind (v) mismatch after restart",
        )
        np.testing.assert_allclose(
            np.asarray(driver_a.state.p_s.data),
            np.asarray(driver_b2.state.p_s.data),
            atol=atol, rtol=0,
            err_msg="Surface pressure (p_s) mismatch after restart",
        )
        np.testing.assert_allclose(
            np.asarray(driver_a.q_v),
            np.asarray(driver_b2.q_v),
            atol=atol, rtol=0,
            err_msg="Specific humidity (q_v) mismatch after restart",
        )

    def test_carry_aux_survives_checkpoint(self, tmp_path):
        """Held radiation fields must round-trip through checkpoint.

        With rad_update_steps=3, the held radiation tendencies
        (dT_rad, sw/lw surface/TOA fluxes) are part of the carry state
        and must be preserved across restart boundaries.
        """
        dir_out = str(tmp_path / "carry_aux")

        cfg = _make_config(days=1, output_dir=dir_out, checkpoint_days=1)
        driver = ModelDriver(cfg)
        driver.setup()
        status = driver.run()
        assert status == "COMPLETED"

        # Verify carry_aux was populated (rad_update_steps=3 means
        # held fields are nontrivial after the first radiation call)
        assert driver._carry_aux, "carry_aux should be populated after run"

        held_keys = [
            "held_dT_rad",
            "held_sw_net_sfc",
            "held_lw_net_sfc",
            "held_sw_up_toa",
            "held_lw_up_toa",
            "held_sw_down_toa",
        ]
        for key in held_keys:
            assert key in driver._carry_aux, f"Missing carry_aux key: {key}"

        # Load checkpoint and verify carry_aux round-trips
        ckpt_files = sorted(glob.glob(str(tmp_path / "carry_aux" / "checkpoint_day_*.npz")))
        assert len(ckpt_files) >= 1, "No checkpoint file found"

        driver2 = ModelDriver(cfg)
        driver2.setup()
        step, day = driver2.load_checkpoint(ckpt_files[-1])

        for key in held_keys:
            original = np.asarray(driver._carry_aux[key])
            restored = np.asarray(driver2._carry_aux[key])
            np.testing.assert_array_equal(
                original, restored,
                err_msg=f"carry_aux[{key!r}] not preserved through checkpoint",
            )
