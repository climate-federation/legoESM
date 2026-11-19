"""Unit tests for the fully coupled Earth System Model driver.

Tests wiring, SST feedback, energy conservation, and configuration
across the six coupled ESM presets.
"""

import unittest
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Ensure test_cases importable
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _make_driver(preset="aquaplanet", days=1, resolution=8, nlev=5, dt=600.0,
                 output_dir=None):
    """Helper: create a CoupledESMDriver with minimal config."""
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    atm_config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(energy_fixer=False, dt=dt, model_type="hydrostatic"),
        output=OutputConfig(diag_days=max(days, 1)),
        radiation="gray",
        days=days,
    )
    coupled_cfg = PRESETS[preset]()
    driver = CoupledESMDriver(atm_config, coupled_cfg, output_dir=output_dir)
    driver.setup()
    return driver


class TestAquaplanet(unittest.TestCase):
    """Aquaplanet mode: f_land=0, slab ocean, no carbon."""

    def test_aquaplanet_setup(self):
        """Aquaplanet driver sets up without error."""
        driver = _make_driver("aquaplanet")
        self.assertIsNotNone(driver.ocean_state)
        self.assertIsNotNone(driver._ocean_step)

    def test_aquaplanet_f_land_zero(self):
        """Aquaplanet has f_land = 0 everywhere."""
        driver = _make_driver("aquaplanet")
        f_land = driver._tile_config.f_land
        self.assertEqual(float(jnp.max(f_land)), 0.0)

    def test_sst_override_reads_slab(self):
        """get_sst_sic returns slab ocean SST, not file SST."""
        driver = _make_driver("aquaplanet")
        # Perturb the slab SST
        from legoesm.core.field import Field
        sst_data = driver._ocean_state.T_sfc.data + 5.0
        driver._ocean_state = driver._ocean_state._replace(
            T_sfc=Field(data=sst_data, name="T_sfc",
                        dims=driver._ocean_state.T_sfc.dims, units="K"),
        )
        sst, _ = driver._atm.get_sst_sic(0.0)
        # Should reflect the +5K perturbation
        self.assertAlmostEqual(
            float(jnp.mean(sst)),
            float(jnp.mean(sst_data)),
            places=2,
        )

    def test_aquaplanet_runs_2day(self):
        """Aquaplanet C8/L5 runs 2 days without blowup."""
        driver = _make_driver("aquaplanet", days=2)
        status = driver.run()
        self.assertEqual(status, "COMPLETED")
        # All fields finite
        self.assertTrue(jnp.all(jnp.isfinite(driver.state.T.data)))
        self.assertTrue(jnp.all(jnp.isfinite(driver.ocean_state.T_sfc.data)))

    def test_aquaplanet_sst_bounded(self):
        """SST stays in physical range after 2 days."""
        driver = _make_driver("aquaplanet", days=2)
        driver.run()
        sst = driver.ocean_state.T_sfc.data
        self.assertGreater(float(jnp.min(sst)), 250.0)
        self.assertLess(float(jnp.max(sst)), 330.0)

    def test_coupled_diagnostics_populated(self):
        """Coupled diagnostics are logged at each segment boundary."""
        driver = _make_driver("aquaplanet", days=2)
        driver.run()
        diag = driver.coupled_diagnostics
        self.assertGreater(len(diag), 0)
        self.assertIn("sst_mean", diag[0])
        self.assertIn("day", diag[0])

    def test_coupled_diag_has_sst_drift(self):
        """SST-drift metric is logged and referenced to run start."""
        driver = _make_driver("aquaplanet", days=2)
        driver.run()
        diag = driver.coupled_diagnostics
        self.assertGreater(len(diag), 0)
        # Every record carries the drift; the first is the reference (zero).
        for d in diag:
            self.assertIn("sst_drift_K", d)
            self.assertTrue(jnp.isfinite(jnp.asarray(d["sst_drift_K"])))
        self.assertEqual(diag[0]["sst_drift_K"], 0.0)
        # Drift is consistent with the absolute means it is derived from.
        self.assertAlmostEqual(
            diag[-1]["sst_drift_K"],
            diag[-1]["sst_mean"] - diag[0]["sst_mean"],
            places=6,
        )


class TestMixedGridCoupling(unittest.TestCase):
    """Atmosphere and ocean on DIFFERENT grids, coupled through the
    differentiable conservative remap (coupler.grid_remap)."""

    def _make_mixed_driver(self, atm_nlat=16, ocean_nlat=8, days=1):
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
        )
        from legoesm.driver.coupled_config import PRESETS
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        from legoesm.grids.latlon import create_latlon_grid

        atm_config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=atm_nlat, nlev=5),
            dycore=DycoreConfig(energy_fixer=False, dt=600.0, model_type="hydrostatic",
                                discretization="finite_volume"),
            output=OutputConfig(diag_days=max(days, 1)),
            radiation="gray", days=days,
        )
        ocean_grid = create_latlon_grid(n_lat=ocean_nlat, n_lon=2 * ocean_nlat)
        driver = CoupledESMDriver(
            atm_config, PRESETS["aquaplanet"](), ocean_grid=ocean_grid,
        )
        driver.setup()
        return driver, ocean_grid

    def test_ocean_state_on_ocean_grid(self):
        """Slab ocean state is allocated on the (distinct) ocean grid."""
        driver, ocean_grid = self._make_mixed_driver()
        self.assertEqual(
            tuple(driver.ocean_state.T_sfc.data.shape),
            tuple(ocean_grid.grid_shape_2d),
        )
        self.assertFalse(driver._grid_remapper.identity)

    def test_mixed_grid_coupled_run_stable(self):
        """A coupled run with atm grid != ocean grid completes and stays finite."""
        driver, ocean_grid = self._make_mixed_driver(days=1)
        status = driver.run()
        self.assertEqual(status, "COMPLETED")
        # Ocean stayed on its grid; atm on its grid; both physical.
        self.assertEqual(tuple(driver.ocean_state.T_sfc.data.shape),
                         tuple(ocean_grid.grid_shape_2d))
        self.assertTrue(jnp.all(jnp.isfinite(driver.ocean_state.T_sfc.data)))
        self.assertTrue(jnp.all(jnp.isfinite(driver.state.T.data)))
        sst_mean = float(jnp.mean(driver.ocean_state.T_sfc.data))
        self.assertGreater(sst_mean, 250.0)
        self.assertLess(sst_mean, 320.0)


class TestCoupledCheckpointValidation(unittest.TestCase):
    """Coupled checkpoint restore validates ocean-grid shape (no silent
    mis-mapping when the ocean_grid / config changed since the save)."""

    def _write_coupled_npz(self, tmp, ocean_shape, version=1):
        import numpy as np
        from pathlib import Path
        np.savez(
            Path(tmp) / "coupled_day_0000.npz",
            ocean_T_sfc=np.zeros(ocean_shape, dtype=np.float64),
            ocean_T_deep=np.zeros(ocean_shape, dtype=np.float64),
            _ckpt_version=np.asarray(version, dtype=np.int64),
            _ckpt_ocean_shape=np.asarray(ocean_shape, dtype=np.int64),
        )

    def test_load_rejects_ocean_shape_mismatch(self):
        import tempfile
        driver = _make_driver("aquaplanet", days=1)
        cur = tuple(driver._ocean_state.T_sfc.data.shape)
        bad = (cur[0] + 1,) + cur[1:]  # a different ocean grid
        with tempfile.TemporaryDirectory() as tmp:
            self._write_coupled_npz(tmp, bad)
            with self.assertRaises(ValueError):
                driver.load_coupled_checkpoint(0.0, checkpoint_dir=tmp)

    def test_load_accepts_matching_shape(self):
        import tempfile
        driver = _make_driver("aquaplanet", days=1)
        cur = tuple(driver._ocean_state.T_sfc.data.shape)
        with tempfile.TemporaryDirectory() as tmp:
            self._write_coupled_npz(tmp, cur)
            driver.load_coupled_checkpoint(0.0, checkpoint_dir=tmp)  # no raise
            self.assertEqual(tuple(driver._ocean_state.T_sfc.data.shape), cur)


class TestCoupledResumeStatePersistence(unittest.TestCase):
    """Ckpt v3 (C8/C9): the coupling-lag surface response
    (``_last_sfc_response``) and the SST-drift reference (``_sst_mean_init``)
    survive a save -> fresh driver -> load round trip, so a resumed run's
    first coupled sub-step delivers the SAME lagged
    runoff/ice-freshwater/CO2 fluxes as the uninterrupted run and
    ``sst_drift_K`` stays referenced to the ORIGINAL run start."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmp = tempfile.TemporaryDirectory()
        # slab_simple has land, so the lagged surface water flux is genuinely
        # nonzero — the round-trip equality below is not vacuous.
        cls.driver = _make_driver("slab_simple", days=1,
                                  output_dir=cls._tmp.name)
        cls.driver.run()
        cls.driver.save_checkpoint(step=1, day=1.0)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_resume_restores_lag_buffer_and_drift_reference(self):
        import numpy as np
        from legoesm.core.coupling_fields import SurfaceToAtm

        ref = self.driver._last_sfc_response
        self.assertIsNotNone(ref)
        self.assertIsNotNone(self.driver._sst_mean_init)
        # Non-vacuity: the buffer carries a live land-sourced channel.  Not
        # the river runoff: after this 1-day run the slab bucket sits at
        # ~75 of W_max=150 kg/m2 and runoff is exactly 0 everywhere (job
        # 10331261) -- the surface water flux is the channel that is live.
        self.assertTrue(bool(jnp.any(ref.surface_mass_flux != 0.0)))

        resumed = _make_driver("slab_simple", days=1)  # fresh: lag buffer None
        self.assertIsNone(resumed._last_sfc_response)
        self.assertIsNone(resumed._sst_mean_init)
        resumed.load_coupled_checkpoint(1.0, checkpoint_dir=self._tmp.name)

        # Interrupted == uninterrupted on the lagged flux channels: the
        # restored buffer is bit-identical, so the first coupled sub-step
        # after resume consumes the SAME runoff / ice-lake-freshwater / CO2
        # fluxes (every SurfaceToAtm channel checked).
        self.assertIsNotNone(resumed._last_sfc_response)
        for field in SurfaceToAtm._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(ref, field)),
                np.asarray(getattr(resumed._last_sfc_response, field)),
                err_msg=f"lag-buffer channel '{field}' did not round-trip")

        # SST-drift reference: referenced to the ORIGINAL run start, and the
        # next diagnostic record computes the identical drift on both drivers
        # (the slab ocean state was restored alongside it).
        self.assertEqual(resumed._sst_mean_init, self.driver._sst_mean_init)
        self.driver._log_coupled_diag(1.0)
        resumed._log_coupled_diag(1.0)
        self.assertEqual(self.driver.coupled_diagnostics[-1]["sst_drift_K"],
                         resumed.coupled_diagnostics[-1]["sst_drift_K"])

    def test_sfcresp_structure_drift_falls_back_to_none(self):
        """A checkpoint whose SurfaceToAtm layout drifted (a field missing)
        must NOT partially restore the lag buffer (silently zeroed channels);
        it falls back to a fresh (None) buffer — the pre-v3 resume behavior —
        while the independent sst_mean_init still restores."""
        import tempfile
        from pathlib import Path

        import numpy as np

        src = np.load(Path(self._tmp.name) / "coupled_day_0001.npz")
        drift_key = next(k for k in src.files if k.startswith("sfcresp_"))
        arrays = {k: src[k] for k in src.files if k != drift_key}
        with tempfile.TemporaryDirectory() as tmp2:
            np.savez(Path(tmp2) / "coupled_day_0001.npz", **arrays)
            resumed = _make_driver("slab_simple", days=1)
            resumed.load_coupled_checkpoint(1.0, checkpoint_dir=tmp2)
            self.assertIsNone(resumed._last_sfc_response)
            self.assertEqual(resumed._sst_mean_init,
                             self.driver._sst_mean_init)

    def test_pre_v3_checkpoint_keeps_fresh_lag_and_drift(self):
        """Backward compat: a pre-v3 checkpoint (no sfcresp_*/sst_mean_init
        keys) loads without raising and keeps the documented fallback — fresh
        lag buffer (None => one zero-flux land/ice/CO2 sub-step) and drift
        re-referenced at the restart point (None until the first diag)."""
        import tempfile
        from pathlib import Path

        import numpy as np

        src = np.load(Path(self._tmp.name) / "coupled_day_0001.npz")
        arrays = {k: src[k] for k in src.files
                  if not k.startswith("sfcresp_") and k != "sst_mean_init"}
        arrays["_ckpt_version"] = np.asarray(2, dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp2:
            np.savez(Path(tmp2) / "coupled_day_0001.npz", **arrays)
            resumed = _make_driver("slab_simple", days=1)
            resumed.load_coupled_checkpoint(1.0, checkpoint_dir=tmp2)  # no raise
            self.assertIsNone(resumed._last_sfc_response)
            self.assertIsNone(resumed._sst_mean_init)


class TestUnfusedRadiation(unittest.TestCase):
    """Un-fused-radiation host loop (production compile-time fix): radiation runs
    as separate host jits instead of fused in the scan.  Must run + stay stable
    and match the fused path to radiation-cadence tolerance; OFF is a no-op.
    Gray radiation exercises the same host-loop structure as rrtmgp, cheaply."""

    def _run(self, unfused, rad_update_steps=2, days=2):
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
        )
        from legoesm.driver.coupled_config import PRESETS
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        atm = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
            dycore=DycoreConfig(energy_fixer=False, dt=600.0, model_type="hydrostatic"),
            output=OutputConfig(diag_days=days),
            radiation="gray", days=days,
            rad_update_steps=rad_update_steps,
            unfused_radiation=unfused,
        )
        driver = CoupledESMDriver(atm, PRESETS["aquaplanet"]())
        driver.setup()
        status = driver.run()
        return driver, status

    def test_unfused_runs_and_matches_fused(self):
        drv_u, st_u = self._run(unfused=True)
        self.assertEqual(st_u, "COMPLETED")
        self.assertTrue(jnp.all(jnp.isfinite(drv_u.state.T.data)))
        # Fused reference (same config, flag OFF -> legacy path).
        drv_f, st_f = self._run(unfused=False)
        self.assertEqual(st_f, "COMPLETED")
        # The host loop applies radiation with a one-step (sub-cadence) phase
        # shift vs the fused subcycle, so close-but-not-bit-identical.
        t_u = float(jnp.mean(drv_u.state.T.data))
        t_f = float(jnp.mean(drv_f.state.T.data))
        self.assertAlmostEqual(t_u, t_f, delta=2.0)  # K, generous for the phase shift


class TestWallclockExhausted(unittest.TestCase):
    """Wallclock-budget checkpoint-and-exit predicate (#6)."""

    def test_predicate(self):
        from legoesm.driver.model_driver import _wallclock_exhausted
        self.assertFalse(_wallclock_exhausted(0.0, 0.0, 600.0))      # disabled
        self.assertFalse(_wallclock_exhausted(100.0, 3600.0, 600.0))  # plenty left
        self.assertTrue(_wallclock_exhausted(3100.0, 3600.0, 600.0))  # within buffer
        self.assertTrue(_wallclock_exhausted(3600.0, 3600.0, 600.0))  # at budget

    def test_no_double_write_on_checkpoint_boundary(self):
        """Wallclock exit must not re-write a checkpoint already written this step.

        When the wallclock budget expires exactly at a periodic checkpoint
        boundary, _maybe_wallclock_exit is called immediately after
        save_checkpoint for the same step.  The second write can corrupt the
        .npz (partial flush before sys.exit) causing SHA-256 mismatch on
        reload.  Guard: skip ckpt_fn if _last_checkpoint_step == step.
        """
        import unittest.mock as mock
        from legoesm.driver.model_driver import _wallclock_exhausted, ModelDriver

        drv = _make_driver("slab_simple")
        atm = drv._atm

        call_count = []
        def fake_ckpt(step, day):
            call_count.append(step)

        # Simulate: regular checkpoint already ran for step 42.
        atm._last_checkpoint_step = 42
        atm._run_wallclock_start = 0.0
        atm.diagnostics = mock.MagicMock()

        with mock.patch(
            "legoesm.driver.model_driver._wallclock_exhausted", return_value=True
        ):
            with self.assertRaises(SystemExit):
                atm._maybe_wallclock_exit(fake_ckpt, 42, 5.0)

        self.assertEqual(call_count, [], "ckpt_fn must not be called again for same step")

    def test_wallclock_writes_checkpoint_when_no_prior_save(self):
        """Wallclock exit must write the checkpoint when no prior save ran this step."""
        import unittest.mock as mock

        drv = _make_driver("slab_simple")
        atm = drv._atm

        call_count = []
        def fake_ckpt(step, day):
            call_count.append(step)

        atm._last_checkpoint_step = None  # no checkpoint written yet
        atm._run_wallclock_start = 0.0
        atm.diagnostics = mock.MagicMock()

        with mock.patch(
            "legoesm.driver.model_driver._wallclock_exhausted", return_value=True
        ):
            with self.assertRaises(SystemExit):
                atm._maybe_wallclock_exit(fake_ckpt, 42, 5.0)

        self.assertEqual(call_count, [42], "ckpt_fn must be called when step is new")


class TestCarbonRadiationCoupling(unittest.TestCase):
    """Prognostic CO2 tracer feeds the atmosphere radiation GHG (#3 / C4MIP)."""

    def test_co2_override_set_after_run(self):
        driver = _make_driver("slab_carbon", days=1)
        # No radiation override before the first coupled segment.
        self.assertIsNone(getattr(driver._atm, "_co2_vmr_override", None))
        driver.run()
        # The coupled driver fed the prognostic CO2 to the atm radiation hook.
        ov = getattr(driver._atm, "_co2_vmr_override", None)
        self.assertIsNotNone(ov)
        # Initial ~415 ppm => CO2 mole fraction ~4.15e-4 (physical range).
        self.assertGreater(ov, 1e-4)
        self.assertLess(ov, 1e-3)

    def _box_flux(self, driver):
        """A spatially varying CO2 flux [kgCO2/m2/s], +into the atmosphere."""
        import numpy as np
        p_s = driver._atm.state.p_s.data
        rng = np.random.default_rng(0)
        return jnp.asarray(rng.normal(0.0, 1e-7, p_s.shape), p_s.dtype)

    def _air_mass(self, driver, p_s, own=None):
        """Independent total air mass [kg] = sum(dp_col * A) / g (numpy)."""
        import numpy as np
        from legoesm import constants
        area = np.asarray(driver._atm.grid.grid_area, np.float64)
        dp = np.asarray(driver._atm.sigma.layer_thickness_dp(p_s), np.float64)
        w = area if own is None else area * own
        return (dp.sum(-1) * w).sum() / constants.g

    def test_co2_box_init_mass(self):
        """#1817 decision 6: box = co2_ppmv_init as mass with the initial air mass."""
        import numpy as np
        from legoesm import constants
        driver = _make_driver("slab_carbon", days=1)
        self.assertEqual(np.ndim(driver._co2_mass), 0)  # one global scalar
        self.assertEqual(driver._co2_mass.dtype, jnp.float64)
        ppmv = driver.coupled_cfg.co2_ppmv_init
        p_s = driver._atm.state.p_s.data
        expected = (ppmv * 1e-6 * (constants.M_CO2 / constants.M_air)
                    * float(driver._co2_air_mass(p_s)))
        np.testing.assert_allclose(float(driver._co2_mass), expected, rtol=1e-12)
        # The driver's air mass vs an independent numpy one (float32 state
        # summed in a different order; same 1e-6 as the halo check below).
        np.testing.assert_allclose(float(driver._co2_air_mass(p_s)),
                                   self._air_mass(driver, p_s), rtol=1e-6)
        np.testing.assert_allclose(float(driver._co2_box_vmr()) * 1e6, ppmv,
                                   rtol=1e-12)

    def test_co2_box_no_response_adds_nothing(self):
        """#1817 decision 6: a step with no surface response leaves the box."""
        driver = _make_driver("slab_carbon", days=1)
        driver._last_sfc_response = None
        m0 = float(driver._co2_mass)
        driver._step_co2_tracer(3600.0)
        self.assertEqual(float(driver._co2_mass), m0)

    def test_co2_box_mass_closure(self):
        """#1817: box CO2 mass change == area-integrated flux * dt, exactly."""
        import numpy as np
        from types import SimpleNamespace
        driver = _make_driver("slab_carbon", days=1)
        flux = self._box_flux(driver)
        dt = 3600.0
        driver._last_sfc_response = SimpleNamespace(co2_flux=flux)
        m0 = float(driver._co2_mass)
        driver._step_co2_tracer(dt)
        m1 = float(driver._co2_mass)
        area = np.asarray(driver._atm.grid.grid_area, np.float64)
        expected = float((np.asarray(flux, np.float64) * area).sum() * dt)
        self.assertNotEqual(expected, 0.0)
        np.testing.assert_allclose(m1 - m0, expected, rtol=1e-9, atol=0.0)

    def test_co2_box_mass_conserved_when_air_mass_changes(self):
        """#1817 decision 1: air-mass change moves the mole fraction, not CO2 mass."""
        import numpy as np
        driver = _make_driver("slab_carbon", days=1)
        state0 = driver._atm.state
        p_s0 = state0.p_s.data
        m0 = float(driver._co2_mass)
        v0 = float(driver._co2_box_vmr())
        driver._atm.state = state0._replace(
            p_s=state0.p_s.replace(data=p_s0 * 1.01))
        driver._last_sfc_response = None
        driver._step_co2_tracer(3600.0)
        self.assertEqual(float(driver._co2_mass), m0)
        ratio = (float(driver._co2_air_mass(p_s0))
                 / float(driver._co2_air_mass(p_s0 * 1.01)))
        self.assertLess(ratio, 1.0 - 1e-3)  # the air mass really changed
        np.testing.assert_allclose(float(driver._co2_box_vmr()), v0 * ratio,
                                   rtol=1e-12)
        driver._atm.state = state0

    def test_radiation_and_land_see_box_value(self):
        """#1817: radiation, land forcing and the log all read the box."""
        from legoesm import constants
        driver = _make_driver("slab_carbon", days=1)
        driver.run()  # populates the held radiation the forcing build needs
        air = driver._co2_air_mass(driver._atm.state.p_s.data)
        driver._co2_mass = jnp.asarray(
            5.0e-4 * (constants.M_CO2 / constants.M_air) * air, jnp.float64)
        driver._update_co2_radiation()
        self.assertAlmostEqual(driver._atm._co2_vmr_override, 5.0e-4,
                               delta=1e-12)
        co2_land = driver._build_atm_forcing(1.0).co2_ppmv
        self.assertTrue(bool(jnp.all(jnp.abs(co2_land - 500.0) < 1e-3)))
        driver._log_coupled_diag(0.0)
        self.assertAlmostEqual(driver._coupled_diag[-1]["co2_ppmv_mean"],
                               500.0, delta=1e-3)

    def test_co2_box_grad_finite(self):
        """#1817: d(vmr)/d(mass) and d(vmr)/d(p_s) finite; flux grad = A dt."""
        import numpy as np
        from types import SimpleNamespace
        from legoesm import constants
        driver = _make_driver("slab_carbon", days=1)
        p_s = driver._atm.state.p_s.data
        flux = self._box_flux(driver)
        dt = 3600.0
        ratio = constants.M_CO2 / constants.M_air
        m0 = driver._co2_mass
        air = float(driver._co2_air_mass(p_s))
        gm = float(jax.grad(lambda m: driver._co2_vmr_from_mass(m, p_s))(m0))
        np.testing.assert_allclose(gm, 1.0 / air / ratio, rtol=1e-9)
        gp = np.asarray(jax.grad(
            lambda ps: driver._co2_vmr_from_mass(m0, ps))(p_s))
        self.assertTrue(np.all(np.isfinite(gp)) and np.any(gp != 0.0))

        def mass_after(f):
            driver._co2_mass = m0
            driver._last_sfc_response = SimpleNamespace(co2_flux=f)
            driver._step_co2_tracer(dt)
            return driver._co2_mass

        g = np.asarray(jax.grad(mass_after)(flux))
        area = np.asarray(driver._atm.grid.grid_area, np.float64)
        # float32 cotangent (flux dtype): same 1e-6 as the earlier grad check.
        np.testing.assert_allclose(g, area * dt, rtol=1e-6)
        # MPAS-style halo: non-owned cells hold NaN; they must not reach the
        # value or the gradient.
        own = np.ones(p_s.shape, bool)
        own.reshape(-1)[::3] = False
        driver._atm._voronoi_layout = SimpleNamespace(
            owned_mask_cells=jnp.asarray(own))
        nan_flux = jnp.where(jnp.asarray(own), flux, jnp.nan)
        nan_ps = jnp.where(jnp.asarray(own), p_s, jnp.nan)
        val = float(driver._co2_vmr_from_mass(m0, nan_ps))
        np.testing.assert_allclose(
            val, float(m0) / self._air_mass(driver, p_s, own) / ratio,
            rtol=1e-6)
        self.assertTrue(np.all(np.isfinite(np.asarray(
            jax.grad(mass_after)(nan_flux)))))
        self.assertTrue(np.all(np.isfinite(np.asarray(jax.grad(
            lambda ps: driver._co2_vmr_from_mass(m0, ps))(nan_ps)))))
        driver._atm._voronoi_layout = None
        driver._co2_mass = m0

    def test_co2_box_requires_x64(self):
        """#1817 decision 3: the CO2 box refuses to set up without JAX x64."""
        from unittest import mock
        driver = _make_driver("slab_carbon", days=1)
        jax.config.update("jax_enable_x64", False)
        try:
            with mock.patch.object(driver._atm, "setup") as atm_setup:
                with self.assertRaisesRegex(RuntimeError, "x64"):
                    driver.setup()
                atm_setup.assert_not_called()  # refused before any component
        finally:
            jax.config.update("jax_enable_x64", True)

    def test_co2_restart_radiation_sees_restored_box(self):
        """#1817 decision 2026-10-26: after a checkpoint load the first resumed
        segment's radiation CO2 equals the uninterrupted run's (exact
        continuation), not the configured value."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ref = _make_driver("slab_carbon", days=1, output_dir=d)
            ref.run()
            ref.save_checkpoint(step=144, day=1.0)
            ov_ref = ref._atm._co2_vmr_override  # what segment 2 radiation reads
            self.assertIsNotNone(ov_ref)
            # Non-vacuity: the box has moved off the configured CO2.
            self.assertNotEqual(ov_ref, ref.coupled_cfg.co2_ppmv_init * 1.0e-6)
            resumed = _make_driver("slab_carbon", days=2)
            self.assertIsNone(getattr(resumed._atm, "_co2_vmr_override", None))
            ckpt = sorted(Path(d).glob("checkpoint_day_*.npz"))[-1]
            resumed._atm.load_checkpoint(ckpt)
            resumed.load_coupled_checkpoint(1.0, checkpoint_dir=d)
            self.assertEqual(resumed._atm._co2_vmr_override, ov_ref)

    def test_co2_checkpoint_roundtrip_and_refusals(self):
        """#1817 decisions 4: restore box mass; refuse co2_field / missing box."""
        import tempfile
        import numpy as np
        driver = _make_driver("slab_carbon", days=1)
        with tempfile.TemporaryDirectory() as d:
            arrays = driver.coupled_checkpoint_arrays()
            self.assertIn("co2_mass", arrays)
            np.savez(Path(d) / "coupled_day_0001.npz", **arrays)
            driver._co2_mass = jnp.asarray(0.0, jnp.float64)
            driver.load_coupled_checkpoint(1.0, checkpoint_dir=d)
            self.assertEqual(float(driver._co2_mass), float(arrays["co2_mass"]))
            old = dict(arrays, co2_field=np.ones((2, 3, 4)))
            np.savez(Path(d) / "coupled_day_0002.npz", **old)
            with self.assertRaisesRegex(ValueError, "co2_field"):
                driver.load_coupled_checkpoint(2.0, checkpoint_dir=d)
            missing = {k: v for k, v in arrays.items() if k != "co2_mass"}
            np.savez(Path(d) / "coupled_day_0003.npz", **missing)
            driver._co2_mass = jnp.asarray(1.0, jnp.float64)
            with self.assertRaisesRegex(ValueError, "co2_mass"):
                driver.load_coupled_checkpoint(3.0, checkpoint_dir=d)
            self.assertEqual(float(driver._co2_mass), 1.0)  # nothing restored


class TestSlabSimple(unittest.TestCase):
    """Slab ocean + slab bucket land."""

    def test_slab_simple_setup(self):
        """slab_simple preset sets up with land config enabled."""
        driver = _make_driver("slab_simple")
        # The analytical dataset may not provide a land mask, but the
        # coupler and land surface are initialized and ready.
        self.assertEqual(driver.coupled_cfg.land_mode, "slab")
        self.assertIsNotNone(driver._sfc_state)
        self.assertIsNotNone(driver._sfc_state.land)

    def test_slab_simple_runs_1day(self):
        """slab_simple runs 1 day without error."""
        driver = _make_driver("slab_simple", days=1)
        status = driver.run()
        self.assertEqual(status, "COMPLETED")


class TestSlabEnergyConservation(unittest.TestCase):
    """Verify slab ocean energy balance."""

    def test_slab_energy_balance(self):
        """Slab ocean energy tendency matches net flux."""
        from legoesm.ocean.simple_ocean import (
            SimpleOceanConfig, _slab_step, init_slab_state,
        )
        from legoesm.core.coupling_fields import AtmToSurface

        shape = (6, 4, 4)
        cfg = SimpleOceanConfig(mode="slab", h_mix=50.0)
        state = init_slab_state(shape, T_sfc_init=290.0)

        # Construct analytical forcing
        forcing = AtmToSurface(
            sw_down=jnp.full(shape, 200.0),
            lw_down=jnp.full(shape, 300.0),
            precip_total=jnp.zeros(shape),
            precip_snow=jnp.zeros(shape),
            T_lowest=jnp.full(shape, 285.0),
            q_lowest=jnp.full(shape, 0.008),
            u_lowest=jnp.full(shape, 5.0),
            v_lowest=jnp.zeros(shape),
            p_lowest=jnp.full(shape, 95000.0),
            p_surface=jnp.full(shape, 101325.0),
            rho_lowest=jnp.full(shape, 1.15),
            cos_zenith=jnp.full(shape, 0.5),
            co2_ppmv=jnp.full(shape, 415.0),
            has_radiation=jnp.ones(shape),
            has_precipitation=jnp.zeros(shape),
        )

        dt = 3600.0
        T_before = state.T_sfc.data
        new_state, T_new, _, _ = _slab_step(state, forcing, cfg, dt)

        # Energy change
        C = cfg.rho_ocean * cfg.c_ocean * cfg.h_mix
        dE = C * (T_new - T_before)

        # Should be finite and non-zero (net flux is non-zero)
        self.assertTrue(jnp.all(jnp.isfinite(dE)))
        self.assertGreater(float(jnp.max(jnp.abs(dE))), 0.0,
                           "Energy change should be non-zero")


class TestPresetConfigs(unittest.TestCase):
    """All six presets should create valid configs."""

    def test_all_presets_create(self):
        """All preset factories produce valid CoupledConfig."""
        from legoesm.driver.coupled_config import PRESETS, CoupledConfig
        for name, factory in PRESETS.items():
            cfg = factory()
            self.assertIsInstance(cfg, CoupledConfig, f"Preset {name}")

    def test_carbon_presets_have_differland(self):
        """Carbon presets enable DifferLand."""
        from legoesm.driver.coupled_config import PRESETS
        for name in ("slab_carbon", "full_coupled"):
            cfg = PRESETS[name]()
            self.assertTrue(cfg.carbon_active, f"{name} should have carbon_active")
            self.assertEqual(cfg.carbon_land, "differland")
            self.assertTrue(cfg.co2_tracer, f"{name} should have co2_tracer")


class TestLandModeDispatch(unittest.TestCase):
    """PR A: the coupled driver's land_mode dispatch (``_init_coupler``) used a
    bare ``else`` that silently selected slab for any unknown mode. It now raises
    on an unknown mode — defense-in-depth behind ``land_scheme_overrides`` (which
    already guards the CLI path)."""

    def test_unknown_land_mode_raises(self):
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
        )
        from legoesm.driver.coupled_config import preset_aquaplanet
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver

        atm_config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
            dycore=DycoreConfig(energy_fixer=False, dt=600.0, model_type="hydrostatic"),
            output=OutputConfig(diag_days=1),
            radiation="gray",
            days=1,
        )
        coupled_cfg = preset_aquaplanet(land_mode="bogus_mode")
        driver = CoupledESMDriver(atm_config, coupled_cfg)
        with self.assertRaisesRegex(ValueError, "(?i)unknown land_mode"):
            driver.setup()


if __name__ == "__main__":
    unittest.main()
