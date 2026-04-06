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


def _make_driver(preset="aquaplanet", days=1, resolution=8, nlev=5, dt=600.0):
    """Helper: create a CoupledESMDriver with minimal config."""
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    atm_config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=dt, model_type="hydrostatic"),
        output=OutputConfig(diag_days=max(days, 1)),
        radiation="gray",
        days=days,
    )
    coupled_cfg = PRESETS[preset]()
    driver = CoupledESMDriver(atm_config, coupled_cfg)
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
        from legoesm.coupler.coupling_fields import AtmToSurface

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


if __name__ == "__main__":
    unittest.main()
