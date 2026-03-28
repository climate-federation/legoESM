"""Tests for the coupled Earth System Driver.

Validates:
- EarthSystemDriver initializes atmosphere + coupler
- Coupled run completes without errors
- Surface state is initialized correctly
- Diagnostics are collected
"""

from __future__ import annotations

import pytest
import jax.numpy as jnp

from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig, OutputConfig
from legoesm.driver.earth_system_driver import EarthSystemDriver


class TestEarthSystemDriver:

    def _make_config(self):
        return ExperimentConfig(
            grid=GridConfig(resolution=8, nlev=5),
            dycore=DycoreConfig(dt=600.0),
            output=OutputConfig(diag_days=1),
            days=1,
            dataset="analytical",
            precision="fp64",
        )

    def test_setup_succeeds(self, tmp_path):
        """EarthSystemDriver.setup() initializes all components."""
        cfg = self._make_config()
        driver = EarthSystemDriver(cfg, output_dir=tmp_path)
        driver.setup()

        assert driver.state is not None
        assert driver.surface_state is not None
        assert driver._step_surface is not None
        assert driver._tile_config is not None

    def test_coupled_run_completes(self, tmp_path):
        """1-day coupled run completes without error."""
        cfg = self._make_config()
        driver = EarthSystemDriver(cfg, output_dir=tmp_path)
        driver.setup()
        status = driver.run()
        assert status == "COMPLETED"

    def test_surface_state_initialized(self, tmp_path):
        """Surface state has land, ice, and lake components."""
        cfg = self._make_config()
        driver = EarthSystemDriver(cfg, output_dir=tmp_path)
        driver.setup()

        sfc = driver.surface_state
        assert hasattr(sfc, 'land')
        assert hasattr(sfc, 'ice')
        assert hasattr(sfc, 'lake')

    def test_diagnostics_available(self, tmp_path):
        """Diagnostics are collected during coupled run."""
        cfg = self._make_config()
        driver = EarthSystemDriver(cfg, output_dir=tmp_path)
        driver.setup()
        driver.run()

        diag = driver.diagnostics
        assert diag is not None
        assert len(diag.times) >= 1
