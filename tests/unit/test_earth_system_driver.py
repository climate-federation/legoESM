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


class TestAtmToSurfaceConstruction:
    """Ensure AtmToSurface is always constructed with all required fields."""

    def test_all_fields_present(self):
        """AtmToSurface must have exactly 16 fields (height appended)."""
        from legoesm.core.coupling_fields import AtmToSurface

        shape = (6, 4, 4)
        ones = jnp.ones(shape)
        zeros = jnp.zeros(shape)

        forcing = AtmToSurface(
            sw_down=zeros,
            lw_down=zeros,
            precip_total=zeros,
            precip_snow=zeros,
            T_lowest=280.0 * ones,
            q_lowest=5e-3 * ones,
            u_lowest=5.0 * ones,
            v_lowest=2.0 * ones,
            p_lowest=1e5 * ones,
            p_surface=1.013e5 * ones,
            rho_lowest=1.2 * ones,
            cos_zenith=0.7 * ones,
            co2_ppmv=400.0 * ones,
            has_radiation=ones,
            has_precipitation=zeros,
        )
        assert len(forcing) == 16
        # Append-only ABI: all fifteen existing positional slots stay fixed.
        assert forcing._fields == (
            "sw_down", "lw_down", "precip_total", "precip_snow", "T_lowest",
            "q_lowest", "u_lowest", "v_lowest", "p_lowest", "p_surface",
            "rho_lowest", "cos_zenith", "co2_ppmv", "has_radiation",
            "has_precipitation", "z_lowest")
        assert hasattr(forcing, 'has_precipitation')
        assert hasattr(forcing, 'has_radiation')

    def test_missing_has_precipitation_raises(self):
        """Omitting has_precipitation must raise TypeError, not silently succeed."""
        from legoesm.core.coupling_fields import AtmToSurface

        shape = (6, 4, 4)
        ones = jnp.ones(shape)
        zeros = jnp.zeros(shape)

        with pytest.raises(TypeError):
            AtmToSurface(
                sw_down=zeros,
                lw_down=zeros,
                precip_total=zeros,
                precip_snow=zeros,
                T_lowest=280.0 * ones,
                q_lowest=5e-3 * ones,
                u_lowest=5.0 * ones,
                v_lowest=2.0 * ones,
                p_lowest=1e5 * ones,
                p_surface=1.013e5 * ones,
                rho_lowest=1.2 * ones,
                cos_zenith=0.7 * ones,
                co2_ppmv=400.0 * ones,
                has_radiation=ones,
                # has_precipitation intentionally missing
            )
