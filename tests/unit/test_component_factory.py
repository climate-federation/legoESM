"""Tests for the driver-level component factory.

Validates that:
- Shallow-water, hydrostatic, and nonhydrostatic solvers resolve
  correctly for cubed-sphere grids.
- Unsupported combinations fail fast with precise error messages.
- The ModelDriver._create_dycore() delegates to the factory.
- The factory is consistent with the dynamics/__init__.py registry.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig
from legoesm.driver.component_factory import (
    create_atmosphere_dycore,
    compute_diffusion,
    supported_matrix,
    _DRIVER_SUPPORTED,
    DiffusionCoeffs,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_config(
    model_type="hydrostatic",
    discretization="centered",
    grid_type="cubed_sphere",
    resolution=8,
    nlev=5,
    dt=300.0,
) -> ExperimentConfig:
    """Build a minimal ExperimentConfig for testing."""
    return ExperimentConfig(
        grid=GridConfig(
            grid_type=grid_type,
            resolution=resolution,
            nlev=nlev,
        ),
        dycore=DycoreConfig(
            model_type=model_type,
            discretization=discretization,
            dt=dt,
        ),
        days=1,
    )


def _make_cubed_sphere_grid(resolution=8):
    """Create a real cubed-sphere grid for factory integration tests."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    return create_cubed_sphere(resolution)


def _make_sigma(nlev=5):
    """Create a sigma coordinate."""
    from legoesm.grids.vertical import create_sigma_coordinate
    return create_sigma_coordinate(nlev)


# =========================================================================
# 1. Solver resolution — cubed-sphere C-D grid
# =========================================================================

class TestCDGridResolution:
    """Factory resolves the correct CDGrid solver for each model_type."""

    def test_shallow_water_creates_sw_model(self):
        config = _make_config(model_type="shallow_water")
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()
        model = create_atmosphere_dycore(config, grid, sigma)

        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import CDGridShallowWaterModel
        assert isinstance(model, CDGridShallowWaterModel)

    def test_hydrostatic_creates_pe_model(self):
        config = _make_config(model_type="hydrostatic")
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()
        model = create_atmosphere_dycore(config, grid, sigma)

        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import CDGridPrimitiveEquationModel
        assert isinstance(model, CDGridPrimitiveEquationModel)

    def test_nonhydrostatic_creates_ce_model(self):
        config = _make_config(model_type="nonhydrostatic")
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()
        model = create_atmosphere_dycore(config, grid, sigma)

        from legoesm.atmosphere.dynamics.compressible_euler_cdgrid import CDGridCompressibleEulerModel
        assert isinstance(model, CDGridCompressibleEulerModel)

    def test_cdgrid_alias_resolves_same_as_centered(self):
        """discretization='cdgrid' is an alias for 'centered' on cubed_sphere."""
        config_c = _make_config(discretization="centered")
        config_d = _make_config(discretization="cdgrid")
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()

        model_c = create_atmosphere_dycore(config_c, grid, sigma)
        model_d = create_atmosphere_dycore(config_d, grid, sigma)
        assert type(model_c) is type(model_d)


# =========================================================================
# 2. Diffusion coefficients
# =========================================================================

class TestDiffusionCoeffs:
    """compute_diffusion returns physically sensible values."""

    def test_returns_named_tuple(self):
        grid = _make_cubed_sphere_grid()
        dc = DycoreConfig(dt=600.0)
        diff = compute_diffusion(grid, dc)
        assert isinstance(diff, DiffusionCoeffs)

    def test_A_h_positive(self):
        grid = _make_cubed_sphere_grid()
        dc = DycoreConfig(dt=600.0)
        diff = compute_diffusion(grid, dc)
        assert diff.A_h > 0.0

    def test_hyperdiff_positive(self):
        grid = _make_cubed_sphere_grid()
        dc = DycoreConfig(dt=600.0, hyperdiff_scale=1.0)
        diff = compute_diffusion(grid, dc)
        assert diff.hyperdiff > 0.0

    def test_div_damp_positive(self):
        grid = _make_cubed_sphere_grid()
        dc = DycoreConfig(dt=600.0, div_damp_scale=1.0)
        diff = compute_diffusion(grid, dc)
        assert diff.div_damp > 0.0

    def test_zero_scale_gives_zero(self):
        grid = _make_cubed_sphere_grid()
        dc = DycoreConfig(dt=600.0, hyperdiff_scale=0.0, div_damp_scale=0.0)
        diff = compute_diffusion(grid, dc)
        assert diff.hyperdiff == 0.0
        assert diff.div_damp == 0.0


# =========================================================================
# 3. Unsupported combinations fail fast
# =========================================================================

class TestFailFast:
    """Unsupported (model_type, discretization, grid_type) must raise."""

    def test_spectral_on_cubed_sphere_fails(self):
        config = _make_config(
            model_type="hydrostatic",
            discretization="spectral",
            grid_type="cubed_sphere",
        )
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()
        with pytest.raises(ValueError, match="Unsupported atmosphere"):
            create_atmosphere_dycore(config, grid, sigma)

    def test_cdgrid_on_gaussian_fails(self):
        config = _make_config(
            model_type="hydrostatic",
            discretization="cdgrid",
            grid_type="gaussian",
        )
        grid = MagicMock()
        sigma = _make_sigma()
        with pytest.raises(ValueError, match="Unsupported atmosphere"):
            create_atmosphere_dycore(config, grid, sigma)

    def test_unknown_grid_type_fails(self):
        config = _make_config(
            model_type="hydrostatic",
            discretization="centered",
            grid_type="triangulated",
        )
        grid = MagicMock()
        sigma = _make_sigma()
        with pytest.raises(ValueError, match="Unsupported atmosphere"):
            create_atmosphere_dycore(config, grid, sigma)

    def test_error_message_includes_suggestions(self):
        """Error message should list what IS available for the grid."""
        config = _make_config(
            model_type="hydrostatic",
            discretization="spectral",
            grid_type="cubed_sphere",
        )
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()
        with pytest.raises(ValueError, match="cubed_sphere.*supported"):
            create_atmosphere_dycore(config, grid, sigma)


# =========================================================================
# 4. ModelDriver delegates to factory
# =========================================================================

class TestDriverDelegation:
    """ModelDriver._create_dycore uses the factory."""

    def test_driver_creates_hydrostatic_model(self):
        """Default config (hydrostatic/centered/cubed_sphere) works."""
        config = _make_config(model_type="hydrostatic")
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()

        from legoesm.driver.model_driver import ModelDriver
        driver = ModelDriver(config, output_dir="/tmp/test_driver")
        driver.grid = grid
        driver.sigma = sigma
        driver._create_dycore()

        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import CDGridPrimitiveEquationModel
        assert isinstance(driver.model, CDGridPrimitiveEquationModel)
        assert driver._hyperdiff > 0.0

    def test_driver_creates_shallow_water_model(self):
        config = _make_config(model_type="shallow_water")
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()

        from legoesm.driver.model_driver import ModelDriver
        driver = ModelDriver(config, output_dir="/tmp/test_driver_sw")
        driver.grid = grid
        driver.sigma = sigma
        driver._create_dycore()

        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import CDGridShallowWaterModel
        assert isinstance(driver.model, CDGridShallowWaterModel)

    def test_driver_creates_nonhydrostatic_model(self):
        config = _make_config(model_type="nonhydrostatic")
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()

        from legoesm.driver.model_driver import ModelDriver
        driver = ModelDriver(config, output_dir="/tmp/test_driver_nh")
        driver.grid = grid
        driver.sigma = sigma
        driver._create_dycore()

        from legoesm.atmosphere.dynamics.compressible_euler_cdgrid import CDGridCompressibleEulerModel
        assert isinstance(driver.model, CDGridCompressibleEulerModel)


# =========================================================================
# 5. Supported matrix consistency
# =========================================================================

class TestSupportedMatrix:
    """The factory's supported matrix is consistent."""

    def test_supported_matrix_returns_list(self):
        mat = supported_matrix()
        assert isinstance(mat, list)
        assert len(mat) > 0

    def test_all_entries_have_required_keys(self):
        for entry in supported_matrix():
            assert "model_type" in entry
            assert "discretization" in entry
            assert "grid_type" in entry
            assert "solver" in entry

    def test_all_three_dynamics_levels_present(self):
        """Shallow-water, hydrostatic, and nonhydrostatic must all appear."""
        model_types = {e["model_type"] for e in supported_matrix()}
        assert "shallow_water" in model_types
        assert "hydrostatic" in model_types
        assert "nonhydrostatic" in model_types

    def test_cubed_sphere_has_all_three_levels(self):
        cs_entries = [e for e in supported_matrix() if e["grid_type"] == "cubed_sphere"]
        cs_types = {e["model_type"] for e in cs_entries}
        assert cs_types >= {"shallow_water", "hydrostatic", "nonhydrostatic"}


# =========================================================================
# 6. Config default preserves existing behavior
# =========================================================================

class TestDefaultBehavior:
    """DycoreConfig defaults produce the same model as before the refactor."""

    def test_default_config_creates_hydrostatic_cdgrid(self):
        """Default ExperimentConfig should create hydrostatic PE on cubed_sphere."""
        config = ExperimentConfig()
        grid = _make_cubed_sphere_grid(config.grid.resolution)
        sigma = _make_sigma(config.grid.nlev)
        model = create_atmosphere_dycore(config, grid, sigma)

        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import CDGridPrimitiveEquationModel
        assert isinstance(model, CDGridPrimitiveEquationModel)
