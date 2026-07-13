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
        # FV3 single-implementation M2 (2026-07-13): the cube SW factory now
        # returns the FV3-faithful edge-midpoint core (FV3EdgeShallowWaterModel
        # + fv3_sw_tendencies), the model the Williamson matrix validates —
        # NOT the legacy corner-corner CDGridShallowWaterModel (which cannot
        # stabilize the cube W5/W6 wave class).  See
        # docs/architecture/fv3_single_implementation_program.md (Phase-1 M2).
        config = _make_config(model_type="shallow_water")
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()
        model = create_atmosphere_dycore(config, grid, sigma)

        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            FV3EdgeShallowWaterModel, CDGridShallowWaterModel)
        assert isinstance(model, FV3EdgeShallowWaterModel)
        assert not isinstance(model, CDGridShallowWaterModel)
        # The validated preset must be wired (div_damp + damp_v present),
        # else the factory would advertise the faithful core with unstable
        # defaults (the codex M2 scope-correction: swapping the class alone
        # is insufficient).
        assert model.config.div_damp > 0.0
        assert model.config.damp_v > 0.0

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

    def test_compute_diffusion_is_pure_no_warning(self, caplog):
        """compute_diffusion itself never warns — the guard lives at dispatch."""
        import logging
        grid = _make_cubed_sphere_grid()
        with caplog.at_level(logging.WARNING):
            compute_diffusion(grid, DycoreConfig(dt=600.0, hyperdiff_scale=1e9))
        assert not any("max stable" in r.message for r in caplog.records)


# =========================================================================
# 2b. Diffusive-CFL guard (warn-only, solver-aware)
# =========================================================================

class TestDiffusiveCFLGuard:
    """warn_if_diffusion_unstable fires only for solvers that use the coeff."""

    def test_explicit_solver_warns_on_oversized_hyperdiff(self, caplog):
        import logging
        from legoesm.driver.component_factory import warn_if_diffusion_unstable
        grid = _make_cubed_sphere_grid()
        diff = compute_diffusion(grid, DycoreConfig(dt=600.0, hyperdiff_scale=1e5))
        with caplog.at_level(logging.WARNING):
            warn_if_diffusion_unstable("cdgrid_shallow_water", diff, grid, 600.0)
        assert any("hyperdiff" in r.message and "max stable" in r.message
                   for r in caplog.records)

    def test_default_coeffs_do_not_warn(self, caplog):
        import logging
        from legoesm.driver.component_factory import warn_if_diffusion_unstable
        grid = _make_cubed_sphere_grid()
        diff = compute_diffusion(grid, DycoreConfig(dt=600.0))  # tuned defaults
        with caplog.at_level(logging.WARNING):
            warn_if_diffusion_unstable("cdgrid_primitive_equations", diff, grid, 600.0)
        assert not any("max stable" in r.message for r in caplog.records)

    def test_spectral_solver_does_not_cry_wolf(self, caplog):
        """Spectral recomputes its own implicit hyperdiff → no warn even for an
        absurd FV diff.hyperdiff (codex issue: warn only on consumed coeffs)."""
        import logging
        from legoesm.driver.component_factory import warn_if_diffusion_unstable
        grid = _make_cubed_sphere_grid()
        diff = compute_diffusion(grid, DycoreConfig(dt=600.0, hyperdiff_scale=1e9))
        with caplog.at_level(logging.WARNING):
            warn_if_diffusion_unstable("spectral_primitive_equations", diff, grid, 600.0)
        assert not any("max stable" in r.message for r in caplog.records)

    def test_divdamp_only_checked_for_primitive_equations(self, caplog):
        import logging
        from legoesm.driver.component_factory import warn_if_diffusion_unstable
        grid = _make_cubed_sphere_grid()
        diff = compute_diffusion(grid, DycoreConfig(dt=600.0, div_damp_scale=1e9))
        # Shallow water never forwards div_damp → silent.
        with caplog.at_level(logging.WARNING):
            warn_if_diffusion_unstable("cdgrid_shallow_water", diff, grid, 600.0)
        assert not any("div_damp" in r.message for r in caplog.records)
        caplog.clear()
        # Primitive equations forwards div_damp → warns.
        with caplog.at_level(logging.WARNING):
            warn_if_diffusion_unstable("cdgrid_primitive_equations", diff, grid, 600.0)
        assert any("div_damp" in r.message for r in caplog.records)


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
# 3b. Any global grid (built via create_grid) feeds a compatible atmosphere
# =========================================================================

class TestAtmosphereOnEveryGlobalGrid:
    """The user ask "instantiate any grid for the atmosphere": each global grid
    from the uniform factory builds a *compatible* dycore (cheapest per grid)."""

    @pytest.mark.parametrize(
        "grid_type,resolution,model_type,discretization,grid_kwargs,expected_cls,grid_attr",
        [
            # M2 (2026-07-13): cube SW factory returns the FV3-faithful
            # edge-midpoint core, not the legacy corner-corner CDGrid model.
            ("cubed_sphere", 8, "shallow_water", "cdgrid", {},
             "FV3EdgeShallowWaterModel", "grid"),
            pytest.param(
                "gaussian", 21, "shallow_water", "spectral", {},
                "SpectralShallowWaterModel", "grid",
                marks=pytest.mark.skipif(
                    not jax.config.read("jax_enable_x64"),
                    reason="spectral/Gaussian needs JAX_ENABLE_X64=1",
                ),
            ),
            ("latlon", 16, "shallow_water", "latlon_cgrid", {},
             "CGridLatLonShallowWaterModel", "grid"),
            # mpas exposes no shallow_water in the driver matrix -> hydrostatic;
            # the TRiSK mesh is held on the model as `.mesh`, not `.grid`.
            ("mpas", 1, "hydrostatic", "mpas", {"lloyd_iterations": 2},
             "MPASPrimitiveEquationModel", "mesh"),
        ],
    )
    def test_dycore_builds_on_factory_grid(
        self, grid_type, resolution, model_type, discretization,
        grid_kwargs, expected_cls, grid_attr,
    ):
        from legoesm.grids.factory import create_grid

        grid = create_grid(grid_type, resolution, **grid_kwargs)
        nlev = 5 if model_type == "shallow_water" else 2
        config = _make_config(
            model_type=model_type,
            discretization=discretization,
            grid_type=grid_type,
            resolution=resolution,
            nlev=nlev,
        )
        model = create_atmosphere_dycore(config, grid, _make_sigma(nlev))
        # Right *class* for this (grid, model_type, discretization) — not merely
        # something step-capable (every dycore has .step), so a key wired to the
        # wrong solver is caught.
        assert type(model).__name__ == expected_cls
        assert hasattr(model, "step")
        # The exact factory grid object is handed through to the model.
        assert getattr(model, grid_attr) is grid


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

        # M2 (2026-07-13): driver now builds the FV3-faithful edge-midpoint
        # cube SW core, not the legacy corner-corner CDGrid model.
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            FV3EdgeShallowWaterModel)
        assert isinstance(driver.model, FV3EdgeShallowWaterModel)

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


# =========================================================================
# 7. Lat-lon C-grid conservation_fixer=False
# =========================================================================

class TestLatLonConservationFixer:
    """conservation_fixer=False must disable the mass fixer."""

    def test_conservation_fixer_false_disables_fix_mass(self):
        """conservation_fixer=False should produce fix_mass=False."""
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(16)
        sigma = _make_sigma(5)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic",
                discretization="finite_volume",
                conservation_fixer=False,
                fix_mass=True,
            ),
        )
        model = create_atmosphere_dycore(config, grid, sigma)
        assert model.config.fix_mass is False, (
            "conservation_fixer=False must override fix_mass to False"
        )

    def test_conservation_fixer_true_preserves_fix_mass(self):
        """conservation_fixer=True (default) should keep fix_mass=True."""
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(16)
        sigma = _make_sigma(5)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic",
                discretization="finite_volume",
                fix_mass=True,
            ),
        )
        model = create_atmosphere_dycore(config, grid, sigma)
        assert model.config.fix_mass is True

    def test_conservation_fixer_false_propagates_to_driver_config(self):
        """conservation_fixer=False must also set fix_mass=False in the
        driver's DycoreConfig so that the compiled-segment driver-level
        mass fixer (fix_ps_mass_target) is also disabled."""
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.driver.model_driver import ModelDriver
        grid = create_latlon_grid(16)
        sigma = _make_sigma(5)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic",
                discretization="finite_volume",
                conservation_fixer=False,
                fix_mass=True,
            ),
            days=1,
        )
        driver = ModelDriver.__new__(ModelDriver)
        driver.config = config
        driver.grid = grid
        driver.sigma = sigma
        driver._create_dycore()
        # After _create_dycore, the driver config must have fix_mass=False
        assert driver.config.dycore.fix_mass is False, (
            "conservation_fixer=False must propagate to "
            "cfg.dycore.fix_mass=False for the compiled driver path"
        )


# =========================================================================
# 8. Lat-lon pole-cell CFL safeguards
# =========================================================================

class TestLatLonUnsupportedKnobs:
    """Factory must reject unsupported hyperdiff/div_damp on latlon_cgrid."""

    def test_hyperdiff_scale_accepted(self):
        """hyperdiff_scale is used by the driver for moisture smoothing,
        not by the lat-lon dycore.  It should not be rejected."""
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(16)
        sigma = _make_sigma(5)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic",
                discretization="finite_volume",
                hyperdiff_scale=2.0,
            ),
        )
        model = create_atmosphere_dycore(config, grid, sigma)
        assert model is not None

    def test_div_damp_scale_rejected(self):
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(16)
        sigma = _make_sigma(5)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic",
                discretization="finite_volume",
                div_damp_scale=2.0,
            ),
        )
        with pytest.raises(ValueError, match="divergence"):
            create_atmosphere_dycore(config, grid, sigma)

    def test_default_scales_accepted(self):
        """Default hyperdiff_scale=1.0, div_damp_scale=1.0 must not raise."""
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(16)
        sigma = _make_sigma(5)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic",
                discretization="finite_volume",
            ),
        )
        model = create_atmosphere_dycore(config, grid, sigma)
        assert model is not None

    def test_zero_scales_accepted(self):
        """hyperdiff_scale=0.0, div_damp_scale=0.0 (disabled) must not raise."""
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(16)
        sigma = _make_sigma(5)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic",
                discretization="finite_volume",
                hyperdiff_scale=0.0,
                div_damp_scale=0.0,
            ),
        )
        model = create_atmosphere_dycore(config, grid, sigma)
        assert model is not None


class TestLatLonPoleCFL:
    """Factory must clamp dt and A_h for the explicit C-grid lat-lon solver."""

    def test_unsafe_dt_is_clamped(self):
        """dt=600 on a 16x32 grid must be clamped to the pole-cell limit."""
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(16)
        sigma = _make_sigma(5)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic",
                discretization="finite_volume",
                dt=600.0,
            ),
        )
        model = create_atmosphere_dycore(config, grid, sigma)
        assert hasattr(model, "effective_dt")
        assert model.effective_dt < 600.0, (
            f"dt should have been clamped but effective_dt={model.effective_dt}"
        )

    def test_A_h_is_clamped(self):
        """A_h must not exceed the pole-cell diffusive CFL limit."""
        from legoesm.core.cfl import pole_cell_dx, max_laplacian_viscosity
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(16)
        sigma = _make_sigma(5)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic",
                discretization="finite_volume",
                dt=600.0,
            ),
        )
        model = create_atmosphere_dycore(config, grid, sigma)
        dx_pole = pole_cell_dx(grid)
        A_h_max = max_laplacian_viscosity(dx_pole, model.effective_dt)
        assert model.config.A_h <= A_h_max * 1.01, (
            f"A_h={model.config.A_h:.2e} exceeds limit {A_h_max:.2e}"
        )

    def test_clamped_model_runs_stable(self):
        """Multi-step stability with factory-clamped parameters."""
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.atmosphere.held_suarez import held_suarez_init_latlon
        grid = create_latlon_grid(16)
        sigma = _make_sigma(5)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic",
                discretization="finite_volume",
                dt=600.0,
            ),
        )
        model = create_atmosphere_dycore(config, grid, sigma)
        state = held_suarez_init_latlon(grid, sigma)
        dt = model.effective_dt
        for _ in range(10):
            state = model.step_with_physics(state, dt)
        assert jnp.all(jnp.isfinite(state.T.data))


# =========================================================================
# 9. create_model() grid-aware routing for lat-lon finite_volume
# =========================================================================

class TestCreateModelGridAware:
    """create_model() grid-aware routing for lat-lon."""

    def test_axis_resolution_reroutes_on_latlon(self):
        """Axis-based resolution (name=None, config with finite_volume)
        on LatLonGrid must produce CGridLatLon, not CDGrid.
        """
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics import create_model

        grid = create_latlon_grid(16)
        sigma = create_sigma_coordinate(5)
        # name=None triggers axis resolution → reroute on LatLonGrid
        model = create_model(
            legoesm_config={
                "atmosphere.dynamics": "hydrostatic",
                "atmosphere.discretization": "finite_volume",
            },
            grid=grid, sigma_coord=sigma,
        )
        from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationModel,
        )
        assert isinstance(model, CGridLatLonPrimitiveEquationModel), (
            f"Expected CGridLatLonPrimitiveEquationModel, got {type(model).__name__}"
        )

    def test_explicit_cdgrid_name_not_rerouted(self):
        """Explicit name='cdgrid_primitive_equations' must NOT be rerouted
        even when grid is LatLonGrid — the caller asked for that solver.
        """
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics import create_model
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel,
        )

        grid = create_latlon_grid(16)
        sigma = create_sigma_coordinate(5)
        # Explicit name= → no rerouting.  CDGrid model will get a
        # LatLonGrid it can't handle, which is the caller's problem;
        # the point is that the factory does not silently swap solvers.
        # We test the type check path by catching the expected error.
        try:
            model = create_model(
                name="cdgrid_primitive_equations",
                grid=grid, sigma_coord=sigma,
            )
            # If it succeeds, it must be the CDGrid model, not lat-lon.
            assert isinstance(model, CDGridPrimitiveEquationModel), (
                f"Explicit cdgrid request rerouted to {type(model).__name__}"
            )
        except (TypeError, AttributeError, ValueError):
            # Expected: CDGrid model can't handle LatLonGrid.
            pass
