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

        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            FV3EdgeShallowWaterModel, CDGridShallowWaterModel)
        assert isinstance(model, FV3EdgeShallowWaterModel)
        assert not isinstance(model, CDGridShallowWaterModel)
        # The validated preset must be wired VERBATIM (codex M2: the class
        # swap alone is insufficient) — config == williamson_cli_calibration(
        # grid.n) with only the three driver-exposed overrides applied.
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            williamson_cli_calibration, CDGridShallowWaterConfig)
        expected = williamson_cli_calibration(grid.n)._replace(
            use_conservation_fixer=config.dycore.conservation_fixer,
            fix_mass=config.dycore.fix_mass,
            time_integrator=(CDGridShallowWaterConfig().time_integrator
                             if config.dycore.time_integrator == "auto"
                             else config.dycore.time_integrator),
        )
        assert model.config == expected
        assert model.config.hyperdiff_coeff > 0.0
        assert model.config.div_damp > 0.0
        assert model.config.damp_v > 0.0

    def test_shallow_water_rejects_nondefault_diffusion_scale(self):
        # codex M2: the driver diffusion knobs have no effect on the fixed
        # cube-SW preset; a non-default scale must fail loudly, not silently
        # no-op.
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig)
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()
        for knob in ("hyperdiff_scale", "a_h_scale", "div_damp_scale"):
            config = ExperimentConfig(
                grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
                dycore=DycoreConfig(model_type="shallow_water",
                                    discretization="centered", dt=300.0,
                                    **{knob: 2.0}),
                days=1)
            with pytest.raises(ValueError, match=knob):
                create_atmosphere_dycore(config, grid, sigma)

    def test_shallow_water_rejects_resolution_mismatch(self):
        # codex M2: calibration uses grid.n; a grid/config resolution mismatch
        # must raise rather than build C24 damping on a C48 grid.
        config = _make_config(model_type="shallow_water", resolution=16)
        grid = _make_cubed_sphere_grid(resolution=8)  # grid.n=8 != cfg 16
        sigma = _make_sigma()
        with pytest.raises(ValueError, match="does not match"):
            create_atmosphere_dycore(config, grid, sigma)

    def test_hydrostatic_creates_pe_model(self):
        config = _make_config(model_type="hydrostatic")
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()
        model = create_atmosphere_dycore(config, grid, sigma)

        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import CDGridPrimitiveEquationModel
        assert isinstance(model, CDGridPrimitiveEquationModel)

    def test_nonhydrostatic_creates_ce_model(self):
        config = _make_config(model_type="nonhydrostatic")
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()
        model = create_atmosphere_dycore(config, grid, sigma)

        from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import CDGridCompressibleEulerModel
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
            # cdgrid_primitive_equations still forwards diff.hyperdiff to the
            # FV operator, so the guard applies.  (Cube SW was removed from the
            # explicit set in M2 — it uses a fixed validated preset that owns
            # its own hyperdiff, so the raw diff.hyperdiff is discarded; see
            # test_shallow_water_does_not_cry_wolf below.)
            warn_if_diffusion_unstable("cdgrid_primitive_equations", diff, grid, 600.0)
        assert any("hyperdiff" in r.message and "max stable" in r.message
                   for r in caplog.records)

    def test_shallow_water_does_not_cry_wolf(self, caplog):
        # M2 (codex): cube SW uses a fixed validated preset that owns its
        # hyperdiff, so an oversized generic diff.hyperdiff must NOT warn
        # (the model discards it).
        import logging
        from legoesm.driver.component_factory import warn_if_diffusion_unstable
        grid = _make_cubed_sphere_grid()
        diff = compute_diffusion(grid, DycoreConfig(dt=600.0, hyperdiff_scale=1e5))
        with caplog.at_level(logging.WARNING):
            warn_if_diffusion_unstable("cdgrid_shallow_water", diff, grid, 600.0)
        assert not any("max stable" in r.message for r in caplog.records)

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
            # Non-hydrostatic branches take (grid, height_coord, terrain_metric),
            # not sigma_coord — these two rows caught the factory passing
            # sigma_coord= to constructors that do not accept it.
            pytest.param(
                "gaussian", 21, "nonhydrostatic", "spectral", {},
                "SpectralCompressibleEulerModel", "grid",
                marks=pytest.mark.skipif(
                    not jax.config.read("jax_enable_x64"),
                    reason="spectral/Gaussian needs JAX_ENABLE_X64=1",
                ),
            ),
            ("mpas", 1, "nonhydrostatic", "mpas", {"lloyd_iterations": 2},
             "MPASCompressibleEulerModel", "mesh"),
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

        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import CDGridPrimitiveEquationModel
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
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            FV3EdgeShallowWaterModel)
        assert isinstance(driver.model, FV3EdgeShallowWaterModel)

    def test_driver_shallow_water_run_raises_clear_error(self):
        # codex M2: the factory builds the SW model for component-registry
        # use, but ModelDriver cannot RUN it — _init_state builds a
        # hydrostatic PE state, not a SW state.  BOTH public entries (setup,
        # run) must reject SW loudly BEFORE any dycore/scale-guard, and the
        # _init_state backstop must too — not crash cryptically at first step.
        config = _make_config(model_type="shallow_water")
        from legoesm.driver.model_driver import ModelDriver

        # Public run() rejects at the door, even without setup().
        driver = ModelDriver(config, output_dir="/tmp/test_driver_sw_run")
        with pytest.raises(NotImplementedError, match="not runnable via ModelDriver"):
            driver.run()
        # Public setup() rejects before any dycore/scale-guard construction.
        driver2 = ModelDriver(config, output_dir="/tmp/test_driver_sw_setup")
        with pytest.raises(NotImplementedError, match="not runnable via ModelDriver"):
            driver2.setup()
        # _init_state backstop still guards a direct call.
        driver3 = ModelDriver(config, output_dir="/tmp/test_driver_sw_init")
        with pytest.raises(NotImplementedError, match="not runnable via ModelDriver"):
            driver3._init_state()

    def test_driver_creates_nonhydrostatic_model(self):
        config = _make_config(model_type="nonhydrostatic")
        grid = _make_cubed_sphere_grid()
        sigma = _make_sigma()

        from legoesm.driver.model_driver import ModelDriver
        driver = ModelDriver(config, output_dir="/tmp/test_driver_nh")
        driver.grid = grid
        driver.sigma = sigma
        driver._create_dycore()

        from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import CDGridCompressibleEulerModel
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

        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import CDGridPrimitiveEquationModel
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
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon
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
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
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
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
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


# =========================================================================
# 10. time_integrator="auto" default resolves per-dycore
# =========================================================================

class TestTimeIntegratorAutoDefault:
    """DycoreConfig defaults to "auto"; every factory branch that forwards
    an integrator maps "auto" to its dycore's own stable default.

    Pins the fix for the MPAS trap: a direct ``DycoreConfig()`` (i.e. not
    via the run_amip CLI, whose default was already "auto") used to carry
    the global "ssp_rk3", which the MPAS PE dycore documents as UNSTABLE
    with hyperdiffusion at production dt (diverges within ~3 steps at
    dt=600)."""

    def test_dycore_config_default_is_auto(self):
        assert DycoreConfig().time_integrator == "auto"

    def test_mpas_default_resolves_to_dycore_default(self):
        """Factory on MPAS + default DycoreConfig → the MPAS dycore's own
        default integrator (the scan-folded large-stability SSP-RK54)."""
        from legoesm.grids.factory import create_grid
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationConfig,
        )
        grid = create_grid("mpas", 1, lloyd_iterations=2)
        config = _make_config(
            model_type="hydrostatic", discretization="mpas",
            grid_type="mpas", nlev=2,
        )
        assert config.dycore.time_integrator == "auto"
        model = create_atmosphere_dycore(config, grid, _make_sigma(2))
        expected = MPASPrimitiveEquationConfig().time_integrator
        assert expected == "ssp_rk54_scan"  # the documented stable default
        assert model.config.time_integrator == expected

    def test_cube_pe_default_resolves_to_dycore_default(self):
        """Cubed-sphere PE maps "auto" to its own default (ssp_rk3) — the
        pre-flip behaviour for the scientific validation suite."""
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationConfig,
        )
        config = _make_config(model_type="hydrostatic")
        grid = _make_cubed_sphere_grid()
        model = create_atmosphere_dycore(config, grid, _make_sigma())
        assert model.config.time_integrator == (
            CDGridPrimitiveEquationConfig().time_integrator
        )
        # "auto" itself must never leak into a model config (it is not a
        # dispatch_integrator key).
        assert model.config.time_integrator != "auto"

    def test_explicit_integrator_forwarded_verbatim(self):
        """An explicit scheme name is forwarded untouched (integrator-
        sensitivity runs stay possible)."""
        config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic",
                discretization="centered",
                dt=300.0,
                time_integrator="ssp_rk3_scan",
            ),
        )
        grid = _make_cubed_sphere_grid()
        model = create_atmosphere_dycore(config, grid, _make_sigma())
        assert model.config.time_integrator == "ssp_rk3_scan"


# =========================================================================
# 11. fix_mass forwarding on every branch with a fix_mass config field
# =========================================================================

class TestFixMassForwarding:
    """DycoreConfig.fix_mass must reach the dycore config on ALL branches
    that support it.  The cdgrid-CE and spectral-PE branches previously
    DROPPED it (silently ignoring fix_mass=True); the spectral-NH and
    MPAS-NH branches forward it (pinned here so they cannot regress)."""

    @pytest.mark.parametrize("flag", [True, False])
    def test_cdgrid_compressible_euler_forwards_fix_mass(self, flag):
        config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
            dycore=DycoreConfig(
                model_type="nonhydrostatic", discretization="centered",
                dt=300.0, fix_mass=flag,
            ),
        )
        grid = _make_cubed_sphere_grid()
        model = create_atmosphere_dycore(config, grid, _make_sigma())
        assert model.config.fix_mass is flag
        assert model.config.anchor_mass_to_initial is flag

    @pytest.mark.parametrize("flag", [True, False])
    @pytest.mark.skipif(
        not jax.config.read("jax_enable_x64"),
        reason="spectral/Gaussian needs JAX_ENABLE_X64=1",
    )
    def test_spectral_pe_forwards_fix_mass(self, flag):
        from legoesm.grids.factory import create_grid
        grid = create_grid("gaussian", 21)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="gaussian", resolution=21, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic", discretization="spectral",
                dt=300.0, fix_mass=flag,
            ),
        )
        model = create_atmosphere_dycore(config, grid, _make_sigma())
        assert model.config.fix_mass is flag
        assert model.config.anchor_mass_to_initial is flag

    @pytest.mark.parametrize("flag", [True, False])
    @pytest.mark.skipif(
        not jax.config.read("jax_enable_x64"),
        reason="spectral/Gaussian needs JAX_ENABLE_X64=1",
    )
    def test_spectral_nh_forwards_fix_mass(self, flag):
        from legoesm.grids.factory import create_grid
        grid = create_grid("gaussian", 21)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="gaussian", resolution=21, nlev=2),
            dycore=DycoreConfig(
                model_type="nonhydrostatic", discretization="spectral",
                dt=300.0, fix_mass=flag,
            ),
        )
        model = create_atmosphere_dycore(config, grid, _make_sigma(2))
        assert model.config.fix_mass is flag
        assert model.config.anchor_mass_to_initial is flag

    @pytest.mark.parametrize("flag", [True, False])
    def test_mpas_nh_forwards_fix_mass(self, flag):
        from legoesm.grids.factory import create_grid
        grid = create_grid("mpas", 1, lloyd_iterations=2)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="mpas", resolution=1, nlev=2),
            dycore=DycoreConfig(
                model_type="nonhydrostatic", discretization="mpas",
                dt=300.0, fix_mass=flag,
            ),
        )
        model = create_atmosphere_dycore(config, grid, _make_sigma(2))
        assert model.config.fix_mass is flag
        assert model.config.anchor_mass_to_initial is flag

    # conservation_fixer=False must OVERRIDE fix_mass=True on every
    # forwarding branch (the lat-lon contract; codex 2026-07-12 —
    # unconditional forwarding ignored the master conservation switch).
    def test_cdgrid_ce_conservation_fixer_false_overrides(self):
        config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=5),
            dycore=DycoreConfig(
                model_type="nonhydrostatic", discretization="centered",
                dt=300.0, fix_mass=True, conservation_fixer=False,
            ),
        )
        model = create_atmosphere_dycore(
            config, _make_cubed_sphere_grid(), _make_sigma())
        assert model.config.fix_mass is False
        assert model.config.anchor_mass_to_initial is False

    @pytest.mark.skipif(
        not jax.config.read("jax_enable_x64"),
        reason="spectral/Gaussian needs JAX_ENABLE_X64=1",
    )
    def test_spectral_pe_conservation_fixer_false_overrides(self):
        from legoesm.grids.factory import create_grid
        grid = create_grid("gaussian", 21)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="gaussian", resolution=21, nlev=5),
            dycore=DycoreConfig(
                model_type="hydrostatic", discretization="spectral",
                dt=300.0, fix_mass=True, conservation_fixer=False,
            ),
        )
        model = create_atmosphere_dycore(config, grid, _make_sigma())
        assert model.config.fix_mass is False
        assert model.config.anchor_mass_to_initial is False

    @pytest.mark.skipif(
        not jax.config.read("jax_enable_x64"),
        reason="spectral/Gaussian needs JAX_ENABLE_X64=1",
    )
    def test_spectral_nh_conservation_fixer_false_overrides(self):
        from legoesm.grids.factory import create_grid
        grid = create_grid("gaussian", 21)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="gaussian", resolution=21, nlev=2),
            dycore=DycoreConfig(
                model_type="nonhydrostatic", discretization="spectral",
                dt=300.0, fix_mass=True, conservation_fixer=False,
            ),
        )
        model = create_atmosphere_dycore(config, grid, _make_sigma(2))
        assert model.config.fix_mass is False
        assert model.config.anchor_mass_to_initial is False

    def test_mpas_nh_conservation_fixer_false_overrides(self):
        from legoesm.grids.factory import create_grid
        grid = create_grid("mpas", 1, lloyd_iterations=2)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="mpas", resolution=1, nlev=2),
            dycore=DycoreConfig(
                model_type="nonhydrostatic", discretization="mpas",
                dt=300.0, fix_mass=True, conservation_fixer=False,
            ),
        )
        model = create_atmosphere_dycore(config, grid, _make_sigma(2))
        assert model.config.fix_mass is False
        assert model.config.anchor_mass_to_initial is False

    # codex round 2: the MPAS hydrostatic PE and plane NH branches
    # pre-dated the audit but had the same ungated forwarding.
    def test_mpas_pe_conservation_fixer_false_overrides(self):
        from legoesm.grids.factory import create_grid
        grid = create_grid("mpas", 1, lloyd_iterations=2)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="mpas", resolution=1, nlev=2),
            dycore=DycoreConfig(
                model_type="hydrostatic", discretization="mpas",
                dt=300.0, fix_mass=True, conservation_fixer=False,
            ),
        )
        model = create_atmosphere_dycore(config, grid, _make_sigma(2))
        assert model.config.fix_mass is False

    def test_plane_conservation_fixer_false_overrides(self):
        from legoesm.grids.plane import create_plane_grid
        from legoesm.grids.vertical import create_height_coordinate
        config = ExperimentConfig(
            grid=GridConfig(grid_type="plane", resolution=8, nlev=6),
            dycore=DycoreConfig(
                model_type="nonhydrostatic", discretization="plane",
                dt=1.0, fix_mass=True, conservation_fixer=False,
            ),
        )
        grid = create_plane_grid(
            nx=8, ny=8, nlev=6, dx=10.0e3, dy=10.0e3, dtype=jnp.float64,
        )
        sigma = create_height_coordinate(6, H=30.0e3)
        model = create_atmosphere_dycore(config, grid, sigma)
        assert model.config.fix_mass is False
        assert model.config.anchor_mass_to_initial is False


# =========================================================================
# 12. Lat-lon SW polar-filter passthrough
# =========================================================================

class TestLatLonSWPolarFilterForwarding:
    """use_polar_filter (+ its two parameters) must reach the SW config.

    The dt/CFL relaxation in the factory already assumed the filter was
    ON (dt lifted to the equatorial CFL) while the SW model silently ran
    without it — the filter flags were only forwarded on the PE branch."""

    def test_polar_filter_reaches_sw_config(self):
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(16)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
            dycore=DycoreConfig(
                model_type="shallow_water", discretization="finite_volume",
                dt=600.0,
                use_polar_filter=True,
                polar_filter_cutoff_deg=65.0,
                polar_filter_max_wave_speed=250.0,
            ),
        )
        model = create_atmosphere_dycore(config, grid, _make_sigma(5))
        assert model.config.use_polar_filter is True
        assert model.config.polar_filter_cutoff_deg == 65.0
        assert model.config.polar_filter_max_wave_speed == 250.0
        # The masks (cell rows AND v-face rows) must actually be built.
        assert model._polar_mask is not None
        assert model._polar_mask_v is not None

    def test_polar_filter_off_by_default_on_sw(self):
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(16)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=5),
            dycore=DycoreConfig(
                model_type="shallow_water", discretization="finite_volume",
                dt=600.0,
            ),
        )
        model = create_atmosphere_dycore(config, grid, _make_sigma(5))
        assert model.config.use_polar_filter is False
        assert model._polar_mask is None
        assert model._polar_mask_v is None


# =========================================================================
# 9. moisture_flux_form must never be SILENTLY inert (#1354)
# =========================================================================

class TestMoistureFluxFormNotSilentlyInert:
    """``DycoreConfig.moisture_flux_form`` is a user-facing field (and a
    ``run_amip`` CLI flag).  A lane that does not wire it must RAISE at factory
    time — accepting the flag and then running the advective ``-(u.grad q)``
    transport is the "silently inert config field" defect this repo hardens
    against (CLAUDE.md *Dispatch*).
    """

    def test_helper_raises_on_true(self):
        from legoesm.driver.component_factory import (
            refuse_unwired_moisture_flux_form,
        )
        dc = DycoreConfig(moisture_flux_form=True)
        with pytest.raises(ValueError, match="moisture_flux_form"):
            refuse_unwired_moisture_flux_form(dc, "SOME LANE")

    def test_helper_names_the_lane(self):
        from legoesm.driver.component_factory import (
            refuse_unwired_moisture_flux_form,
        )
        dc = DycoreConfig(moisture_flux_form=True)
        with pytest.raises(ValueError, match="MPAS non-hydrostatic"):
            refuse_unwired_moisture_flux_form(
                dc, "MPAS non-hydrostatic (compressible Euler)")

    def test_helper_is_a_noop_when_false(self):
        from legoesm.driver.component_factory import (
            refuse_unwired_moisture_flux_form,
        )
        # Default (False) must not raise — otherwise every existing run breaks.
        refuse_unwired_moisture_flux_form(DycoreConfig(), "SOME LANE")
        refuse_unwired_moisture_flux_form(
            DycoreConfig(moisture_flux_form=False), "SOME LANE")

    @pytest.mark.parametrize("model_type", ["hydrostatic", "nonhydrostatic"])
    def test_mpas_factory_refuses_unwired_lane(self, model_type):
        """End-to-end through the factory: the MPAS lane(s) that do NOT
        implement flux-form transport refuse the flag instead of ignoring it."""
        from legoesm.grids.factory import create_grid

        grid = create_grid("mpas", 1, lloyd_iterations=2)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="mpas", resolution=1, nlev=2),
            dycore=DycoreConfig(
                model_type=model_type, discretization="mpas", dt=300.0,
                moisture_flux_form=True,
            ),
        )
        with pytest.raises(ValueError, match="moisture_flux_form"):
            create_atmosphere_dycore(config, grid, _make_sigma(2))
