"""Advertised-schemes-are-buildable contract for the unified driver.

``ExperimentConfig.validate_strict()`` advertises physics scheme literals
and ``_DRIVER_SUPPORTED`` advertises (model_type, discretization,
grid_type) triples.  The unified ModelDriver path must either build every
advertised combination or fail fast with an explicit
``NotImplementedError`` pointing at the per-model-type bridge factory —
never a misleading ``KeyError``/``AttributeError`` from deep inside the
build (audit 2026-06-10: microphysics p3/ml_emulator raised
``KeyError: Unknown scheme`` and the SFNO/U-Cast rows advertised
``cubed_sphere``, a grid they can never construct on).
"""

import pytest

from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
from legoesm.driver.component_factory import (
    _DRIVER_SUPPORTED,
    create_atmosphere_dycore,
)
from legoesm.driver.physics_pipeline import (
    _PIPELINE_UNSUPPORTED_CONVECTION,
    _PIPELINE_UNSUPPORTED_MICROPHYSICS,
    _resolve_convection,
    _resolve_gwd,
    _resolve_microphysics,
    _resolve_turbulence,
)

# Mirrors of the membership tuples in ExperimentConfig.validate_strict
# (driver/config.py).  Keep in sync — test_validate_strict_coverage.py
# guards the validate_strict side; this file guards that everything
# validate_strict accepts is honest at build time.
CONVECTION_LITERALS = (
    "sbm", "dca", "kuo", "mass_flux", "edmf", "zhang_mcfarlane",
    "kain_fritsch", "emanuel", "tiedtke", "bechtold", "none",
)
MICROPHYSICS_LITERALS = (
    "none", "kessler", "sundqvist", "seifert_beheng",
    "morrison", "thompson", "p3", "sdm", "fast_sbm", "ml_emulator",
)
TURBULENCE_LITERALS = (
    "smagorinsky", "louis", "tke", "mynn25", "clubb_lite", "clubb",
    "holtslag_boville", "ysu", "edmf", "none",
)
GWD_LITERALS = (
    "rayleigh", "lindzen", "mcfarlane", "hines",
    "prognostic_spectral", "e3sm_cam", "ml_emulator", "none",
)

# Shrink-only baselines: wiring a scheme into the unified pipeline may
# REMOVE entries here; ADDING one means a newly advertised scheme is
# unbuildable through the production driver — wire it or make that a
# conscious, reviewed decision.
EXPECTED_UNSUPPORTED_CONVECTION = frozenset(
    {"zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke", "bechtold"}
)
EXPECTED_UNSUPPORTED_MICROPHYSICS = frozenset({"p3", "ml_emulator"})


def _cfg(**physics) -> ExperimentConfig:
    """Minimal strict-valid ExperimentConfig with physics overrides."""
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
        dycore=DycoreConfig(
            model_type="hydrostatic", discretization="cdgrid",
        ),
        **physics,
    )
    config.validate_strict()
    return config


# =========================================================================
# 1. Scheme literals: resolve or explicit NotImplementedError
# =========================================================================

class TestSchemeLiteralsResolveOrExplicitNIE:

    @pytest.mark.parametrize("scheme", CONVECTION_LITERALS)
    def test_convection(self, scheme):
        config = _cfg(convection=scheme)
        if scheme in _PIPELINE_UNSUPPORTED_CONVECTION:
            with pytest.raises(NotImplementedError, match="bridge"):
                _resolve_convection(config)
        else:
            conv_fn, _ = _resolve_convection(config)
            assert callable(conv_fn)

    @pytest.mark.parametrize("scheme", MICROPHYSICS_LITERALS)
    def test_microphysics(self, scheme):
        config = _cfg(microphysics=scheme)
        if scheme == "none":
            assert _resolve_microphysics(config) == (None, None)
        elif scheme in _PIPELINE_UNSUPPORTED_MICROPHYSICS:
            with pytest.raises(NotImplementedError, match="bridge"):
                _resolve_microphysics(config)
        else:
            micro_fn, micro_config = _resolve_microphysics(config)
            assert callable(micro_fn)
            assert micro_config is not None

    @pytest.mark.parametrize("scheme", TURBULENCE_LITERALS)
    def test_turbulence(self, scheme):
        config = _cfg(turbulence=scheme)
        if scheme == "none":
            assert _resolve_turbulence(config) == (None, None)
        else:
            turb_fn, _ = _resolve_turbulence(config)
            assert callable(turb_fn)

    @pytest.mark.parametrize("scheme", GWD_LITERALS)
    def test_gravity_wave_drag(self, scheme):
        config = _cfg(gravity_wave_drag=scheme)
        if scheme == "none":
            assert _resolve_gwd(config) == (None, None)
        else:
            gwd_fn, _ = _resolve_gwd(config)
            assert callable(gwd_fn)

    def test_unsupported_sets_shrink_only(self):
        assert _PIPELINE_UNSUPPORTED_CONVECTION <= EXPECTED_UNSUPPORTED_CONVECTION
        assert (
            _PIPELINE_UNSUPPORTED_MICROPHYSICS
            <= EXPECTED_UNSUPPORTED_MICROPHYSICS
        )
        # The unsupported sets must stay inside the advertised literal sets
        # (an entry outside them would be pure dead configuration).
        assert _PIPELINE_UNSUPPORTED_CONVECTION <= set(CONVECTION_LITERALS)
        assert _PIPELINE_UNSUPPORTED_MICROPHYSICS <= set(MICROPHYSICS_LITERALS)


# =========================================================================
# 2. Data-driven driver rows: Gaussian-only, and they must construct
# =========================================================================

def _gaussian_grid_and_sigma(nlev=8):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import make_hybrid_levels
    return create_gaussian_grid(n_max=8), make_hybrid_levels(nlev)


class TestDataDrivenRowsBuildable:

    def test_data_driven_rows_are_gaussian_only(self):
        """SFNO is a spherical-harmonic operator and the U-Cast PE bridge
        packs the spectral PE state — neither can construct on a grid
        without SH transforms (e.g. cubed_sphere, which has no ``n_sh``)."""
        for (model_type, disc, grid_type), solver in _DRIVER_SUPPORTED.items():
            if disc in ("sfno", "u_cast"):
                assert grid_type == "gaussian", (
                    f"({model_type}, {disc}, {grid_type}) -> {solver}: "
                    f"data-driven solvers are Gaussian-only"
                )

    @pytest.mark.parametrize(
        "model_type,disc,expected_cls_module,expected_cls_name",
        [
            ("shallow_water", "sfno",
             "legoesm.atmosphere.dynamics.neural.sfno_sw", "SFNOShallowWaterModel"),
            ("hydrostatic", "sfno",
             "legoesm.atmosphere.dynamics.neural.sfno_pe",
             "SFNOPrimitiveEquationModel"),
            ("hydrostatic", "u_cast",
             "legoesm.atmosphere.dynamics.neural.ucast_pe",
             "UCastPrimitiveEquationModel"),
        ],
    )
    def test_advertised_data_driven_builds(
        self, model_type, disc, expected_cls_module, expected_cls_name,
    ):
        import importlib

        nlev = 8
        config = ExperimentConfig(
            grid=GridConfig(grid_type="gaussian", resolution=8, nlev=nlev),
            dycore=DycoreConfig(model_type=model_type, discretization=disc),
        )
        grid, sigma = _gaussian_grid_and_sigma(nlev)
        model = create_atmosphere_dycore(config, grid, sigma)
        expected_cls = getattr(
            importlib.import_module(expected_cls_module), expected_cls_name,
        )
        assert isinstance(model, expected_cls)

    def test_ucast_driver_default_sizes_channels_to_nlev(self):
        """The driver must size the default U-Cast network to 4*nlev+2
        (the ERA5-13-level default of 54 would fail the model ctor)."""
        from legoesm.ml.channel_packing import PE3DChannelSpec

        nlev = 8
        config = ExperimentConfig(
            grid=GridConfig(grid_type="gaussian", resolution=8, nlev=nlev),
            dycore=DycoreConfig(
                model_type="hydrostatic", discretization="u_cast",
            ),
        )
        grid, sigma = _gaussian_grid_and_sigma(nlev)
        model = create_atmosphere_dycore(config, grid, sigma)
        n_expected = PE3DChannelSpec(nlev=nlev).n_channels
        assert model.config.ucast_config.in_channels == n_expected
        assert model.config.ucast_config.out_channels == n_expected
