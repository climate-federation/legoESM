"""ExperimentConfig.surface_bulk_scheme="none": no surface bulk exchange.

Validation, the YAML/CLI mapping, and the single-device lat-lon pipeline
honouring it (it used to apply constant-coefficient fluxes even with every
physics scheme off, while the multi-device lat-band lane applied none).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig, VALID_SURFACE_BULK)

_REPO = Path(__file__).resolve().parents[2]
_DT = 100.0
_DRY = dict(turbulence="none", radiation="none", convection="none",
            microphysics="none", gravity_wave_drag="none", cloud_scheme="none")


def test_none_is_a_valid_scheme_on_a_dry_config():
    assert "none" in VALID_SURFACE_BULK
    ExperimentConfig(surface_bulk_scheme="none", **_DRY).validate_strict()


@pytest.mark.parametrize("over", [
    {"turbulence": "louis"},
    {"physics_parameterization": "ml"},
    {"slab_land_active": True},
    {"use_multilayer_land": True},
    {"surface_tiled": True},
    {"C_H": 0.002},
    {"C_E": 0.002},
    {"surface_gustiness_zi": 600.0},
    {"surface_thermo_convention": "aerobulk"},
    {"surface_z_ref_model_level": True},
    {"surface_ocean_q_sfc_saline": True},
])
def test_none_rejected_next_to_a_surface_flux_owner(over):
    cfg = ExperimentConfig(surface_bulk_scheme="none", **{**_DRY, **over})
    with pytest.raises(ValueError, match="surface_bulk_scheme='none'"):
        cfg.validate_strict()


def test_none_cannot_be_coupled_to_an_ocean():
    from legoesm.driver.air_sea_consistency import validate_air_sea_consistency
    with pytest.raises(ValueError, match="cannot be coupled"):
        validate_air_sea_consistency(
            ExperimentConfig(surface_bulk_scheme="none", **_DRY), None)


def test_run_coupled_cli_refuses_none():
    from scripts.run.run_coupled import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--surface-bulk-scheme", "none"])


def test_held_suarez_template_selects_none():
    from legoesm.config import Config
    cfg = Config.from_yaml(
        _REPO / "config/templates/3d_idealized/held_suarez.yaml"
    ).to_experiment_config()
    assert cfg.surface_bulk_scheme == "none"
    cfg.validate_strict()


def test_run_amip_cli_round_trips_none():
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args)
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--surface-bulk-scheme", "none"]), parser)
    assert build_config_from_args(args).surface_bulk_scheme == "none"


def _one_step_T(tmp_path, sbs):
    """One Held-Suarez step on the single-device lat-lon driver, and the same
    step taken by the bare dycore with only the Held-Suarez forcing."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_forcing_latlon)
    from legoesm.driver.model_driver import ModelDriver
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(model_type="hydrostatic",
                            discretization="latlon_cgrid", dt=_DT, fix_mass=True),
        output=OutputConfig(output_dir=str(tmp_path / sbs), diag_days=0,
                            checkpoint_days=0),
        days=_DT / 86400.0, dataset="analytical", held_suarez_forcing=True,
        fix_moisture=True, enable_latlon_spmd=False, distributed=False,
        n_devices=1, surface_bulk_scheme=sbs, **_DRY)
    d = ModelDriver(cfg, output_dir=cfg.output.output_dir)
    d.setup()
    bare = d.model.step(d.state, _DT, physics_fn=held_suarez_forcing_latlon)
    assert d.run(compiled=True) == "COMPLETED"
    return (np.asarray(d.state.T.data, np.float64),
            np.asarray(bare.T.data, np.float64))


def test_single_device_driver_applies_no_surface_flux_under_none(tmp_path):
    drv, bare = _one_step_T(tmp_path, "none")
    # One fp32 rounding step at ~300 K is 3.05e-5 K; the constant-coefficient
    # bulk kick this replaces is ~9e-3 K at the bottom level.
    assert np.abs(drv - bare).max() < 1e-4
    drv_c, bare_c = _one_step_T(tmp_path, "constant")
    assert np.abs(drv_c - bare_c).max() > 1e-3


# ------------------------------------------------- pipeline step, every grid

def _grids():
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.voronoi import create_voronoi_mesh
    return {"cubed_sphere": lambda: create_cubed_sphere(4),
            "latlon": lambda: create_latlon_grid(8, 16),
            "mpas": lambda: create_voronoi_mesh(3, lloyd_iterations=8)}


@pytest.fixture(scope="module", params=["cubed_sphere", "latlon", "mpas"])
def pipe_grid(request):
    return _grids()[request.param]()


def _pipe(grid, sbs, **over):
    from tests.unit.test_prescribed_surface_fluxes_driver_path import _sigma
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    cfg = ExperimentConfig(diurnal_cycle=False, surface_bulk_scheme=sbs,
                           **{**_DRY, **over})
    return build_physics_pipeline(grid, _sigma(), cfg)


def test_pipeline_step_has_no_surface_exchange_under_none(pipe_grid):
    """Warm, moist-deficient surface: "constant" heats and moistens the bottom
    level, "none" leaves every tendency and surface flux at exactly zero."""
    from tests.unit.test_prescribed_surface_fluxes_driver_path import _step
    out = _step(_pipe(pipe_grid, "none"))
    for name in ("shflx", "lhflx", "evap_sfc", "dT_dt", "dq_v_dt",
                 "du_dt", "dv_dt"):
        assert np.all(np.asarray(getattr(out, name)) == 0.0), name
    ref = _step(_pipe(pipe_grid, "constant"))
    assert float(np.abs(np.asarray(ref.dT_dt)[..., -1]).max()) > 0.0
    assert float(np.abs(np.asarray(ref.dq_v_dt)[..., -1]).max()) > 0.0


def test_pipeline_step_refuses_a_second_surface_source_under_none(pipe_grid):
    import jax.numpy as jnp
    from tests.unit.test_prescribed_surface_fluxes_driver_path import (
        _inputs, _step)
    pipe = _pipe(pipe_grid, "none")
    _, s2 = _inputs(pipe)
    with pytest.raises(ValueError, match="no surface fluxes at all"):
        _step(pipe, sfc_shflx_override=jnp.zeros(s2),
              sfc_lhflx_override=jnp.zeros(s2))
    with pytest.raises(ValueError, match="no surface fluxes at all"):
        _step(_pipe(pipe_grid, "none", turbulence="louis"))
