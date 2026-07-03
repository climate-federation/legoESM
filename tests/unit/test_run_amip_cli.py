"""CLI coverage for the real AMIP entrypoint."""

from __future__ import annotations

import pytest

from scripts.run.run_amip import build_arg_parser, build_config_from_args, _postprocess_args


def test_build_config_includes_joint_physics_parameterization_flags():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--radiation", "rrtmgp",
        "--convection", "mass_flux",
        "--turbulence", "louis",
        "--gravity-wave-drag", "rayleigh",
        "--time-var", "month",
        "--lat-var", "ylat",
        "--lon-var", "xlon",
        "--held-suarez-forcing",
        "--physics-parameterization", "ml",
        "--physics-parameterization-checkpoint", "physics.eqx",
        "--physics-parameterization-stats", "physics_stats.npz",
        "--physics-parameterization-hidden-dim", "192",
        "--physics-parameterization-layers", "4",
        "--physics-parameterization-seed", "7",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.radiation == "rrtmg"
    assert cfg.convection == "mass_flux"
    assert cfg.turbulence == "louis"
    assert cfg.gravity_wave_drag == "rayleigh"
    assert cfg.time_var == "month"
    assert cfg.lat_var == "ylat"
    assert cfg.lon_var == "xlon"
    assert cfg.held_suarez_forcing is True
    assert cfg.physics_parameterization == "ml"
    assert cfg.physics_parameterization_checkpoint == "physics.eqx"
    assert cfg.physics_parameterization_stats == "physics_stats.npz"
    assert cfg.physics_parameterization_hidden_dim == 192
    assert cfg.physics_parameterization_layers == 4
    assert cfg.physics_parameterization_seed == 7


def test_build_config_includes_surfdata_path():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--land-mask-file", "sftlf.nc",
        "--surfdata", "legoesm_surfdata.nc",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.land_mask_path == "sftlf.nc"
    assert cfg.surfdata_path == "legoesm_surfdata.nc"


def test_build_config_surfdata_defaults_empty():
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.surfdata_path == ""


def test_surfdata_without_land_mask_warns(capsys):
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--surfdata", "sd.nc"])
    _postprocess_args(args, parser)
    assert "ignored without --land-mask-file" in capsys.readouterr().out


def test_joint_parameterization_requires_mass_flux_and_louis():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--physics-parameterization", "ml",
        "--physics-parameterization-checkpoint", "physics.eqx",
        "--physics-parameterization-stats", "physics_stats.npz",
    ])
    with pytest.raises(SystemExit):
        _postprocess_args(args, parser)


def test_spectral_postprocess_promotes_gaussian_grid():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--discretization", "spectral",
        "--truncation", "42",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.grid.grid_type == "gaussian"
    assert cfg.grid.resolution == 42
    assert cfg.dycore.discretization == "spectral"


def test_ic_era5_threads_through_to_config():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--ic", "era5",
        "--ic-path", "/tmp/era5_test.zarr",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.ic == "era5"
    assert cfg.ic_path == "/tmp/era5_test.zarr"


def test_ic_default_is_backward_compatible():
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.ic == "default"
    assert cfg.ic_path == ""


def test_ic_era5_without_ic_path_fails():
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--ic", "era5"])
    with pytest.raises(SystemExit):
        _postprocess_args(args, parser)
