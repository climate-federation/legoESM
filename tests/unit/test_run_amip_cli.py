"""CLI coverage for the real AMIP entrypoint."""

from __future__ import annotations

import pytest

from legoesm import constants
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


def test_enable_latlon_spmd_flag_flows_to_config():
    """--enable-latlon-spmd round-trips into ExperimentConfig (default off)."""
    parser = build_arg_parser()
    base = ["--dataset", "analytical", "--time-var", "month",
            "--lat-var", "ylat", "--lon-var", "xlon"]
    cfg_off = build_config_from_args(_postprocess_args(
        parser.parse_args(base), parser))
    assert cfg_off.enable_latlon_spmd is False

    cfg_on = build_config_from_args(_postprocess_args(
        parser.parse_args(base + ["--enable-latlon-spmd"]), parser))
    assert cfg_on.enable_latlon_spmd is True


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


def test_precision_flag_flows_to_config_and_validates():
    parser = build_arg_parser()
    for mode in ("fp32", "fp64", "mixed"):
        args = parser.parse_args(["--dataset", "analytical",
                                  "--precision", mode])
        args = _postprocess_args(args, parser)
        cfg = build_config_from_args(args)
        assert cfg.precision == mode
        cfg.validate_strict()  # membership-validated scheme Literal


def test_precision_default_is_fp32():
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]),
                             parser)
    assert build_config_from_args(args).precision == "fp32"


def test_precision_unknown_mode_rejected():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "analytical", "--precision", "bf16"])


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


# ---------------------------------------------------------------------------
# --aerosol-ccn grid gating: MPAS is now wired (Phase C); spectral standalone
# is still blocked.
# ---------------------------------------------------------------------------

_AEROSOL_CCN_BASE = [
    "--dataset", "analytical",
    "--aerosol-ccn",
    "--aerosol-forcing", "external",
    "--microphysics", "morrison",
]


def test_aerosol_ccn_allowed_on_mpas():
    parser = build_arg_parser()
    args = parser.parse_args(_AEROSOL_CCN_BASE + [
        "--grid-type", "voronoi",
        "--discretization", "mpas",
    ])
    # Must NOT raise now that the MPAS combined-physics path fills the
    # specified-Nc field.
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.nc_from_aerosol is True


def test_aerosol_ccn_still_blocked_on_spectral():
    parser = build_arg_parser()
    args = parser.parse_args(_AEROSOL_CCN_BASE + [
        "--discretization", "spectral",
    ])
    with pytest.raises(SystemExit):
        _postprocess_args(args, parser)


def test_aerosol_ccn_requires_external_forcing():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical", "--aerosol-ccn",
        "--microphysics", "morrison", "--grid-type", "voronoi",
        "--discretization", "mpas",
    ])
    with pytest.raises(SystemExit):
        _postprocess_args(args, parser)


def test_issue484_new_amip_flags_flow_to_config():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--hyperdiff-scale", "1.25",
        "--div-damp-scale", "0.75",
        "--no-conservation-fixer",
        "--no-fix-mass",
        "--max-wallclock-seconds", "7200",
        "--restart-buffer-seconds", "900",
        "--checkpoint-format", "zarr",
        "--seed", "123",
        "--forcing-update-days", "2.5",
        "--solar-s0", "1362.5",
        "--tau-equator", "8.1",
        "--tau-pole", "2.2",
        "--unfused-radiation",
        "--rrtmgp-gpoint-batch-size", "16",
        "--volcanic-aerosol-lw",
        "--t-ice-k", str(constants.T_freeze_ocean + 0.25),
        "--albedo-ice", "0.7",
        "--albedo-ocean", "0.08",
        "--sfc-emissivity", "0.96",
        "--emissivity-ice", "0.94",
        "--k-bl-max-per-day", "1.5",
        "--k-free-per-day", "0.2",
        "--aerosol-forcing", "external",
        "--microphysics", "morrison",
        "--nc-from-aerosol",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.dycore.hyperdiff_scale == 1.25
    assert cfg.dycore.div_damp_scale == 0.75
    assert cfg.dycore.conservation_fixer is False
    assert cfg.dycore.fix_mass is False
    assert cfg.output.max_wallclock_seconds == 7200
    assert cfg.output.restart_buffer_seconds == 900
    assert cfg.output.checkpoint_format == "zarr"
    assert cfg.seed == 123
    assert cfg.forcing_update_days == 2.5
    assert cfg.S_0 == 1362.5
    assert cfg.tau_equator == 8.1
    assert cfg.tau_pole == 2.2
    assert cfg.unfused_radiation is True
    assert cfg.rrtmgp_gpoint_batch_size == 16
    assert cfg.volcanic_aerosol_lw is True
    assert cfg.T_ice == constants.T_freeze_ocean + 0.25
    assert cfg.albedo_ice == 0.7
    assert cfg.albedo_ocean == 0.08
    assert cfg.sfc_emissivity == 0.96
    assert cfg.emissivity_ice == 0.94
    assert cfg.k_BL_max_per_day == 1.5
    assert cfg.k_free_per_day == 0.2
    assert cfg.nc_from_aerosol is True
    cfg.validate_strict()


def test_tuned_slab_knobs_flow_to_config():
    """The tuned air-sea + cloud knobs (mirroring run_coupled) round-trip into
    ExperimentConfig so AMIP can run with the tuned slab parameters; the defaults
    keep the prior AMIP behaviour (constant / 0 / None / off)."""
    parser = build_arg_parser()
    d = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert d.surface_bulk_scheme == "constant"
    assert d.surface_gustiness_zi == 0.0
    assert d.cloud_q_c_diagnostic is None
    assert d.cloud_rh_crit is None
    assert d.convective_cloud is False

    c = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical",
        "--surface-bulk-scheme", "coare3",
        "--gustiness-zi", "300",
        "--q-c-diagnostic", "3e-4",
        "--rh-crit", "0.8",
        "--convective-cloud",
    ]), parser))
    assert c.surface_bulk_scheme == "coare3"
    assert c.surface_gustiness_zi == 300.0
    assert c.cloud_q_c_diagnostic == pytest.approx(3e-4)
    assert c.cloud_rh_crit == 0.8
    assert c.convective_cloud is True
