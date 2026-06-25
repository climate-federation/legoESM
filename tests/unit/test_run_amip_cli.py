"""CLI coverage for the real AMIP entrypoint."""

from __future__ import annotations

import io
import sys

import pytest

from legoesm import constants
from scripts.run.run_amip import (
    _postprocess_args,
    _print_forcing_activity,
    build_arg_parser,
    build_config_from_args,
)


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


def test_convection_cli_choices_match_config_single_source():
    """--convection CLI choices MUST equal the driver config's authoritative
    VALID_CONVECTION_SCHEMES.  Regression guard: a stale hardcoded CLI choices
    list ({none,sbm,dca,kuo,mass_flux,edmf}) once rejected --convection tiedtke
    while ExperimentConfig.validate_strict accepted it, so every tiedtke AMIP
    job died at argparse."""
    from legoesm.driver.config import VALID_CONVECTION_SCHEMES

    parser = build_arg_parser()
    conv = next(a for a in parser._actions if a.dest == "convection")
    assert set(conv.choices) == set(VALID_CONVECTION_SCHEMES)
    # the profile-prognostic schemes wired in #477 must be selectable
    for scheme in ("tiedtke", "bechtold", "zhang_mcfarlane",
                   "kain_fritsch", "emanuel"):
        assert scheme in conv.choices


def test_convection_tiedtke_parses_and_flows_to_config():
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--convection", "tiedtke"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.convection == "tiedtke"


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


def test_surface_tiled_flags_flow_to_config():
    """--surface-tiled / --surface-z0-land round-trip into ExperimentConfig
    and validate together with --slab-land-active + --turbulence louis."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "louis",
        "--surface-bulk-scheme", "coare3",
        "--slab-land-active",
        "--surface-tiled",
        "--surface-z0-land", "0.15",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.surface_tiled is True
    assert cfg.surface_z0_land == 0.15
    assert cfg.slab_land_active is True
    assert cfg.turbulence == "louis"
    # The full combination must be self-consistent under strict validation.
    assert cfg.validate_strict() is None


def test_surface_tiled_defaults_off():
    """Without the flag, tiling is off (byte-identical legacy behaviour)."""
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.surface_tiled is False


def test_surface_tiled_requires_louis_rejected():
    """--surface-tiled with a non-louis scheme fails strict validation."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "holtslag_boville",
        "--slab-land-active",
        "--surface-tiled",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="surface_tiled.*louis"):
        cfg.validate_strict()


def test_soil_bucket_flags_flow_to_config():
    """--land-soil-bucket and its parameters round-trip into ExperimentConfig
    and validate together with an active land tile."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "louis",
        "--slab-land-active",
        "--land-soil-bucket",
        "--land-bucket-w-max", "120.0",
        "--land-beta-min", "0.2",
        "--land-bucket-w-init-frac", "0.4",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)

    assert cfg.land_soil_bucket is True
    assert cfg.land_bucket_w_max == 120.0
    assert cfg.land_beta_min == 0.2
    assert cfg.land_bucket_w_init_frac == 0.4
    assert cfg.validate_strict() is None


def test_soil_bucket_defaults_off():
    """Without the flag, the bucket is off (byte-identical legacy behaviour)."""
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.land_soil_bucket is False


def test_soil_bucket_requires_active_land_rejected():
    """--land-soil-bucket without an active land tile fails strict validation."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "louis",
        "--land-soil-bucket",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="land_soil_bucket.*land tile"):
        cfg.validate_strict()


def test_land_stomatal_beta_flag_flows_to_config():
    """--land-stomatal-beta round-trips and validates with the bucket on."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "louis",
        "--slab-land-active",
        "--land-soil-bucket",
        "--land-stomatal-beta",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.land_stomatal_beta is True
    assert cfg.validate_strict() is None


def test_land_stomatal_beta_defaults_off():
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    assert build_config_from_args(args).land_stomatal_beta is False


def test_land_stomatal_beta_requires_bucket_rejected():
    """--land-stomatal-beta without the bucket (no beta_soil) is rejected."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "louis",
        "--slab-land-active",
        "--land-stomatal-beta",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="land_stomatal_beta.*land_soil_bucket"):
        cfg.validate_strict()


def test_soil_bucket_rejects_bad_beta_min():
    """beta_min outside (0, 1] is rejected."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--turbulence", "louis",
        "--slab-land-active",
        "--land-soil-bucket",
        "--land-beta-min", "1.5",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="land_beta_min"):
        cfg.validate_strict()


def test_bechtold_cape_threshold_flows_to_config():
    """--bechtold-cape-threshold round-trips into ExperimentConfig and the
    Bechtold convection config (the coarse-resolution precip-deficit lever)."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--convection", "bechtold",
        "--bechtold-cape-threshold", "10.0",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.bechtold_cape_threshold == 10.0
    assert cfg.validate_strict() is None


def test_bechtold_cape_threshold_defaults_to_scheme_default():
    """Unset → matches BechtoldConfig.cape_threshold (byte-identical)."""
    parser = build_arg_parser()
    args = _postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser)
    cfg = build_config_from_args(args)
    assert cfg.bechtold_cape_threshold == 70.0


def test_bechtold_cape_threshold_rejects_negative():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--bechtold-cape-threshold", "-5.0",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    with pytest.raises(ValueError, match="bechtold_cape_threshold"):
        cfg.validate_strict()


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


def test_max_wallclock_seconds_threads_to_config():
    """--max-wallclock-seconds must wire into OutputConfig.max_wallclock_seconds."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--max-wallclock-seconds", "41400",
        "--checkpoint-days", "30",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.output.max_wallclock_seconds == 41400.0
    assert cfg.output.checkpoint_days == 30


def test_slurm_ntasks_1_does_not_trigger_distributed(monkeypatch):
    """SLURM_NTASKS=1 (set in every sbatch job) must NOT set distributed=True.
    Regression guard: prior bug made all single-task GPU sbatch jobs enter the
    MPI path and crash on SingleRankLayout.ownership."""
    import os
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    monkeypatch.delenv("PMI_SIZE", raising=False)
    monkeypatch.delenv("MPI_LOCALNRANKS", raising=False)
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    args = _postprocess_args(args, parser)
    assert not args.distributed, "SLURM_NTASKS=1 must not trigger distributed mode"


def test_slurm_ntasks_4_triggers_distributed(monkeypatch):
    """SLURM_NTASKS>1 (real multi-task MPI job) MUST set distributed=True."""
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    monkeypatch.delenv("PMI_SIZE", raising=False)
    monkeypatch.delenv("MPI_LOCALNRANKS", raising=False)
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    args = _postprocess_args(args, parser)
    assert args.distributed, "SLURM_NTASKS=4 must trigger distributed mode"


# ---------------------------------------------------------------------------
# G4a: _print_forcing_activity summary
# ---------------------------------------------------------------------------

def _capture_forcing_activity(extra_args: list[str]) -> str:
    """Parse args, run _print_forcing_activity, return captured stdout."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"] + extra_args)
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        _print_forcing_activity(args)
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_forcing_activity_all_channels_active_rrtmg():
    """All external forcing channels should print ACTIVE with rrtmg."""
    out = _capture_forcing_activity([
        "--radiation", "rrtmg",
        "--ghg-forcing", "external", "--ghg-file", "/tmp/ghg.nc",
        "--ozone-forcing", "external", "--ozone-file", "/tmp/o3.nc",
        "--aerosol-forcing", "external", "--aerosol-file", "/tmp/aer.nc",
        "--volcanic-aerosol-file", "/tmp/vol.nc",
        "--solar-source", "file", "--solar-file", "/tmp/solar.nc",
    ])
    assert "SST/SIC" in out and "ACTIVE" in out
    # Each radiation-gated channel should be ACTIVE
    assert out.count("ACTIVE") >= 4
    assert "inert" not in out


def test_forcing_activity_gray_marks_channels_inert():
    """With gray radiation, GHG/ozone/aerosol channels must be inert."""
    out = _capture_forcing_activity([
        "--radiation", "gray",
        "--ghg-forcing", "external", "--ghg-file", "/tmp/ghg.nc",
        "--ozone-forcing", "external", "--ozone-file", "/tmp/o3.nc",
    ])
    # GHG and ozone are radiation-gated → inert under gray
    assert "inert" in out or "gray radiation" in out


def test_forcing_activity_solar_constant_label():
    """Default solar (constant) should be labeled as constant, not ACTIVE file."""
    out = _capture_forcing_activity(["--radiation", "rrtmg"])
    assert "constant S_0" in out or "constant" in out


def test_slab_land_active_flag_threads_to_config():
    """--slab-land-active must reach ExperimentConfig.slab_land_active."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical", "--slab-land-active"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.slab_land_active is True


def test_slab_land_active_default_is_false():
    """slab_land_active must default to False (passive land is the safe default)."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.slab_land_active is False


def test_gustiness_zi_threads_to_config():
    """--gustiness-zi and --surface-bulk-scheme must reach ExperimentConfig."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--surface-bulk-scheme", "coare3",
        "--gustiness-zi", "300",
        "--turbulence", "holtslag_boville",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.surface_bulk_scheme == "coare3"
    assert cfg.surface_gustiness_zi == 300.0


def test_q_c_diagnostic_threads_to_config():
    """--q-c-diagnostic must reach ExperimentConfig.cloud_q_c_diagnostic."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dataset", "analytical",
        "--q-c-diagnostic", "3e-4",
    ])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.cloud_q_c_diagnostic == pytest.approx(3e-4)


def test_gustiness_defaults_off():
    """Gustiness and q_c_diagnostic disabled by default — opt-in only."""
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", "analytical"])
    args = _postprocess_args(args, parser)
    cfg = build_config_from_args(args)
    assert cfg.surface_gustiness_zi is None
    assert cfg.cloud_q_c_diagnostic is None
