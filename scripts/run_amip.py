#!/usr/bin/env python
"""AMIP simulation via the canonical ``ModelDriver`` path.

Usage:
    JAX_ENABLE_X64=1 python scripts/run_amip.py \
        --dataset analytical --days 30 --resolution 16 --dt 600

    JAX_ENABLE_X64=1 python scripts/run_amip.py \
        --dataset cobe --forcing-path /path/to/MODEL.SST.COBE-SST2.nc \
        --radiation rrtmgp --convection sbm --turbulence louis \
        --days 30 --resolution 16 --dt 600
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Enable line-buffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AMIP simulation with prescribed SST/SIC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Forcing
    parser.add_argument("--dataset", type=str, default="analytical",
                        choices=["cobe", "hadisst", "custom", "analytical"])
    parser.add_argument("--forcing-path", type=str, default=None)
    parser.add_argument("--sic-path", type=str, default=None,
                        help="Separate SIC file (for ICON split SST/SIC files)")
    parser.add_argument("--sst-var", type=str, default=None)
    parser.add_argument("--sic-var", type=str, default=None)
    parser.add_argument("--time-var", type=str, default=None)
    parser.add_argument("--lat-var", type=str, default=None)
    parser.add_argument("--lon-var", type=str, default=None)
    parser.add_argument("--sst-offset", type=float, default=None)
    parser.add_argument("--sic-scale", type=float, default=None)

    # Grid
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--nlev", type=int, default=40)
    parser.add_argument("--vertical-coord", type=str, default="hybrid",
                        choices=["sigma", "hybrid"])
    parser.add_argument("--p-top", type=float, default=None)
    parser.add_argument("--stretching", type=float, default=None)
    parser.add_argument("--grid-type", type=str, default="cubed_sphere",
                        choices=["cubed_sphere", "gaussian", "latlon", "voronoi"])
    parser.add_argument("--discretization", type=str, default="centered",
                        choices=["centered", "finite_volume", "cgrid", "mpas", "spectral"])
    parser.add_argument("--truncation", type=int, default=None,
                        help="Spectral truncation (T21, T42, etc.). Sets grid_type=gaussian.")

    # Integration
    parser.add_argument("--start-day", type=float, default=0.0)
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--diag-days", type=int, default=5)

    # Output
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoint-days", type=int, default=0)
    parser.add_argument("--restart-from", type=str, default=None)

    # Radiation
    parser.add_argument("--radiation", type=str, default="gray",
                        choices=["gray", "rrtmg", "rrtmgp"])
    parser.add_argument("--rad-update-steps", type=int, default=1)
    parser.add_argument("--diurnal-cycle", action="store_true", default=False)
    parser.add_argument("--co2-ppmv", type=float, default=415.0)
    parser.add_argument("--ch4-ppbv", type=float, default=1900.0)
    parser.add_argument("--n2o-ppbv", type=float, default=332.0)
    parser.add_argument("--ozone-source", type=str, default="standard",
                        choices=["standard", "analytical", "none"])
    parser.add_argument("--ozone-forcing", type=str, default="inline",
                        choices=["inline", "external", "off"])
    parser.add_argument("--ozone-file", type=str, default="")
    parser.add_argument("--ghg-forcing", type=str, default="constant",
                        choices=["constant", "external"])
    parser.add_argument("--ghg-file", type=str, default="",
                        help="Path to a time-varying GHG forcing file")

    # Solar
    parser.add_argument("--solar-source", type=str, default="constant",
                        choices=["constant", "file", "spectral_file"])
    parser.add_argument("--solar-file", type=str, default="")
    parser.add_argument("--solar-tsi-var", type=str, default="tsi",
                        help="Variable name for TSI in solar forcing file (e.g. 'TSI' for CMIP6 files)")
    parser.add_argument("--solar-spectral-var", type=str,
                        default="solar_fraction_by_gpt")

    # Aerosol
    parser.add_argument("--aerosol-forcing", type=str, default="off",
                        choices=["off", "external"])
    parser.add_argument("--aerosol-file", type=str, default="")
    parser.add_argument("--aerosol-reference-aod", type=float, default=0.03)
    parser.add_argument("--volcanic-aerosol-file", type=str, default="")
    parser.add_argument("--volcanic-aerosol-scale", type=float, default=1.0)

    # Subgrid physics
    parser.add_argument("--convection", type=str, default="sbm",
                        choices=[
                            "none", "sbm", "dca", "kuo", "mass_flux", "edmf",
                        ])
    parser.add_argument("--turbulence", type=str, default="none",
                        choices=[
                            "none", "smagorinsky", "louis", "tke",
                            "clubb_lite", "holtslag_boville", "ysu", "edmf",
                        ])
    parser.add_argument("--gravity-wave-drag", type=str, default="none",
                        choices=[
                            "none", "rayleigh", "lindzen", "mcfarlane",
                            "hines", "prognostic_spectral", "ml_emulator",
                        ])
    parser.add_argument("--held-suarez-forcing", action="store_true", default=False)
    parser.add_argument("--sbm-tau-c", type=float, default=7200.0)
    parser.add_argument("--sbm-rh-ref", type=float, default=0.7)
    parser.add_argument("--sbm-cape-threshold", type=float, default=70.0)

    # Joint ML physics parameterization
    parser.add_argument("--physics-parameterization", type=str, default="none",
                        choices=["none", "ml"])
    parser.add_argument("--physics-parameterization-checkpoint", type=str, default="")
    parser.add_argument("--physics-parameterization-stats", type=str, default="")
    parser.add_argument("--physics-parameterization-hidden-dim", type=int, default=128)
    parser.add_argument("--physics-parameterization-layers", type=int, default=3)
    parser.add_argument("--physics-parameterization-seed", type=int, default=0)

    # Clouds & microphysics
    parser.add_argument("--clouds", type=str, default="none",
                        choices=["none", "sundqvist", "xu_randall"])
    parser.add_argument("--microphysics", type=str, default="none",
                        choices=["none", "kessler", "sundqvist",
                                 "seifert_beheng", "morrison", "thompson"])

    # Topography
    parser.add_argument("--topography", type=str, default="flat")
    parser.add_argument("--topo-smoothing", type=int, default=4)
    parser.add_argument("--topo-edge-blend", type=float, default=0.3)

    # Surface / diagnostics
    parser.add_argument("--monthly-means", action="store_true", default=False)

    # Moisture conservation
    parser.add_argument("--fix-moisture", action="store_true", default=False)

    # CMIP
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--cmip-output", action="store_true", default=False)
    parser.add_argument("--clear-sky-diag", action="store_true", default=False)

    # Performance
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp64", "mixed"],
                        help="Precision mode: fp32, fp64, or mixed")
    parser.add_argument("--gradient-checkpoint", action="store_true", default=False,
                        help="Enable gradient checkpointing for O(sqrt(N)) AD memory")
    parser.add_argument("--profile", type=int, default=0, metavar="N_STEPS",
                        help="Profile first N steps with jax.profiler and exit")

    # Distributed / MPI
    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Enable MPI distributed execution (auto-detected from environment)")
    parser.add_argument("--ensemble-size", type=int, default=1)

    # Visualization
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Generate diagnostic plots after simulation completes")

    return parser


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    grid_config = GridConfig(
        grid_type=args.grid_type,
        resolution=args.resolution,
        nlev=args.nlev,
        vertical_coord=args.vertical_coord,
        p_top_Pa=args.p_top or 200.0,
        stretching=args.stretching or 2.0,
    )

    dycore_config = DycoreConfig(
        discretization=args.discretization,
        dt=args.dt,
    )

    output_config = OutputConfig(
        output_dir=args.output or "",
        diag_days=args.diag_days,
        checkpoint_days=args.checkpoint_days,
        monthly_means=args.monthly_means,
        cmip_output=args.cmip_output,
        clear_sky_diag=args.clear_sky_diag,
    )

    return ExperimentConfig(
        grid=grid_config,
        dycore=dycore_config,
        output=output_config,
        days=args.days,
        start_day=args.start_day,
        dataset=args.dataset,
        forcing_path=args.forcing_path or "",
        sic_path=args.sic_path or "",
        sst_var=args.sst_var or "",
        sic_var=args.sic_var or "",
        time_var=args.time_var or "",
        lat_var=args.lat_var or "",
        lon_var=args.lon_var or "",
        sst_offset=args.sst_offset or 0.0,
        sic_scale=args.sic_scale or 1.0,
        radiation=args.radiation,
        rad_update_steps=args.rad_update_steps,
        diurnal_cycle=args.diurnal_cycle,
        co2_ppmv=args.co2_ppmv,
        ch4_ppbv=args.ch4_ppbv,
        n2o_ppbv=args.n2o_ppbv,
        ozone_source=args.ozone_source,
        ozone_forcing=args.ozone_forcing,
        ozone_file=args.ozone_file,
        ghg_forcing=args.ghg_forcing,
        ghg_file=args.ghg_file,
        solar_source=args.solar_source,
        solar_file=args.solar_file,
        solar_tsi_var=args.solar_tsi_var,
        solar_spectral_var=args.solar_spectral_var,
        aerosol_forcing=args.aerosol_forcing,
        aerosol_file=args.aerosol_file,
        aerosol_reference_aod=args.aerosol_reference_aod,
        volcanic_aerosol_file=args.volcanic_aerosol_file,
        volcanic_aerosol_scale=args.volcanic_aerosol_scale,
        cloud_scheme=args.clouds,
        microphysics=args.microphysics,
        convection=args.convection,
        turbulence=args.turbulence,
        gravity_wave_drag=args.gravity_wave_drag,
        fix_moisture=args.fix_moisture,
        topography=args.topography,
        topo_smoothing=args.topo_smoothing,
        topo_edge_blend=args.topo_edge_blend,
        dynamic_albedo=False,
        experiment=args.experiment,
        start_year=args.start_year,
        sbm_tau_c=args.sbm_tau_c,
        sbm_RH_ref=args.sbm_rh_ref,
        sbm_cape_threshold=args.sbm_cape_threshold,
        held_suarez_forcing=args.held_suarez_forcing,
        physics_parameterization=args.physics_parameterization,
        physics_parameterization_checkpoint=args.physics_parameterization_checkpoint,
        physics_parameterization_stats=args.physics_parameterization_stats,
        physics_parameterization_hidden_dim=args.physics_parameterization_hidden_dim,
        physics_parameterization_layers=args.physics_parameterization_layers,
        physics_parameterization_seed=args.physics_parameterization_seed,
        precision=args.precision,
        gradient_checkpoint=args.gradient_checkpoint,
        distributed=args.distributed,
        ensemble_size=args.ensemble_size,
    )


def _postprocess_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    # Auto-detect MPI environment
    if not args.distributed and any(
        key in os.environ for key in (
            "OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
            "SLURM_NTASKS", "MPI_LOCALNRANKS",
        )
    ):
        args.distributed = True

    # Backward-compatible alias
    if args.radiation == "rrtmgp":
        args.radiation = "rrtmg"

    if (args.forcing_path is None and args.restart_from is None
            and args.dataset != "analytical"):
        parser.error("--forcing-path required (unless --dataset analytical or --restart-from)")
    if args.solar_source in ("file", "spectral_file") and not args.solar_file:
        parser.error("--solar-file required when --solar-source is file/spectral_file")
    if args.ghg_forcing == "external" and not args.ghg_file:
        parser.error("--ghg-file required when --ghg-forcing is external")
    if args.physics_parameterization == "ml":
        if args.convection != "mass_flux" or args.turbulence != "louis":
            parser.error(
                "--physics-parameterization ml currently requires "
                "--convection mass_flux and --turbulence louis"
            )
        if not args.physics_parameterization_checkpoint or not args.physics_parameterization_stats:
            parser.error(
                "--physics-parameterization ml requires both "
                "--physics-parameterization-checkpoint and "
                "--physics-parameterization-stats"
            )

    # Auto-configure spectral runs
    if args.discretization == "spectral" or args.truncation is not None:
        args.discretization = "spectral"
        args.grid_type = "gaussian"
        if args.truncation is not None:
            args.resolution = args.truncation

    return args


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args = _postprocess_args(args, parser)
    config = build_config_from_args(args)

    from legoesm.driver.model_driver import ModelDriver

    driver = ModelDriver(config)
    print("Setup...")
    driver.setup()

    _is_root = (driver._mpi_rank is None or driver._mpi_rank == 0)

    start_step = 0
    start_day = None
    if args.restart_from:
        restart_path = Path(args.restart_from)
        if not restart_path.exists():
            print(f"ERROR: restart file not found: {restart_path}", file=sys.stderr)
            sys.exit(1)
        if _is_root:
            print(f"Loading checkpoint: {restart_path}")
        start_step, start_day = driver.load_checkpoint(restart_path)
        if _is_root:
            print(f"  Resumed at step={start_step}, day={start_day:.2f}")

    if args.profile > 0:
        import jax

        profile_dir = str(Path(driver.output_dir) / "jax_profile")
        if _is_root:
            print(f"Profiling {args.profile} steps -> {profile_dir}")
        n_profile_days = args.profile * args.dt / 86400.0
        driver.config = driver.config._replace(days=int(n_profile_days + 1))
        with jax.profiler.trace(profile_dir):
            driver.run(start_step=start_step, start_day=start_day)
        if _is_root:
            print(f"Profile saved to {profile_dir}")
            print("View with: tensorboard --logdir " + profile_dir)
        return

    if _is_root:
        print("Running...")
    driver.run(start_step=start_step, start_day=start_day)
    if _is_root:
        print(f"Complete. Output: {driver.output_dir}")

    if args.plot and _is_root:
        sys.path.insert(0, str(Path(__file__).parent))
        from plot_amip import plot_amip as _plot_amip

        _plot_amip(driver.output_dir, show=False)


if __name__ == "__main__":
    main()
