"""Unified CLI for the initialized pure-physics legoESM slab workflow."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig, OutputConfig
from legoesm.ml.s2s.legoesm_slab.campaign import case_dir_for_start
from legoesm.ml.s2s.legoesm_slab.postprocess import postprocess_campaign, postprocess_case
from legoesm.ml.s2s.legoesm_slab.preparation import (
    DEFAULT_ARCO_ERA5_STORE,
    PreparationConfig,
    prepare_legoesm_case,
)
from legoesm.ml.s2s.legoesm_slab.rollout import (
    DEFAULT_EXPORT_PRESSURE_LEVELS,
    ForecastExportConfig,
    run_legoesm_case,
)
from legoesm.ml.s2s.paths import LEGOESM_SLAB_RESULTS_ROOT


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _absolute_start_day(start_time: str) -> float:
    timestamp = datetime.fromisoformat(start_time)
    return (
        float(timestamp.timetuple().tm_yday - 1)
        + float(timestamp.hour) / 24.0
        + float(timestamp.minute) / 1440.0
        + float(timestamp.second) / 86400.0
    )


def _case_dir_from_args(args: argparse.Namespace) -> Path:
    if getattr(args, "case_dir", None) is not None:
        return Path(args.case_dir)
    return case_dir_for_start(
        base_output_dir=Path(args.output_dir),
        start_time=args.start_time,
        forecast_days=int(args.forecast_days),
    )


def _build_experiment_config(args: argparse.Namespace, case_dir: Path) -> ExperimentConfig:
    forcing_path = case_dir / "_prepared" / f"surface_forcing_{datetime.fromisoformat(args.start_time).strftime('%Y%m%d')}_{int(args.forecast_days)}d.nc"
    grid = GridConfig(
        grid_type="cubed_sphere",
        resolution=int(args.resolution),
        nlev=int(args.nlev),
        vertical_coord=str(args.vertical_coord),
        p_top_Pa=float(args.p_top),
        stretching=float(args.stretching),
    )
    dycore = DycoreConfig(
        discretization="cdgrid",
        dt=float(args.dt),
        A_h_scale=float(args.A_h_scale),
        hyperdiff_scale=float(args.hyperdiff_scale),
        div_damp_scale=float(args.div_damp_scale),
        sponge_sigma=float(args.sponge_sigma),
        sponge_tau_sec=float(args.sponge_tau_sec),
    )
    output = OutputConfig(
        output_dir=str(case_dir),
        diag_days=1,
    )
    return ExperimentConfig(
        grid=grid,
        dycore=dycore,
        output=output,
        days=int(args.forecast_days),
        start_day=_absolute_start_day(args.start_time),
        dataset="custom",
        forcing_path=str(forcing_path),
        sst_var="sea_surface_temperature",
        sic_var="sea_ice_cover",
        time_var="time",
        lat_var="latitude",
        lon_var="longitude",
        radiation=str(args.radiation),
        rad_update_steps=int(args.rad_update_steps),
        diurnal_cycle=bool(args.diurnal_cycle),
        ozone_source=str(args.ozone_source),
        cloud_scheme=str(args.clouds),
        microphysics=str(args.microphysics),
        convection=str(args.convection),
        turbulence=str(args.turbulence),
        sat_adjust_without_microphysics=bool(args.sat_adjust_without_microphysics),
        sbm_tau_c=float(args.sbm_tau_c),
        sbm_RH_ref=float(args.sbm_RH_ref),
        sbm_cape_threshold=float(args.sbm_cape_threshold),
        C_H=float(args.C_H),
        C_E=float(args.C_E),
        sigma_b=float(args.sigma_b),
        k_BL_max_per_day=float(args.k_BL_max_per_day),
        k_free_per_day=float(args.k_free_per_day),
        topography=str(args.topography),
        start_year=datetime.fromisoformat(args.start_time).year,
        precision=str(args.precision),
        debug_precision=bool(args.debug_precision),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialized pure-physics legoESM slab S2S workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-case", help="Prepare one initialized ARCO/ERA5 case bundle")
    prepare.add_argument("--start-time", required=True, help="Initialization timestamp, e.g. 2022-01-01T00:00:00")
    prepare.add_argument("--forecast-days", type=int, default=42)
    prepare.add_argument("--case-dir", type=Path, default=None)
    prepare.add_argument("--output-dir", type=Path, default=LEGOESM_SLAB_RESULTS_ROOT)
    prepare.add_argument("--era5-store", default=DEFAULT_ARCO_ERA5_STORE)
    prepare.add_argument("--resolution", type=int, default=16)
    prepare.add_argument("--nlev", type=int, default=20)
    prepare.add_argument("--dt", type=float, default=450.0)
    prepare.add_argument("--A-h-scale", dest="A_h_scale", type=float, default=0.0)
    prepare.add_argument("--hyperdiff-scale", type=float, default=0.0)
    prepare.add_argument("--div-damp-scale", type=float, default=1.0)
    prepare.add_argument("--sponge-sigma", type=float, default=0.15)
    prepare.add_argument("--sponge-tau-sec", type=float, default=3600.0)
    prepare.add_argument("--vertical-coord", choices=("sigma", "hybrid"), default="hybrid")
    prepare.add_argument("--p-top", type=float, default=200.0)
    prepare.add_argument("--stretching", type=float, default=2.0)
    prepare.add_argument("--forcing-resolution-deg", type=float, default=1.0)
    prepare.add_argument("--evaluation-resolution-deg", type=float, default=5.0)
    prepare.add_argument("--pressure-levels", default="200,500,700,850")
    prepare.add_argument("--cdgrid-balance-steps", type=int, default=5)
    prepare.add_argument("--cubedsphere-edge-blend-strength", type=float, default=0.0)
    prepare.add_argument("--cubedsphere-edge-blend-width", type=int, default=0)
    prepare.add_argument(
        "--seed-cloud-liquid-from-era5",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    prepare.add_argument("--radiation", choices=("gray", "rrtmgp", "rrtmg"), default="rrtmgp")
    prepare.add_argument("--rad-update-steps", type=int, default=1)
    prepare.add_argument("--diurnal-cycle", action=argparse.BooleanOptionalAction, default=True)
    prepare.add_argument("--ozone-source", choices=("standard", "analytical", "none"), default="standard")
    prepare.add_argument("--clouds", choices=("none", "sundqvist", "xu_randall"), default="sundqvist")
    prepare.add_argument("--microphysics", choices=("none", "sundqvist", "kessler"), default="sundqvist")
    prepare.add_argument("--convection", choices=("none", "sbm", "kuo", "mass_flux", "edmf", "dca"), default="sbm")
    prepare.add_argument("--sbm-tau-c", type=float, default=7200.0)
    prepare.add_argument("--sbm-RH-ref", type=float, default=0.7)
    prepare.add_argument("--sbm-cape-threshold", type=float, default=70.0)
    prepare.add_argument(
        "--sat-adjust-without-microphysics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    prepare.add_argument("--C-H", dest="C_H", type=float, default=0.0044)
    prepare.add_argument("--C-E", dest="C_E", type=float, default=0.0044)
    prepare.add_argument("--sigma-b", type=float, default=0.7)
    prepare.add_argument("--k-BL-max-per-day", dest="k_BL_max_per_day", type=float, default=1.0)
    prepare.add_argument("--k-free-per-day", dest="k_free_per_day", type=float, default=0.1)
    prepare.add_argument("--turbulence", choices=("none", "smagorinsky", "louis", "tke", "clubb_lite", "holtslag_boville", "ysu", "edmf"), default="louis")
    prepare.add_argument("--topography", default="flat")
    prepare.add_argument("--precision", choices=("fp32", "fp64", "mixed", "mixed_fp64_storage"), default="fp32")
    prepare.add_argument("--debug-precision", action="store_true")
    prepare.add_argument("--ocean-h-mix", type=float, default=50.0)
    prepare.add_argument("--coupling-dt", type=float, default=3600.0)

    run_case = subparsers.add_parser("run-case", help="Run the paired coupled/uncoupled initialized case")
    run_case.add_argument("--case-dir", type=Path, required=True)
    run_case.add_argument("--experiments", default="coupled,uncoupled")
    run_case.add_argument("--pressure-levels", default=None)
    run_case.add_argument("--evaluation-resolution-deg", type=float, default=None)
    run_case.add_argument("--compiled", action="store_true", help="Use the compiled scan path instead of the default uncompiled validation path")

    post_case = subparsers.add_parser("postprocess-case", help="Score and plot one paired case")
    post_case.add_argument("--case-dir", type=Path, required=True)
    post_case.add_argument("--snapshot-days", default="1,15,29,42")

    post_campaign = subparsers.add_parser("postprocess-campaign", help="Aggregate paired case metrics across a campaign")
    post_campaign.add_argument("--case-dirs", nargs="+", required=True)
    post_campaign.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-case":
        case_dir = _case_dir_from_args(args)
        pressure_levels = _parse_csv_ints(args.pressure_levels)
        experiment_config = _build_experiment_config(args, case_dir)
        prep_config = PreparationConfig(
            era5_store=args.era5_store,
            forcing_resolution_deg=float(args.forcing_resolution_deg),
            evaluation_resolution_deg=float(args.evaluation_resolution_deg),
            pressure_levels=pressure_levels,
            cdgrid_balance_steps=int(args.cdgrid_balance_steps),
            cubedsphere_edge_blend_strength=float(args.cubedsphere_edge_blend_strength),
            cubedsphere_edge_blend_width=int(args.cubedsphere_edge_blend_width),
            seed_cloud_liquid_from_era5=bool(args.seed_cloud_liquid_from_era5),
        )
        outputs = prepare_legoesm_case(
            start_time=args.start_time,
            forecast_days=int(args.forecast_days),
            case_dir=case_dir,
            experiment_config=experiment_config,
            config=prep_config,
            coupled={
                "ocean_mode": "slab",
                "land_mode": "slab",
                "f_land_mode": "analytical",
                "ocean_h_mix": float(args.ocean_h_mix),
                "coupling_dt": float(args.coupling_dt),
            },
        )
        print(f"Prepared case: {case_dir}")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
        return

    if args.command == "run-case":
        pressure_levels = None if args.pressure_levels is None else _parse_csv_ints(args.pressure_levels)
        export_config = None
        if pressure_levels is not None or args.evaluation_resolution_deg is not None:
            export_config = ForecastExportConfig(
                pressure_levels=pressure_levels or DEFAULT_EXPORT_PRESSURE_LEVELS,
                evaluation_resolution_deg=float(args.evaluation_resolution_deg or 5.0),
            )
        outputs = run_legoesm_case(
            case_dir=args.case_dir,
            experiments=_parse_csv_strings(args.experiments),
            export_config=export_config,
            compiled=bool(args.compiled),
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return

    if args.command == "postprocess-case":
        outputs = postprocess_case(
            args.case_dir,
            snapshot_days=_parse_csv_ints(args.snapshot_days),
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return

    if args.command == "postprocess-campaign":
        outputs = postprocess_campaign(args.case_dirs, output_dir=args.output_dir)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return

    raise RuntimeError(f"Unhandled command: {args.command!r}")


__all__ = ["build_arg_parser", "main"]
