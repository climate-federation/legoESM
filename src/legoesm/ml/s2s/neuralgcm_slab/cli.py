"""Unified CLI for NeuralGCM slab-ocean experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Sequence

import xarray as xr

from legoesm.ml.s2s.paths import NEURALGCM_SLAB_RESULTS_ROOT
from legoesm.ml.s2s.neuralgcm_slab.campaign import (
    campaign_dir_for_year,
    case_dir_for_start,
    checkpoint_tag,
    filter_start_times_to_year,
    generate_semimonthly_start_times,
    surface_forcing_filename,
)
from legoesm.ml.s2s.neuralgcm_slab.ensemble import (
    PerturbationConfig,
    apply_initial_perturbations,
    member_name,
    member_seed,
)
from legoesm.ml.s2s.neuralgcm_slab.evaluation import DEFAULT_S2S_WINDOWS, LeadTimeWindow
from legoesm.ml.s2s.neuralgcm_slab.neuralgcm_backend import (
    DEFAULT_NEURALGCM_CHECKPOINT,
    NeuralGCMBackend,
    NeuralGCMBackendConfig,
)
from legoesm.ml.s2s.neuralgcm_slab.postprocess import (
    postprocess_campaign,
    postprocess_campaign_center_crps,
    postprocess_case,
    postprocess_case_dirs_center_crps,
)
from legoesm.ml.s2s.neuralgcm_slab.preparation import (
    DEFAULT_ARCO_ERA5_STORE,
    DEFAULT_PREP_CHECKPOINT,
    PreparationConfig,
    prepare_neuralgcm_case,
)
from legoesm.ml.s2s.neuralgcm_slab.slab_coupling import SlabCouplingConfig, rollout_coupled_daily


def _open_dataset(path: str | Path) -> xr.Dataset:
    path = Path(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def _save_dataset(dataset: xr.Dataset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zarr":
        dataset.to_zarr(path, mode="w")
    else:
        dataset.to_netcdf(path)
    return path


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(chunk.strip()) for chunk in text.split(",") if chunk.strip())


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    return tuple(chunk.strip() for chunk in text.split(",") if chunk.strip())


def _parse_windows(values: Sequence[str] | None) -> tuple[LeadTimeWindow, ...]:
    if not values:
        return DEFAULT_S2S_WINDOWS
    windows: list[LeadTimeWindow] = []
    for value in values:
        parts = value.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid window {value!r}. Expected NAME:START:END, for example wk3_4:15:28."
            )
        name, start, end = parts
        try:
            start_day = int(start)
            end_day = int(end)
        except ValueError as exc:
            raise ValueError(
                f"Invalid window {value!r}. START and END must be integers."
            ) from exc
        if start_day <= 0 or end_day <= 0 or start_day > end_day:
            raise ValueError(
                f"Invalid window {value!r}. Require 0 < START <= END."
            )
        windows.append(LeadTimeWindow(name, start_day, end_day))
    return tuple(windows)


def _infer_forecast_days(dataset: xr.Dataset, *, time_name: str = "time") -> int:
    if time_name not in dataset.dims:
        raise ValueError(f"Dataset has no {time_name!r} dimension to infer forecast length")
    return int(dataset.sizes[time_name])


def _prepared_dir(case_dir: str | Path) -> Path:
    return Path(case_dir) / "_prepared"


def _prepared_paths(case_dir: str | Path) -> dict[str, Path]:
    prepared_dir = _prepared_dir(case_dir)
    return {
        "initial": prepared_dir / "initial.nc",
        "forcing": prepared_dir / "forcing.nc",
        "radiation": prepared_dir / "radiation.nc",
        "truth": prepared_dir / "truth.nc",
    }


def _publish_surface_forcing_alias(*, case_dir: Path, start_time: str, forecast_days: int, forcing_path: Path) -> Path:
    target = case_dir / surface_forcing_filename(start_time=start_time, forecast_days=int(forecast_days))
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(forcing_path.resolve())
    except OSError:
        shutil.copy2(forcing_path, target)
    return target


def _default_case_dir(
    *,
    start_time: str,
    checkpoint: str,
    forecast_days: int,
    base_output_dir: str | Path,
) -> Path:
    return case_dir_for_start(
        base_output_dir=base_output_dir,
        start_time=start_time,
        checkpoint=checkpoint,
        forecast_days=forecast_days,
    )


def _prepare_case(
    *,
    start_time: str,
    forecast_days: int,
    case_dir: Path,
    prep_checkpoint: str,
    era5_store: str,
    overwrite: bool,
) -> dict[str, Path]:
    prepared_dir = case_dir / "_prepared"
    prepared = _prepared_paths(case_dir)
    if (not overwrite) and all(path.exists() for path in prepared.values()):
        _publish_surface_forcing_alias(
            case_dir=case_dir,
            start_time=start_time,
            forecast_days=forecast_days,
            forcing_path=prepared["forcing"],
        )
        return prepared
    prepare_neuralgcm_case(
        start_time=start_time,
        forecast_days=forecast_days,
        output_dir=prepared_dir,
        config=PreparationConfig(
            era5_store=era5_store,
            checkpoint=prep_checkpoint,
        ),
    )
    prepared = _prepared_paths(case_dir)
    _publish_surface_forcing_alias(
        case_dir=case_dir,
        start_time=start_time,
        forecast_days=forecast_days,
        forcing_path=prepared["forcing"],
    )
    return prepared


def _run_rollout_core(
    *,
    initial: xr.Dataset,
    forcing: xr.Dataset,
    radiation: xr.Dataset,
    checkpoint: str,
    rng_seed: int,
    mode: str,
    forecast_days: int,
) -> dict[str, xr.Dataset]:
    backend = NeuralGCMBackend(
        NeuralGCMBackendConfig(
            checkpoint=checkpoint,
            rng_seed=int(rng_seed),
        )
    )
    experiments = EXPERIMENTS if mode == "both" else (mode,)
    outputs: dict[str, xr.Dataset] = {}
    for experiment in experiments:
        outputs[experiment] = rollout_coupled_daily(
            backend,
            initial,
            forcing,
            radiation,
            forecast_days=forecast_days,
            config=SlabCouplingConfig(),
            coupled=(experiment == "coupled"),
        )
    return outputs


EXPERIMENTS = ("coupled", "uncoupled")


def _run_prepare(args: argparse.Namespace) -> None:
    outputs = prepare_neuralgcm_case(
        start_time=args.start_time,
        forecast_days=int(args.forecast_days),
        output_dir=args.output_dir,
        config=PreparationConfig(
            era5_store=args.era5_store,
            checkpoint=args.checkpoint,
        ),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def _run_rollout(args: argparse.Namespace) -> None:
    initial = _open_dataset(args.initial)
    forcing = _open_dataset(args.forcing)
    radiation = _open_dataset(args.radiation)
    forecast_days = int(args.forecast_days) if args.forecast_days is not None else _infer_forecast_days(forcing)
    outputs = _run_rollout_core(
        initial=initial,
        forcing=forcing,
        radiation=radiation,
        checkpoint=args.checkpoint,
        rng_seed=int(args.rng_seed),
        mode=args.mode,
        forecast_days=forecast_days,
    )
    output_path = Path(args.output)
    for experiment, dataset in outputs.items():
        if args.mode == "both":
            target = output_path.with_name(f"{output_path.stem}_{experiment}{output_path.suffix}")
        else:
            target = output_path
        _save_dataset(dataset, target)
        print(f"saved {experiment}: {target}")


def _run_member(args: argparse.Namespace) -> None:
    case_dir = Path(args.case_dir)
    prepared = _prepared_paths(case_dir)
    initial = _open_dataset(prepared["initial"])
    forcing = _open_dataset(prepared["forcing"])
    radiation = _open_dataset(prepared["radiation"])
    forecast_days = _infer_forecast_days(forcing)

    perturbation_metadata: dict[str, object] = {"perturbation": "none"}
    if args.initial_perturbation == "correlated_gaussian":
        initial, perturbation_metadata = apply_initial_perturbations(
            initial,
            member_index=int(args.member_index),
            config=PerturbationConfig(),
        )

    seed = int(args.rng_seed) if args.rng_seed is not None else member_seed(int(args.member_index))
    outputs = _run_rollout_core(
        initial=initial,
        forcing=forcing,
        radiation=radiation,
        checkpoint=args.checkpoint,
        rng_seed=seed,
        mode=args.mode,
        forecast_days=forecast_days,
    )

    member_dir = case_dir / member_name(int(args.member_index))
    member_dir.mkdir(parents=True, exist_ok=True)
    for experiment, dataset in outputs.items():
        target = member_dir / f"{experiment}.nc"
        _save_dataset(dataset, target)
        print(f"saved {experiment}: {target}")

    metadata = {
        "member_index": int(args.member_index),
        "seed": int(seed),
        "checkpoint": str(args.checkpoint),
        "mode": str(args.mode),
        "initial_perturbation": str(args.initial_perturbation),
        "perturbation_metadata": perturbation_metadata,
    }
    metadata_dir = case_dir / "_member_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / f"{member_name(int(args.member_index))}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_ensemble_inference(args: argparse.Namespace) -> None:
    case_dir = (
        Path(args.case_dir)
        if args.case_dir is not None
        else _default_case_dir(
            start_time=args.start_time,
            checkpoint=args.checkpoint,
            forecast_days=int(args.forecast_days),
            base_output_dir=args.base_output_dir,
        )
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = case_dir / "metrics" / "window_ensemble_metrics.csv"
    if args.skip_existing and metrics_path.exists():
        print(f"skip existing case: {case_dir}")
        return

    _prepare_case(
        start_time=args.start_time,
        forecast_days=int(args.forecast_days),
        case_dir=case_dir,
        prep_checkpoint=args.prep_checkpoint,
        era5_store=args.era5_store,
        overwrite=not args.skip_existing,
    )

    for member_index in range(int(args.n_members)):
        member_args = argparse.Namespace(
            case_dir=case_dir,
            member_index=member_index,
            checkpoint=args.checkpoint,
            mode=args.mode,
            rng_seed=None,
            initial_perturbation=args.initial_perturbation,
        )
        _run_member(member_args)

    windows = _parse_windows(args.window)
    plot_members = _parse_csv_ints(args.plot_members)
    snapshot_days = _parse_csv_ints(args.snapshot_days)
    postprocess_case(
        case_dir,
        n_members=int(args.n_members),
        plot_member_indices=plot_members,
        windows=windows,
        snapshot_days=snapshot_days,
        make_snapshots=bool(args.make_snapshots),
        make_gifs=bool(args.make_gifs),
    )
    print(f"completed case: {case_dir}")


def _resolve_case_dirs(args: argparse.Namespace) -> list[Path]:
    if args.case_dir:
        return [Path(case_dir) for case_dir in args.case_dir]
    if getattr(args, "campaign_dir", None):
        case_dirs: list[Path] = []
        for campaign_dir in args.campaign_dir:
            campaign_path = Path(campaign_dir)
            matches = sorted(
                path for path in campaign_path.glob("inference_*")
                if path.is_dir()
            )
            if not matches:
                raise ValueError(
                    f"No inference_* case directories found under campaign dir {campaign_path}"
                )
            case_dirs.extend(matches)
        return case_dirs
    if args.year is None:
        raise ValueError(
            "postprocess-campaign requires --case-dir, --campaign-dir, or --year"
        )
    start_times = filter_start_times_to_year(
        generate_semimonthly_start_times(
            int(args.year),
            months=_parse_csv_ints(args.months),
            days=_parse_csv_ints(args.days),
        ),
        forecast_days=int(args.forecast_days),
        year=int(args.year),
    )
    campaign_dir = campaign_dir_for_year(
        base_output_dir=args.base_output_dir,
        year=int(args.year),
        days=_parse_csv_ints(args.days),
        forecast_days=int(args.forecast_days),
    )
    return [
        _default_case_dir(
            start_time=start_time,
            checkpoint=args.checkpoint,
            forecast_days=int(args.forecast_days),
            base_output_dir=campaign_dir,
        )
        for start_time in start_times
    ]


def _run_postprocess_case(args: argparse.Namespace) -> None:
    outputs = postprocess_case(
        args.case_dir,
        n_members=int(args.n_members),
        plot_member_indices=_parse_csv_ints(args.plot_members),
        windows=_parse_windows(args.window),
        snapshot_days=_parse_csv_ints(args.snapshot_days),
        make_snapshots=bool(args.make_snapshots),
        make_gifs=bool(args.make_gifs),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def _run_postprocess_campaign(args: argparse.Namespace) -> None:
    case_dirs = _resolve_case_dirs(args)
    outputs = postprocess_campaign(
        case_dirs,
        output_dir=args.output_dir,
        windows=_parse_windows(args.window),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def _run_postprocess_center_crps(args: argparse.Namespace) -> None:
    postprocess_campaign_center_crps(
        campaign_dirs=args.campaign_dir,
        output_dir=args.output_dir,
        fields=_parse_csv_strings(args.fields),
        windows=_parse_windows(args.window),
    )
    print(f"saved center comparison under: {args.output_dir}")


def _run_postprocess_center_crps_from_cases(args: argparse.Namespace) -> None:
    outputs = postprocess_case_dirs_center_crps(
        case_dirs=args.case_dir,
        output_dir=args.output_dir,
        fields=_parse_csv_strings(args.fields),
        windows=_parse_windows(args.window),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"saved center comparison under: {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified NeuralGCM slab-ocean workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare a real NeuralGCM case from ARCO ERA5")
    prepare_parser.add_argument("--start-time", required=True, help="Initialization time in ISO format")
    prepare_parser.add_argument("--forecast-days", type=int, default=42)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--checkpoint", default=DEFAULT_PREP_CHECKPOINT)
    prepare_parser.add_argument("--era5-store", default=DEFAULT_ARCO_ERA5_STORE)
    prepare_parser.set_defaults(func=_run_prepare)

    rollout_parser = subparsers.add_parser("rollout", help="Run a single prepared NeuralGCM slab-ocean rollout")
    rollout_parser.add_argument("--initial", type=Path, required=True)
    rollout_parser.add_argument("--forcing", type=Path, required=True)
    rollout_parser.add_argument("--radiation", type=Path, required=True)
    rollout_parser.add_argument("--output", type=Path, required=True)
    rollout_parser.add_argument("--checkpoint", default=DEFAULT_NEURALGCM_CHECKPOINT)
    rollout_parser.add_argument("--rng-seed", type=int, default=0)
    rollout_parser.add_argument("--forecast-days", type=int, default=None)
    rollout_parser.add_argument("--mode", choices=("both", *EXPERIMENTS), default="both")
    rollout_parser.set_defaults(func=_run_rollout)

    member_parser = subparsers.add_parser("run-member", help="Run one NeuralGCM ensemble member from a prepared case")
    member_parser.add_argument("--case-dir", type=Path, required=True)
    member_parser.add_argument("--member-index", type=int, required=True)
    member_parser.add_argument("--checkpoint", default=DEFAULT_NEURALGCM_CHECKPOINT)
    member_parser.add_argument("--mode", choices=("both", *EXPERIMENTS), default="both")
    member_parser.add_argument("--rng-seed", type=int, default=None)
    member_parser.add_argument(
        "--initial-perturbation",
        choices=("none", "correlated_gaussian"),
        default="correlated_gaussian",
    )
    member_parser.set_defaults(func=_run_member)

    inference_parser = subparsers.add_parser("ensemble-inference", help="Prepare, run, and postprocess one NeuralGCM ensemble case")
    inference_parser.add_argument("--start-time", required=True)
    inference_parser.add_argument("--forecast-days", type=int, default=42)
    inference_parser.add_argument("--case-dir", type=Path, default=None)
    inference_parser.add_argument("--base-output-dir", type=Path, default=NEURALGCM_SLAB_RESULTS_ROOT)
    inference_parser.add_argument("--prep-checkpoint", default=DEFAULT_PREP_CHECKPOINT)
    inference_parser.add_argument("--checkpoint", default=DEFAULT_NEURALGCM_CHECKPOINT)
    inference_parser.add_argument("--era5-store", default=DEFAULT_ARCO_ERA5_STORE)
    inference_parser.add_argument("--n-members", type=int, default=4)
    inference_parser.add_argument("--mode", choices=("both", *EXPERIMENTS), default="both")
    inference_parser.add_argument(
        "--initial-perturbation",
        choices=("none", "correlated_gaussian"),
        default="correlated_gaussian",
    )
    inference_parser.add_argument("--plot-members", default="0,1")
    inference_parser.add_argument("--snapshot-days", default="1,15,29,42")
    inference_parser.add_argument("--window", action="append", default=[])
    inference_parser.add_argument("--skip-existing", action="store_true")
    inference_parser.add_argument("--make-snapshots", action=argparse.BooleanOptionalAction, default=True)
    inference_parser.add_argument("--make-gifs", action=argparse.BooleanOptionalAction, default=True)
    inference_parser.set_defaults(func=_run_ensemble_inference)

    case_parser = subparsers.add_parser("postprocess-case", help="Postprocess one ensemble case into metrics and plots")
    case_parser.add_argument("--case-dir", type=Path, required=True)
    case_parser.add_argument("--n-members", type=int, required=True)
    case_parser.add_argument("--plot-members", default="0,1")
    case_parser.add_argument("--snapshot-days", default="1,15,29,42")
    case_parser.add_argument("--window", action="append", default=[])
    case_parser.add_argument("--make-snapshots", action=argparse.BooleanOptionalAction, default=True)
    case_parser.add_argument("--make-gifs", action=argparse.BooleanOptionalAction, default=True)
    case_parser.set_defaults(func=_run_postprocess_case)

    campaign_parser = subparsers.add_parser("postprocess-campaign", help="Aggregate multiple init dates")
    campaign_parser.add_argument("--case-dir", action="append", default=[])
    campaign_parser.add_argument(
        "--campaign-dir",
        action="append",
        default=[],
        help="Campaign directory containing inference_YYYYMMDD case subdirectories",
    )
    campaign_parser.add_argument("--output-dir", type=Path, required=True)
    campaign_parser.add_argument("--year", type=int, default=None)
    campaign_parser.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12")
    campaign_parser.add_argument("--days", default="1,15")
    campaign_parser.add_argument("--forecast-days", type=int, default=42)
    campaign_parser.add_argument("--checkpoint", default=DEFAULT_NEURALGCM_CHECKPOINT)
    campaign_parser.add_argument("--base-output-dir", type=Path, default=NEURALGCM_SLAB_RESULTS_ROOT)
    campaign_parser.add_argument("--window", action="append", default=[])
    campaign_parser.set_defaults(func=_run_postprocess_campaign)

    center_parser = subparsers.add_parser("postprocess-center-crps", help="Compare NeuralGCM campaign CRPS against ChaosBench centers")
    center_parser.add_argument("--campaign-dir", action="append", required=True)
    center_parser.add_argument("--output-dir", type=Path, required=True)
    center_parser.add_argument("--fields", default="t-850,z-500,q-700")
    center_parser.add_argument("--window", action="append", default=[])
    center_parser.set_defaults(func=_run_postprocess_center_crps)

    center_cases_parser = subparsers.add_parser(
        "postprocess-center-crps-from-cases",
        help="Compare NeuralGCM raw-case CRPS against ChaosBench centers",
    )
    center_cases_parser.add_argument("--case-dir", action="append", required=True)
    center_cases_parser.add_argument("--output-dir", type=Path, required=True)
    center_cases_parser.add_argument("--fields", default="t-850,z-500,q-700")
    center_cases_parser.add_argument("--window", action="append", default=[])
    center_cases_parser.set_defaults(func=_run_postprocess_center_crps_from_cases)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


__all__ = ["build_parser", "main"]
