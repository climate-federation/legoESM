#!/usr/bin/env python
"""Run semimonthly NeuralGCM slab campaigns with SFNO-style output layout."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from legoesm.ml.s2s.paths import NEURALGCM_SLAB_RESULTS_ROOT
from legoesm.ml.s2s.neuralgcm_slab import (
    DEFAULT_ARCO_ERA5_STORE,
    DEFAULT_NEURALGCM_CHECKPOINT,
    DEFAULT_PREP_CHECKPOINT,
    campaign_dir_for_year,
    case_dir_for_start,
    center_compare_dir,
    filter_start_times_to_year,
    generate_semimonthly_start_times,
)


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(chunk.strip()) for chunk in text.split(",") if chunk.strip())


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _resolve_runtime_selector(
    *,
    python_executable: str | None,
    conda_env: str | None,
) -> list[str]:
    if python_executable and conda_env:
        raise ValueError("Use only one of --python or --conda-env.")
    if python_executable:
        return [python_executable]
    if conda_env:
        return ["conda", "run", "--no-capture-output", "-n", conda_env, "python"]
    return [sys.executable]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semimonthly NeuralGCM slab campaigns")
    parser.add_argument("--years", default="2022,2023")
    parser.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--days", default="1,15")
    parser.add_argument("--forecast-days", type=int, default=42)
    parser.add_argument("--n-members", type=int, default=4)
    parser.add_argument("--checkpoint", default=DEFAULT_NEURALGCM_CHECKPOINT)
    parser.add_argument("--prep-checkpoint", default=DEFAULT_PREP_CHECKPOINT)
    parser.add_argument("--era5-store", default=DEFAULT_ARCO_ERA5_STORE)
    parser.add_argument("--base-output-dir", type=Path, default=NEURALGCM_SLAB_RESULTS_ROOT)
    parser.add_argument("--mode", choices=("both", "coupled", "uncoupled"), default="both")
    parser.add_argument(
        "--initial-perturbation",
        choices=("none", "correlated_gaussian"),
        default="correlated_gaussian",
    )
    parser.add_argument("--plot-members", default="0,1")
    parser.add_argument("--snapshot-days", default="1,15,29,42")
    parser.add_argument("--window", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--run-center-crps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--python", default=None, help="Explicit Python executable for commands")
    parser.add_argument("--conda-env", default=None, help="Conda environment name for commands")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_path = Path(__file__).resolve().parent / "neuralgcm_slab.py"
    python_prefix = _resolve_runtime_selector(
        python_executable=args.python,
        conda_env=args.conda_env,
    )
    years = _parse_csv_ints(args.years)
    months = _parse_csv_ints(args.months)
    days = _parse_csv_ints(args.days)
    campaign_dirs: list[Path] = []

    for year in years:
        campaign_dir = campaign_dir_for_year(
            base_output_dir=args.base_output_dir,
            year=int(year),
            days=days,
            forecast_days=int(args.forecast_days),
        )
        campaign_dirs.append(campaign_dir)
        start_times = filter_start_times_to_year(
            generate_semimonthly_start_times(year, months=months, days=days),
            forecast_days=int(args.forecast_days),
            year=year,
        )
        if not start_times:
            raise ValueError(f"No valid start times remain for year {year}")
        print(
            f"year={year} n_cases={len(start_times)} first={start_times[0]} last={start_times[-1]}",
            flush=True,
        )

        case_dirs: list[Path] = []
        for start_time in start_times:
            case_dir = case_dir_for_start(
                base_output_dir=campaign_dir,
                start_time=start_time,
                checkpoint=args.checkpoint,
                forecast_days=int(args.forecast_days),
            )
            case_dirs.append(case_dir)
            cmd = [
                *python_prefix,
                str(script_path),
                "ensemble-inference",
                "--start-time",
                start_time,
                "--forecast-days",
                str(args.forecast_days),
                "--case-dir",
                str(case_dir),
                "--checkpoint",
                str(args.checkpoint),
                "--prep-checkpoint",
                str(args.prep_checkpoint),
                "--era5-store",
                str(args.era5_store),
                "--n-members",
                str(args.n_members),
                "--mode",
                str(args.mode),
                "--initial-perturbation",
                str(args.initial_perturbation),
                "--plot-members",
                str(args.plot_members),
                "--snapshot-days",
                str(args.snapshot_days),
            ]
            for window in args.window:
                cmd.extend(["--window", window])
            if args.skip_existing:
                cmd.append("--skip-existing")
            _run(cmd, dry_run=bool(args.dry_run))

        cmd = [
            *python_prefix,
            str(script_path),
            "postprocess-campaign",
            "--output-dir",
            str(campaign_dir),
        ]
        for case_dir in case_dirs:
            cmd.extend(["--case-dir", str(case_dir)])
        for window in args.window:
            cmd.extend(["--window", window])
        _run(cmd, dry_run=bool(args.dry_run))

    if args.run_center_crps:
        center_dir = center_compare_dir(
            base_output_dir=args.base_output_dir,
            years=years,
            forecast_days=int(args.forecast_days),
        )
        cmd = [
            *python_prefix,
            str(script_path),
            "postprocess-center-crps",
            "--output-dir",
            str(center_dir),
        ]
        for campaign_dir in campaign_dirs:
            cmd.extend(["--campaign-dir", str(campaign_dir)])
        for window in args.window:
            cmd.extend(["--window", window])
        _run(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
