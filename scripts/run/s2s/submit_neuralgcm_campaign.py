#!/usr/bin/env python
"""Submit NeuralGCM slab campaigns to Slurm with one final dependent aggregation job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legoesm.ml.s2s.paths import NEURALGCM_SLAB_RESULTS_ROOT
from legoesm.ml.s2s.neuralgcm_slab import (
    DEFAULT_ARCO_ERA5_STORE,
    DEFAULT_NEURALGCM_CHECKPOINT,
    DEFAULT_PREP_CHECKPOINT,
    campaign_dir_for_year,
    case_dir_for_start,
    center_compare_dir,
    checkpoint_tag,
    filter_start_times_to_year,
    generate_semimonthly_start_times,
)


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(chunk.strip()) for chunk in text.split(",") if chunk.strip())


def _sbatch_directives(
    *,
    job_name: str,
    output_path: Path,
    error_path: Path,
    partition: str,
    time_limit: str,
    cpus_per_task: int,
    mem: str,
    nodes: int,
    account: str | None,
    gres: str | None,
) -> list[str]:
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH -N {int(nodes)}",
        f"#SBATCH --cpus-per-task={int(cpus_per_task)}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={output_path}",
        f"#SBATCH --error={error_path}",
    ]
    if account:
        lines.append(f"#SBATCH --account={account}")
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    return lines


def _runtime_command(
    *,
    python_executable: str | None,
    conda_env: str | None,
    command: list[str],
) -> list[str]:
    if python_executable:
        return [python_executable, *command]
    if conda_env:
        return ["conda", "run", "--no-capture-output", "-n", conda_env, "python", *command]
    return [sys.executable, *command]


def _resolve_runtime_selector(
    *,
    python_executable: str | None,
    conda_env: str | None,
) -> tuple[str | None, str | None]:
    if python_executable and conda_env:
        raise ValueError("Use only one of --python or --conda-env.")
    if python_executable:
        return python_executable, None
    if conda_env:
        return None, conda_env
    return sys.executable, None


def _write_job_script(
    *,
    script_path: Path,
    repo_root: Path,
    run_commands: list[list[str]],
    directives: list[str],
    cuda_module: str | None,
) -> Path:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    body = directives + [
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo_root))}",
        'export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"',
        'export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-$$}"',
    ]
    if cuda_module:
        body.append(
            f"if command -v module >/dev/null 2>&1; then module load {shlex.quote(cuda_module)}; fi"
        )
    body.extend(
        [
            "echo HOST=$(hostname)",
            "echo START=$(date --iso-8601=seconds)",
            "echo PYTHONPATH=${PYTHONPATH}",
        ]
    )
    for run_command in run_commands:
        body.append(f"echo RUN_CMD={shlex.quote(shlex.join(run_command))}")
        body.append(shlex.join(run_command))
    body.append("echo END=$(date --iso-8601=seconds)")
    script_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return script_path


def _submit_job(script_path: Path, *, dependency: str | None, dry_run: bool) -> str:
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(script_path))
    print(shlex.join(cmd), flush=True)
    if dry_run:
        return f"dryrun-{script_path.stem}"
    output = subprocess.check_output(cmd, text=True).strip()
    return output.split(";", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit NeuralGCM slab campaigns to Slurm")
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
    parser.add_argument("--fields", default="t-850,z-500,q-700")
    parser.add_argument("--window", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--refresh-uncoupled-only", action="store_true")
    parser.add_argument("--partition-case", default="glab1")
    parser.add_argument("--partition-post", default="glab1")
    parser.add_argument("--account", default="glab")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--case-time", default="2-00:00:00")
    parser.add_argument("--post-time", default="04:00:00")
    parser.add_argument("--case-cpus", type=int, default=12)
    parser.add_argument("--post-cpus", type=int, default=4)
    parser.add_argument("--case-mem", default="150G")
    parser.add_argument("--post-mem", default="24G")
    parser.add_argument("--case-gres", default="gpu:1")
    parser.add_argument("--python", default=None, help="Explicit Python executable for batch jobs")
    parser.add_argument("--conda-env", default=None, help="Conda environment name for batch jobs")
    parser.add_argument("--cuda-module", default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--jobs-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    years = _parse_csv_ints(args.years)
    months = _parse_csv_ints(args.months)
    days = _parse_csv_ints(args.days)
    repo_root = args.repo_root.resolve()
    base_output_dir = args.base_output_dir.resolve()
    python_executable, conda_env = _resolve_runtime_selector(
        python_executable=args.python,
        conda_env=args.conda_env,
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = (args.jobs_dir or (base_output_dir / "slurm_jobs")).resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)

    tag = checkpoint_tag(args.checkpoint)
    script_entry = repo_root / "scripts" / "run" / "s2s" / "neuralgcm_slab.py"
    submission_mode = "refresh_uncoupled_only" if args.refresh_uncoupled_only else "full_campaign"

    manifest: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "prep_checkpoint": str(args.prep_checkpoint),
        "era5_store": str(args.era5_store),
        "forecast_days": int(args.forecast_days),
        "n_members": int(args.n_members),
        "years": list(years),
        "submission_mode": submission_mode,
        "campaign_dirs": {},
        "case_jobs": {},
        "center_job": None,
    }

    all_case_job_ids: list[str] = []
    campaign_dirs: dict[int, Path] = {}

    for year in years:
        start_times = filter_start_times_to_year(
            generate_semimonthly_start_times(year, months=months, days=days),
            forecast_days=int(args.forecast_days),
            year=year,
        )
        if not start_times:
            raise ValueError(f"No valid start times remain for year {year}")

        campaign_dir = campaign_dir_for_year(
            base_output_dir=base_output_dir,
            year=int(year),
            days=days,
            forecast_days=int(args.forecast_days),
        )
        campaign_dir.mkdir(parents=True, exist_ok=True)
        campaign_dirs[int(year)] = campaign_dir
        manifest["campaign_dirs"][str(year)] = str(campaign_dir)

        year_job_dir = jobs_dir / str(year)
        year_job_dir.mkdir(parents=True, exist_ok=True)
        case_job_info: list[dict[str, str]] = []

        for start_time in start_times:
            case_dir = case_dir_for_start(
                base_output_dir=campaign_dir,
                start_time=start_time,
                checkpoint=args.checkpoint,
                forecast_days=int(args.forecast_days),
            )
            start_tag = start_time[:10].replace("-", "")
            job_prefix = "ngcmu" if args.refresh_uncoupled_only else "ngcm"
            job_name = f"{job_prefix}_{start_tag}"
            script_path = year_job_dir / f"{job_name}.sbatch"
            output_path = year_job_dir / f"{job_name}.out"
            error_path = year_job_dir / f"{job_name}.err"
            run_commands: list[list[str]] = []
            if args.refresh_uncoupled_only:
                for member_index in range(int(args.n_members)):
                    member_command = [
                        str(script_entry),
                        "run-member",
                        "--case-dir",
                        str(case_dir),
                        "--member-index",
                        str(member_index),
                        "--checkpoint",
                        str(args.checkpoint),
                        "--mode",
                        "uncoupled",
                        "--initial-perturbation",
                        str(args.initial_perturbation),
                    ]
                    run_commands.append(
                        _runtime_command(
                            python_executable=python_executable,
                            conda_env=conda_env,
                            command=member_command,
                        )
                    )
                postprocess_command = [
                    str(script_entry),
                    "postprocess-case",
                    "--case-dir",
                    str(case_dir),
                    "--n-members",
                    str(args.n_members),
                    "--plot-members",
                    str(args.plot_members),
                    "--snapshot-days",
                    str(args.snapshot_days),
                ]
                for window in args.window:
                    postprocess_command.extend(["--window", window])
                run_commands.append(
                    _runtime_command(
                        python_executable=python_executable,
                        conda_env=conda_env,
                        command=postprocess_command,
                    )
                )
            else:
                command = [
                    str(script_entry),
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
                    command.extend(["--window", window])
                if args.skip_existing:
                    command.append("--skip-existing")
                run_commands.append(
                    _runtime_command(
                        python_executable=python_executable,
                        conda_env=conda_env,
                        command=command,
                    )
                )

            directives = _sbatch_directives(
                job_name=job_name,
                output_path=output_path,
                error_path=error_path,
                partition=args.partition_case,
                time_limit=args.case_time,
                cpus_per_task=int(args.case_cpus),
                mem=str(args.case_mem),
                nodes=int(args.nodes),
                account=args.account,
                gres=args.case_gres,
            )
            _write_job_script(
                script_path=script_path,
                repo_root=repo_root,
                run_commands=run_commands,
                directives=directives,
                cuda_module=args.cuda_module,
            )
            job_id = _submit_job(script_path, dependency=None, dry_run=bool(args.dry_run))
            all_case_job_ids.append(job_id)
            case_job_info.append(
                {
                    "start_time": start_time,
                    "case_dir": str(case_dir),
                    "job_id": job_id,
                    "script": str(script_path),
                }
            )

        manifest["case_jobs"][str(year)] = case_job_info

    center_dir = center_compare_dir(
        base_output_dir=base_output_dir,
        years=years,
        forecast_days=int(args.forecast_days),
    )
    center_name = "ngcmu_center" if args.refresh_uncoupled_only else "ngcm_center"
    center_script = jobs_dir / f"{center_name}.sbatch"
    center_output = jobs_dir / f"{center_name}.out"
    center_error = jobs_dir / f"{center_name}.err"
    center_run_commands: list[list[str]] = []

    for year in years:
        campaign_command = [
            str(script_entry),
            "postprocess-campaign",
            "--output-dir",
            str(campaign_dirs[int(year)]),
        ]
        for info in manifest["case_jobs"][str(year)]:
            campaign_command.extend(["--case-dir", str(info["case_dir"])])
        for window in args.window:
            campaign_command.extend(["--window", window])
        center_run_commands.append(
            _runtime_command(
                python_executable=python_executable,
                conda_env=conda_env,
                command=campaign_command,
            )
        )

    center_command = [
        str(script_entry),
        "postprocess-center-crps",
        "--output-dir",
        str(center_dir),
        "--fields",
        str(args.fields),
    ]
    for year in years:
        center_command.extend(["--campaign-dir", str(campaign_dirs[int(year)])])
    for window in args.window:
        center_command.extend(["--window", window])
    center_run_commands.append(
        _runtime_command(
            python_executable=python_executable,
            conda_env=conda_env,
            command=center_command,
        )
    )

    center_directives = _sbatch_directives(
        job_name=center_name,
        output_path=center_output,
        error_path=center_error,
        partition=args.partition_post,
        time_limit=args.post_time,
        cpus_per_task=int(args.post_cpus),
        mem=str(args.post_mem),
        nodes=1,
        account=args.account,
        gres=None,
    )
    _write_job_script(
        script_path=center_script,
        repo_root=repo_root,
        run_commands=center_run_commands,
        directives=center_directives,
        cuda_module=args.cuda_module,
    )
    center_dependency = "afterok:" + ":".join(all_case_job_ids) if all_case_job_ids else None
    center_job_id = _submit_job(
        center_script,
        dependency=center_dependency,
        dry_run=bool(args.dry_run),
    )
    manifest["center_job"] = {
        "job_id": center_job_id,
        "center_dir": str(center_dir),
        "campaign_dirs": [str(campaign_dirs[int(year)]) for year in years],
        "script": str(center_script),
        "dependency": center_dependency,
        "n_case_jobs": len(all_case_job_ids),
    }

    manifest_suffix = "_uncoupled_refresh" if args.refresh_uncoupled_only else ""
    manifest_path = jobs_dir / f"submission_manifest_{tag}_{int(args.forecast_days)}d{manifest_suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"submitted {len(all_case_job_ids)} case jobs across {len(years)} year(s)")
    print(f"center job: {center_job_id}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
