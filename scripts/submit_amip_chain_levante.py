#!/usr/bin/env python
"""Submit a chained sequence of SLURM jobs for a multi-year AMIP run on Levante.

Each calendar year is split into 5 segments of 73 days (5 × 73 = 365).
No segment crosses a year boundary, so per-year aerosol/volcanic forcing
files are always valid for the full segment.  Jobs are submitted with
--dependency=afterok so the chain stops automatically on failure.

Usage::

    # Full 36-year AMIP (1979–2014), dry run to preview all jobs:
    python scripts/submit_amip_chain_levante.py \\
        --start-year 1979 --end-year 2014 \\
        --ic-zarr /scratch/b/b309178/era5_ic_1979-01-01.zarr \\
        --base-output /scratch/b/b309178/amip_chain \\
        --dry-run

    # Submit for real:
    python scripts/submit_amip_chain_levante.py \\
        --start-year 1979 --end-year 2014 \\
        --ic-zarr /scratch/b/b309178/era5_ic_1979-01-01.zarr \\
        --base-output /scratch/b/b309178/amip_chain

    # Short test: 1 year only
    python scripts/submit_amip_chain_levante.py \\
        --start-year 1979 --end-year 1979 \\
        --ic-zarr /scratch/b/b309178/era5_ic_1979-01-01.zarr \\
        --base-output /scratch/b/b309178/amip_chain_test

Checkpoint naming convention
-----------------------------
Each segment saves one checkpoint at ``checkpoint_day_0073.npz`` (73 elapsed
days).  The chain script uses this fixed name to construct restart paths, so
``--checkpoint-days`` must be ≤ 73 (or set to 73).  The default is 73.

Output layout::

    {base_output}/
      y1979_s00/   # year 1979, segment 0 (days   0–73)
      y1979_s01/   # year 1979, segment 1 (days  73–146)
      y1979_s02/   # year 1979, segment 2 (days 146–219)
      y1979_s03/   # year 1979, segment 3 (days 219–292)
      y1979_s04/   # year 1979, segment 4 (days 292–365)
      y1980_s00/   # year 1980, segment 0 (days   0–73)
      ...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO   = Path(__file__).resolve().parents[1]
_PYTHON = "/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python"
_CONDA  = "/work/bd1083/b309178/mambaforge/etc/profile.d/conda.sh"
_LOGS   = Path("/work/bd1083/b309178/logs")

_SEG_DAYS   = 73          # 5 × 73 = 365 exactly
_N_SEGS     = 5
_CHECKPOINT_FILE = f"checkpoint_day_{_SEG_DAYS:04d}.npz"


def _seg_output(base: Path, year: int, seg: int) -> Path:
    return base / f"y{year:04d}_s{seg:02d}"


def _sbatch_script(
    year: int,
    seg: int,
    output_dir: Path,
    restart_from: str,
    restart_start_day: float | None,
    ic_zarr: str,
    dt: float,
    radiation: str,
    gwd: str,
    account: str,
    time_limit: str,
    dry_run: bool,
    distributed: bool = False,
    n_ranks: int = 2,
) -> str:
    """Return the sbatch script content for one segment."""
    job_name = f"lego-{year}-{seg:02d}"
    log_base = _LOGS / f"amip-{year}-{seg:02d}-%j"

    # Multi-node: allocate n_ranks nodes (1 MPI rank per node, 4 GPUs each)
    nodes_line = f"#SBATCH --nodes={n_ranks}" if distributed else "#SBATCH --nodes=1"
    ntasks_line = (f"#SBATCH --ntasks={n_ranks}\n#SBATCH --ntasks-per-node=1"
                   if distributed else "")

    levante_args = [
        f"--year {year}",
        f"--days {_SEG_DAYS}",
        f"--dt {dt}",
        "--diag-days 5",
        f"--checkpoint-days {_SEG_DAYS}",
        f"--radiation {radiation}",
        "--rad-update-steps 6",
        "--convection sbm",
        "--turbulence louis",
        f"--gravity-wave-drag {gwd}",
        "--clouds sundqvist",
        "--microphysics sundqvist",
        "--monthly-means",
        "--cmip-output",
        "--clear-sky-diag",
        f"--output {output_dir}",
    ]
    if distributed:
        levante_args.append(f"--distributed --n-ranks {n_ranks}")

    if restart_from:
        levante_args.append(f"--restart-from {restart_from}")
        if restart_start_day is not None:
            levante_args.append(f"--restart-start-day {restart_start_day}")
    elif ic_zarr:
        levante_args.append(f"--ic-zarr {ic_zarr}")

    cmd = (
        f"{_PYTHON} {_REPO}/scripts/run_amip_levante.py \\\n    "
        + " \\\n    ".join(levante_args)
    )

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition=gpu
{nodes_line}
{ntasks_line + chr(10) if ntasks_line else ""}#SBATCH --gpus-per-node=a100_80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time={time_limit}
#SBATCH --output={log_base}.out
#SBATCH --error={log_base}.err
#SBATCH --mail-type=FAIL

set -euo pipefail

mkdir -p {_LOGS}

echo "=============================="
echo "legoESM AMIP chain: year {year}, seg {seg:02d}"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Output:  {output_dir}"
echo "Started: $(date)"
echo "=============================="

source {_CONDA}
conda activate diffesm

cd {_REPO}

{cmd}

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--end-year",   type=int, default=2014)
    parser.add_argument("--ic-zarr",    type=str, default="",
                        help="ERA5 IC Zarr for the first segment of --start-year")
    parser.add_argument("--base-output", type=str, required=True,
                        help="Base directory; segments go in y{year}_s{seg:02d}/ subdirs")
    parser.add_argument("--dt",         type=float, default=150.0)
    parser.add_argument("--radiation",  type=str,   default="rrtmg",
                        choices=["rrtmg", "gray"])
    parser.add_argument("--gwd",        type=str,   default="rayleigh",
                        help="Gravity-wave-drag scheme: none, rayleigh, lindzen, mcfarlane")
    parser.add_argument("--account",    type=str,   default="bd1083")
    parser.add_argument("--time-limit", type=str,   default="12:00:00")
    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Enable multi-node MPI (1 rank per node, srun launcher)")
    parser.add_argument("--n-ranks",   type=int,   default=2,
                        help="Number of MPI ranks / nodes (default 2, giving 6 GPUs)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print all sbatch scripts and dependency graph; do not submit")
    args = parser.parse_args(argv)

    base = Path(args.base_output)
    years = range(args.start_year, args.end_year + 1)
    total = len(years) * _N_SEGS

    print(f"legoESM AMIP chain: {args.start_year}–{args.end_year}  "
          f"({len(years)} years × {_N_SEGS} segments = {total} jobs)")
    print(f"  Base output : {base}")
    print(f"  dt          : {args.dt}s  radiation: {args.radiation}  gwd: {args.gwd}")
    print(f"  SLURM       : account={args.account}  time={args.time_limit}"
          f"  distributed: {args.distributed}"
          + (f"  n_ranks={args.n_ranks}" if args.distributed else ""))
    print(f"  Dry run     : {args.dry_run}")
    print()

    prev_job_id: str | None = None
    prev_output: Path | None = None

    for year in years:
        for seg in range(_N_SEGS):
            output_dir = _seg_output(base, year, seg)

            # Determine IC / restart source
            if year == args.start_year and seg == 0:
                restart_from      = ""
                restart_start_day = None
                ic_zarr           = args.ic_zarr
            elif seg == 0:
                # Year boundary: load last checkpoint of previous year, reset day to 0
                assert prev_output is not None
                restart_from      = str(prev_output / _CHECKPOINT_FILE)
                restart_start_day = 0.0
                ic_zarr           = ""
            else:
                # Within-year continuation: day counter comes from checkpoint as-is
                assert prev_output is not None
                restart_from      = str(prev_output / _CHECKPOINT_FILE)
                restart_start_day = None
                ic_zarr           = ""

            script = _sbatch_script(
                year=year, seg=seg,
                output_dir=output_dir,
                restart_from=restart_from,
                restart_start_day=restart_start_day,
                ic_zarr=ic_zarr,
                dt=args.dt,
                radiation=args.radiation,
                gwd=args.gwd,
                account=args.account,
                time_limit=args.time_limit,
                dry_run=args.dry_run,
                distributed=args.distributed,
                n_ranks=args.n_ranks,
            )

            dep_flag = f"--dependency=afterok:{prev_job_id} " if prev_job_id else ""

            if args.dry_run:
                print(f"--- y{year:04d}_s{seg:02d}  dep={prev_job_id or 'none'} ---")
                print(script[:400] + "...\n")
                prev_job_id = f"DRY{year}{seg:02d}"
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".sh", delete=False
                ) as f:
                    f.write(script)
                    tmp = f.name

                sbatch_cmd = ["sbatch", "--parsable"]
                if prev_job_id:
                    sbatch_cmd.append(f"--dependency=afterok:{prev_job_id}")
                sbatch_cmd.append(tmp)

                result = subprocess.run(
                    sbatch_cmd, capture_output=True, text=True
                )
                if result.returncode != 0:
                    print(f"ERROR submitting y{year}_s{seg:02d}:", result.stderr)
                    return 1
                job_id = result.stdout.strip()
                prev_job_id = job_id
                print(f"  y{year:04d}_s{seg:02d}  job={job_id:>12s}  dep={dep_flag.strip() or 'none':20s}  out={output_dir}")

            prev_output = output_dir

    if not args.dry_run:
        print(f"\nChain submitted: {total} jobs.  Final job ID: {prev_job_id}")
        print(f"Monitor with:  squeue -u $USER -o '%.18i %.9T %.10M %j'")
        print(f"Concatenate after completion:")
        print(f"  python scripts/concat_amip_output.py --base {base}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
