#!/usr/bin/env python
"""Generate and optionally submit Levante strong/weak scaling jobs.

This driver prepares SLURM batch jobs for DKRZ Levante (Atos BullSequana XH2000)
to benchmark strong and weak scaling on:
- CPU partition (single-node + multi-node)
- GPU partition (single-node + multi-node)

It uses the existing `scripts/run_parallel_validation.py` benchmark runner and
focuses on reproducible job generation, submission, and bookkeeping.

Default behavior:
- Writes job scripts and a manifest to `results/levante_scaling/<tag>/`
- Does NOT submit jobs unless `--submit` is passed

Examples
--------
Create scripts only:
    .venv/bin/python scripts/run_levante_scaling.py --account bb1234

Create + submit:
    .venv/bin/python scripts/run_levante_scaling.py --account bb1234 --submit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


LEVANTE_FACTS = {
    "model": "Atos BullSequana XH2000",
    "total_cores": 390_000,
    "network": {
        "fabric": "NVIDIA Mellanox InfiniBand HDR 100G/200G",
        "bandwidth_per_node_gbit_s": 100,
    },
    "storage_lustre_petabyte": 130,
    "cpu_partition": {
        "nodes_total": 2_982,
        "amd_7763_nodes": {
            "standard_mem_256g": 2_670,
            "high_mem_512g": 294,
            "fat_mem_1024g": 18,
        },
        "peak_petaflops": 14.0,
        "main_memory_tb": 852,
    },
    "gpu_partition": {
        "nodes_total": 60,
        "node_layout": "2x AMD 7763 + 4x NVIDIA A100",
        "a100_80g_nodes": 56,
        "a100_40g_nodes": 4,
        "peak_petaflops": 2.8,
        "main_memory_tb": 30,
        "gpu_memory_tb": 5,
    },
}


@dataclass(frozen=True)
class JobSpec:
    mode: str
    nodes: int
    partition: str
    ntasks_per_node: int
    gpus_per_node: int
    mpi_ranks_csv: str
    scaling_cpu_devices_csv: str
    scaling_gpu_devices_csv: str


def _parse_int_csv(text: str, *, label: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError(f"{label} entries must be >= 1, got {value!r}")
        values.append(value)
    if not values:
        raise ValueError(f"{label} must include at least one integer")
    return sorted(dict.fromkeys(values))


def _parse_kv_csv(text: str) -> dict[str, str]:
    text = text.strip()
    if not text:
        return {}
    raw = text.split(";") if ";" in text else text.split(",")
    out: dict[str, str] = {}
    for token in raw:
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--extra-env entries must be KEY=VALUE, got {token!r}")
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid empty env key in {token!r}")
        out[key] = value
    return out


def _parse_mode_csv(text: str) -> list[str]:
    modes: list[str] = []
    for token in text.split(","):
        mode = token.strip().lower()
        if not mode:
            continue
        if mode not in {"cpu", "gpu"}:
            raise ValueError(f"Unsupported mode {mode!r}; expected cpu and/or gpu")
        modes.append(mode)
    if not modes:
        raise ValueError("--modes must include at least one of cpu,gpu")
    return sorted(dict.fromkeys(modes))


def _power_of_two_points(max_value: int) -> list[int]:
    out = [1]
    value = 1
    while value * 2 <= max_value:
        value *= 2
        out.append(value)
    if out[-1] != max_value:
        out.append(max_value)
    return sorted(dict.fromkeys(out))


def _default_mpi_points(nodes: int, ranks_per_node: int) -> list[int]:
    total = nodes * ranks_per_node
    points = set(_power_of_two_points(total))
    points.add(ranks_per_node)
    points.add(total)
    return sorted(p for p in points if p <= total)


def _default_cpu_device_points(ranks_per_node: int, cap: int = 32) -> list[int]:
    max_devices = min(ranks_per_node, cap)
    return _power_of_two_points(max_devices)


def _default_gpu_device_points(gpus_per_node: int) -> list[int]:
    return _power_of_two_points(gpus_per_node)


def _run_cmd(cmd: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "status": "pass" if proc.returncode == 0 else "fail",
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "status": "fail",
            "returncode": 1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _shell_join(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _build_parallel_validation_cmd(
    *,
    python_exe: str,
    output_dir: Path,
    job: JobSpec,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        python_exe,
        "scripts/run_parallel_validation.py",
        "--output",
        str(output_dir),
        "--python",
        python_exe,
        "--mpi-ranks",
        job.mpi_ranks_csv,
        "--mpi-launcher",
        args.mpi_launcher,
        "--scaling-backends",
        job.mode,
        "--scaling-workload",
        args.scaling_workload,
        "--scaling-dt",
        f"{args.scaling_dt}",
        "--scaling-cpu-devices",
        job.scaling_cpu_devices_csv,
        "--scaling-gpu-devices",
        job.scaling_gpu_devices_csv,
        "--scaling-strong-grid",
        str(args.scaling_strong_grid),
        "--scaling-weak-base-grid",
        str(args.scaling_weak_base_grid),
        "--scaling-iterations",
        str(args.scaling_iterations),
        "--scaling-warmup",
        str(args.scaling_warmup),
        "--timeout-sec",
        str(args.timeout_sec),
        "--mpi-scaling",
        "--mpi-scaling-strong-grid",
        str(args.mpi_scaling_strong_grid),
        "--mpi-scaling-weak-base-grid",
        str(args.mpi_scaling_weak_base_grid),
        "--mpi-scaling-iterations",
        str(args.mpi_scaling_iterations),
        "--mpi-scaling-warmup",
        str(args.mpi_scaling_warmup),
    ]
    if args.mpi_extra_args.strip():
        cmd.extend(["--mpi-extra-args", args.mpi_extra_args.strip()])
    if args.mpi_mca.strip():
        cmd.extend(["--mpi-mca", args.mpi_mca.strip()])
    if args.mpi_interface.strip():
        cmd.extend(["--mpi-interface", args.mpi_interface.strip()])
    if args.mpi_env.strip():
        cmd.extend(["--mpi-env", args.mpi_env.strip()])
    if args.x64:
        cmd.append("--x64")
    return cmd


def _render_sbatch_script(
    *,
    job: JobSpec,
    args: argparse.Namespace,
    job_name: str,
    script_path: Path,
    run_cmd: list[str],
    repo_root: Path,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={job.partition}",
        f"#SBATCH --nodes={job.nodes}",
        f"#SBATCH --ntasks-per-node={job.ntasks_per_node}",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --time={args.time_limit}",
        f"#SBATCH --output={script_path.with_suffix('.out')}",
        f"#SBATCH --error={script_path.with_suffix('.err')}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    if args.qos:
        lines.append(f"#SBATCH --qos={args.qos}")
    if args.constraint:
        lines.append(f"#SBATCH --constraint={args.constraint}")
    if args.exclusive:
        lines.append("#SBATCH --exclusive")
    if job.mode == "gpu":
        lines.append(f"#SBATCH --gpus-per-node={job.gpus_per_node}")
        if args.gpu_constraint:
            lines.append(f"#SBATCH --constraint={args.gpu_constraint}")

    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_root))}",
            'export PYTHONUNBUFFERED=1',
            'export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/results/.mplconfig}"',
            'mkdir -p "$MPLCONFIGDIR"',
        ],
    )
    if job.mode == "gpu":
        lines.append('export JAX_PLATFORMS="${JAX_PLATFORMS:-gpu,cpu}"')
    else:
        lines.append('export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"')

    if args.module_load.strip():
        for module_name in [m.strip() for m in args.module_load.split(",") if m.strip()]:
            lines.append(f"module load {shlex.quote(module_name)}")

    extra_env = _parse_kv_csv(args.extra_env)
    for key, value in sorted(extra_env.items()):
        lines.append(f"export {key}={shlex.quote(value)}")

    lines.extend(
        [
            "",
            f"echo '[levante-scaling] mode={job.mode} nodes={job.nodes} ntasks_per_node={job.ntasks_per_node}'",
            f"echo '[levante-scaling] command: {_shell_join(run_cmd)}'",
            _shell_join(run_cmd),
        ],
    )
    return "\n".join(lines) + "\n"


def _build_jobs(args: argparse.Namespace) -> list[JobSpec]:
    modes = _parse_mode_csv(args.modes)
    jobs: list[JobSpec] = []

    if "cpu" in modes:
        cpu_nodes = _parse_int_csv(args.cpu_nodes, label="--cpu-nodes")
        cpu_dev_points = (
            _parse_int_csv(args.cpu_device_points, label="--cpu-device-points")
            if args.cpu_device_points.strip()
            else _default_cpu_device_points(args.cpu_ranks_per_node)
        )
        for n_nodes in cpu_nodes:
            mpi_points = (
                _parse_int_csv(args.cpu_mpi_ranks, label="--cpu-mpi-ranks")
                if args.cpu_mpi_ranks.strip()
                else _default_mpi_points(n_nodes, args.cpu_ranks_per_node)
            )
            jobs.append(
                JobSpec(
                    mode="cpu",
                    nodes=n_nodes,
                    partition=args.cpu_partition,
                    ntasks_per_node=args.cpu_ranks_per_node,
                    gpus_per_node=0,
                    mpi_ranks_csv=",".join(str(x) for x in mpi_points),
                    scaling_cpu_devices_csv=",".join(str(x) for x in cpu_dev_points),
                    scaling_gpu_devices_csv="1",
                ),
            )

    if "gpu" in modes:
        gpu_nodes = _parse_int_csv(args.gpu_nodes, label="--gpu-nodes")
        gpu_dev_points = (
            _parse_int_csv(args.gpu_device_points, label="--gpu-device-points")
            if args.gpu_device_points.strip()
            else _default_gpu_device_points(args.gpus_per_node)
        )
        for n_nodes in gpu_nodes:
            mpi_points = (
                _parse_int_csv(args.gpu_mpi_ranks, label="--gpu-mpi-ranks")
                if args.gpu_mpi_ranks.strip()
                else _default_mpi_points(n_nodes, args.gpu_ranks_per_node)
            )
            jobs.append(
                JobSpec(
                    mode="gpu",
                    nodes=n_nodes,
                    partition=args.gpu_partition,
                    ntasks_per_node=args.gpu_ranks_per_node,
                    gpus_per_node=args.gpus_per_node,
                    mpi_ranks_csv=",".join(str(x) for x in mpi_points),
                    scaling_cpu_devices_csv="1",
                    scaling_gpu_devices_csv=",".join(str(x) for x in gpu_dev_points),
                ),
            )

    return jobs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate/submit Levante strong+weak scaling benchmark jobs.",
    )
    parser.add_argument("--output", type=str, default="results/levante_scaling", help="Base output directory.")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag (default: UTC timestamp).")
    parser.add_argument("--submit", action="store_true", help="Submit generated sbatch scripts.")
    parser.add_argument("--python", type=str, default=".venv/bin/python", help="Python executable for benchmark command.")
    parser.add_argument("--modes", type=str, default="cpu,gpu", help="Comma-separated modes: cpu,gpu.")

    parser.add_argument("--account", type=str, default="", help="SLURM account.")
    parser.add_argument("--qos", type=str, default="", help="SLURM QoS.")
    parser.add_argument("--constraint", type=str, default="", help="Optional SLURM constraint.")
    parser.add_argument("--gpu-constraint", type=str, default="", help="Optional GPU-specific SLURM constraint.")
    parser.add_argument("--time-limit", type=str, default="01:00:00", help="SLURM time limit (HH:MM:SS).")
    parser.add_argument("--exclusive", action=argparse.BooleanOptionalAction, default=True, help="Request exclusive nodes.")
    parser.add_argument("--module-load", type=str, default="", help="Comma-separated modules to load in sbatch jobs.")
    parser.add_argument("--extra-env", type=str, default="", help="Extra env exports KEY=VALUE[,KEY=VALUE].")

    parser.add_argument("--cpu-partition", type=str, default="compute", help="CPU partition name.")
    parser.add_argument("--cpu-nodes", type=str, default="1,2,4,8", help="CPU node counts.")
    parser.add_argument("--cpu-ranks-per-node", type=int, default=128, help="MPI ranks per CPU node.")
    parser.add_argument("--cpu-device-points", type=str, default="", help="Override CPU local-device points for non-MPI scaling.")
    parser.add_argument("--cpu-mpi-ranks", type=str, default="", help="Override CPU MPI rank points.")

    parser.add_argument("--gpu-partition", type=str, default="gpu", help="GPU partition name.")
    parser.add_argument("--gpu-nodes", type=str, default="1,2,4", help="GPU node counts.")
    parser.add_argument("--gpus-per-node", type=int, default=4, help="GPUs per Levante GPU node.")
    parser.add_argument("--gpu-ranks-per-node", type=int, default=4, help="MPI ranks per GPU node (typically 1 per GPU).")
    parser.add_argument("--gpu-device-points", type=str, default="", help="Override GPU local-device points for non-MPI scaling.")
    parser.add_argument("--gpu-mpi-ranks", type=str, default="", help="Override GPU MPI rank points.")

    parser.add_argument("--mpi-launcher", type=str, default="auto", help="MPI launcher (auto/mpirun/mpiexec).")
    parser.add_argument("--mpi-extra-args", type=str, default="", help="Extra args passed to MPI launcher.")
    parser.add_argument("--mpi-mca", type=str, default="", help="OpenMPI MCA overrides KEY=VALUE[,KEY=VALUE].")
    parser.add_argument("--mpi-interface", type=str, default="", help="OpenMPI btl_tcp_if_include value.")
    parser.add_argument("--mpi-env", type=str, default="", help="MPI subprocess env KEY=VALUE[,KEY=VALUE].")

    parser.add_argument("--scaling-workload", type=str, default="atmosphere_sw", choices=("halo", "atmosphere_sw"))
    parser.add_argument("--scaling-dt", type=float, default=300.0)
    parser.add_argument("--scaling-strong-grid", type=int, default=256)
    parser.add_argument("--scaling-weak-base-grid", type=int, default=256)
    parser.add_argument("--scaling-iterations", type=int, default=20)
    parser.add_argument("--scaling-warmup", type=int, default=3)
    parser.add_argument("--mpi-scaling-strong-grid", type=int, default=96)
    parser.add_argument("--mpi-scaling-weak-base-grid", type=int, default=96)
    parser.add_argument("--mpi-scaling-iterations", type=int, default=12)
    parser.add_argument("--mpi-scaling-warmup", type=int, default=2)
    parser.add_argument("--timeout-sec", type=float, default=3600.0)
    parser.add_argument("--x64", action="store_true", help="Enable float64 in scaling runs.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    tag = args.tag.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (repo_root / args.output / tag).resolve()
    jobs_dir = root / "jobs"
    outputs_dir = root / "runs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(args)
    if not jobs:
        raise RuntimeError("No jobs generated. Check --modes and node-count options.")

    _json_dump(root / "levante_facts.json", LEVANTE_FACTS)

    manifest: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "output_root": str(root),
        "submit_requested": bool(args.submit),
        "jobs": [],
    }

    for idx, job in enumerate(jobs, start=1):
        run_out = outputs_dir / f"{idx:02d}_{job.mode}_n{job.nodes}"
        run_out.mkdir(parents=True, exist_ok=True)
        run_cmd = _build_parallel_validation_cmd(
            python_exe=args.python,
            output_dir=run_out,
            job=job,
            args=args,
        )
        job_name = f"lev_{job.mode}_n{job.nodes}"
        script_path = jobs_dir / f"{idx:02d}_{job_name}.sbatch"
        script_text = _render_sbatch_script(
            job=job,
            args=args,
            job_name=job_name,
            script_path=script_path,
            run_cmd=run_cmd,
            repo_root=repo_root,
        )
        script_path.write_text(script_text, encoding="utf-8")

        record = {
            "index": idx,
            "mode": job.mode,
            "nodes": job.nodes,
            "partition": job.partition,
            "ntasks_per_node": job.ntasks_per_node,
            "gpus_per_node": job.gpus_per_node,
            "mpi_ranks": job.mpi_ranks_csv,
            "scaling_cpu_devices": job.scaling_cpu_devices_csv,
            "scaling_gpu_devices": job.scaling_gpu_devices_csv,
            "script": str(script_path),
            "run_output": str(run_out),
            "command": _shell_join(run_cmd),
            "submit": {"status": "not_submitted"},
        }

        if args.submit:
            submit_res = _run_cmd(["sbatch", str(script_path)], cwd=repo_root)
            record["submit"] = submit_res

        manifest["jobs"].append(record)

    _json_dump(root / "manifest.json", manifest)

    print("=" * 72)
    print("Levante Scaling Job Generator")
    print("=" * 72)
    print(f"Output root: {root}")
    print(f"Jobs generated: {len(jobs)}")
    print(f"Submit mode: {args.submit}")
    print(f"Manifest: {root / 'manifest.json'}")
    print(f"Levante facts: {root / 'levante_facts.json'}")
    print("=" * 72)

    if args.submit:
        submitted = [j for j in manifest["jobs"] if j["submit"]["status"] == "pass"]
        failed = [j for j in manifest["jobs"] if j["submit"]["status"] != "pass"]
        print(f"Submitted: {len(submitted)}")
        print(f"Failed submissions: {len(failed)}")
        for rec in failed:
            print(f"  - {rec['script']}: {rec['submit'].get('stderr', '').strip()}")
        return 0 if not failed else 1

    print("No jobs submitted. Re-run with --submit to queue on Levante.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
