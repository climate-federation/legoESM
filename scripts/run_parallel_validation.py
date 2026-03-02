#!/usr/bin/env python
"""Validate MPI runtime and run strong/weak scaling benchmarks.

This script performs three checks:
1. Parallel unit tests (no MPI required)
2. MPI distributed test cases (if launcher + mpi4py + mpi4jax are available)
3. Scaling benchmark suite (strong + weak scaling on CPU/GPU backends)

Outputs:
  - <output>/results.json
  - <output>/report.md
  - <output>/logs/*.log
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any


def _parse_int_list(csv_text: str, *, label: str) -> list[int]:
    values = []
    for token in csv_text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError(f"{label} entries must be >=1, got {value!r}")
        values.append(value)
    if not values:
        raise ValueError(f"{label} must contain at least one integer, got {csv_text!r}")
    return values


def _parse_str_list(csv_text: str) -> list[str]:
    return [token.strip() for token in csv_text.split(",") if token.strip()]


def _parse_kv_list(csv_text: str, *, label: str) -> list[tuple[str, str]]:
    """Parse KEY=VALUE items preserving order.

    Delimiters:
      - comma: ``A=1,B=2``
      - semicolon: ``A=1;B=2`` (useful when values include commas)
    """
    items: list[tuple[str, str]] = []
    if not csv_text.strip():
        return items
    if ";" in csv_text:
        tokens = csv_text.split(";")
    elif csv_text.count("=") == 1:
        # Single KEY=VALUE item; allow commas inside the value.
        tokens = [csv_text]
    else:
        tokens = csv_text.split(",")

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"{label} entries must be KEY=VALUE, got {token!r}",
            )
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"{label} contains empty key in {token!r}")
        items.append((key, value))
    return items


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _merge_xla_flags(existing: str | None, n_devices: int) -> str:
    host_flag = f"--xla_force_host_platform_device_count={n_devices}"
    if not existing:
        return host_flag
    if host_flag in existing:
        return existing
    return f"{existing} {host_flag}"


def _extract_json_line(stdout: str) -> dict[str, Any] | None:
    """Parse the last JSON object emitted to stdout, if present."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _run_cmd(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        return {
            "status": "timeout",
            "returncode": 124,
            "elapsed_s": elapsed,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"Command timed out after {timeout_sec} seconds.",
        }

    elapsed = time.perf_counter() - started
    status = "pass" if proc.returncode == 0 else "fail"
    return {
        "status": status,
        "returncode": int(proc.returncode),
        "elapsed_s": elapsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _write_log(path: Path, cmd: list[str], result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"Command: {shlex.join(cmd)}\n")
        f.write(f"Status: {result['status']}\n")
        f.write(f"Return code: {result['returncode']}\n")
        f.write(f"Elapsed (s): {result['elapsed_s']:.3f}\n\n")
        f.write("=== STDOUT ===\n")
        f.write(result.get("stdout", ""))
        f.write("\n\n=== STDERR ===\n")
        f.write(result.get("stderr", ""))


def _device_count(platform: str) -> int:
    import jax

    try:
        return len(jax.devices(platform))
    except Exception:
        return 0


def _probe_host() -> dict[str, Any]:
    import jax

    device_counts = {
        "cpu": _device_count("cpu"),
        "gpu": _device_count("gpu"),
        "tpu": _device_count("tpu"),
    }

    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "process_count": int(jax.process_count()),
        "local_device_count": int(jax.local_device_count()),
        "local_devices": [f"{d.platform}:{d.id}" for d in jax.local_devices()],
        "device_counts": device_counts,
        "mpirun_path": shutil.which("mpirun"),
        "mpiexec_path": shutil.which("mpiexec"),
        "has_mpi4py": importlib.util.find_spec("mpi4py") is not None,
        "has_mpi4jax": importlib.util.find_spec("mpi4jax") is not None,
    }


def _run_scaling_worker(
    backend_target: str,
    case_type: str,
    n_devices: int,
    grid_size: int,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    try:
        import jax
        import jax.numpy as jnp

        from legoesm.grids.halo import pad_halo, set_halo_backend
        from legoesm.parallel.mesh import create_device_mesh, shard_pytree

        active_backend = jax.default_backend().lower()
        if backend_target and active_backend != backend_target:
            return {
                "status": "skipped",
                "reason": (
                    f"Requested backend '{backend_target}' but runtime backend is "
                    f"'{active_backend}'."
                ),
            }

        if 6 % n_devices != 0:
            return {
                "status": "skipped",
                "reason": f"n_devices={n_devices} does not evenly divide 6 faces.",
            }

        set_halo_backend("local")
        config = create_device_mesh(n_devices=n_devices)
        if int(config.n_devices) != int(n_devices):
            return {
                "status": "skipped",
                "reason": (
                    f"Requested {n_devices} device(s), runtime resolved {config.n_devices}."
                ),
            }

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (6, grid_size, grid_size), dtype=jnp.float32)
        x = shard_pytree(x, config)

        @jax.jit
        def kernel(field):
            halo = pad_halo(field)
            return 0.25 * (
                halo
                + jnp.roll(halo, shift=1, axis=1)
                + jnp.roll(halo, shift=-1, axis=1)
                + jnp.roll(halo, shift=1, axis=2)
            )

        # Compile + first execution cost.
        t_compile0 = time.perf_counter()
        y = kernel(x)
        y.block_until_ready()
        compile_time_s = time.perf_counter() - t_compile0

        for _ in range(max(0, warmup)):
            y = kernel(y)
            y.block_until_ready()

        t_steady0 = time.perf_counter()
        for _ in range(max(1, iterations)):
            y = kernel(y)
        y.block_until_ready()
        steady_total_s = time.perf_counter() - t_steady0

        step_count = max(1, iterations)
        steady_ms_per_step = 1000.0 * steady_total_s / step_count
        cells_per_step = 6.0 * float(grid_size) * float(grid_size)
        throughput_global_mcells_s = (
            (cells_per_step * step_count / steady_total_s) / 1.0e6
            if steady_total_s > 0.0
            else 0.0
        )
        process_count = max(1, int(jax.process_count()))
        throughput_per_device_mcells_s = throughput_global_mcells_s / float(config.n_devices)
        throughput_per_rank_mcells_s = throughput_global_mcells_s / float(process_count)

        return {
            "status": "pass",
            "backend_target": backend_target,
            "active_backend": active_backend,
            "case_type": case_type,
            "n_devices": int(config.n_devices),
            "process_count": process_count,
            "local_device_count": int(jax.local_device_count()),
            "grid_size": int(grid_size),
            "iterations": int(step_count),
            "warmup": int(max(0, warmup)),
            "compile_time_s": compile_time_s,
            "steady_total_s": steady_total_s,
            "steady_ms_per_step": steady_ms_per_step,
            "cells_per_step": cells_per_step,
            "throughput_global_mcells_s": throughput_global_mcells_s,
            "throughput_per_device_mcells_s": throughput_per_device_mcells_s,
            "throughput_per_rank_mcells_s": throughput_per_rank_mcells_s,
        }
    except Exception as exc:  # pragma: no cover - exercised by orchestrator mode.
        return {
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _run_scaling_point(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    backend: str,
    case_type: str,
    n_devices: int,
    grid_size: int,
) -> dict[str, Any]:
    cmd = [
        args.python,
        "scripts/run_parallel_validation.py",
        "--scaling-worker",
        "--backend-target",
        backend,
        "--case-type",
        case_type,
        "--n-devices",
        str(n_devices),
        "--grid-size",
        str(grid_size),
        "--iterations",
        str(args.scaling_iterations),
        "--warmup",
        str(args.scaling_warmup),
    ]

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = backend
    if backend == "cpu":
        env["XLA_FLAGS"] = _merge_xla_flags(env.get("XLA_FLAGS"), n_devices)
    if args.x64:
        env["JAX_ENABLE_X64"] = "1"

    run = _run_cmd(cmd, env=env, timeout_sec=args.timeout_sec)
    log_path = output_dir / "logs" / f"scaling_{backend}_{case_type}_n{n_devices}.log"
    _write_log(log_path, cmd, run)

    worker_payload = _extract_json_line(run.get("stdout", ""))
    if worker_payload is None:
        worker_payload = {
            "status": "fail",
            "error": "No worker JSON payload found in stdout.",
            "stdout_tail": _tail(run.get("stdout", "")),
            "stderr_tail": _tail(run.get("stderr", "")),
        }

    return {
        "backend": backend,
        "case_type": case_type,
        "n_devices_requested": int(n_devices),
        "grid_size": int(grid_size),
        "command_status": run["status"],
        "command_returncode": run["returncode"],
        "elapsed_s": run["elapsed_s"],
        "log": str(log_path),
        "worker": worker_payload,
    }


def _case_status(entries: list[dict[str, Any]]) -> str:
    statuses = {entry.get("status", "fail") for entry in entries}
    if not entries:
        return "skipped"
    if "fail" in statuses:
        return "fail"
    if "pass" in statuses:
        return "pass"
    return "skipped"


def _evaluate_scaling_case(
    entries: list[dict[str, Any]],
    *,
    case_type: str,
    compile_time_max_s: float,
    strong_min_efficiency: float,
    weak_max_step_growth: float,
    weak_min_per_device_throughput_ratio: float,
) -> dict[str, Any]:
    """Attach derived metrics/checks and return case-level status."""
    evaluated = []
    baseline = None
    baseline_devices = None

    for entry in sorted(entries, key=lambda e: e["n_devices_requested"]):
        worker = entry["worker"]
        status = worker.get("status", "fail")
        checks = {}
        derived = {}
        pass_flag = status == "pass"

        if status == "pass":
            compile_time = float(worker["compile_time_s"])
            step_ms = float(worker["steady_ms_per_step"])
            per_dev = float(worker["throughput_per_device_mcells_s"])
            checks["compile_time_ok"] = compile_time <= compile_time_max_s

            if baseline is None:
                baseline = worker
                baseline_devices = max(1, int(worker["n_devices"]))
                derived["speedup_vs_baseline"] = 1.0
                derived["efficiency_vs_baseline"] = 1.0
                derived["step_growth_vs_baseline"] = 1.0
                derived["per_device_throughput_ratio_vs_baseline"] = 1.0
                checks["case_scaling_ok"] = True
            else:
                base_step_ms = float(baseline["steady_ms_per_step"])
                base_per_dev = float(baseline["throughput_per_device_mcells_s"])
                device_ratio = max(1.0, float(worker["n_devices"]) / float(baseline_devices))
                speedup = base_step_ms / max(step_ms, 1.0e-12)
                efficiency = speedup / device_ratio
                step_growth = step_ms / max(base_step_ms, 1.0e-12)
                per_dev_ratio = per_dev / max(base_per_dev, 1.0e-12)

                derived["speedup_vs_baseline"] = speedup
                derived["efficiency_vs_baseline"] = efficiency
                derived["step_growth_vs_baseline"] = step_growth
                derived["per_device_throughput_ratio_vs_baseline"] = per_dev_ratio

                if case_type == "strong":
                    checks["case_scaling_ok"] = efficiency >= strong_min_efficiency
                else:
                    checks["case_scaling_ok"] = (
                        step_growth <= weak_max_step_growth
                        and per_dev_ratio >= weak_min_per_device_throughput_ratio
                    )

            pass_flag = all(bool(v) for v in checks.values())
            status = "pass" if pass_flag else "fail"

        evaluated.append(
            {
                **entry,
                "status": status,
                "checks": checks,
                "derived": derived,
            },
        )

    return {
        "status": _case_status(evaluated),
        "entries": evaluated,
    }


def _run_scaling_suite(args: argparse.Namespace, output_dir: Path, host_info: dict[str, Any]) -> dict[str, Any]:
    backends = _parse_str_list(args.scaling_backends)
    cpu_devices = _parse_int_list(args.scaling_cpu_devices, label="--scaling-cpu-devices")
    gpu_devices = _parse_int_list(args.scaling_gpu_devices, label="--scaling-gpu-devices")

    thresholds = {
        "compile_time_max_s": float(args.scaling_compile_time_max_s),
        "strong_min_efficiency": float(args.strong_min_efficiency),
        "weak_max_step_growth": float(args.weak_max_step_growth),
        "weak_min_per_device_throughput_ratio": float(
            args.weak_min_per_device_throughput_ratio,
        ),
    }

    backend_results = []
    any_pass = False
    any_fail = False

    for backend in backends:
        backend = backend.lower()
        if backend not in {"cpu", "gpu"}:
            backend_results.append(
                {
                    "backend": backend,
                    "status": "skipped",
                    "reason": f"Unsupported benchmark backend '{backend}'.",
                    "cases": [],
                },
            )
            continue

        available = int(host_info.get("device_counts", {}).get(backend, 0))
        if backend == "gpu" and available < 1:
            backend_results.append(
                {
                    "backend": backend,
                    "status": "skipped",
                    "reason": "No local GPU devices detected.",
                    "cases": [],
                },
            )
            continue

        requested_devices = cpu_devices if backend == "cpu" else gpu_devices
        if not requested_devices:
            backend_results.append(
                {
                    "backend": backend,
                    "status": "skipped",
                    "reason": "No device counts requested for backend.",
                    "cases": [],
                },
            )
            continue

        baseline_devices = min(requested_devices)
        case_results = []

        for case_type in ("strong", "weak"):
            raw_entries = []
            for n_devices in requested_devices:
                if case_type == "strong":
                    grid_size = int(args.scaling_strong_grid)
                else:
                    scale = (float(n_devices) / float(baseline_devices)) ** 0.5
                    grid_size = max(4, int(round(float(args.scaling_weak_base_grid) * scale)))

                raw_entries.append(
                    _run_scaling_point(
                        args=args,
                        output_dir=output_dir,
                        backend=backend,
                        case_type=case_type,
                        n_devices=n_devices,
                        grid_size=grid_size,
                    ),
                )

            evaluated = _evaluate_scaling_case(
                raw_entries,
                case_type=case_type,
                compile_time_max_s=thresholds["compile_time_max_s"],
                strong_min_efficiency=thresholds["strong_min_efficiency"],
                weak_max_step_growth=thresholds["weak_max_step_growth"],
                weak_min_per_device_throughput_ratio=thresholds[
                    "weak_min_per_device_throughput_ratio"
                ],
            )
            case_results.append(
                {
                    "case_type": case_type,
                    "status": evaluated["status"],
                    "entries": evaluated["entries"],
                },
            )

        backend_status = _case_status(case_results)
        if backend_status == "pass":
            any_pass = True
        if backend_status == "fail":
            any_fail = True

        backend_results.append(
            {
                "backend": backend,
                "available_devices": available,
                "requested_devices": requested_devices,
                "status": backend_status,
                "cases": case_results,
            },
        )

    if any_fail:
        status = "fail"
    elif any_pass:
        status = "pass"
    else:
        status = "skipped"

    return {
        "status": status,
        "thresholds": thresholds,
        "backends": backend_results,
    }


def _run_mpi_suite(args: argparse.Namespace, output_dir: Path, host_info: dict[str, Any]) -> dict[str, Any]:
    launcher_arg = args.mpi_launcher.strip()
    if launcher_arg and launcher_arg.lower() != "auto":
        launcher = shutil.which(launcher_arg) or launcher_arg
    else:
        launcher = host_info.get("mpirun_path") or host_info.get("mpiexec_path")
    missing = []
    if launcher is None:
        missing.append("MPI launcher (mpirun/mpiexec)")
    if not host_info.get("has_mpi4py", False):
        missing.append("mpi4py")
    if not host_info.get("has_mpi4jax", False):
        missing.append("mpi4jax")

    if missing:
        return {
            "status": "skipped",
            "reason": "Missing MPI prerequisites.",
            "missing": missing,
            "runs": [],
        }

    try:
        mca_pairs = _parse_kv_list(args.mpi_mca, label="--mpi-mca")
        env_overrides = _parse_kv_list(args.mpi_env, label="--mpi-env")
    except ValueError as exc:
        return {
            "status": "fail",
            "launcher": launcher,
            "error": str(exc),
            "runs": [],
        }

    launcher_args = shlex.split(args.mpi_extra_args) if args.mpi_extra_args.strip() else []
    interface = args.mpi_interface.strip()

    base_mpi_cmd = [launcher]
    if args.mpi_allow_run_as_root:
        base_mpi_cmd.append("--allow-run-as-root")
    if interface:
        base_mpi_cmd.extend(["--mca", "btl_tcp_if_include", interface])
    for key, value in mca_pairs:
        base_mpi_cmd.extend(["--mca", key, value])
    base_mpi_cmd.extend(launcher_args)

    ranks = _parse_int_list(args.mpi_ranks, label="--mpi-ranks")
    runs = []
    all_pass = True
    for n_ranks in ranks:
        cmd = [
            *base_mpi_cmd,
            "-np",
            str(n_ranks),
            args.python,
            "-m",
            "pytest",
            "-q",
            "tests/distributed/test_halo_mpi.py",
        ]
        env = dict(os.environ)
        for key, value in env_overrides:
            env[key] = value
        if args.x64:
            env["JAX_ENABLE_X64"] = "1"

        run = _run_cmd(cmd, env=env, timeout_sec=args.timeout_sec)
        log_path = output_dir / "logs" / f"mpi_np{n_ranks}.log"
        _write_log(log_path, cmd, run)

        passed = run["returncode"] == 0
        all_pass = all_pass and passed
        runs.append(
            {
                "n_ranks": n_ranks,
                "status": "pass" if passed else run["status"],
                "returncode": run["returncode"],
                "elapsed_s": run["elapsed_s"],
                "log": str(log_path),
                "stdout_tail": _tail(run.get("stdout", "")),
                "stderr_tail": _tail(run.get("stderr", "")),
            },
        )

    return {
        "status": "pass" if all_pass else "fail",
        "launcher": launcher,
        "launcher_args": launcher_args,
        "interface": interface,
        "mca": {k: v for k, v in mca_pairs},
        "env_overrides": {k: v for k, v in env_overrides},
        "runs": runs,
    }


def _run_parallel_unit_tests(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    cmd = [
        args.python,
        "-m",
        "pytest",
        "-q",
        "tests/unit/test_parallel.py",
    ]
    env = dict(os.environ)
    if args.x64:
        env["JAX_ENABLE_X64"] = "1"
    run = _run_cmd(cmd, env=env, timeout_sec=args.timeout_sec)
    log_path = output_dir / "logs" / "unit_test_parallel.log"
    _write_log(log_path, cmd, run)
    return {
        "status": "pass" if run["returncode"] == 0 else run["status"],
        "returncode": run["returncode"],
        "elapsed_s": run["elapsed_s"],
        "log": str(log_path),
        "stdout_tail": _tail(run.get("stdout", "")),
        "stderr_tail": _tail(run.get("stderr", "")),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    host = payload["host"]
    unit = payload["parallel_unit_tests"]
    mpi = payload["mpi_validation"]
    scaling = payload["scaling_validation"]

    lines = [
        "# Parallel Validation Report",
        "",
        f"- Timestamp (UTC): {payload['timestamp_utc']}",
        f"- Overall pass: {payload['overall_pass']}",
        "",
        "## Host",
        "",
        f"- Backend: {host['backend']}",
        f"- Local devices: {host['local_device_count']} ({', '.join(host['local_devices'])})",
        f"- Device counts: cpu={host['device_counts']['cpu']}, gpu={host['device_counts']['gpu']}, tpu={host['device_counts']['tpu']}",
        f"- JAX process count: {host['process_count']}",
        f"- mpirun: {host.get('mpirun_path') or 'not found'}",
        f"- mpiexec: {host.get('mpiexec_path') or 'not found'}",
        f"- mpi4py: {host['has_mpi4py']}",
        f"- mpi4jax: {host['has_mpi4jax']}",
        "",
        "## Unit Tests",
        "",
        f"- Status: {unit['status']}",
        f"- Return code: {unit['returncode']}",
        f"- Elapsed (s): {unit['elapsed_s']:.2f}",
        f"- Log: `{unit['log']}`",
        "",
        "## MPI Validation",
        "",
        f"- Status: {mpi['status']}",
    ]
    if mpi.get("launcher"):
        lines.append(f"- Launcher: {mpi['launcher']}")
    if mpi.get("launcher_args"):
        lines.append(f"- Launcher args: `{shlex.join(mpi['launcher_args'])}`")
    if mpi.get("interface"):
        lines.append(f"- Interface include: {mpi['interface']}")
    if mpi.get("mca"):
        mca_text = ", ".join(f"{k}={v}" for k, v in mpi["mca"].items())
        lines.append(f"- MCA overrides: {mca_text}")
    if mpi.get("env_overrides"):
        env_text = ", ".join(f"{k}={v}" for k, v in mpi["env_overrides"].items())
        lines.append(f"- Env overrides: {env_text}")
    if mpi.get("error"):
        lines.append(f"- Error: {mpi['error']}")
    if mpi["status"] == "skipped":
        lines.append(f"- Reason: {mpi.get('reason', 'N/A')}")
        missing = ", ".join(mpi.get("missing", []))
        lines.append(f"- Missing: {missing or 'N/A'}")
    for run in mpi.get("runs", []):
        lines.append(
            f"- np={run['n_ranks']}: {run['status']} "
            f"(rc={run['returncode']}, {run['elapsed_s']:.2f}s) log=`{run['log']}`"
        )

    thresholds = scaling.get("thresholds", {})
    lines.extend(
        [
            "",
            "## Scaling Validation",
            "",
            f"- Status: {scaling['status']}",
            (
                "- Thresholds: "
                f"compile<= {thresholds.get('compile_time_max_s', 'N/A')}s, "
                f"strong_eff>= {thresholds.get('strong_min_efficiency', 'N/A'):.3f}, "
                f"weak_step_growth<= {thresholds.get('weak_max_step_growth', 'N/A'):.3f}, "
                f"weak_per_dev_tput_ratio>= "
                f"{thresholds.get('weak_min_per_device_throughput_ratio', 'N/A'):.3f}"
            ),
        ],
    )

    for backend in scaling.get("backends", []):
        lines.extend(
            [
                "",
                f"### Backend: {backend['backend'].upper()}",
                "",
                f"- Status: {backend['status']}",
            ],
        )
        if backend.get("reason"):
            lines.append(f"- Reason: {backend['reason']}")
            continue
        lines.append(
            f"- Requested devices: {backend.get('requested_devices', [])}; "
            f"available: {backend.get('available_devices', 'N/A')}"
        )

        for case in backend.get("cases", []):
            lines.append(f"- Case `{case['case_type']}`: {case['status']}")
            for run in case.get("entries", []):
                worker = run.get("worker", {})
                status = run.get("status", worker.get("status", "fail"))
                line = (
                    f"  n={run['n_devices_requested']} grid={run['grid_size']}: {status}"
                )
                if status == "pass":
                    line += (
                        f", compile={worker['compile_time_s']:.3f}s, "
                        f"steady={worker['steady_ms_per_step']:.3f}ms/step, "
                        f"tput={worker['throughput_global_mcells_s']:.2f} Mcells/s, "
                        f"per_device={worker['throughput_per_device_mcells_s']:.2f}, "
                        f"per_rank={worker['throughput_per_rank_mcells_s']:.2f}"
                    )
                    derived = run.get("derived", {})
                    if derived:
                        line += (
                            f", speedup={derived.get('speedup_vs_baseline', 0.0):.2f}x, "
                            f"eff={derived.get('efficiency_vs_baseline', 0.0):.2f}, "
                            f"step_growth={derived.get('step_growth_vs_baseline', 0.0):.2f}, "
                            f"per_dev_ratio="
                            f"{derived.get('per_device_throughput_ratio_vs_baseline', 0.0):.2f}"
                        )
                elif status == "skipped":
                    line += f", reason={worker.get('reason', 'N/A')}"
                else:
                    line += f", error={worker.get('error', 'unknown')}"
                line += f" (log=`{run['log']}`)"
                lines.append(line)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _main(args: argparse.Namespace) -> int:
    if args.scaling_worker:
        payload = _run_scaling_worker(
            backend_target=args.backend_target,
            case_type=args.case_type,
            n_devices=args.n_devices,
            grid_size=args.grid_size,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload.get("status") in {"pass", "skipped"} else 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    host_info = _probe_host()
    unit = _run_parallel_unit_tests(args, output_dir)
    mpi = _run_mpi_suite(args, output_dir, host_info)
    scaling = _run_scaling_suite(args, output_dir, host_info)

    overall_pass = (
        unit["status"] == "pass"
        and mpi["status"] in {"pass", "skipped"}
        and scaling["status"] in {"pass", "skipped"}
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": overall_pass,
        "host": host_info,
        "parallel_unit_tests": unit,
        "mpi_validation": mpi,
        "scaling_validation": scaling,
    }

    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = output_dir / "report.md"
    _write_report(report_path, payload)

    print("=" * 72)
    print("legoESM Parallel Validation")
    print("=" * 72)
    print(f"Output: {output_dir}")
    print(f"Overall pass: {overall_pass}")
    print(f"Unit tests: {unit['status']}")
    print(f"MPI validation: {mpi['status']}")
    print(f"Scaling validation: {scaling['status']}")
    print(f"Results JSON: {json_path}")
    print(f"Report:       {report_path}")
    print("=" * 72)
    return 0 if overall_pass else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate MPI runtime and run strong/weak scaling benchmarks.",
    )

    # Main mode arguments.
    parser.add_argument(
        "--output",
        type=str,
        default="results/parallel_validation",
        help="Output directory for validation reports.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used for subprocess checks.",
    )
    parser.add_argument(
        "--mpi-ranks",
        type=str,
        default="2,3,6",
        help="Comma-separated MPI process counts to test.",
    )
    parser.add_argument(
        "--mpi-launcher",
        type=str,
        default="auto",
        help="MPI launcher executable path/name (default: auto detect mpirun/mpiexec).",
    )
    parser.add_argument(
        "--mpi-extra-args",
        type=str,
        default="",
        help="Extra raw args passed to MPI launcher before -np (shell-split).",
    )
    parser.add_argument(
        "--mpi-mca",
        type=str,
        default="",
        help=(
            "MCA overrides KEY=VALUE (OpenMPI). "
            "Use comma or semicolon separators; semicolon supports comma in values."
        ),
    )
    parser.add_argument(
        "--mpi-interface",
        type=str,
        default="",
        help="Network interface include value for OpenMPI btl_tcp_if_include.",
    )
    parser.add_argument(
        "--mpi-env",
        type=str,
        default="",
        help=(
            "Env overrides KEY=VALUE for MPI subprocesses. "
            "Use comma or semicolon separators."
        ),
    )
    parser.add_argument(
        "--mpi-allow-run-as-root",
        action="store_true",
        help="Pass --allow-run-as-root to the MPI launcher (container/root setups).",
    )
    parser.add_argument(
        "--scaling-backends",
        type=str,
        default="cpu,gpu",
        help="Comma-separated scaling benchmark backends (cpu,gpu).",
    )
    parser.add_argument(
        "--scaling-cpu-devices",
        type=str,
        default="1,2,3,6",
        help="Comma-separated CPU device counts for scaling benchmarks.",
    )
    parser.add_argument(
        "--scaling-gpu-devices",
        type=str,
        default="1,2,3,6",
        help="Comma-separated GPU device counts for scaling benchmarks.",
    )
    parser.add_argument(
        "--scaling-strong-grid",
        type=int,
        default=256,
        help="Fixed grid size n for strong scaling (field: 6 x n x n).",
    )
    parser.add_argument(
        "--scaling-weak-base-grid",
        type=int,
        default=256,
        help=(
            "Baseline grid size at minimum device count for weak scaling. "
            "Grid scales as sqrt(device_ratio)."
        ),
    )
    parser.add_argument(
        "--scaling-iterations",
        type=int,
        default=20,
        help="Timed steady-state iterations per scaling point.",
    )
    parser.add_argument(
        "--scaling-warmup",
        type=int,
        default=3,
        help="Warmup iterations before steady-state timing.",
    )
    parser.add_argument(
        "--scaling-compile-time-max-s",
        type=float,
        default=30.0,
        help="Maximum allowed compile time per scaling point [s].",
    )
    parser.add_argument(
        "--strong-min-efficiency",
        type=float,
        default=0.12,
        help="Minimum strong-scaling parallel efficiency vs baseline.",
    )
    parser.add_argument(
        "--weak-max-step-growth",
        type=float,
        default=2.50,
        help="Maximum allowed weak-scaling steady-step growth vs baseline.",
    )
    parser.add_argument(
        "--weak-min-per-device-throughput-ratio",
        type=float,
        default=0.35,
        help="Minimum weak-scaling per-device throughput ratio vs baseline.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=1800.0,
        help="Command timeout in seconds for subprocess checks.",
    )
    parser.add_argument(
        "--x64",
        action="store_true",
        help="Run validation subprocesses with JAX_ENABLE_X64=1.",
    )

    # Internal worker mode.
    parser.add_argument("--scaling-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend-target", type=str, default="cpu", help=argparse.SUPPRESS)
    parser.add_argument("--case-type", type=str, default="strong", help=argparse.SUPPRESS)
    parser.add_argument("--n-devices", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--grid-size", type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument("--iterations", type=int, default=20, help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=3, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    raise SystemExit(_main(_build_parser().parse_args()))
