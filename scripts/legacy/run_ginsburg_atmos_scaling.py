#!/usr/bin/env python
"""Paper-grade atmosphere scaling benchmark runner for Ginsburg (SLURM).

This script orchestrates repeated atmosphere scaling runs using
`scripts/run_parallel_validation.py` and collates paper-ready artifacts:

  - raw point-level metrics CSV
  - repeat-aggregated summary CSV
  - markdown summary table
  - runtime metadata (SLURM/env/toolchain snapshots)

Supports CPU-only (MPI rank scaling), GPU-only, or both in one run.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any


def _parse_int_csv(text: str, *, label: str) -> list[int]:
    vals: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError(f"{label} entries must be >= 1, got {value!r}")
        vals.append(value)
    if not vals:
        raise ValueError(f"{label} must contain at least one integer")
    return vals


def _parse_mode_csv(text: str) -> list[str]:
    modes: list[str] = []
    for token in text.split(","):
        tok = token.strip().lower()
        if not tok:
            continue
        if tok not in {"cpu", "gpu"}:
            raise ValueError(f"Unsupported mode {tok!r}; expected cpu and/or gpu")
        modes.append(tok)
    if not modes:
        raise ValueError("--modes must include at least one of cpu,gpu")
    return modes


def _parse_kv_csv(text: str) -> dict[str, str]:
    text = text.strip()
    if not text:
        return {}
    tokens = text.split(";") if ";" in text else text.split(",")
    out: dict[str, str] = {}
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--extra-env entries must be KEY=VALUE, got {token!r}")
        k, v = token.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"--extra-env contains empty key in {token!r}")
        out[k] = v
    return out


def _append_pythonpath(env: dict[str, str], path: Path) -> None:
    pp = env.get("PYTHONPATH", "").strip()
    path_str = str(path)
    if not pp:
        env["PYTHONPATH"] = path_str
        return
    parts = pp.split(":")
    if path_str in parts:
        env["PYTHONPATH"] = pp
        return
    env["PYTHONPATH"] = f"{path_str}:{pp}"


def _run_capture(cmd: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - t0
        return {
            "status": "pass" if proc.returncode == 0 else "fail",
            "returncode": int(proc.returncode),
            "elapsed_s": float(elapsed),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "status": "fail",
            "returncode": 1,
            "elapsed_s": float(time.perf_counter() - t0),
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _parse_csv_tokens(text: str) -> list[str]:
    items: list[str] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            items.append(token)
    return items


def _missing_modules_in_python(python_exe: str, modules: list[str]) -> list[str]:
    if not modules:
        return []
    cmd = [
        python_exe,
        "-c",
        (
            "import importlib.util, json, sys; "
            "mods=sys.argv[1:]; "
            "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
            "print(json.dumps(missing))"
        ),
        *modules,
    ]
    res = _run_capture(cmd)
    if res["status"] != "pass":
        raise RuntimeError(
            "Failed to probe python environment for modules.\n"
            f"Command: {shlex.join(cmd)}\n"
            f"stderr: {res['stderr']}",
        )
    try:
        parsed = json.loads((res.get("stdout") or "").strip() or "[]")
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not parse module probe output as JSON.\n"
            f"stdout: {res.get('stdout','')}",
        ) from exc
    if not isinstance(parsed, list):
        raise RuntimeError(f"Unexpected probe payload type: {type(parsed).__name__}")
    return [str(x) for x in parsed]


def _ensure_required_packages(
    *,
    python_exe: str,
    install_missing: bool,
    pip_spec_map: dict[str, str],
    pip_install_extra_args: str,
    repo_root: Path,
    need_mpi: bool,
) -> dict[str, Any]:
    required_modules = ["numpy", "pytest", "jax", "jaxlib"]
    if need_mpi:
        required_modules.extend(["mpi4py", "mpi4jax"])

    missing = _missing_modules_in_python(python_exe, required_modules)
    info: dict[str, Any] = {
        "python": python_exe,
        "required_modules": required_modules,
        "missing_before_install": missing,
        "installed_specs": [],
        "status": "pass" if not missing else "pending",
    }
    if not missing:
        return info

    if not install_missing:
        info["status"] = "fail"
        raise RuntimeError(
            "Missing required Python packages in benchmark environment: "
            f"{', '.join(missing)}. "
            "Re-run with --install-missing or pre-install dependencies.",
        )

    install_specs: list[str] = []
    seen: set[str] = set()
    for mod in missing:
        spec = pip_spec_map.get(mod, mod)
        if spec not in seen:
            install_specs.append(spec)
            seen.add(spec)

    pip_cmd = [python_exe, "-m", "pip", "install"]
    if pip_install_extra_args.strip():
        pip_cmd.extend(shlex.split(pip_install_extra_args))
    pip_cmd.extend(install_specs)

    res = _run_capture(pip_cmd, cwd=repo_root)
    info["installed_specs"] = install_specs
    info["pip_command"] = pip_cmd
    info["pip_returncode"] = res.get("returncode")
    if res["status"] != "pass":
        info["status"] = "fail"
        info["pip_stderr"] = res.get("stderr", "")
        raise RuntimeError(
            "Automatic dependency installation failed.\n"
            f"Command: {shlex.join(pip_cmd)}\n"
            f"stderr:\n{res.get('stderr','')}",
        )

    missing_after = _missing_modules_in_python(python_exe, required_modules)
    info["missing_after_install"] = missing_after
    if missing_after:
        info["status"] = "fail"
        raise RuntimeError(
            "Dependency installation completed but required modules are still missing: "
            f"{', '.join(missing_after)}",
        )
    info["status"] = "pass"
    return info


def _q(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    vals = sorted(float(v) for v in values)
    pos = (len(vals) - 1) * quantile
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _extract_scaling_rows(payload: dict[str, Any], *, mode: str, repeat: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    scaling = payload.get("scaling_validation", {})
    for backend in scaling.get("backends", []):
        backend_name = str(backend.get("backend", "unknown"))
        for case in backend.get("cases", []):
            case_type = str(case.get("case_type", "unknown"))
            for entry in case.get("entries", []):
                worker = entry.get("worker", {})
                derived = entry.get("derived", {})
                rows.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "suite": "local_scaling",
                        "backend": backend_name,
                        "case_type": case_type,
                        "scale_value": int(entry.get("n_devices_requested", 0)),
                        "grid_size": int(entry.get("grid_size", 0)),
                        "status": str(entry.get("status", "fail")),
                        "compile_time_s": _safe_float(worker.get("compile_time_s")),
                        "steady_ms_per_step": _safe_float(worker.get("steady_ms_per_step")),
                        "steady_total_s": _safe_float(worker.get("steady_total_s")),
                        "throughput_global_mcells_s": _safe_float(worker.get("throughput_global_mcells_s")),
                        "throughput_per_device_mcells_s": _safe_float(worker.get("throughput_per_device_mcells_s")),
                        "throughput_per_rank_mcells_s": _safe_float(worker.get("throughput_per_rank_mcells_s")),
                        "efficiency_vs_baseline": _safe_float(derived.get("efficiency_vs_baseline")),
                        "step_growth_vs_baseline": _safe_float(derived.get("step_growth_vs_baseline")),
                        "throughput_ratio_vs_baseline": _safe_float(
                            derived.get("per_device_throughput_ratio_vs_baseline"),
                        ),
                    },
                )

    mpi_scaling = payload.get("mpi_validation", {}).get("scaling", {})
    for case in mpi_scaling.get("cases", []):
        case_type = str(case.get("case_type", "unknown"))
        for entry in case.get("entries", []):
            worker = entry.get("worker", {})
            derived = entry.get("derived", {})
            rows.append(
                {
                    "mode": mode,
                    "repeat": repeat,
                    "suite": "mpi_scaling",
                    "backend": "mpi",
                    "case_type": case_type,
                    "scale_value": int(entry.get("n_ranks_requested", 0)),
                    "grid_size": int(entry.get("grid_size", 0)),
                    "status": str(entry.get("status", "fail")),
                    "compile_time_s": _safe_float(worker.get("compile_time_s")),
                    "steady_ms_per_step": _safe_float(worker.get("steady_ms_per_step")),
                    "steady_total_s": _safe_float(worker.get("steady_total_s")),
                    "throughput_global_mcells_s": _safe_float(worker.get("throughput_global_mcells_s")),
                    "throughput_per_device_mcells_s": _safe_float(worker.get("throughput_per_device_mcells_s")),
                    "throughput_per_rank_mcells_s": _safe_float(worker.get("throughput_per_rank_mcells_s")),
                    "efficiency_vs_baseline": _safe_float(derived.get("efficiency_vs_baseline")),
                    "step_growth_vs_baseline": _safe_float(derived.get("step_growth_vs_baseline")),
                    "throughput_ratio_vs_baseline": _safe_float(
                        derived.get("per_rank_throughput_ratio_vs_baseline"),
                    ),
                },
            )

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "mode",
        "repeat",
        "suite",
        "backend",
        "case_type",
        "scale_value",
        "grid_size",
        "status",
        "compile_time_s",
        "steady_ms_per_step",
        "steady_total_s",
        "throughput_global_mcells_s",
        "throughput_per_device_mcells_s",
        "throughput_per_rank_mcells_s",
        "efficiency_vs_baseline",
        "step_growth_vs_baseline",
        "throughput_ratio_vs_baseline",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        key = (
            r["mode"],
            r["suite"],
            r["backend"],
            r["case_type"],
            int(r["scale_value"]),
            int(r["grid_size"]),
        )
        grouped.setdefault(key, []).append(r)

    summary_rows: list[dict[str, Any]] = []
    metric_names = [
        "compile_time_s",
        "steady_ms_per_step",
        "throughput_global_mcells_s",
        "throughput_per_device_mcells_s",
        "throughput_per_rank_mcells_s",
        "efficiency_vs_baseline",
        "step_growth_vs_baseline",
        "throughput_ratio_vs_baseline",
    ]
    for key, group in sorted(grouped.items()):
        mode, suite, backend, case_type, scale_value, grid_size = key
        status_counts: dict[str, int] = {}
        for g in group:
            s = str(g.get("status", "fail"))
            status_counts[s] = status_counts.get(s, 0) + 1
        agg: dict[str, Any] = {
            "mode": mode,
            "suite": suite,
            "backend": backend,
            "case_type": case_type,
            "scale_value": scale_value,
            "grid_size": grid_size,
            "n_samples": len(group),
            "n_pass": status_counts.get("pass", 0),
            "n_fail": status_counts.get("fail", 0),
            "n_skipped": status_counts.get("skipped", 0),
        }
        for metric in metric_names:
            vals = [float(v) for v in (g.get(metric) for g in group) if isinstance(v, (float, int))]
            if vals:
                agg[f"{metric}_median"] = statistics.median(vals)
                agg[f"{metric}_p10"] = _q(vals, 0.10)
                agg[f"{metric}_p90"] = _q(vals, 0.90)
            else:
                agg[f"{metric}_median"] = ""
                agg[f"{metric}_p10"] = ""
                agg[f"{metric}_p90"] = ""
        summary_rows.append(agg)

    return summary_rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cols = [
        "mode",
        "suite",
        "backend",
        "case_type",
        "scale_value",
        "grid_size",
        "n_samples",
        "n_pass",
        "n_fail",
        "n_skipped",
        "compile_time_s_median",
        "compile_time_s_p10",
        "compile_time_s_p90",
        "steady_ms_per_step_median",
        "steady_ms_per_step_p10",
        "steady_ms_per_step_p90",
        "throughput_global_mcells_s_median",
        "throughput_global_mcells_s_p10",
        "throughput_global_mcells_s_p90",
        "throughput_per_device_mcells_s_median",
        "throughput_per_device_mcells_s_p10",
        "throughput_per_device_mcells_s_p90",
        "throughput_per_rank_mcells_s_median",
        "throughput_per_rank_mcells_s_p10",
        "throughput_per_rank_mcells_s_p90",
        "efficiency_vs_baseline_median",
        "efficiency_vs_baseline_p10",
        "efficiency_vs_baseline_p90",
        "step_growth_vs_baseline_median",
        "step_growth_vs_baseline_p10",
        "step_growth_vs_baseline_p90",
        "throughput_ratio_vs_baseline_median",
        "throughput_ratio_vs_baseline_p10",
        "throughput_ratio_vs_baseline_p90",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    repo_root: Path,
    run_records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Ginsburg Atmosphere Scaling Benchmark")
    lines.append("")
    lines.append(f"- UTC timestamp: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Repository: `{repo_root}`")
    lines.append(f"- Python: `{args.python}`")
    lines.append(f"- Modes: `{args.modes}`")
    lines.append(f"- Repeats: {args.repeats}")
    lines.append(f"- Workload: `atmosphere_sw`")
    lines.append(
        f"- Auto-install missing deps: `{args.install_missing}`"
        + (
            f" (extra pip args: `{args.pip_install_extra_args}`)"
            if args.pip_install_extra_args.strip()
            else ""
        ),
    )
    lines.append("")

    lines.append("## Run Status")
    lines.append("")
    lines.append("| mode | repeat | returncode | status | output_dir |")
    lines.append("|---|---:|---:|---|---|")
    for rec in run_records:
        lines.append(
            "| "
            f"{rec['mode']} | {rec['repeat']} | {rec['returncode']} | "
            f"{rec['status']} | `{rec['output_dir']}` |",
        )
    lines.append("")

    lines.append("## Scaling Summary (Median Across Repeats)")
    lines.append("")
    lines.append(
        "| mode | suite | case | scale | grid | n_pass/n | compile(s) | step(ms) | "
        "global tput (Mcells/s) | per-rank tput | per-device tput |",
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        lines.append(
            "| "
            f"{r['mode']} | {r['suite']} | {r['case_type']} | "
            f"{r['scale_value']} | {r['grid_size']} | "
            f"{r['n_pass']}/{r['n_samples']} | "
            f"{r['compile_time_s_median'] if r['compile_time_s_median'] != '' else 'NA'} | "
            f"{r['steady_ms_per_step_median'] if r['steady_ms_per_step_median'] != '' else 'NA'} | "
            f"{r['throughput_global_mcells_s_median'] if r['throughput_global_mcells_s_median'] != '' else 'NA'} | "
            f"{r['throughput_per_rank_mcells_s_median'] if r['throughput_per_rank_mcells_s_median'] != '' else 'NA'} | "
            f"{r['throughput_per_device_mcells_s_median'] if r['throughput_per_device_mcells_s_median'] != '' else 'NA'} |",
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_metadata(output_root: Path, repo_root: Path) -> None:
    meta_dir = output_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    env_capture = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(("SLURM_", "CUDA", "NCCL", "OMPI", "PMI", "JAX_", "XLA_"))
    }
    (meta_dir / "env.json").write_text(json.dumps(env_capture, indent=2), encoding="utf-8")

    cmds = {
        "git_head.txt": ["git", "rev-parse", "HEAD"],
        "git_status.txt": ["git", "status", "--short"],
        "python_version.txt": [sys.executable, "--version"],
        "mpirun_version.txt": ["mpirun", "--version"],
        "nvidia_smi.txt": ["nvidia-smi"],
        "lscpu.txt": ["lscpu"],
    }
    for fname, cmd in cmds.items():
        res = _run_capture(cmd, cwd=repo_root)
        text = (
            f"Command: {shlex.join(cmd)}\n"
            f"Status: {res['status']}\n"
            f"Return code: {res['returncode']}\n\n"
            "=== STDOUT ===\n"
            f"{res['stdout']}\n\n=== STDERR ===\n{res['stderr']}\n"
        )
        (meta_dir / fname).write_text(text, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run repeated atmosphere scaling benchmarks on Ginsburg via run_parallel_validation.py",
    )
    p.add_argument("--output-root", type=str, default="", help="Benchmark output root.")
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable.")
    p.add_argument("--modes", type=str, default="cpu,gpu", help="Comma-separated: cpu,gpu")
    p.add_argument("--repeats", type=int, default=3, help="Number of repeated benchmark runs per mode.")
    p.add_argument("--x64", action="store_true", help="Enable JAX x64 mode.")
    p.add_argument("--strict", action="store_true", help="Return non-zero if any run fails.")
    p.add_argument(
        "--install-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-install missing required python packages in --python environment.",
    )
    p.add_argument(
        "--pip-spec-map",
        type=str,
        default="numpy=numpy,pytest=pytest,jax=jax,jaxlib=jaxlib,mpi4py=mpi4py,mpi4jax=mpi4jax",
        help=(
            "Mapping module=package_spec for auto-install (comma/semicolon separated). "
            "Use to pin versions or GPU-specific wheels."
        ),
    )
    p.add_argument(
        "--pip-install-extra-args",
        type=str,
        default="",
        help="Extra raw args passed to `python -m pip install` (shell-split).",
    )

    p.add_argument("--cpu-scaling-devices", type=str, default="1,2,4,8")
    p.add_argument("--gpu-scaling-devices", type=str, default="1,2,4")
    p.add_argument("--cpu-mpi-ranks", type=str, default="1,2,4,8")
    p.add_argument("--gpu-mpi-ranks", type=str, default="1,2,4")
    p.add_argument("--disable-mpi", action="store_true", help="Disable MPI scaling sub-suite.")

    p.add_argument("--scaling-dt", type=float, default=300.0)
    p.add_argument("--scaling-iterations", type=int, default=30)
    p.add_argument("--scaling-warmup", type=int, default=5)
    p.add_argument("--cpu-strong-grid", type=int, default=192)
    p.add_argument("--cpu-weak-base-grid", type=int, default=192)
    p.add_argument("--gpu-strong-grid", type=int, default=256)
    p.add_argument("--gpu-weak-base-grid", type=int, default=256)
    p.add_argument("--mpi-strong-grid", type=int, default=128)
    p.add_argument("--mpi-weak-base-grid", type=int, default=128)
    p.add_argument("--mpi-scaling-iterations", type=int, default=16)
    p.add_argument("--mpi-scaling-warmup", type=int, default=3)
    p.add_argument("--timeout-sec", type=float, default=7200.0)

    # Default thresholds are permissive: this script is for benchmark data collection.
    p.add_argument("--scaling-compile-time-max-s", type=float, default=1.0e9)
    p.add_argument("--strong-min-efficiency", type=float, default=0.0)
    p.add_argument("--weak-max-step-growth", type=float, default=1.0e9)
    p.add_argument("--weak-min-per-device-throughput-ratio", type=float, default=0.0)

    p.add_argument("--mpi-launcher", type=str, default="mpirun")
    p.add_argument("--mpi-extra-args", type=str, default="")
    p.add_argument("--mpi-mca", type=str, default="")
    p.add_argument("--mpi-interface", type=str, default="")
    p.add_argument("--mpi-env", type=str, default="")
    p.add_argument("--mpi-allow-run-as-root", action="store_true")
    p.add_argument(
        "--extra-env",
        type=str,
        default="",
        help="Extra KEY=VALUE env pairs for subprocesses (comma/semicolon-separated).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    modes = _parse_mode_csv(args.modes)
    cpu_devices = _parse_int_csv(args.cpu_scaling_devices, label="--cpu-scaling-devices")
    gpu_devices = _parse_int_csv(args.gpu_scaling_devices, label="--gpu-scaling-devices")
    cpu_ranks = _parse_int_csv(args.cpu_mpi_ranks, label="--cpu-mpi-ranks")
    gpu_ranks = _parse_int_csv(args.gpu_mpi_ranks, label="--gpu-mpi-ranks")
    extra_env = _parse_kv_csv(args.extra_env)
    pip_spec_map = _parse_kv_csv(args.pip_spec_map)

    repo_root = Path(__file__).resolve().parents[1]
    out_root = (
        Path(args.output_root)
        if args.output_root
        else Path("results") / f"ginsburg_atmos_scaling_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    _collect_metadata(out_root, repo_root)
    install_info = _ensure_required_packages(
        python_exe=args.python,
        install_missing=bool(args.install_missing),
        pip_spec_map=pip_spec_map,
        pip_install_extra_args=args.pip_install_extra_args,
        repo_root=repo_root,
        need_mpi=not bool(args.disable_mpi),
    )
    (out_root / "metadata" / "python_dependency_install.json").write_text(
        json.dumps(install_info, indent=2),
        encoding="utf-8",
    )

    run_records: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for mode in modes:
        devices_csv = ",".join(str(v) for v in (cpu_devices if mode == "cpu" else gpu_devices))
        ranks_csv = ",".join(str(v) for v in (cpu_ranks if mode == "cpu" else gpu_ranks))
        strong_grid = args.cpu_strong_grid if mode == "cpu" else args.gpu_strong_grid
        weak_base_grid = args.cpu_weak_base_grid if mode == "cpu" else args.gpu_weak_base_grid

        for repeat in range(1, args.repeats + 1):
            run_out = out_root / mode / f"repeat_{repeat:02d}"
            run_out.mkdir(parents=True, exist_ok=True)

            cmd = [
                args.python,
                "scripts/run_parallel_validation.py",
                "--output",
                str(run_out),
                "--python",
                args.python,
                "--scaling-backends",
                mode,
                "--scaling-workload",
                "atmosphere_sw",
                "--scaling-dt",
                str(args.scaling_dt),
                "--scaling-cpu-devices",
                devices_csv if mode == "cpu" else "1",
                "--scaling-gpu-devices",
                devices_csv if mode == "gpu" else "1",
                "--scaling-strong-grid",
                str(strong_grid),
                "--scaling-weak-base-grid",
                str(weak_base_grid),
                "--scaling-iterations",
                str(args.scaling_iterations),
                "--scaling-warmup",
                str(args.scaling_warmup),
                "--scaling-compile-time-max-s",
                str(args.scaling_compile_time_max_s),
                "--strong-min-efficiency",
                str(args.strong_min_efficiency),
                "--weak-max-step-growth",
                str(args.weak_max_step_growth),
                "--weak-min-per-device-throughput-ratio",
                str(args.weak_min_per_device_throughput_ratio),
                "--timeout-sec",
                str(args.timeout_sec),
                "--mpi-ranks",
                ranks_csv,
                "--mpi-launcher",
                args.mpi_launcher,
                "--mpi-extra-args",
                args.mpi_extra_args,
                "--mpi-mca",
                args.mpi_mca,
                "--mpi-interface",
                args.mpi_interface,
                "--mpi-env",
                args.mpi_env,
                "--mpi-scaling-strong-grid",
                str(args.mpi_strong_grid),
                "--mpi-scaling-weak-base-grid",
                str(args.mpi_weak_base_grid),
                "--mpi-scaling-iterations",
                str(args.mpi_scaling_iterations),
                "--mpi-scaling-warmup",
                str(args.mpi_scaling_warmup),
            ]
            if args.disable_mpi:
                cmd.append("--no-mpi-scaling")
            if args.mpi_allow_run_as_root:
                cmd.append("--mpi-allow-run-as-root")
            if args.x64:
                cmd.append("--x64")

            env = dict(os.environ)
            _append_pythonpath(env, repo_root)
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("NUMEXPR_NUM_THREADS", "1")
            env["JAX_PLATFORMS"] = mode
            for k, v in extra_env.items():
                env[k] = v

            t0 = time.perf_counter()
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            elapsed = time.perf_counter() - t0

            driver_log = (
                f"Command: {shlex.join(cmd)}\n"
                f"Return code: {proc.returncode}\n"
                f"Elapsed (s): {elapsed:.3f}\n\n"
                "=== STDOUT ===\n"
                f"{proc.stdout}\n\n=== STDERR ===\n{proc.stderr}\n"
            )
            (run_out / "driver_run.log").write_text(driver_log, encoding="utf-8")

            payload_path = run_out / "results.json"
            payload = {}
            if payload_path.exists():
                try:
                    payload = json.loads(payload_path.read_text(encoding="utf-8"))
                except Exception as exc:  # pragma: no cover
                    payload = {"parse_error": f"{type(exc).__name__}: {exc}"}

            rec = {
                "mode": mode,
                "repeat": repeat,
                "status": "pass" if proc.returncode == 0 and payload_path.exists() else "fail",
                "returncode": int(proc.returncode),
                "elapsed_s": float(elapsed),
                "output_dir": str(run_out),
                "results_json": str(payload_path),
                "command": cmd,
            }
            run_records.append(rec)
            all_rows.extend(_extract_scaling_rows(payload, mode=mode, repeat=repeat))

            print(
                f"[{mode} repeat {repeat}/{args.repeats}] "
                f"returncode={proc.returncode} elapsed={elapsed:.1f}s output={run_out}",
                flush=True,
            )

    _write_csv(out_root / "paper_scaling_raw.csv", all_rows)
    summary_rows = _summarize(all_rows)
    _write_summary_csv(out_root / "paper_scaling_summary.csv", summary_rows)
    _write_markdown(
        out_root / "paper_scaling_report.md",
        args=args,
        repo_root=repo_root,
        run_records=run_records,
        summary_rows=summary_rows,
    )
    (out_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "args": vars(args),
                "repo_root": str(repo_root),
                "dependency_install": install_info,
                "runs": run_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    n_fail = sum(1 for r in run_records if r["status"] != "pass")
    print("=" * 72)
    print(f"Output root: {out_root}")
    print(f"Raw CSV:     {out_root / 'paper_scaling_raw.csv'}")
    print(f"Summary CSV: {out_root / 'paper_scaling_summary.csv'}")
    print(f"Report MD:   {out_root / 'paper_scaling_report.md'}")
    print(f"Runs: {len(run_records)}, failed: {n_fail}")
    print("=" * 72)

    if args.strict and n_fail > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
