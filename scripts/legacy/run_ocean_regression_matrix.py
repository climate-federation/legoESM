#!/usr/bin/env python
"""Run an ocean regression matrix and gate stability/conservation metrics.

This script orchestrates `run_ocean_tests.py` and optionally
`run_ocean_realistic.py`, then consumes their machine-readable
`summary.json` outputs and applies pass/fail thresholds plus warning tiers.

Outputs:
  - results/ocean_regression/results.json
  - results/ocean_regression/report.md
  - per-run logs in results/ocean_regression/logs/
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


CASE_SANITY_LIMITS = {
    "ocean_tests": {
        "rest_state": {"speed_max_mps": 0.1, "ssh_abs_max_m": 0.1},
        "gravity_wave": {"speed_max_mps": 20.0, "ssh_abs_max_m": 2.0},
        "wind_gyre": {"speed_max_mps": 20.0, "ssh_abs_max_m": 5.0},
    },
    "ocean_realistic": {
        "stommel_gyre": {"speed_max_mps": 150.0, "ssh_abs_max_m": 10.0},
        "baroclinic_adjustment": {"speed_max_mps": 80.0, "ssh_abs_max_m": 20.0},
        "kelvin_wave": {"speed_max_mps": 5.0, "ssh_abs_max_m": 1.0},
    },
}


def _parse_int_list(csv_text: str, *, label: str) -> list[int]:
    values = []
    for token in csv_text.split(","):
        token = token.strip()
        if token:
            value = int(token)
            if value < 1:
                raise ValueError(f"{label} entries must be >= 1, got {value!r}")
            values.append(value)
    if not values:
        raise ValueError(f"{label} must contain at least one integer, got {csv_text!r}")
    return values


def _pair_levels(resolutions: list[int], levels: list[int]) -> list[int]:
    if len(levels) == 1:
        return [levels[0] for _ in resolutions]
    if len(levels) != len(resolutions):
        raise ValueError(
            "levels must have length 1 or match resolutions length "
            f"(got resolutions={len(resolutions)}, levels={len(levels)})",
        )
    return levels


def _build_matrix(args, output_root: Path) -> list[dict]:
    matrix = []

    std_res = _parse_int_list(args.std_resolutions, label="--std-resolutions")
    std_lev = _pair_levels(std_res, _parse_int_list(args.std_levels, label="--std-levels"))
    for n, l in zip(std_res, std_lev):
        run_id = f"ocean_tests_C{n}_L{l}"
        outdir = output_root / "runs" / run_id
        cmd = [
            args.python,
            "scripts/run_ocean_tests.py",
            "--resolution", str(n),
            "--levels", str(l),
            "--dt", str(args.std_dt),
            "--days", str(args.std_days),
            "--test", "all",
            "--output", str(outdir),
        ]
        if args.runtime_checks:
            cmd.append("--runtime-checks")
        if args.x64:
            cmd.append("--x64")
        matrix.append(
            {
                "id": run_id,
                "suite": "ocean_tests",
                "resolution": n,
                "levels": l,
                "output_dir": str(outdir),
                "cmd": cmd,
            },
        )

    if args.include_realistic:
        real_res = _parse_int_list(
            args.realistic_resolutions, label="--realistic-resolutions",
        )
        real_lev = _pair_levels(
            real_res, _parse_int_list(args.realistic_levels, label="--realistic-levels"),
        )
        for n, l in zip(real_res, real_lev):
            run_id = f"ocean_realistic_C{n}_L{l}"
            outdir = output_root / "runs" / run_id
            cmd = [
                args.python,
                "scripts/run_ocean_realistic.py",
                "--resolution", str(n),
                "--levels", str(l),
                "--dt", str(args.realistic_dt),
                "--days", str(args.realistic_days),
                "--test", "all",
                "--output", str(outdir),
            ]
            if args.runtime_checks:
                cmd.append("--runtime-checks")
            if args.x64:
                cmd.append("--x64")
            matrix.append(
                {
                    "id": run_id,
                    "suite": "ocean_realistic",
                    "resolution": n,
                    "levels": l,
                    "output_dir": str(outdir),
                    "cmd": cmd,
                },
            )

    return matrix


def _effective_limits(
    suite: str,
    case_name: str,
    thresholds: dict,
) -> dict:
    """Return case-specific limits, falling back to global defaults."""
    limits = {
        "speed_max_mps": float(thresholds["speed_max_mps"]),
        "ssh_abs_max_m": float(thresholds["ssh_abs_max_m"]),
    }
    if not thresholds.get("use_case_profiles", True):
        return limits

    suite_limits = CASE_SANITY_LIMITS.get(suite, {})
    case_limits = suite_limits.get(case_name, {})
    if "speed_max_mps" in case_limits:
        limits["speed_max_mps"] = float(case_limits["speed_max_mps"])
    if "ssh_abs_max_m" in case_limits:
        limits["ssh_abs_max_m"] = float(case_limits["ssh_abs_max_m"])
    return limits


def _is_near_limit(value: float, limit: float, warn_fraction: float) -> bool:
    """True if value is within warning band (below fail limit, but close)."""
    if limit <= 0.0 or warn_fraction <= 0.0:
        return False
    if value > limit:
        return False
    return value >= warn_fraction * limit


def _evaluate_summary(summary: dict, thresholds: dict, suite: str) -> dict:
    case_results = {}
    overall_pass = True
    overall_has_warnings = False
    for case_name, metrics in summary.get("cases", {}).items():
        heat = abs(float(metrics.get("heat_drift_rel", 0.0)))
        salt = abs(float(metrics.get("salt_drift_rel", 0.0)))
        eta = abs(float(metrics.get("volume_mean_eta_drift", 0.0)))
        ssh_min = float(metrics.get("SSH_min", 0.0))
        ssh_max = float(metrics.get("SSH_max", 0.0))
        ssh_abs = max(abs(ssh_min), abs(ssh_max))

        speed = metrics.get("speed_max", None)
        if speed is None:
            u_max = float(metrics.get("u_max", 0.0))
            v_max = float(metrics.get("v_max", 0.0))
            speed = (u_max * u_max + v_max * v_max) ** 0.5
        speed = abs(float(speed))

        finite_ok = bool(metrics.get("all_finite", False))
        land_ok = bool(metrics.get("land_zero", False))

        limits = _effective_limits(suite, case_name, thresholds)
        speed_limit = limits["speed_max_mps"]
        ssh_limit = limits["ssh_abs_max_m"]
        warn_fraction = float(thresholds.get("warn_fraction", 0.8))
        speed_ok = (speed <= speed_limit) if speed_limit > 0.0 else True
        ssh_ok = (ssh_abs <= ssh_limit) if ssh_limit > 0.0 else True

        checks = {
            "all_finite": finite_ok,
            "land_zero": land_ok,
            "heat_drift_ok": heat <= thresholds["heat_drift_rel"],
            "salt_drift_ok": salt <= thresholds["salt_drift_rel"],
            "eta_drift_ok": eta <= thresholds["eta_mean_drift_m"],
            "speed_ok": speed_ok,
            "ssh_abs_ok": ssh_ok,
        }

        warnings = {
            "heat_drift_warn": _is_near_limit(heat, thresholds["heat_drift_rel"], warn_fraction),
            "salt_drift_warn": _is_near_limit(salt, thresholds["salt_drift_rel"], warn_fraction),
            "eta_drift_warn": _is_near_limit(eta, thresholds["eta_mean_drift_m"], warn_fraction),
            "speed_warn": _is_near_limit(speed, speed_limit, warn_fraction),
            "ssh_abs_warn": _is_near_limit(ssh_abs, ssh_limit, warn_fraction),
        }
        warning_items = [name for name, enabled in warnings.items() if enabled]

        case_pass = all(checks.values())
        case_has_warnings = len(warning_items) > 0
        overall_pass = overall_pass and case_pass
        overall_has_warnings = overall_has_warnings or case_has_warnings
        case_results[case_name] = {
            "pass": case_pass,
            "checks": checks,
            "has_warnings": case_has_warnings,
            "warnings": warning_items,
            "metrics": metrics,
            "derived": {
                "speed_max_mps": speed,
                "ssh_abs_max_m": ssh_abs,
            },
            "limits": limits,
        }

    case_errors = summary.get("case_errors", {}) or {}
    for case_name, err_msg in case_errors.items():
        limits = _effective_limits(suite, case_name, thresholds)
        case_results[case_name] = {
            "pass": False,
            "checks": {
                "all_finite": False,
                "land_zero": False,
                "heat_drift_ok": False,
                "salt_drift_ok": False,
                "eta_drift_ok": False,
                "speed_ok": False,
                "ssh_abs_ok": False,
                "case_error": False,
            },
            "has_warnings": False,
            "warnings": [],
            "error": str(err_msg),
            "metrics": {},
            "derived": {
                "speed_max_mps": None,
                "ssh_abs_max_m": None,
            },
            "limits": limits,
        }
        overall_pass = False

    if not case_results:
        overall_pass = False

    return {
        "pass": overall_pass,
        "has_warnings": overall_has_warnings,
        "cases": case_results,
    }


def _run_one(
    entry: dict,
    cwd: Path,
    logs_dir: Path,
    timeout_s: float | None = None,
) -> dict:
    run_id = entry["id"]
    cmd = entry["cmd"]
    log_path = logs_dir / f"{run_id}.log"
    summary_path = Path(entry["output_dir"]) / "summary.json"

    # Avoid consuming stale summaries from previous runs.
    if summary_path.exists():
        summary_path.unlink()

    timed_out = False
    timeout_msg = None
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        timeout_msg = (
            f"timed out after {timeout_s:.1f}s"
            if timeout_s is not None else "timed out"
        )
        stdout = exc.stdout or ""
        if exc.stderr:
            stdout += str(exc.stderr)
        proc = SimpleNamespace(returncode=124, stdout=stdout)
    wall_s = time.time() - t0

    log_text = []
    log_text.append(f"$ {' '.join(shlex.quote(tok) for tok in cmd)}\n")
    if timeout_msg:
        log_text.append(f"[matrix] {timeout_msg}\n")
    log_text.append(proc.stdout or "")
    log_path.write_text("".join(log_text))

    summary = None
    summary_error = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError as exc:
            summary_error = f"invalid summary.json: {exc}"
    else:
        summary_error = "summary.json not found"

    return {
        **entry,
        "returncode": int(proc.returncode),
        "wall_time_s": float(wall_s),
        "log_path": str(log_path),
        "summary_path": str(summary_path),
        "summary": summary,
        "summary_error": summary_error,
        "timed_out": bool(timed_out),
        "timeout_error": timeout_msg,
    }


def _write_report(results: dict, report_path: Path) -> None:
    def _fmt_num(value) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.2e}"
        except (TypeError, ValueError):
            return "n/a"

    lines = []
    lines.append("# Ocean Regression Matrix")
    lines.append("")
    lines.append(f"- Generated: {results['generated_utc']}")
    lines.append(f"- Overall pass: `{results['overall_pass']}`")
    lines.append(f"- Overall warnings: `{results.get('overall_has_warnings', False)}`")
    th = results["thresholds"]
    lines.append(
        "- Thresholds: "
        f"|heat drift| <= {th['heat_drift_rel']:.2e}, "
        f"|salt drift| <= {th['salt_drift_rel']:.2e}, "
        f"|mean eta drift| <= {th['eta_mean_drift_m']:.2e} m, "
        f"speed <= {th['speed_max_mps']:.2e} m/s, "
        f"|SSH| <= {th['ssh_abs_max_m']:.2e} m",
    )
    lines.append(f"- Case profiles enabled: `{results.get('case_profiles_enabled', True)}`")
    lines.append("")
    lines.append("## Run Status")
    lines.append("")
    lines.append("| Run | Suite | Status | Return | Wall (s) |")
    lines.append("|---|---|---|---:|---:|")
    for run in results["runs"]:
        status = run["status"]
        lines.append(
            f"| {run['id']} | {run['suite']} | {status} | "
            f"{run['returncode']} | {run['wall_time_s']:.1f} |",
        )

    lines.append("")
    lines.append("## Case Checks")
    lines.append("")
    lines.append("| Run | Case | Pass | Warn | Warnings | Error | finite | land | speed_ok | ssh_ok | heat | salt | eta | speed | speed_lim | ssh_abs | ssh_lim |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for run in results["runs"]:
        if not run.get("evaluation"):
            continue
        for case_name, case in run["evaluation"]["cases"].items():
            m = case["metrics"]
            d = case.get("derived", {})
            lim = case.get("limits", {})
            warns = ",".join(case.get("warnings", [])) or "-"
            err = str(case.get("error", "-")).replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {run['id']} | {case_name} | {case['pass']} | "
                f"{case.get('has_warnings', False)} | {warns} | {err} | "
                f"{case['checks'].get('all_finite', '-')} | {case['checks'].get('land_zero', '-')} | "
                f"{case['checks'].get('speed_ok', True)} | "
                f"{case['checks'].get('ssh_abs_ok', True)} | "
                f"{_fmt_num(m.get('heat_drift_rel', None))} | "
                f"{_fmt_num(m.get('salt_drift_rel', None))} | "
                f"{_fmt_num(m.get('volume_mean_eta_drift', None))} | "
                f"{_fmt_num(d.get('speed_max_mps', None))} | "
                f"{_fmt_num(lim.get('speed_max_mps', None))} | "
                f"{_fmt_num(d.get('ssh_abs_max_m', None))} | "
                f"{_fmt_num(lim.get('ssh_abs_max_m', None))} |",
            )

    lines.append("")
    lines.append("## Logs")
    lines.append("")
    for run in results["runs"]:
        lines.append(f"- `{run['id']}`: `{run['log_path']}`")

    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ocean regression matrix")
    parser.add_argument(
        "--output",
        type=str,
        default="results/ocean_regression",
        help="Output directory for matrix results",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run the test scripts",
    )
    parser.add_argument(
        "--runtime-checks",
        action="store_true",
        help="Enable host-side runtime checks in child runs",
    )
    parser.add_argument(
        "--x64",
        action="store_true",
        help="Pass --x64 to child scripts",
    )
    parser.add_argument(
        "--include-realistic",
        action="store_true",
        help="Also run realistic ocean suite (longer wall-clock)",
    )
    parser.add_argument("--std-resolutions", type=str, default="8,16")
    parser.add_argument("--std-levels", type=str, default="10,20")
    parser.add_argument("--std-dt", type=float, default=3600.0)
    parser.add_argument("--std-days", type=float, default=5.0)
    parser.add_argument("--realistic-resolutions", type=str, default="8")
    parser.add_argument("--realistic-levels", type=str, default="10")
    parser.add_argument("--realistic-dt", type=float, default=1800.0)
    parser.add_argument("--realistic-days", type=float, default=10.0)
    parser.add_argument("--heat-drift-max", type=float, default=1.0e-3)
    parser.add_argument("--salt-drift-max", type=float, default=1.0e-3)
    parser.add_argument("--eta-drift-max", type=float, default=1.0e-3)
    parser.add_argument(
        "--speed-max",
        type=float,
        default=2.0e2,
        help="Maximum allowed case speed [m/s]. <=0 disables this check.",
    )
    parser.add_argument(
        "--ssh-max",
        type=float,
        default=5.0e1,
        help="Maximum allowed absolute SSH [m]. <=0 disables this check.",
    )
    parser.add_argument(
        "--no-case-profiles",
        action="store_true",
        help="Disable built-in case-specific speed/SSH limits",
    )
    parser.add_argument(
        "--warn-fraction",
        type=float,
        default=0.8,
        help=(
            "Warning threshold as fraction of fail limit "
            "(e.g. 0.8 warns at 80%% of limit). <=0 disables warnings."
        ),
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=0.0,
        help="Per-run command timeout in seconds. <=0 disables timeout.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop matrix on first failed run",
    )
    args = parser.parse_args()
    if args.std_dt <= 0.0:
        raise ValueError(f"--std-dt must be > 0, got {args.std_dt!r}")
    if args.std_days <= 0.0:
        raise ValueError(f"--std-days must be > 0, got {args.std_days!r}")
    if args.include_realistic and args.realistic_dt <= 0.0:
        raise ValueError(f"--realistic-dt must be > 0, got {args.realistic_dt!r}")
    if args.include_realistic and args.realistic_days <= 0.0:
        raise ValueError(f"--realistic-days must be > 0, got {args.realistic_days!r}")
    if args.heat_drift_max < 0.0:
        raise ValueError(f"--heat-drift-max must be >= 0, got {args.heat_drift_max!r}")
    if args.salt_drift_max < 0.0:
        raise ValueError(f"--salt-drift-max must be >= 0, got {args.salt_drift_max!r}")
    if args.eta_drift_max < 0.0:
        raise ValueError(f"--eta-drift-max must be >= 0, got {args.eta_drift_max!r}")
    if args.warn_fraction > 1.0:
        raise ValueError(
            f"--warn-fraction must be <= 1.0 (or <=0 to disable), got {args.warn_fraction!r}",
        )

    cwd = Path.cwd()
    output_root = Path(args.output)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    matrix = _build_matrix(args, output_root)
    if not matrix:
        print("No runs requested.")
        return 1

    thresholds = {
        "heat_drift_rel": float(args.heat_drift_max),
        "salt_drift_rel": float(args.salt_drift_max),
        "eta_mean_drift_m": float(args.eta_drift_max),
        "speed_max_mps": float(args.speed_max),
        "ssh_abs_max_m": float(args.ssh_max),
        "use_case_profiles": not bool(args.no_case_profiles),
        "warn_fraction": float(args.warn_fraction),
    }

    run_results = []
    overall_pass = True

    print("=" * 72)
    print("legoESM Ocean Regression Matrix")
    print("=" * 72)
    print(f"Runs: {len(matrix)}")
    print(f"Output: {output_root}")
    print(
        "Thresholds: "
        f"|heat|<={thresholds['heat_drift_rel']:.2e}, "
        f"|salt|<={thresholds['salt_drift_rel']:.2e}, "
        f"|eta|<={thresholds['eta_mean_drift_m']:.2e} m, "
        f"speed<={thresholds['speed_max_mps']:.2e} m/s, "
        f"|SSH|<={thresholds['ssh_abs_max_m']:.2e} m",
    )
    print(f"Case profiles enabled: {thresholds['use_case_profiles']}")
    print(f"Warn fraction: {thresholds['warn_fraction']}")
    timeout_s = float(args.timeout_sec) if float(args.timeout_sec) > 0.0 else None
    print(f"Per-run timeout: {timeout_s if timeout_s is not None else 'disabled'}")

    for idx, entry in enumerate(matrix, start=1):
        print(f"\n[{idx}/{len(matrix)}] {entry['id']}")
        run = _run_one(entry, cwd=cwd, logs_dir=logs_dir, timeout_s=timeout_s)
        evaluation = None
        if run["summary"] is not None:
            evaluation = _evaluate_summary(run["summary"], thresholds, run["suite"])
        run["evaluation"] = evaluation

        if run.get("timed_out", False):
            run["status"] = "command_timeout"
            overall_pass = False
            print(f"  FAIL: {run.get('timeout_error') or 'command timeout'}")
            if run["summary_error"] and run["summary"] is None:
                print(f"  Summary error: {run['summary_error']}")
        elif run["returncode"] != 0:
            run["status"] = "command_failed"
            overall_pass = False
            print(f"  FAIL: command exit code {run['returncode']}")
            if run["summary_error"] and run["summary"] is None:
                print(f"  Summary error: {run['summary_error']}")
            if evaluation is not None and not evaluation["pass"]:
                failed = []
                for case_name, case in evaluation["cases"].items():
                    if case.get("error"):
                        failed.append(f"{case_name}(error)")
                        continue
                    bad_checks = [k for k, ok in case["checks"].items() if not ok]
                    if bad_checks:
                        failed.append(f"{case_name}({','.join(bad_checks)})")
                if failed:
                    print(f"  Failed checks: {', '.join(failed)}")
        elif run["summary"] is None:
            run["status"] = "summary_failed"
            overall_pass = False
            print(f"  FAIL: {run['summary_error']}")
        else:
            if evaluation["pass"]:
                if evaluation.get("has_warnings", False):
                    run["status"] = "pass_with_warnings"
                    print("  PASS (with warnings)")
                    warned = []
                    for case_name, case in evaluation["cases"].items():
                        if case.get("warnings"):
                            warned.append(f"{case_name}({','.join(case['warnings'])})")
                    if warned:
                        print(f"  Warning checks: {', '.join(warned)}")
                else:
                    run["status"] = "pass"
                    print("  PASS")
            else:
                run["status"] = "metric_failed"
                overall_pass = False
                print("  FAIL: metric thresholds not met")
                failed = []
                for case_name, case in evaluation["cases"].items():
                    if case.get("error"):
                        failed.append(f"{case_name}(error)")
                        continue
                    bad_checks = [k for k, ok in case["checks"].items() if not ok]
                    if bad_checks:
                        failed.append(f"{case_name}({','.join(bad_checks)})")
                if failed:
                    print(f"  Failed checks: {', '.join(failed)}")

        run_results.append(run)
        if args.fail_fast and run["status"] in {
            "command_timeout", "command_failed", "summary_failed", "metric_failed",
        }:
            print("  Stopping early due to --fail-fast")
            break

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "generated_utc": timestamp,
        "overall_pass": overall_pass,
        "overall_has_warnings": any(
            bool((run.get("evaluation") or {}).get("has_warnings", False))
            for run in run_results
        ),
        "thresholds": thresholds,
        "case_profiles_enabled": thresholds["use_case_profiles"],
        "runs": run_results,
    }

    results_path = output_root / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    report_path = output_root / "report.md"
    _write_report(payload, report_path)

    print("\n" + "=" * 72)
    print(f"Overall pass: {overall_pass}")
    print(f"Overall warnings: {payload['overall_has_warnings']}")
    print(f"Results JSON: {results_path}")
    print(f"Report:       {report_path}")
    print("=" * 72)

    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
