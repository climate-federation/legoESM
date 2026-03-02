#!/usr/bin/env python
"""Run an ocean regression matrix and gate stability/conservation metrics.

This script orchestrates `run_ocean_tests.py` and optionally
`run_ocean_realistic.py`, then consumes their machine-readable
`summary.json` outputs and applies simple pass/fail thresholds.

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


def _parse_int_list(csv_text: str) -> list[int]:
    values = []
    for token in csv_text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError(f"Expected at least one integer, got {csv_text!r}")
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

    std_res = _parse_int_list(args.std_resolutions)
    std_lev = _pair_levels(std_res, _parse_int_list(args.std_levels))
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
        real_res = _parse_int_list(args.realistic_resolutions)
        real_lev = _pair_levels(real_res, _parse_int_list(args.realistic_levels))
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


def _evaluate_summary(summary: dict, thresholds: dict) -> dict:
    case_results = {}
    overall_pass = True
    for case_name, metrics in summary.get("cases", {}).items():
        heat = abs(float(metrics.get("heat_drift_rel", 0.0)))
        salt = abs(float(metrics.get("salt_drift_rel", 0.0)))
        eta = abs(float(metrics.get("volume_mean_eta_drift", 0.0)))
        finite_ok = bool(metrics.get("all_finite", False))
        land_ok = bool(metrics.get("land_zero", False))

        checks = {
            "all_finite": finite_ok,
            "land_zero": land_ok,
            "heat_drift_ok": heat <= thresholds["heat_drift_rel"],
            "salt_drift_ok": salt <= thresholds["salt_drift_rel"],
            "eta_drift_ok": eta <= thresholds["eta_mean_drift_m"],
        }
        case_pass = all(checks.values())
        overall_pass = overall_pass and case_pass
        case_results[case_name] = {
            "pass": case_pass,
            "checks": checks,
            "metrics": metrics,
        }

    if not case_results:
        overall_pass = False

    return {
        "pass": overall_pass,
        "cases": case_results,
    }


def _run_one(entry: dict, cwd: Path, logs_dir: Path) -> dict:
    run_id = entry["id"]
    cmd = entry["cmd"]
    log_path = logs_dir / f"{run_id}.log"
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    wall_s = time.time() - t0

    log_text = []
    log_text.append(f"$ {' '.join(shlex.quote(tok) for tok in cmd)}\n")
    log_text.append(proc.stdout or "")
    log_path.write_text("".join(log_text))

    summary_path = Path(entry["output_dir"]) / "summary.json"
    summary = None
    summary_error = None
    if proc.returncode == 0 and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError as exc:
            summary_error = f"invalid summary.json: {exc}"
    elif proc.returncode == 0:
        summary_error = "summary.json not found"

    return {
        **entry,
        "returncode": int(proc.returncode),
        "wall_time_s": float(wall_s),
        "log_path": str(log_path),
        "summary_path": str(summary_path),
        "summary": summary,
        "summary_error": summary_error,
    }


def _write_report(results: dict, report_path: Path) -> None:
    lines = []
    lines.append("# Ocean Regression Matrix")
    lines.append("")
    lines.append(f"- Generated: {results['generated_utc']}")
    lines.append(f"- Overall pass: `{results['overall_pass']}`")
    th = results["thresholds"]
    lines.append(
        "- Thresholds: "
        f"|heat drift| <= {th['heat_drift_rel']:.2e}, "
        f"|salt drift| <= {th['salt_drift_rel']:.2e}, "
        f"|mean eta drift| <= {th['eta_mean_drift_m']:.2e} m",
    )
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
    lines.append("| Run | Case | Pass | all_finite | land_zero | heat | salt | eta |")
    lines.append("|---|---|---|---|---|---:|---:|---:|")
    for run in results["runs"]:
        if not run.get("evaluation"):
            continue
        for case_name, case in run["evaluation"]["cases"].items():
            m = case["metrics"]
            lines.append(
                f"| {run['id']} | {case_name} | {case['pass']} | "
                f"{case['checks']['all_finite']} | {case['checks']['land_zero']} | "
                f"{m.get('heat_drift_rel', 0.0):.2e} | "
                f"{m.get('salt_drift_rel', 0.0):.2e} | "
                f"{m.get('volume_mean_eta_drift', 0.0):.2e} |",
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
        "--fail-fast",
        action="store_true",
        help="Stop matrix on first failed run",
    )
    args = parser.parse_args()

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
        f"|eta|<={thresholds['eta_mean_drift_m']:.2e} m",
    )

    for idx, entry in enumerate(matrix, start=1):
        print(f"\n[{idx}/{len(matrix)}] {entry['id']}")
        run = _run_one(entry, cwd=cwd, logs_dir=logs_dir)

        if run["returncode"] != 0:
            run["status"] = "command_failed"
            run["evaluation"] = None
            overall_pass = False
            print(f"  FAIL: command exit code {run['returncode']}")
        elif run["summary"] is None:
            run["status"] = "summary_failed"
            run["evaluation"] = None
            overall_pass = False
            print(f"  FAIL: {run['summary_error']}")
        else:
            evaluation = _evaluate_summary(run["summary"], thresholds)
            run["evaluation"] = evaluation
            if evaluation["pass"]:
                run["status"] = "pass"
                print("  PASS")
            else:
                run["status"] = "metric_failed"
                overall_pass = False
                print("  FAIL: metric thresholds not met")

        run_results.append(run)
        if args.fail_fast and run["status"] != "pass":
            print("  Stopping early due to --fail-fast")
            break

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "generated_utc": timestamp,
        "overall_pass": overall_pass,
        "thresholds": thresholds,
        "runs": run_results,
    }

    results_path = output_root / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    report_path = output_root / "report.md"
    _write_report(payload, report_path)

    print("\n" + "=" * 72)
    print(f"Overall pass: {overall_pass}")
    print(f"Results JSON: {results_path}")
    print(f"Report:       {report_path}")
    print("=" * 72)

    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
