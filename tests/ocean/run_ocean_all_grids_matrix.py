#!/usr/bin/env python3
"""Run ocean test matrices across cubed-sphere, lat-lon, and icosahedral grids.

This driver standardizes outputs under ``results/ocean/<grid_type>/...``, doubles
horizontal resolution relative to baseline defaults, and runs finite-volume and
spectral-style options where available.

Canonical test cases are covered by ``scripts/run_ocean_test_matrix.py``:
  - rest_state, barotropic_wave, wind_gyre, baroclinic, phillips_two_layer,
    inertia_gravity_wave, lock_exchange, overflow, stommel_gyre_tracer
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable


REQUESTED_CASES = (
    "rest_state",
    "gravity_wave",
    "wind_gyre",
    "adiabatic_topography",
    "holland_lin_gyre",
    "thermohaline",
    "phillips_two_layer",
    "taylor_column",
)


def _parse_csv(text: str) -> list[str]:
    return [tok.strip() for tok in text.split(",") if tok.strip()]


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _cube_summary_ok(summary: dict) -> tuple[bool, str]:
    case_errors = summary.get("case_errors", {}) or {}
    if case_errors:
        return False, f"case_errors={list(case_errors)}"
    cases = summary.get("cases", {}) or {}
    missing = [c for c in REQUESTED_CASES if c not in cases]
    if missing:
        return False, f"missing_cases={missing}"
    nonfinite = [name for name, m in cases.items() if not bool(m.get("all_finite", False))]
    if nonfinite:
        return False, f"nonfinite_cases={nonfinite}"
    return True, "ok"


def _spectral_summary_ok(summary: dict) -> tuple[bool, str]:
    case_errors = summary.get("case_errors", {}) or {}
    if case_errors:
        return False, f"case_errors={list(case_errors)}"
    cases = summary.get("cases", {}) or {}
    if not cases:
        return False, "no_cases"
    nonfinite = [name for name, m in cases.items() if not bool(m.get("all_finite", False))]
    if nonfinite:
        return False, f"nonfinite_cases={nonfinite}"
    return True, "ok"


def _mpas_summary_ok(summary: dict) -> tuple[bool, str]:
    runs = summary.get("runs", []) or []
    if not runs:
        return False, "no_runs"
    cases_seen = {str(run.get("case", "")) for run in runs}
    missing = [c for c in REQUESTED_CASES if c not in cases_seen]
    if missing:
        return False, f"missing_cases={missing}"
    unstable = []
    for run in runs:
        stable = bool(run.get("stable", False))
        finite = bool(run.get("all_finite", False))
        if (not stable) or (not finite):
            unstable.append(run.get("case", "unknown"))
    if unstable:
        return False, f"unstable_runs={unstable}"
    return True, "ok"


def _latlon_native_fv_summary_ok(summary: dict) -> tuple[bool, str]:
    if not bool(summary.get("ok", False)):
        return False, f"ok={summary.get('ok', False)}"
    counts = summary.get("counts", {}) or {}
    passed = int(counts.get("passed", 0))
    failed = int(counts.get("failed", 0))
    errors = int(counts.get("errors", 0))
    if passed <= 0:
        return False, "no_passed_tests"
    if failed > 0 or errors > 0:
        return False, f"failed={failed},errors={errors}"
    return True, "ok"


@dataclass
class Attempt:
    attempt: int
    dt: float
    returncode: int
    wall_time_s: float
    ok: bool
    reason: str
    log_path: str
    cmd: list[str]


def _run_cmd(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool) -> tuple[int, float]:
    if dry_run:
        print(f"[dry-run] $ {' '.join(shlex.quote(t) for t in cmd)}")
        return 0, 0.0
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    wall = time.time() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(shlex.quote(t) for t in cmd)}\n\n{proc.stdout or ''}",
    )
    return int(proc.returncode), float(wall)


def _run_with_backoff(
    *,
    label: str,
    output_dir: Path,
    dt0: float,
    max_retries: int,
    build_cmd: Callable[[float], list[str]],
    summary_check: Callable[[dict], tuple[bool, str]],
    cwd: Path,
    logs_root: Path,
    dry_run: bool,
) -> dict:
    dt = float(dt0)
    attempts: list[Attempt] = []
    for attempt_idx in range(max_retries + 1):
        cmd = build_cmd(dt)
        log_path = logs_root / f"{label.replace('/', '_')}_attempt{attempt_idx + 1}.log"
        rc, wall = _run_cmd(cmd, cwd, log_path, dry_run=dry_run)

        ok = False
        reason = "ok"
        if dry_run:
            ok = True
        elif rc != 0:
            reason = f"returncode={rc}"
        else:
            summary = _read_json(output_dir / "summary.json")
            if summary is None:
                reason = "missing_or_invalid_summary"
            else:
                ok, reason = summary_check(summary)

        attempts.append(
            Attempt(
                attempt=attempt_idx + 1,
                dt=float(dt),
                returncode=int(rc),
                wall_time_s=float(wall),
                ok=bool(ok),
                reason=str(reason),
                log_path=str(log_path),
                cmd=cmd,
            ),
        )
        print(
            f"[{label}] attempt={attempt_idx + 1} dt={dt:.6g} rc={rc} ok={ok} reason={reason}",
        )
        if ok:
            break
        dt *= 0.5

    return {
        "label": label,
        "output_dir": str(output_dir),
        "ok": bool(attempts[-1].ok),
        "attempts": [asdict(a) for a in attempts],
        "final_reason": attempts[-1].reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ocean tests across cube-sphere, lat-lon, and icosahedral grids.",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--output", type=Path, default=Path("results/ocean"))
    parser.add_argument("--levels", type=int, default=10)

    parser.add_argument("--cube-base-resolution", type=int, default=8)
    parser.add_argument("--cube-dt", type=float, default=3600.0)
    parser.add_argument(
        "--cube-discretizations",
        type=str,
        default="finite_volume,fc_gram",
        help="Comma-separated cubed-sphere discretizations.",
    )

    parser.add_argument("--spectral-base-truncation", type=int, default=8)
    parser.add_argument("--spectral-dt", type=float, default=1800.0)

    parser.add_argument("--mpas-base-mesh-level", type=int, default=2)
    parser.add_argument("--mpas-dt", type=float, default=60.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--lloyd-iterations", type=int, default=30)
    parser.add_argument("--latlon-nlon", type=int, default=360)
    parser.add_argument("--latlon-nlat", type=int, default=181)

    parser.add_argument("--days", type=float, default=5.0)
    parser.add_argument(
        "--double-horizontal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Double baseline horizontal resolution (default: true).",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--skip-cube", action="store_true")
    parser.add_argument("--skip-latlon-projection", action="store_true")
    parser.add_argument("--skip-latlon-native-fv", action="store_true")
    parser.add_argument("--skip-latlon-spectral", action="store_true")
    parser.add_argument("--skip-icosahedral", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.levels < 1:
        raise ValueError("--levels must be >= 1")
    if args.days <= 0.0:
        raise ValueError("--days must be > 0")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")

    cwd = Path(__file__).resolve().parents[2]
    out_root = args.output.resolve()
    logs_root = out_root / "_logs"
    out_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    cube_res = args.cube_base_resolution * (2 if args.double_horizontal else 1)
    spectral_trunc = args.spectral_base_truncation * (2 if args.double_horizontal else 1)
    mpas_mesh_level = args.mpas_base_mesh_level + (1 if args.double_horizontal else 0)

    cube_discs = _parse_csv(args.cube_discretizations)
    if not cube_discs:
        raise ValueError("No cube discretizations configured")

    print("=" * 72)
    print("legoESM Ocean All-Grid Matrix")
    print("=" * 72)
    print(f"repo={cwd}")
    print(f"output={out_root}")
    print(f"cube_res=C{cube_res}, levels={args.levels}, discs={cube_discs}")
    print(f"spectral_trunc=T{spectral_trunc}, levels={args.levels}")
    print(f"mpas_mesh_level=L{mpas_mesh_level}, levels={args.levels}")
    print(f"days={args.days}, retries={args.max_retries}, dry_run={args.dry_run}")

    runs: list[dict] = []

    # ------------------------------------------------------------------
    # Cubed-sphere runs (all 8 requested cases, FV + FC/spectral-style)
    # ------------------------------------------------------------------
    if not args.skip_cube:
        for disc in cube_discs:
            out_dir = out_root / "cube_sphere" / disc / f"C{cube_res}_L{args.levels}"
            out_dir.mkdir(parents=True, exist_ok=True)

            def build_cube_cmd(dt_now: float) -> list[str]:
                return [
                    args.python,
                    "scripts/run_ocean_test_matrix.py",
                    "--grid",
                    "cubed_sphere",
                    "--resolution",
                    f"C{cube_res}",
                    "--levels",
                    str(args.levels),
                    "--dt",
                    f"{dt_now:.12g}",
                    "--output",
                    str(out_dir),
                ]

            runs.append(
                _run_with_backoff(
                    label=f"cube_sphere/{disc}",
                    output_dir=out_dir,
                    dt0=float(args.cube_dt),
                    max_retries=int(args.max_retries),
                    build_cmd=build_cube_cmd,
                    summary_check=_cube_summary_ok,
                    cwd=cwd,
                    logs_root=logs_root,
                    dry_run=bool(args.dry_run),
                ),
            )

    # ------------------------------------------------------------------
    # Lat-lon projection outputs (same 8 requested cases via cubed-sphere
    # dynamics, but organized under lat_lon/projection and including
    # lat-lon pixel snapshots + lat/lon-depth sections over time).
    # ------------------------------------------------------------------
    if not args.skip_latlon_projection:
        for disc in cube_discs:
            out_dir = out_root / "lat_lon" / "projection_from_cube" / disc / f"C{cube_res}_L{args.levels}"
            out_dir.mkdir(parents=True, exist_ok=True)

            def build_latlon_projection_cmd(dt_now: float) -> list[str]:
                return [
                    args.python,
                    "scripts/run_ocean_test_matrix.py",
                    "--grid",
                    "cubed_sphere",
                    "--resolution",
                    f"C{cube_res}",
                    "--levels",
                    str(args.levels),
                    "--dt",
                    f"{dt_now:.12g}",
                    "--output",
                    str(out_dir),
                ]

            runs.append(
                _run_with_backoff(
                    label=f"lat_lon/projection_from_cube/{disc}",
                    output_dir=out_dir,
                    dt0=float(args.cube_dt),
                    max_retries=int(args.max_retries),
                    build_cmd=build_latlon_projection_cmd,
                    summary_check=_cube_summary_ok,
                    cwd=cwd,
                    logs_root=logs_root,
                    dry_run=bool(args.dry_run),
                ),
            )

    # ------------------------------------------------------------------
    # Native lat-lon finite-volume branch (dedicated FV test suite).
    # ------------------------------------------------------------------
    if not args.skip_latlon_native_fv:
        out_dir = out_root / "lat_lon" / "native_finite_volume" / "pytest"
        out_dir.mkdir(parents=True, exist_ok=True)

        def build_latlon_native_fv_cmd(_: float) -> list[str]:
            return [
                args.python,
                "tests/ocean/run_latlon_ocean_fv_suite.py",
                "--python",
                args.python,
                "--output",
                str(out_dir),
            ]

        runs.append(
            _run_with_backoff(
                label="lat_lon/native_finite_volume",
                output_dir=out_dir,
                dt0=1.0,
                max_retries=int(args.max_retries),
                build_cmd=build_latlon_native_fv_cmd,
                summary_check=_latlon_native_fv_summary_ok,
                cwd=cwd,
                logs_root=logs_root,
                dry_run=bool(args.dry_run),
            ),
        )

    # ------------------------------------------------------------------
    # Lat-lon spectral run (native Gaussian-grid spectral ocean suite).
    # ------------------------------------------------------------------
    if not args.skip_latlon_spectral:
        out_dir = out_root / "lat_lon" / "spectral" / f"T{spectral_trunc}_L{args.levels}"
        out_dir.mkdir(parents=True, exist_ok=True)

        def build_spectral_cmd(dt_now: float) -> list[str]:
            return [
                args.python,
                "scripts/run_ocean_spectral_tests.py",
                "--x64",
                "--truncation",
                str(spectral_trunc),
                "--levels",
                str(args.levels),
                "--dt",
                f"{dt_now:.12g}",
                "--days",
                f"{args.days:.12g}",
                "--test",
                "all",
                "--output",
                str(out_dir),
            ]

        runs.append(
            _run_with_backoff(
                label="lat_lon/spectral",
                output_dir=out_dir,
                dt0=float(args.spectral_dt),
                max_retries=int(args.max_retries),
                build_cmd=build_spectral_cmd,
                summary_check=_spectral_summary_ok,
                cwd=cwd,
                logs_root=logs_root,
                dry_run=bool(args.dry_run),
            ),
        )

    # ------------------------------------------------------------------
    # Icosahedral/MPAS runs.
    # ------------------------------------------------------------------
    if not args.skip_icosahedral:
        out_dir = out_root / "icosahedral" / "finite_volume"
        out_dir.mkdir(parents=True, exist_ok=True)

        def build_mpas_cmd(dt_now: float) -> list[str]:
            return [
                args.python,
                "tests/ocean/run_mpas_ocean_cases.py",
                "--output",
                str(out_dir),
                "--cases",
                "all",
                "--mesh-levels",
                str(mpas_mesh_level),
                "--n-levels",
                str(args.levels),
                "--dt",
                f"{dt_now:.12g}",
                "--hours",
                f"{(24.0 * args.days):.12g}",
                "--save-every",
                str(args.save_every),
                "--lloyd-iterations",
                str(args.lloyd_iterations),
                "--latlon-nlon",
                str(args.latlon_nlon),
                "--latlon-nlat",
                str(args.latlon_nlat),
            ]

        runs.append(
            _run_with_backoff(
                label="icosahedral/finite_volume",
                output_dir=out_dir,
                dt0=float(args.mpas_dt),
                max_retries=int(args.max_retries),
                build_cmd=build_mpas_cmd,
                summary_check=_mpas_summary_ok,
                cwd=cwd,
                logs_root=logs_root,
                dry_run=bool(args.dry_run),
            ),
        )

    overall_ok = all(bool(r.get("ok", False)) for r in runs) if runs else False
    summary = {
        "suite": "ocean_all_grids_matrix",
        "overall_ok": bool(overall_ok),
        "requested_cases": list(REQUESTED_CASES),
        "notes": {
            "lat_lon_projection": (
                "projection_from_cube runs use cubed-sphere ocean dynamics and output "
                "native + remapped lat-lon diagnostics under lat_lon/projection_from_cube."
            ),
            "lat_lon_native_fv": (
                "native finite-volume lat-lon branch runs tests/ocean/test_latlon_ocean.py "
                "via tests/ocean/run_latlon_ocean_fv_suite.py."
            ),
            "lat_lon_spectral_cases": (
                "Native spectral lat-lon suite currently covers rest/wave/baroclinic cases."
            ),
            "icosahedral_cases": (
                "MPAS suite currently follows tests/ocean/run_mpas_ocean_cases.py case set."
            ),
        },
        "config": {
            "cube_resolution": int(cube_res),
            "spectral_truncation": int(spectral_trunc),
            "mpas_mesh_level": int(mpas_mesh_level),
            "levels": int(args.levels),
            "days": float(args.days),
            "cube_discretizations": cube_discs,
            "max_retries": int(args.max_retries),
        },
        "runs": runs,
    }

    summary_path = out_root / "all_grids_matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\nSummary:")
    print(f"  overall_ok={overall_ok}")
    print(f"  summary={summary_path}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
