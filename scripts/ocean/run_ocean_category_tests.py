#!/usr/bin/env python
"""Category runner: ocean tests (finite-volume and spectral)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ocean category tests in finite-volume and/or spectral mode.")
    p.add_argument("--mode", choices=["finite_volume", "spectral", "both"],
                   default="both")
    p.add_argument("--x64", action="store_true")
    p.add_argument("--resolution", type=int, default=24,
                   help="FV cubed-sphere resolution.")
    p.add_argument("--truncation", type=int, default=43,
                   help="Spectral truncation Tn.")
    p.add_argument("--levels", "-l", type=int, default=10)
    p.add_argument("--dt", type=float, default=900.0)
    p.add_argument("--days", "-d", type=float, default=0.2)
    p.add_argument("--fv-test", type=str, default="all")
    p.add_argument("--spectral-test", type=str, default="all")
    p.add_argument("--output-root", "-o", type=str, default="results/ocean")
    return p


def _run(cmd: list[str], cwd: Path) -> int:
    print("Running:", " ".join(cmd))
    return int(subprocess.run(cmd, cwd=cwd, check=False).returncode)


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    output_root = Path(args.output_root)
    statuses: list[int] = []

    if args.mode in ("finite_volume", "both"):
        cmd = [
            sys.executable,
            str(scripts_dir / "run_ocean_tests.py"),
            "--resolution",
            str(args.resolution),
            "--levels",
            str(args.levels),
            "--dt",
            str(args.dt),
            "--days",
            str(args.days),
            "--test",
            args.fv_test,
            "--output",
            str(output_root / "finite_volume"),
        ]
        if args.x64:
            cmd.append("--x64")
        statuses.append(_run(cmd, repo_root))

    if args.mode in ("spectral", "both"):
        cmd = [
            sys.executable,
            str(scripts_dir / "run_ocean_spectral_tests.py"),
            "--truncation",
            str(args.truncation),
            "--levels",
            str(args.levels),
            "--dt",
            str(args.dt),
            "--days",
            str(args.days),
            "--test",
            args.spectral_test,
            "--output",
            str(output_root / "spectral"),
        ]
        if args.x64:
            cmd.append("--x64")
        statuses.append(_run(cmd, repo_root))

    return 0 if all(s == 0 for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())

