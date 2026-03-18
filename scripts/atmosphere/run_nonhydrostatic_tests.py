#!/usr/bin/env python
"""Category runner: non-hydrostatic atmosphere test cases."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run non-hydrostatic cases via run_atmosphere_test_matrix.py")
    p.add_argument(
        "--grid", type=str, default="all",
        choices=["cubed_sphere", "latlon", "icosahedral", "all"])
    p.add_argument("--test", type=str, default=None)
    p.add_argument("--resolution", type=str, default=None)
    p.add_argument(
        "--radiation", type=str, default="gray", choices=["gray", "rrtmgp"])
    p.add_argument("--output", "-o", type=str, default="results/atmosphere")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--list-category-scripts", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    matrix = repo_root / "scripts" / "run_atmosphere_test_matrix.py"

    cmd = [
        sys.executable,
        str(matrix),
        "--only",
        "nh",
        "--grid",
        args.grid,
        "--radiation",
        args.radiation,
        "--output",
        args.output,
    ]
    if args.test:
        cmd.extend(["--test", args.test])
    if args.resolution:
        cmd.extend(["--resolution", args.resolution])
    if args.quick:
        cmd.append("--quick")
    if args.list:
        cmd.append("--list")
    if args.list_category_scripts:
        cmd.append("--list-category-scripts")

    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=repo_root, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

