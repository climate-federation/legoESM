#!/usr/bin/env python
"""Category runner: aquaplanet experiment suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run aquaplanet slab-ocean experiment (gray-radiation RCE).")
    p.add_argument("--days", type=int, default=200)
    p.add_argument("--resolution", type=int, default=16)
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--diag-days", type=int, default=5)
    p.add_argument(
        "--output", "-o", type=str,
        default="results/atmosphere/aquaplanet/slab_ocean")
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "scripts" / "run_rce_slab_ocean.py"

    cmd = [
        sys.executable,
        str(target),
        "--days",
        str(args.days),
        "--resolution",
        str(args.resolution),
        "--nlev",
        str(args.nlev),
        "--diag-days",
        str(args.diag_days),
        "--output",
        args.output,
    ]
    if args.dt is not None:
        cmd.extend(["--dt", str(args.dt)])

    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=repo_root, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

