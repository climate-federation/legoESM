#!/usr/bin/env python
"""Category runner: radiative-convective equilibrium (RCE) experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run RCE slab-ocean and slab-land category tests.")
    p.add_argument("--mode", choices=["all", "ocean", "land"], default="all")
    p.add_argument("--days", type=int, default=100)
    p.add_argument("--resolution", type=int, default=16)
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--diag-days", type=int, default=5)
    p.add_argument("--output-root", type=str, default="results/atmosphere/rce")
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
    if args.mode in ("all", "ocean"):
        cmd = [
            sys.executable,
            str(scripts_dir / "run_rce_slab_ocean.py"),
            "--days",
            str(args.days),
            "--resolution",
            str(args.resolution),
            "--nlev",
            str(args.nlev),
            "--diag-days",
            str(args.diag_days),
            "--output",
            str(output_root / "slab_ocean"),
        ]
        if args.dt is not None:
            cmd.extend(["--dt", str(args.dt)])
        statuses.append(_run(cmd, repo_root))

    if args.mode in ("all", "land"):
        cmd = [
            sys.executable,
            str(scripts_dir / "run_rce_slab_land.py"),
            "--days",
            str(args.days),
            "--resolution",
            str(args.resolution),
            "--nlev",
            str(args.nlev),
            "--diag-days",
            str(args.diag_days),
            "--output",
            str(output_root / "slab_land"),
        ]
        if args.dt is not None:
            cmd.extend(["--dt", str(args.dt)])
        statuses.append(_run(cmd, repo_root))

    return 0 if all(s == 0 for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())

