#!/usr/bin/env python3
"""Run native lat-lon finite-volume ocean tests and emit a summary.json."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _extract_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    if match is None:
        return 0
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run native lat-lon FV ocean pytest suite.",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ocean/lat_lon/native_finite_volume/pytest"),
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="tests/ocean/test_latlon_ocean.py",
    )
    parser.add_argument(
        "--pytest-args",
        type=str,
        default="",
        help="Additional raw args passed to pytest.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [args.python, "-m", "pytest", "-q", args.test_file]
    if args.pytest_args.strip():
        cmd.extend(shlex.split(args.pytest_args))

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    wall_time_s = float(time.time() - t0)
    output_text = proc.stdout or ""

    log_path = out_dir / "pytest_output.log"
    log_path.write_text(
        f"$ {' '.join(shlex.quote(tok) for tok in cmd)}\n\n{output_text}",
    )

    counts = {
        "collected": _extract_int(r"collected\s+(\d+)\s+items", output_text),
        "passed": _extract_int(r"(\d+)\s+passed", output_text),
        "failed": _extract_int(r"(\d+)\s+failed", output_text),
        "errors": _extract_int(r"(\d+)\s+errors?", output_text),
        "skipped": _extract_int(r"(\d+)\s+skipped", output_text),
        "xfailed": _extract_int(r"(\d+)\s+xfailed", output_text),
        "xpassed": _extract_int(r"(\d+)\s+xpassed", output_text),
    }
    ok = (proc.returncode == 0) and (counts["passed"] > 0)

    summary = {
        "suite": "latlon_native_finite_volume_pytest",
        "ok": bool(ok),
        "returncode": int(proc.returncode),
        "test_file": str(args.test_file),
        "counts": counts,
        "wall_time_s": wall_time_s,
        "cmd": cmd,
        "log_path": str(log_path),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    txt_path = out_dir / "summary.txt"
    txt_path.write_text(
        "\n".join(
            [
                "Native Lat-Lon FV Ocean Pytest Summary",
                f"ok={summary['ok']}",
                f"returncode={summary['returncode']}",
                f"collected={counts['collected']}",
                f"passed={counts['passed']}",
                f"failed={counts['failed']}",
                f"errors={counts['errors']}",
                f"skipped={counts['skipped']}",
                f"wall_time_s={wall_time_s:.3f}",
                f"log={log_path}",
            ],
        )
        + "\n",
    )

    print(f"ok={summary['ok']}")
    print(f"summary={summary_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
