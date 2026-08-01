#!/usr/bin/env python
"""Sweep the ocean test tree one file per process, under a bounded CPU set.

#1387: a plain ``pytest tests/ocean`` aborts at ~2% and dies under xdist. The
cause is NOT the ocean code — it is XLA sizing its Eigen/tsl thread pools from
the machine's VISIBLE core count while the user's ``ulimit -u`` bounds threads
across the whole node.  ``pthread_create`` then returns EAGAIN and XLA's CHECK
turns it into ``abort()``, which surfaces as ``Fatal Python error: Aborted``
mid-collection and reads like a test crash::

    F env.cc:93] Check failed: ret == 0 (11 vs. 0) Thread tf_foreach creation
    via pthread_create() failed.

Capping the affinity mask fixes it because ``tsl::port::MaxParallelism`` reads
the mask.  ``TF_NUM_*``/``OMP_NUM_THREADS``/``--xla_cpu_multi_thread_eigen``
do NOT — they never reach that pool.

Usage
-----
    python scripts/validate/sweep_ocean_tests.py                # whole tree
    python scripts/validate/sweep_ocean_tests.py --cores 4      # tighter cap
    python scripts/validate/sweep_ocean_tests.py -k barotropic  # subset
    python scripts/validate/sweep_ocean_tests.py --paths tests/ocean/unit

Writes a machine-readable summary so a sweep can be diffed against a later one
instead of re-read by eye.  Every file gets its own process, so one abort costs
one file rather than the run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = ("tests/ocean",)
# 8 cores was enough to clear every abort observed on a 256-core Levante login
# node; it is a cap on XLA's pool, not a statement about the test's own needs.
DEFAULT_CORES = 8


def thread_headroom() -> tuple[int, int]:
    """(soft RLIMIT_NPROC, visible cores) — the two numbers that collide."""
    import resource
    soft, _hard = resource.getrlimit(resource.RLIMIT_NPROC)
    return soft, os.cpu_count() or 1


def collect_files(paths, pattern: str | None) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        root = REPO / p
        if root.is_file():
            out.append(root)
            continue
        for f in sorted(root.rglob("test_*.py")):
            if pattern is None or pattern in f.name:
                out.append(f)
    return out


def run_one(f: Path, cores: int, k: str | None, timeout: int) -> dict:
    rel = f.relative_to(REPO)
    cmd: list[str] = []
    if cores > 0 and hasattr(os, "sched_setaffinity"):
        cmd += ["taskset", "-c", f"0-{cores - 1}"]
    cmd += [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            str(rel)]
    if k:
        cmd += ["-k", k]
    env = {**os.environ, "JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu"}
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=timeout)
        out, code = proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        out, code = "TIMEOUT", -9
    tail = [ln for ln in out.splitlines()
            if (" passed" in ln or " failed" in ln or " error" in ln
                or "Fatal Python error" in ln)]
    aborted = "Fatal Python error" in out
    return {
        "file": str(rel),
        "returncode": code,
        "aborted": aborted,
        # An exit code alone is not evidence: an abort or an OOM can present as
        # a non-obvious status, so keep pytest's OWN summary line.
        "summary": tail[-1].strip() if tail else "(no pytest summary line)",
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", nargs="*", default=list(DEFAULT_PATHS))
    ap.add_argument("--cores", type=int, default=DEFAULT_CORES,
                    help="CPU-affinity cap per test process (0 = no cap)")
    ap.add_argument("-k", dest="k", default=None, help="pytest -k expression")
    ap.add_argument("--pattern", default=None,
                    help="substring filter on the test FILE name")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="ocean_sweep.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N files (smoke run)")
    args = ap.parse_args()

    soft, cores_visible = thread_headroom()
    print(f"[sweep] RLIMIT_NPROC(soft)={soft}  visible cores={cores_visible}  "
          f"affinity cap={args.cores or 'none'}")
    if args.cores == 0 and soft < cores_visible * 8:
        print("[sweep] WARNING: no affinity cap and the thread budget is tight "
              "relative to the visible core count — this is the #1387 abort "
              "condition. Expect 'Fatal Python error: Aborted'.")

    files = collect_files(args.paths, args.pattern)
    if args.limit:
        files = files[:args.limit]
    print(f"[sweep] {len(files)} files")

    results = []
    for i, f in enumerate(files, 1):
        r = run_one(f, args.cores, args.k, args.timeout)
        results.append(r)
        flag = "ABORT" if r["aborted"] else ("ok" if r["returncode"] == 0
                                             else "FAIL")
        print(f"[{i}/{len(files)}] {flag:5s} {r['file']}  {r['summary']}",
              flush=True)

    aborted = [r for r in results if r["aborted"]]
    failed = [r for r in results if r["returncode"] != 0 and not r["aborted"]]
    Path(args.out).write_text(json.dumps(
        {"cap_cores": args.cores, "rlimit_nproc": soft,
         "visible_cores": cores_visible, "results": results}, indent=2))
    print(f"\n[sweep] {len(results)} files: "
          f"{len(results) - len(failed) - len(aborted)} clean, "
          f"{len(failed)} failed, {len(aborted)} aborted -> {args.out}")
    return 1 if (failed or aborted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
