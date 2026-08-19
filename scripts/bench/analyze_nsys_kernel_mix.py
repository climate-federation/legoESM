"""Kernel-mix attribution from an nsys sqlite export: NCCL vs compute,
counts, duration medians — the comm-bound / launch-bound / compute-bound
discriminator for a captured step loop.

DURATIONS ONLY. nsys kernel COUNTS on this stack are unreliable across
captures (campaign doc, 2026-08-08: two ranks recorded 0 collectives in
runs that demonstrably communicated) — within ONE capture the recorded
rows are usable for a mix breakdown, but never compare counts across
captures.

Usage:
    python scripts/bench/analyze_nsys_kernel_mix.py rank_0.sqlite \
        --tail-frac 0.5
`--tail-frac` keeps only the last fraction of the wall (drops
compile/warmup; the timed loop sits at the end of the bench).
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sqlite", help="nsys --export=sqlite output")
    p.add_argument("--tail-frac", type=float, default=0.5)
    p.add_argument("--tail-ms", type=float, default=None,
                   help="Absolute tail window in ms (overrides --tail-frac); "
                        "size it to steps x ms/step to exclude setup bursts "
                        "(replication all-gathers etc.).")
    args = p.parse_args()

    con = sqlite3.connect(args.sqlite)
    # shortName is the reliable name column on this nsys version
    # (demangledName was NULL for NCCL kernels in prior captures).
    rows = con.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id").fetchall()
    if not rows:
        raise SystemExit("no kernel rows — wrong table or empty capture")
    t0 = min(r[0] for r in rows)
    t1 = max(r[1] for r in rows)
    if args.tail_ms is not None:
        cut = t1 - args.tail_ms * 1e6
    else:
        cut = t1 - (t1 - t0) * args.tail_frac
    tail = [(s, e, n) for s, e, n in rows if s >= cut]
    window_ms = (t1 - cut) / 1e6

    def bucket(name: str) -> str:
        if name.startswith("ncclDevKernel"):
            return "nccl"
        return "compute"

    agg: dict = {}
    for s, e, n in tail:
        b = bucket(n)
        agg.setdefault(b, []).append((e - s) / 1e3)   # us
    print(f"window {window_ms:.1f} ms (tail {args.tail_frac} of wall), "
          f"{len(tail)} kernel rows")
    for b, durs in sorted(agg.items()):
        tot = sum(durs) / 1e3
        print(f"  {b:8s} n={len(durs):6d} total={tot:9.2f} ms "
              f"median={statistics.median(durs):8.2f} us "
              f"share_of_window={100 * tot / window_ms:5.1f}%")
    # top individual kernels by total time
    per_name: dict = {}
    for s, e, n in tail:
        per_name[n] = per_name.get(n, 0.0) + (e - s) / 1e6
    print("top kernels by total ms:")
    for n, tot in sorted(per_name.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {tot:9.2f} ms  {n[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
