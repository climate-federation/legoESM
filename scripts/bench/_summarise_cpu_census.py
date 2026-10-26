#!/usr/bin/env python
"""Table the executed-collective census of the CPU lanes.

Reads the per-arm JSON that analyze_jax_trace_gaps.py --time-by-family
writes and prints two tables: executions per step per rank, and occupied
milliseconds per step.  Counts are the quotable half.  Occupancy INCLUDES
time a collective spends waiting for late partners, so it overstates wire
cost and does not add across families; the isolated per-operation costs come
from the gloo latency ladder instead.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys


def load(out_dir: str) -> list[tuple[str, int, int, dict]]:
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, "census_*.json"))):
        m = re.search(r"census_(\w+)_nd(\d+)\.json$", f)
        if m is None:
            continue
        d = json.load(open(f))
        rows.append((m.group(1), int(m.group(2)), d.get("steps", 1),
                     d.get("families", {})))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def _table(rows, names, title, cell):
    print(f"\n{'lane':>5} {'ranks':>6}"
          + "".join(f"{n[:12]:>14}" for n in names) + f"    ({title})")
    for lane, nd, steps, fams in rows:
        print(f"{lane:>5} {nd:>6}"
              + "".join(cell(fams.get(n, {}), steps) for n in names))


def main() -> int:
    rows = load(sys.argv[1])
    if not rows:
        print("no census files found", file=sys.stderr)
        return 1
    names = sorted({k for *_, f in rows for k in f})
    _table(rows, names, "executions per step per rank",
           lambda d, s: f"{d.get('instructions', 0) / s:>14.1f}")
    _table(rows, names, "ms per step occupied",
           lambda d, s: f"{d.get('ms_per_step', 0.0):>14.2f}")
    print("\nOccupancy includes waiting for late partners, so it overstates "
          "wire cost and is not additive across families. Pair the counts "
          "with the isolated per-operation costs from the gloo ladder.")
    print("CPU_CENSUS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
