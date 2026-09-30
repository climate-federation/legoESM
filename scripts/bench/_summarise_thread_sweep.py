#!/usr/bin/env python
"""Table the per-rank core-share sweep, and say what it does and does not show.

Reports each (ranks, cores-per-rank) cell as the best of its repeats plus
the spread between them, because a difference smaller than the spread is not
a difference.  It states the verdict against the pre-registered outcomes
rather than leaving the reader to eyeball it, and it refuses to call a
winner when the winner is inside the spread.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict


def load(out_dir: str) -> dict[tuple[int, int], list[float]]:
    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(out_dir, "nd*_cpt*_r*.jsonl"))):
        m = re.search(r"nd(\d+)_cpt(\d+)_r(\d+)\.jsonl$", f)
        if m is None:
            continue
        for line in open(f):
            r = json.loads(line)
            ms = r.get("steady_median_ms")
            if ms:
                cells[(int(m.group(1)), int(m.group(2)))].append(float(ms))
    return cells


def _best(cells, nd):
    at = {c: v for (n, c), v in cells.items() if n == nd}
    if not at:
        return None, None, None
    best_c = min(at, key=lambda c: min(at[c]))
    best = min(at[best_c])
    spread = max(at[best_c]) - min(at[best_c]) if len(at[best_c]) > 1 else 0.0
    return best_c, best, spread


def thread_census(out_dir: str) -> dict[int, int]:
    """Worker count actually observed at each core share.

    The pool is sized from the affinity mask on this stack, so the worker
    count should equal the share.  Where it does not, the sweep was varying
    pinning rather than threads at that share and its timing cannot be read
    as a thread result.
    """
    seen: dict[int, int] = {}
    for f in glob.glob(os.path.join(out_dir, "threads_cpt*.json*")):
        m = re.search(r"threads_cpt(\d+)\.json", f)
        if m is None:
            continue
        try:
            rec = json.load(open(f))
        except (OSError, ValueError):
            continue
        seen[int(m.group(1))] = int(rec.get("threads", {}).get("tf_XLAEigen", 0))
    return seen


def main() -> int:
    cells = load(sys.argv[1])
    if not cells:
        print("no arms found", file=sys.stderr)
        return 1
    workers = thread_census(sys.argv[1])
    dead = {c: n for c, n in workers.items() if n != c}
    if workers:
        print("core share -> XLA workers: "
              + ", ".join(f"{c}->{n}" for c, n in sorted(workers.items())))
    if dead:
        print(f"INVALID: the worker count did not follow the core share at "
              f"{sorted(dead)}. At those shares this sweep varied pinning, "
              f"not threads, so no thread conclusion can be drawn from it.")
        return 2
    if not workers:
        print("INVALID: no thread census was recorded, so it is unknown "
              "whether the core share reached the pool at all.")
        return 2
    shares = sorted({c for _, c in cells})
    ranks = sorted({n for n, _ in cells})
    print(f"\n{'ranks':>6}" + "".join(f"{c:>12}" for c in shares)
          + "     (cores/rank; best of repeats, ms)")
    for nd in ranks:
        row = ""
        for c in shares:
            v = cells.get((nd, c))
            row += f"{min(v):>12.1f}" if v else f"{'-':>12}"
        print(f"{nd:>6}{row}")
    print(f"\n{'ranks':>6}" + "".join(f"{c:>12}" for c in shares)
          + "     (repeat spread, ms)")
    for nd in ranks:
        row = ""
        for c in shares:
            v = cells.get((nd, c))
            row += (f"{max(v) - min(v):>12.1f}" if v and len(v) > 1
                    else f"{'-':>12}")
        print(f"{nd:>6}{row}")

    big = max(ranks)
    bc, bv, bs = _best(cells, big)
    cur = cells.get((big, 64))
    print()
    if cur and bc is not None:
        gain = min(cur) - bv
        if bc == 64 or gain <= max(bs, 1e-9):
            print(f"REFUTED at {big} ranks: the current 64 cores/rank is best "
                  f"or every share lands inside the repeat spread "
                  f"({bs:.1f} ms). The thread pool is not the cause; the "
                  f"partition and halo at {big} ranks are next.")
        else:
            print(f"At {big} ranks, {bc} cores/rank beats 64 by {gain:.1f} ms "
                  f"({100 * gain / min(cur):.1f}%), spread {bs:.1f} ms.")
            small = min(ranks)
            sc, _, _ = _best(cells, small)
            if sc is not None and small != big:
                if sc > bc:
                    print(f"CONFIRMED: {small} ranks prefer {sc} cores/rank "
                          f"while {big} prefer {bc} -- the optimum moves with "
                          f"the local problem size, as the mechanism says.")
                else:
                    print(f"MECHANISM NOT CONFIRMED: {small} ranks prefer "
                          f"{sc} cores/rank, not more than {big}'s {bc}. The "
                          f"speed-up is real but the stated reason is not "
                          f"established -- report it as unexplained.")
    print("\nNothing here is adopted. The per-rank core share is a production "
          "setting and changing it is a separate decision.")
    print("CPU_THR_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
