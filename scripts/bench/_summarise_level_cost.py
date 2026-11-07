#!/usr/bin/env python
"""Does the level-count cost anomaly reproduce, and does it survive at scale?

The repo carries a measured table saying the icosahedral core costs about
2.9x more per cell per level at 26 levels than at 32 or 40.  Every MPAS
scaling arm runs 26; production AMIP runs 40.  This checks the table against
its own measurement size first -- a known answer -- and then at the size the
ladder actually runs.

Cost is reported per cell per level so that arms with different level counts
are comparable at all; comparing step times directly would just say that
more levels take longer.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

#: The published table, picoseconds per cell per level (grids/vertical.py).
PUBLISHED = {26: 4755, 32: 1629, 40: 1664}
KNOWN_ANSWER_SUB = 7


def load(out_dir: str):
    cells: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(out_dir, "sub*_nlev*_r*.jsonl"))):
        m = re.search(r"sub(\d+)_nlev(\d+)_r(\d+)\.jsonl$", f)
        if m is None:
            continue
        for line in open(f):
            r = json.loads(line)
            ms = r.get("steady_median_ms")
            n_cells = r.get("n_cells") or r.get("total_cells")
            if ms and n_cells:
                cells[(int(m.group(1)), int(m.group(2)))].append(
                    (float(ms), int(n_cells)))
    return cells


def main() -> int:
    cells = load(sys.argv[1])
    if not cells:
        print("no arms found", file=sys.stderr)
        return 1
    subs = sorted({s for s, _ in cells})
    levs = sorted({n for _, n in cells})

    # Picoseconds per cell per level: ms -> ps is 1e9.
    ps = {}
    for (s, n), vals in cells.items():
        ms, n_cells = min(vals, key=lambda v: v[0])
        ps[(s, n)] = ms * 1e9 / (n_cells * n)

    print(f"\n{'mesh':>5}" + "".join(f"{n:>12}" for n in levs)
          + "      (picoseconds per cell per level)")
    for s in subs:
        print(f"{s:>5}" + "".join(
            f"{ps[(s, n)]:>12.0f}" if (s, n) in ps else f"{'-':>12}"
            for n in levs))
    print(f"{'pub':>5}" + "".join(
        f"{PUBLISHED[n]:>12}" if n in PUBLISHED else f"{'-':>12}"
        for n in levs) + "      (the table this checks)")

    if KNOWN_ANSWER_SUB in subs and 26 in levs:
        cheap = [n for n in levs if n != 26 and (KNOWN_ANSWER_SUB, n) in ps]
        if cheap:
            best = min(ps[(KNOWN_ANSWER_SUB, n)] for n in cheap)
            ratio = ps[(KNOWN_ANSWER_SUB, 26)] / best
            pub_ratio = PUBLISHED[26] / min(PUBLISHED[n] for n in cheap
                                            if n in PUBLISHED)
            print(f"\nknown answer at subdivision {KNOWN_ANSWER_SUB}: "
                  f"26 levels cost {ratio:.2f}x the cheapest count measured "
                  f"here; the table says {pub_ratio:.2f}x.")
            if ratio >= 2.0:
                print("CONFIRMED: the anomaly reproduces. Every MPAS scaling "
                      "number in the figures was taken at 26 levels, so its "
                      "local kernel is roughly that factor more expensive "
                      "than production's 40, and the compute-to-communication "
                      "balance under test is not production's.")
            elif ratio <= 1.3:
                print("REFUTED: the anomaly does NOT reproduce. The table is "
                      "a year-old measurement on one machine and should be "
                      "corrected rather than carried; the ladder's level "
                      "count is then not an issue.")
            else:
                print("PARTIAL: the anomaly is present but much weaker than "
                      "the table claims. Report the measured ratio, not the "
                      "published one.")

    for s in subs:
        if s == KNOWN_ANSWER_SUB:
            continue
        have = [n for n in levs if (s, n) in ps]
        if len(have) >= 2:
            worst = max(have, key=lambda n: ps[(s, n)])
            best = min(have, key=lambda n: ps[(s, n)])
            print(f"\nat subdivision {s}: {worst} levels cost "
                  f"{ps[(s, worst)] / ps[(s, best)]:.2f}x {best} levels per "
                  f"cell per level. Total step time still favours whichever "
                  f"count the model needs -- this is a cost table, not a "
                  f"recommendation, and the level count is the user's call.")
    print("LEV_COST_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
