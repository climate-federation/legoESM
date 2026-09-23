#!/usr/bin/env python
"""Does the best per-rank core share fall as the local problem shrinks?

One node, four ranks, mesh size varied instead of rank count, so the
partition and the rank layout are identical across arms and only the work per
rank moves.  Reports best-of-repeats per cell and decides against the
pre-registered outcomes: the explanation is SUPPORTED only if the smallest
mesh prefers a smaller share than the largest by more than the repeat
spread, and REFUTED if one share wins everywhere.

A share whose worker count did not follow it is INVALID, not evidence: a
sweep over a dead knob varies pinning alone and still moves timings.
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
    for f in sorted(glob.glob(os.path.join(out_dir, "sub*_cpt*_r*.jsonl"))):
        m = re.search(r"sub(\d+)_cpt(\d+)_r(\d+)\.jsonl$", f)
        if m is None:
            continue
        for line in open(f):
            ms = json.loads(line).get("steady_median_ms")
            if ms:
                cells[(int(m.group(1)), int(m.group(2)))].append(float(ms))
    return cells


def census(out_dir: str) -> dict[int, int]:
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
    workers = census(sys.argv[1])
    dead = {c: n for c, n in workers.items() if n != c}
    if workers:
        print("core share -> XLA workers: "
              + ", ".join(f"{c}->{n}" for c, n in sorted(workers.items())))
    if dead or not workers:
        print(f"INVALID: the worker count did not follow the core share "
              f"({sorted(dead) or 'no census recorded'}), so these timings "
              f"are not a thread result.")
        return 2

    subs = sorted({s for s, _ in cells})
    shares = sorted({c for _, c in cells})
    print(f"\n{'mesh':>5}" + "".join(f"{c:>11}" for c in shares)
          + "      (cores/rank; best of repeats, ms)")
    for s in subs:
        row = "".join(f"{min(cells[(s, c)]):>11.1f}" if (s, c) in cells
                      else f"{'-':>11}" for c in shares)
        print(f"{s:>5}{row}")

    # THE PREREGISTERED STATISTIC is not "which share wins" -- picking a
    # winner turns noise into a verdict when two shares are a millisecond
    # apart. It is the PENALTY for keeping the production share:
    #
    #     penalty(size) = time at 64 cores / best time at that size
    #
    # The explanation says that penalty GROWS as the local problem shrinks.
    # A penalty is a ratio of two measured numbers, so its uncertainty is
    # carried from the repeat spread of both.
    PROD = 64
    pen, unc = {}, {}
    for s_ in subs:
        at = {c: v for (ss, c), v in cells.items() if ss == s_}
        if PROD not in at:
            continue
        bc = min(at, key=lambda c: min(at[c]))
        t_prod, t_best = min(at[PROD]), min(at[bc])
        sp_prod = (max(at[PROD]) - min(at[PROD])) if len(at[PROD]) > 1 else 0.0
        sp_best = (max(at[bc]) - min(at[bc])) if len(at[bc]) > 1 else 0.0
        pen[s_] = t_prod / t_best
        # When the production share IS the best, the ratio is 1 by
        # construction: numerator and denominator are the same measurement,
        # so the repeat spread cancels instead of accumulating. Deriving an
        # uncertainty from it anyway invented a tolerance band around a
        # number that cannot vary, and that band swallowed the clearest
        # possible result -- 64 best at every size -- as INCONCLUSIVE.
        unc[s_] = (0.0 if bc == PROD
                   else pen[s_] * (sp_prod / t_prod + sp_best / t_best))
    if len(pen) < 2:
        print("INCONCLUSIVE: fewer than two mesh sizes carry the production "
              "share, so no trend in its penalty can be read.")
        return 0
    print("\npenalty for keeping " + str(PROD) + " cores/rank "
          "(1.00 = it is already best):")
    for s_ in subs:
        if s_ in pen:
            print(f"  sub{s_}: {pen[s_]:.3f} +/- {unc[s_]:.3f}"
                  f"   best share {min({c: v for (ss, c), v in cells.items() if ss == s_}, key=lambda c: min(cells[(s_, c)]))}")

    lo, hi = min(pen), max(pen)
    delta = pen[lo] - pen[hi]
    tol = unc[lo] + unc[hi]
    never_penalised = all(p_ <= 1.0 + unc[s_] for s_, p_ in pen.items())

    if tol > 0 and abs(delta) <= tol:
        # Noise swamps the trend. Said first, because a verdict read off a
        # difference smaller than its own uncertainty is the failure this
        # statistic exists to prevent.
        print(f"\nINCONCLUSIVE: the penalty changes by {delta:+.3f} across "
              f"the mesh range against a combined uncertainty of {tol:.3f}. "
              f"This job cannot separate the outcomes; more repeats or a "
              f"wider size range would be needed.")
    elif never_penalised:
        # Flat at 1 with a spread too small to hide anything: the production
        # share was never beaten at any size, which is the refutation
        # outright and has no trend to test.
        print(f"\nREFUTED: {PROD} cores/rank was never beaten at any mesh "
              f"size, so its penalty cannot rise as the local problem "
              f"shrinks. The LOCAL-SIZE mechanism is dead.\n"
              f"That does not clear the thread pool itself -- one node has no "
              f"inter-node fabric and cannot see the cousin mechanisms. So "
              f"TRUNCATE the multi-node sweep rather than cancelling it: 128 "
              f"ranks at {PROD} against 32, plus a 64-rank control, two "
              f"repeats. That tests the kink directly at a fraction of the "
              f"cost and still yields the best share per rank count, which "
              f"is worth having whatever the mechanism turns out to be.")
    elif delta > tol and pen[lo] > 1.0 + unc[lo]:
        print(f"\nSUPPORTED: the penalty for {PROD} cores/rank rises from "
              f"{pen[hi]:.3f} at the largest mesh to {pen[lo]:.3f} at the "
              f"smallest, a change of {delta:.3f} against a combined "
              f"uncertainty of {tol:.3f}. A share fixed at {PROD} costs more "
              f"as the local problem shrinks, which is where strong scaling "
              f"ends up.")
    else:
        print(f"\nREFUTED, in the opposite direction: the penalty for "
              f"{PROD} cores/rank FALLS as the local problem shrinks "
              f"({pen[hi]:.3f} -> {pen[lo]:.3f}), which the explanation does "
              f"not predict. TRUNCATE the multi-node sweep as above rather "
              f"than cancelling it.")
    print("\nOne node has no inter-node fabric, so this can refute the "
          "explanation but cannot confirm it at real rank counts.")
    print("SHARE_SZ_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
