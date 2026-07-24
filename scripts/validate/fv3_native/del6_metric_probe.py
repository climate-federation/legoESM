#!/usr/bin/env python
"""Does the nord=2 del-6 read BIG_NUMBER junk metric at the cube vertex?

The del-6 divergence-damping flux is fx2 = del6_v(i,j)*(d2(i)-d2(i-1))
over windows reaching is-nord..ie+nord+1 (sw_core d_sw5 / our
d_sw5_duo).  The del-6 edge metrics divg_u/divg_v/del6_u/del6_v are
built full of BIG_NUMBER (1e8) then filled from the extended-grid
metric; gridstruct.py:32 documents the OUTERMOST ring as
"big_number-derived junk".  If the telescoping's read window at the
3-valent vertex touches a junk metric cell, the vertex del-6 flux is
corrupted -> the vertex ring is mis-damped (matches the marginal
day-5 vertex escape while all FIELD inputs are certified faithful).

This builds the live gridstruct and checks, over the exact del-6 read
window near each tile corner, whether any of the four del-6 metric
arrays still carries a BIG_NUMBER-scale value (|v| > 1e7) or a NaN.

Usage: del6_metric_probe.py --n 48 --ng 3 --nord 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
BIG = 1.0e7          # anything this large is the 1e8 sentinel or junk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--nord", type=int, default=2)
    args = ap.parse_args()
    n, ng, nord = args.n, args.ng, args.nord

    sys.path.insert(0, str(REPO / "packages/core"))
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )

    ctx = build_six_face_duo_context(n, ng, use_ext_bundle=True,
                                     oracle_conventions=True, omega=0.0)

    # del-6 read window (matches d_sw5_duo / sw_core): iteration n=1 has
    # nt=nord-1; vc/uc computed over is-1-nt..ie+1+nt reading the metric
    # at those edges.  Widest metric read = is-nord .. ie+nord+1 in each
    # axis.  In full-halo local indices (compute origin at ng): the low
    # ring is ng-nord, the high is ng+n+nord.
    lo = ng - nord
    hi = ng + n + nord
    names = ("divg_u", "divg_v", "del6_u", "del6_v")
    corner_boxes = {
        "SW": (slice(lo, ng + 3), slice(lo, ng + 3)),
        "SE": (slice(ng + n - 2, hi + 1), slice(lo, ng + 3)),
        "NW": (slice(lo, ng + 3), slice(ng + n - 2, hi + 1)),
        "NE": (slice(ng + n - 2, hi + 1), slice(ng + n - 2, hi + 1)),
    }
    total = 0
    for t in range(6):
        gs = ctx["gs6"][t]
        for nm in names:
            arr = np.asarray(gs[nm])
            for cn, (si, sj) in corner_boxes.items():
                blk = arr[si, sj]
                bad = (~np.isfinite(blk)) | (np.abs(blk) > BIG)
                nb = int(bad.sum())
                if nb:
                    total += nb
                    ii, jj = np.where(bad)
                    offs = [(int(si.start + a) - ng, int(sj.start + b) - ng)
                            for a, b in zip(ii[:3], jj[:3])]
                    print(f"face {t+1} {nm} {cn}: {nb} junk/BIG in del-6 "
                          f"read window at compute-rel idx {offs} "
                          f"(vals ~{blk[bad].flat[0]:.2e})")
    if total == 0:
        print("VERDICT: all four del-6 metrics FINITE + sub-BIG over the "
              "nord=%d read window at every vertex (metric NOT the "
              "under-damp source)." % nord)
    else:
        print(f"VERDICT: del-6 metric JUNK — {total} BIG_NUMBER/NaN cells "
              f"in the nord={nord} read window at cube vertices; the "
              f"vertex del-6 flux uses junk metric -> mis-damped vertex "
              f"ring (THE marginal-under-damp source).")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
