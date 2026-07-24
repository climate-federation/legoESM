#!/usr/bin/env python
"""Structural probe: is divg_d ghosted to the del-6 read-depth at the
3-valent cube vertex?

The oracle's d_sw5 divergence del-6 (nord=2) telescopes a purely LOCAL
Laplacian^3 over divg_d that was ext_scalar(divgd,1,1)-ghosted ONCE to
depth nord+1 before the loop (sw_core.F90:1731-1788, dyn_core:652); no
inter-iteration re-exchange, copy_corners skipped in duo mode.  So the
vertex damping REQUIRES divg_d valid over the full del-6 stencil window
at the vertex.

This poisons every divg_d HALO cell with NaN, runs the live
ext_scalar_sixface(divgd, "B") exactly as the stepper does, then checks
whether the cells the nord=2 del-6 n-loop READS near each cube vertex
are finite.  Any NaN survivor in the read window = an incomplete vertex
ghost = the faithful divergence there is fed a garbage neighbour and is
NOT damped correctly -> the vertex ring grows (matches the hourly-oracle
finding: oracle damps the vertex mode, ours amplifies it).

nord=2 read window (per d_sw5_duo / sw_core): iteration n=1 has nt=1 and
computes vc/uc over is-1-nt..ie+1+nt = is-2..ie+2, reading divg_d at
i,i+1 -> divg_d needed over is-2..ie+3 (i.e. 2 rings past the compute
block on the low side, 3 on the high).  We check finiteness over the
B-node index box [is-2 .. ie+3] near each corner.

Usage: divgd_ghost_probe.py --n 12 --ng 3 --nord 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--nord", type=int, default=2)
    args = ap.parse_args()
    n, ng, nord = args.n, args.ng, args.nord

    sys.path.insert(0, str(REPO / "packages/core"))
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.grids.fv3_native_ext_vector import ext_scalar_sixface

    ctx = build_six_face_duo_context(n, ng, use_ext_bundle=True,
                                     oracle_conventions=True, omega=0.0)
    ectx = ctx["ectx"]

    npx = n + 1
    m = n + 2 * ng + 1          # B-node full-halo extent (isd..ied+1)
    # a sharp divg_d in the COMPUTE block (vertex-crossing dipole proxy),
    # NaN everywhere in the halo so any un-refreshed cell stays NaN
    divgd6 = []
    for t in range(6):
        a = np.full((m, m), np.nan)
        cb = slice(ng, ng + npx)
        xb = np.linspace(-1, 1, npx)
        gx, gy = np.meshgrid(xb, xb, indexing="ij")
        a[cb, cb] = np.sin(3 * np.pi * gx) * np.cos(3 * np.pi * gy)
        divgd6.append(a)

    ext_scalar_sixface(divgd6, "B", ectx)

    # del-6 read box near each of the 4 tile corners (B-node local idx,
    # compute origin at ng): low corner needs is-2 = ng-2; high corner
    # needs ie+3 = ng + n + 3.  Check the (nord)-deep L-shaped ring the
    # telescoping actually reads.
    lo = ng - nord          # is-nord ring (=is-2 for nord2)
    hi = ng + n + nord + 1  # ie+nord+1 (=ie+3)
    corners = {
        "SW": (slice(lo, ng + 2), slice(lo, ng + 2)),
        "SE": (slice(ng + n - 1, hi), slice(lo, ng + 2)),
        "NW": (slice(lo, ng + 2), slice(ng + n - 1, hi)),
        "NE": (slice(ng + n - 1, hi), slice(ng + n - 1, hi)),
    }
    total_nan = 0
    for t in range(6):
        for name, (si, sj) in corners.items():
            blk = divgd6[t][si, sj]
            nnan = int(np.isnan(blk).sum())
            if nnan:
                total_nan += nnan
                ii, jj = np.where(np.isnan(blk))
                # report the first few offending local indices
                offs = [(int(si.start + a), int(sj.start + b))
                        for a, b in zip(ii[:4], jj[:4])]
                print(f"face {t+1} {name}: {nnan} NaN in del-6 read box "
                      f"-> stale ghost at {offs}")
    if total_nan == 0:
        print("VERDICT: divg_d ghost COMPLETE over the nord=%d del-6 "
              "read window at every vertex (mechanism NOT ghost-depth)"
              % nord)
    else:
        print(f"VERDICT: divg_d ghost INCOMPLETE — {total_nan} stale "
              f"cells in the del-6 read window at cube vertices; the "
              f"telescoping damps a garbage neighbour -> vertex ring "
              f"under-damped (matches oracle-damps/ours-grows).")
    sys.exit(1 if total_nan else 0)


if __name__ == "__main__":
    main()
