#!/usr/bin/env python
"""Is the case-8 vertex 'spike' in the MODEL STATE or only in the LENS?

Our vertex-growth numbers (day-5 max|V| ~95-135) are read off the
c2l_ord2 -> 1-degree geographic frames the runner writes.  The W2
campaign already hit one trap of exactly this shape: the apparent
"vertex imprint" was a runner diagnostic artifact (central-difference
tangent bases at the panel kink), not a model defect.

This compares, at the SAME frame, the maximum wind computed from
  (a) the RAW six-face D-grid state  (--dump-lattice-days output)
  (b) the c2l-sampled 1-degree frame (--out npz)

If (a) stays ~20-26 m/s while (b) reads 100+, the spike is a
DIAGNOSTIC artifact of the geographic lens near the 3-valent vertex and
the model state is healthy.  If (a) also spikes, the growth is real.

Usage:
  raw_vs_lens_max.py --dump svtx_dump_lat_day5.npz \
                     --frames svtx_dump.npz --day 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True,
                    help="lattice dump npz (raw six-face delp/pt/u/v)")
    ap.add_argument("--frames", required=True,
                    help="runner npz with the c2l 1-degree frames")
    ap.add_argument("--day", type=float, required=True)
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--ng", type=int, default=3)
    args = ap.parse_args()
    n, ng = args.n, args.ng

    sys.path.insert(0, str(REPO / "packages/core"))
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.grids.fv3_native_ext_vector import (
        c2l_ord2_face,
        center_a_matrix,
    )

    d = np.load(args.dump)
    ctx = build_six_face_duo_context(n, ng, use_ext_bundle=True,
                                     oracle_conventions=True, omega=0.0)
    sl = slice(ng, ng + n)
    slb = slice(ng, ng + n + 1)

    raw_max = 0.0
    c2l_native_max = 0.0
    for t in range(1, 7):
        u = np.asarray(d[f"u_t{t}"])
        v = np.asarray(d[f"v_t{t}"])
        # (a) raw D-grid edge winds, compute block only
        uu = u[sl, slb]
        vv = v[slb, sl]
        raw_max = max(raw_max, float(np.nanmax(np.abs(uu))),
                      float(np.nanmax(np.abs(vv))))
        # (b) the c2l lens on the native cube (pre-regrid): this is what
        # the runner then samples to 1 degree
        gs = ctx["gs6"][t - 1]
        amat = center_a_matrix(gs)
        ua, va = c2l_ord2_face(u, v, gs["dx"], gs["dy"], amat, n, ng)
        w = np.hypot(ua[sl, sl], va[sl, sl])
        c2l_native_max = max(c2l_native_max, float(np.nanmax(w)))

    f = np.load(args.frames)
    days = f["times_days"]
    i = int(np.argmin(np.abs(days - args.day)))
    lens_max = float(np.nanmax(np.hypot(f["u"][i], f["v"][i])))

    print(f"day {days[i]:g}")
    print(f"  RAW D-grid max|u|,|v| (model state)   : {raw_max:8.3f} m/s")
    print(f"  c2l on native cube (pre-regrid)       : "
          f"{c2l_native_max:8.3f} m/s")
    print(f"  c2l + 1-degree nearest sample (report): "
          f"{lens_max:8.3f} m/s")
    if lens_max > 3 * max(raw_max, 1.0):
        print("  => VERDICT: the reported spike is a LENS ARTIFACT "
              "(model state is healthy)")
    elif c2l_native_max > 3 * max(raw_max, 1.0):
        print("  => VERDICT: the c2l transform itself inflates it "
              "(tangent-basis artifact at the panel kink)")
    else:
        print("  => VERDICT: the growth is REAL in the model state")


if __name__ == "__main__":
    main()
