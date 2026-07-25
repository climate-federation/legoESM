#!/usr/bin/env python
"""Grid-scale (2*dx) content near cube vertices: ours vs the oracle.

The corner-Lagrange wedge is a ~300x per-step amplifier of 2*dx
structure -- and it is FAITHFUL (ours matches the verbatim Fortran to
4e-14 on a checkerboard), so BOTH codes carry that amplifier.  It only
fires if the field actually has grid-scale content at the 3-valent
vertex.  Halving dt makes our instability much worse, confirming a
per-step (not per-unit-time) compounding.

So the question is no longer "what amplifies?" but "what SEEDS 2*dx
structure at our vertex that the oracle's does not?".

This measures the seed directly and layout-independently.  Both models
carry A-grid winds on the NATIVE cube (theirs as atmos_*.nc ucomp/vcomp
per tile; ours via c2l_ord2 on the raw six-face dump), so no lat-lon
regridding and no face-permutation mapping is needed: the metric is
computed per tile near the four tile corners and MAXIMISED over tiles,
which is invariant under any relabelling of faces.

Metric: amplitude of the checkerboard component,
    g = mean( |f - smooth(f)| )  over a (2k+1)^2 box at each corner,
where smooth() is a 4-neighbour average.  Reported per frame for both
models so the first frame at which ours departs localizes the seed in
time.

Usage:
  vertex_gridscale.py --oracle-dir run_c36 --oracle-prefix atmos_daily \
      --dumps 'c36tb_dump_lat_day*.npz' --n 36
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import netCDF4 as nc

REPO = Path(__file__).resolve().parents[3]


def gridscale(field, k=4):
    """mean |f - 4-neighbour-average| in (2k+1)^2 corner boxes, max over
    the four corners of the tile."""
    f = np.asarray(field, dtype=float)
    sm = np.copy(f)
    sm[1:-1, 1:-1] = 0.25 * (f[:-2, 1:-1] + f[2:, 1:-1]
                             + f[1:-1, :-2] + f[1:-1, 2:])
    r = np.abs(f - sm)
    n = f.shape[0]
    b = min(k, n // 4)
    return max(float(np.nanmean(r[1:1 + b, 1:1 + b])),
               float(np.nanmean(r[1:1 + b, -1 - b:-1])),
               float(np.nanmean(r[-1 - b:-1, 1:1 + b])),
               float(np.nanmean(r[-1 - b:-1, -1 - b:-1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--oracle-prefix", default="atmos_daily")
    ap.add_argument("--dumps", required=True,
                    help="glob for our lattice dumps (…_lat_dayN.npz)")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--oracle-cadence", choices=("days", "hours"),
                    default="days",
                    help="output interval of the oracle file: frame k is "
                         "day k+1 ('days') or hour k+1 ('hours').  Using "
                         "the wrong one silently compares different "
                         "times (a real bug this script had).")
    args = ap.parse_args()
    n, ng = args.n, args.ng

    # ---- oracle: native-tile A-grid winds --------------------------
    fs = sorted(glob.glob(f"{args.oracle_dir}/{args.oracle_prefix}"
                          f".tile*.nc"))
    o_g = None
    for f in fs:
        d = nc.Dataset(f)
        u = np.asarray(d.variables["ucomp"][:]).squeeze()
        v = np.asarray(d.variables["vcomp"][:]).squeeze()
        if u.ndim == 2:
            u = u[None]
            v = v[None]
        nt = u.shape[0]
        if o_g is None:
            o_g = np.zeros(nt)
        for t in range(nt):
            o_g[t] = max(o_g[t], gridscale(u[t]), gridscale(v[t]))
        d.close()

    # ---- ours: c2l on the raw six-face dumps -----------------------
    sys.path.insert(0, str(REPO / "packages/core"))
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.grids.fv3_native_ext_vector import (
        c2l_ord2_face,
        center_a_matrix,
    )

    ctx = build_six_face_duo_context(n, ng, use_ext_bundle=True,
                                     oracle_conventions=True, omega=0.0)
    sl = slice(ng, ng + n)
    ours = {}
    for path in sorted(glob.glob(args.dumps)):
        m = re.search(r"day([\d.]+)\.npz", path)
        day = float(m.group(1)) if m else float("nan")
        d = np.load(path)
        g = 0.0
        for t in range(1, 7):
            gs = ctx["gs6"][t - 1]
            amat = center_a_matrix(gs)
            ua, va = c2l_ord2_face(np.asarray(d[f"u_t{t}"]),
                                   np.asarray(d[f"v_t{t}"]),
                                   gs["dx"], gs["dy"], amat, n, ng)
            g = max(g, gridscale(ua[sl, sl]), gridscale(va[sl, sl]))
        ours[day] = g

    print("# 2dx content near cube vertices (A-grid winds, native cube)")
    print("# day    ORACLE      OURS       ratio")
    for day in sorted(ours):
        # frame k holds day k+1 (daily) or hour k+1 (hourly)
        oi = (int(round(day)) - 1 if args.oracle_cadence == "days"
              else int(round(day * 24.0)) - 1)
        ov = o_g[oi] if o_g is not None and 0 <= oi < len(o_g) else np.nan
        r = ours[day] / ov if ov and np.isfinite(ov) else float("nan")
        print(f"{day:5.1f} {ov:11.4e} {ours[day]:11.4e} {r:9.2f}")
    print("# the first day where OURS >> ORACLE localizes the seed in "
          "time; ratio ~1 early means the seed appears later")


if __name__ == "__main__":
    main()
