#!/usr/bin/env python
"""Per-step GAIN of the corner-Lagrange wedge on a grid-scale vertex mode.

The case-8 vertex instability is amplified PER STEP, not per unit time:
halving dt (228.6 -> 114.3 s) made it much worse and earlier
(day-3 19.5 -> 147.5).  Physical tendencies scale with dt, so a mode
that grows faster when the same physical time is covered in twice as
many steps must be amplified by a fixed-gain operator applied once per
step -- not by a dt-proportional term.

The corner-Lagrange wedge (fill_corner_region / _CornerLagrange.fill)
is exactly such an operator: applied every acoustic step, with
extrapolation weights reaching ~35 (upstream's own comment calls
Lagrange extrapolation "not highly recommended").  Its VALUES are
certified bit-faithful on smooth and day-0-sharp fields, but that says
nothing about its gain on a 2*dx mode sitting at the 3-valent vertex.

This measures the gain directly: seed a grid-scale (checkerboard)
perturbation confined to the vertex-adjacent halo, apply the fill
repeatedly, and report the growth factor per application.  A gain > 1
is a per-step amplifier and explains every observation (dt-worsening,
dissipation-independence, determinism, vertex locality, and why
CORNER_MODE=nearest suppressed it).

Usage: wedge_gain.py --n 36 --ng 3 --iters 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    n, ng = args.n, args.ng

    sys.path.insert(0, str(REPO / "packages/core"))
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )

    ctx = build_six_face_duo_context(n, ng, use_ext_bundle=True,
                                     oracle_conventions=True, omega=0.0)
    ectx = ctx["ectx"]

    m = n + 2 * ng                      # A-stagger full-halo extent
    ii, jj = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    checker = ((-1.0) ** (ii + jj))     # 2*dx grid-scale mode

    print(f"# corner-Lagrange wedge gain, n={n} ng={ng}")
    print("# stagger  iter  max|wedge|   gain/appl   (>1 = per-step "
          "amplifier)")
    for key, (istag, jstag), lbl in (("corner_a3", (0, 0), "A (0,0)"),
                                     ("corner_b3", (1, 1), "B (1,1)"),
                                     ("corner_du3", (0, 1), "D-u(0,1)"),
                                     ("corner_dv3", (1, 0), "D-v(1,0)")):
        op = ectx[key][0]
        # field: zero in the compute block, grid-scale noise in the halo
        # ring the wedge reads from (so we isolate the wedge's own gain)
        f = np.zeros((m + istag, m + jstag))
        f[:] = 0.0
        cb = slice(ng, ng + n)
        f[:, :] = checker[:m + istag, :m + jstag] * 1.0e-6
        f[cb, cb] = 0.0                 # quiet interior
        amps = []
        for k in range(args.iters):
            op.fill(f)
            # the four 3x3 diagonal wedge blocks
            lo = 0
            hi = m + istag
            w = max(
                float(np.max(np.abs(f[lo:ng, lo:ng]))),
                float(np.max(np.abs(f[lo:ng, hi - ng:hi]))),
                float(np.max(np.abs(f[hi - ng:hi, lo:ng]))),
                float(np.max(np.abs(f[hi - ng:hi, hi - ng:hi]))),
            )
            amps.append(w)
        a = np.array(amps)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.exp(np.diff(np.log(np.maximum(a, 1e-300)))).mean()
        print(f"{lbl:10s} {args.iters:4d}  {a[-1]:11.4e}  {g:9.4f}")
    print("# gain > 1 on any stagger => the wedge amplifies a 2dx "
          "vertex mode once per acoustic step (dt-independent), which "
          "is the observed signature")


if __name__ == "__main__":
    main()
