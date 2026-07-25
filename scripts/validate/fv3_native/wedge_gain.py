#!/usr/bin/env python
"""Corner-wedge RESPONSE RATIO on a grid-scale mode, ours vs the Fortran.

The case-8 vertex instability is amplified PER STEP, not per unit time:
halving dt (228.6 -> 114.3 s) made it much worse and earlier (day-3
19.5 -> 147.5).  Physical tendencies scale with dt, so the amplifier
must be a fixed-gain operator applied once per acoustic step.  The
corner-Lagrange wedge is such an operator (extrapolation weights reach
~35; upstream's own comment warns against Lagrange extrapolation), and
replacing it with the nearest compute value suppressed the growth.

NOTE on what is NOT measured here: applying `fill` repeatedly to a
FIXED source is trivially gain 1 -- the fill overwrites the wedge from
the source strips every time.  The physically relevant quantity is the
RESPONSE RATIO  max|wedge| / max|source|  for a grid-scale (2*dx)
source: that is the factor by which halo values exceed interior values,
which is what feeds back through the stencils on the next step.

It also answers the faithfulness question the existing certifications
do not: our wedge matches the Fortran to 3.5e-13 on smooth and
day-0-sharp fields, but a 2*dx checkerboard is a different input class.
With --build-dir the same verbatim-Fortran cornerlag driver is run on
the identical checkerboard and the outputs are diffed.

Usage:
  wedge_gain.py --n 12 --ng 3 --build-dir /path/to/build
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
_STAGS = ((0, 0), (1, 1), (0, 1), (1, 0))
_TAG = {(0, 0): "F00", (1, 1): "F11", (0, 1): "F01", (1, 0): "F10"}
_LBL = {(0, 0): "A  (0,0)", (1, 1): "B  (1,1)",
        (0, 1): "D-u(0,1)", (1, 0): "D-v(1,0)"}


def _driver(build_dir: Path) -> Path:
    exe = build_dir / "cornerlag_driver"
    if not exe.exists():
        src = REPO / "scripts/validate/fv3_native"
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["gfortran", "-O0", "-g", "-fdefault-real-8",
             "-fdefault-double-8", "-o", str(exe),
             str(src / "fv3_cornerlag_shim.F90"),
             str(src / "fv3_cornerlag_extract.F90"),
             str(src / "fv3_cornerlag_driver.F90")],
            check=True, cwd=build_dir)
    return exe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--build-dir", default=None,
                    help="if given, also diff against the verbatim "
                         "Fortran cornerlag driver on the same input")
    args = ap.parse_args()
    n, ng = args.n, args.ng

    sys.path.insert(0, str(REPO / "packages/core"))
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.grids.fv3_native_ext_vector import (
        ext_parity_lonlat_ref,
    )

    ctx = build_six_face_duo_context(n, ng, use_ext_bundle=True,
                                     oracle_conventions=True, omega=0.0)
    ectx = ctx["ectx"]
    keys = {(0, 0): "corner_a3", (1, 1): "corner_b3",
            (0, 1): "corner_du3", (1, 0): "corner_dv3"}

    def checker_field(istag, jstag):
        """2*dx checkerboard over the whole lattice for this stagger."""
        ni = n + 2 * ng + istag
        nj = n + 2 * ng + jstag
        ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
        return ((-1.0) ** (ii + jj)).astype(float)

    def wedge_blocks(f, istag, jstag):
        ni, nj = f.shape
        return (f[0:ng, 0:ng], f[0:ng, nj - ng:nj],
                f[ni - ng:ni, 0:ng], f[ni - ng:ni, nj - ng:nj])

    fields = {}
    print(f"# corner-wedge response on a 2dx checkerboard, n={n} ng={ng}")
    print("# stagger    max|source|   max|wedge|   RESPONSE RATIO")
    for stag in _STAGS:
        f = checker_field(*stag)
        fields[stag] = np.array(f, copy=True)      # pristine copy
        src = float(np.max(np.abs(f)))
        ectx[keys[stag]][0].fill(f)
        w = max(float(np.max(np.abs(b)))
                for b in wedge_blocks(f, *stag))
        print(f"{_LBL[stag]:10s} {src:12.4e} {w:12.4e} {w / src:14.2f}")
    print("# ratio >> 1: the wedge writes halo values far larger than "
          "the interior for a grid-scale mode -> per-step amplification")

    if not args.build_dir:
        return

    # ---- same checkerboard through the verbatim Fortran ------------
    a_lon_w, a_lat_w = (x[0] for x in
                        ext_parity_lonlat_ref(n, ng + 1, "A"))
    exe = _driver(Path(args.build_dir))
    lines = [f"{n} {ng}"]
    lo_w = 1 - (ng + 1)
    m = a_lon_w.shape[0]
    for i in range(m):
        for j in range(m):
            lines.append(f"APT {i + lo_w} {j + lo_w} "
                         f"{a_lon_w[i, j]:.17e} {a_lat_w[i, j]:.17e}")
    for stag in _STAGS:
        f = fields[stag]
        ni, nj = f.shape
        for i in range(ni):
            for j in range(nj):
                lines.append(f"{_TAG[stag]} {i + 1 - ng} {j + 1 - ng} "
                             f"{f[i, j]:.17e} 0.0")
    lines.append("COEF 0 0 0 0")
    for stag in _STAGS:
        lines.append(f"RUN {stag[0]} {stag[1]} 0 0")
    run = subprocess.run([str(exe)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    oracle = {s: {} for s in _STAGS}
    for line in run.stdout.splitlines():
        p = line.split()
        if p and p[0] == "OUT":
            oracle[(int(p[1]), int(p[2]))][(int(p[3]), int(p[4]))] = \
                float(p[5])

    print("\n# ours vs verbatim Fortran on the SAME checkerboard")
    print("# stagger    max|ours-oracle|   rel")
    worst = 0.0
    for stag in _STAGS:
        f = np.array(fields[stag], copy=True)
        ectx[keys[stag]][0].fill(f)
        d = 0.0
        scale = 0.0
        for (fi, fj), val in oracle[stag].items():
            got = f[fi - 1 + ng, fj - 1 + ng]
            d = max(d, abs(got - val))
            scale = max(scale, abs(val))
        rel = d / scale if scale else float("nan")
        worst = max(worst, rel)
        print(f"{_LBL[stag]:10s} {d:18.6e} {rel:10.2e}")
    verdict = ("wedge is FAITHFUL even for grid-scale input"
               if worst < 1e-9 else
               "wedge DIFFERS from the oracle on grid-scale input")
    print(f"# VERDICT: {verdict} (worst rel {worst:.2e})")


if __name__ == "__main__":
    main()
