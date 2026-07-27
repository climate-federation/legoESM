#!/usr/bin/env python
"""Stage-by-stage diff: Zenodo model ext_vector dumps vs ours.

Inputs: --fort-dir (extvec_<stage>_<name>_t<T>.dat from the
instrumented fv_duogrid.F90, first D-grid call) and --ours (npz from
extvec_stage_dump.py).  Both sides run the case-8 two-burst IC; the
oracle run uses do_schmidt (a 180-degree lon rotation, a cube symmetry)
under which the case-8 wind field maps to its NEGATIVE, so dynamic
stages are compared as ours vs sign*theirs with sign picked globally
from the S2 interior fit and required consistent everywhere.

Alignment gate: static dx must match tile-for-tile at ~1e-12 rel or
the comparison ABORTS (tile mapping is then not identity and every
other number would be garbage).

Region masks (Fortran indices, face extent [1..n] per unstaggered
axis): interior / side-halo / corner-wedge; at S3/S4 the corner-wedge
region is EXCLUDED (upstream leaves it undefined there until the
Lagrange fill inside cubed_a2d_halo); at S5geo the outermost wedge
ring (4) is excluded (ours keeps AGRID index-copies there, upstream
leaves it undefined; both dead on the faithful path).  Unwritten
sentinels: -99999. (theirs) and NaN (ours) — coverage mismatches are
REPORTED, not silently dropped.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

SENT = -99999.0


def read_fort(path: Path):
    with open(path) as f:
        ilo, ihi, jlo, jhi = (int(x) for x in f.readline().split())
        a = np.full((ihi - ilo + 1, jhi - jlo + 1), np.nan)
        for line in f:
            i, j, v = line.split()
            a[int(i) - ilo, int(j) - jlo] = float(v)
    return a, ilo, jlo


def load_fort_dir(d: Path):
    out = {}
    for p in sorted(d.glob("extvec_*.dat")):
        m = re.match(r"extvec_(S\d)_(\w+)_t(\d)\.dat", p.name)
        if not m:
            continue
        st, nm, t = m.group(1), m.group(2), int(m.group(3))
        out[(st, t, nm)] = read_fort(p)
    return out


def region_mask(shape, ilo, jlo, n, stag_i=0, stag_j=0):
    """0=interior, 1=side halo, 2=corner wedge (Fortran indexing;
    face extent [1..n+stag] per axis)."""
    ii = np.arange(ilo, ilo + shape[0])[:, None] * np.ones(
        (1, shape[1]), dtype=int)
    jj = np.arange(jlo, jlo + shape[1])[None, :] * np.ones(
        (shape[0], 1), dtype=int)
    di = np.maximum(np.maximum(1 - ii, ii - (n + stag_i)), 0)
    dj = np.maximum(np.maximum(1 - jj, jj - (n + stag_j)), 0)
    reg = np.zeros(shape, dtype=int)
    reg[(di > 0) | (dj > 0)] = 1
    reg[(di > 0) & (dj > 0)] = 2
    ring = np.maximum(di, dj)
    return reg, ring


def compare(name, ours, fort, ilo, jlo, n, sign, stag_i=0, stag_j=0,
            exclude=None, floor=1e-9):
    """Returns (max_rel, argmax_fort_ij, n_cov_mismatch, n_compared)."""
    if ours.shape != fort.shape:
        return ("SHAPE", ours.shape, fort.shape, 0)
    wf = np.isfinite(fort) & (np.abs(fort - SENT) > 0.5)
    wo = np.isfinite(ours)
    reg, ring = region_mask(fort.shape, ilo, jlo, n, stag_i, stag_j)
    keep = np.ones(fort.shape, bool)
    if exclude == "corners":
        keep &= reg != 2
    elif exclude == "ring4corners":
        keep &= ~((reg == 2) & (ring >= 4))
    cov_mismatch = int(np.sum((wf != wo) & keep))
    both = wf & wo & keep
    if not both.any():
        return (0.0, None, cov_mismatch, 0)
    d = np.abs(ours - sign * fort)
    scale = np.maximum(np.maximum(np.abs(ours), np.abs(fort)), floor)
    rel = np.where(both, d / scale, 0.0)
    k = np.unravel_index(np.argmax(rel), rel.shape)
    return (float(rel.max()), (k[0] + ilo, k[1] + jlo), cov_mismatch,
            int(both.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fort-dir", required=True)
    ap.add_argument("--ours", required=True)
    args = ap.parse_args()

    fort = load_fort_dir(Path(args.fort_dir))
    ours = np.load(args.ours, allow_pickle=False)
    n = int(ours["n"])
    if not fort:
        print("FATAL: no extvec_*.dat in", args.fort_dir)
        sys.exit(2)
    print(f"loaded {len(fort)} fortran arrays; ours git_sha="
          f"{ours['git_sha']}")

    # ---- alignment gate: static dx identity, all six tiles ----
    worst = 0.0
    for t in range(1, 7):
        fa, ilo, jlo = fort[("S0", t, "dx")]
        oa = ours[f"S0_t{t}_dx"]
        if oa.shape != fa.shape:
            print(f"FATAL: dx shape t{t} ours {oa.shape} fort {fa.shape}")
            sys.exit(2)
        r = np.abs(oa - fa) / np.maximum(np.abs(fa), 1e-9)
        worst = max(worst, float(r.max()))
    print(f"ALIGNMENT GATE dx identity: max rel {worst:.3e}")
    if worst > 1e-11:
        print("FATAL: tile mapping is not identity — comparison aborted")
        sys.exit(2)

    # ---- sign from S2 interior fit ----
    fa, ilo, jlo = fort[("S2", 1, "ull")]
    oa = ours["S2_t1_ull"]
    reg, _ = region_mask(fa.shape, ilo, jlo, n)
    both = (np.isfinite(fa) & (np.abs(fa - SENT) > 0.5)
            & np.isfinite(oa) & (reg == 0))
    dp = float(np.abs(oa[both] - fa[both]).mean())
    dm = float(np.abs(oa[both] + fa[both]).mean())
    sign = 1.0 if dp <= dm else -1.0
    print(f"SIGN (S2 interior): +1 residual {dp:.3e}, -1 residual "
          f"{dm:.3e} -> sign={sign:+.0f}")

    # ---- per-stage compare ----
    # (stage, name, sign?, stag_i, stag_j, exclude)
    plan = [
        # a-matrix window is is-1..ie+1: its corner region is exactly
        # the 4 corner-DIAGONAL cells where our sin5 recompute is
        # disclosed-different from upstream's floor-poisoned value
        # (center_a_matrix docstring; both dead on the faithful path)
        ("S0", "a11", 1.0, 0, 0, "corners"),
        ("S0", "a12", 1.0, 0, 0, "corners"),
        ("S0", "a21", 1.0, 0, 0, "corners"),
        ("S0", "a22", 1.0, 0, 0, "corners"),
        ("S0", "dy", 1.0, 1, 0, None),
        ("S0", "sinsg5", 1.0, 0, 0, "corners"),
        ("S1", "uin", sign, 0, 1, None),
        ("S1", "vin", sign, 1, 0, None),
        ("S2", "ull", sign, 0, 0, None),
        ("S2", "vll", sign, 0, 0, None),
        ("S3", "ullp1", sign, 0, 0, "corners"),
        ("S3", "vllp1", sign, 0, 0, "corners"),
        ("S4", "ullp1", sign, 0, 0, "corners"),
        ("S4", "vllp1", sign, 0, 0, "corners"),
        ("S5", "ullp1", sign, 0, 0, "ring4corners"),
        ("S5", "vllp1", sign, 0, 0, "ring4corners"),
        ("S6", "uin", sign, 0, 1, None),
        ("S6", "vin", sign, 1, 0, None),
    ]
    print(f"{'stage':6s} {'name':7s} {'tile':4s} {'max_rel':>12s} "
          f"{'argmax(i,j)':>14s} {'cov_mm':>7s} {'ncmp':>8s}")
    stage_worst = {}
    for st, nm, sg, sti, stj, excl in plan:
        for t in range(1, 7):
            key = (st, t, nm)
            okey = f"{st}_t{t}_{nm}"
            if key not in fort or okey not in ours.files:
                print(f"{st:6s} {nm:7s} t{t}   MISSING "
                      f"(fort={key in fort} ours={okey in ours.files})")
                continue
            fa, ilo, jlo = fort[key]
            r = compare(nm, ours[okey], fa, ilo, jlo, n, sg,
                        sti, stj, excl)
            if r[0] == "SHAPE":
                print(f"{st:6s} {nm:7s} t{t}   SHAPE ours={r[1]} "
                      f"fort={r[2]}")
                stage_worst[st] = np.inf
                continue
            mx, arg, cov, ncmp = r
            print(f"{st:6s} {nm:7s} t{t}  {mx:12.4e} {str(arg):>14s} "
                  f"{cov:7d} {ncmp:8d}")
            stage_worst[st] = max(stage_worst.get(st, 0.0), mx)

    print("\n=== per-stage worst rel ===")
    for st in sorted(stage_worst):
        print(f"  {st}: {stage_worst[st]:.4e}")
    print("\nREADING: static S0 should sit at ~1e-12; the FIRST stage "
          "whose worst rel jumps orders above its predecessor is the "
          "defect locus.  IC-formula FP noise between the two codes is "
          "~1e-13 rel; anything > 1e-6 at S1-S6 is a discretization-"
          "level live-path difference.")


if __name__ == "__main__":
    main()
