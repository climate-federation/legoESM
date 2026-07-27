#!/usr/bin/env python
"""Stage-by-stage diff: Zenodo model ext_vector dumps vs ours.

Inputs: --fort-dir (extvec_<stage>_<name>_t<T>.dat from the
instrumented fv_duogrid.F90, first D-grid call) and --ours (npz from
extvec_stage_dump.py).  Both sides run the case-8 two-burst IC; the
oracle run uses do_schmidt (a 180-degree lon rotation about the polar
axis, stretch=1) under which the case-8 wind field maps to its
NEGATIVE, so dynamic stages are compared as ours vs sign*theirs with
sign picked globally from the S2 interior fit and required consistent
everywhere.

ALIGNMENT GATE (codex r1 P1-6): tile-for-tile, index-for-index
Cartesian agrid agreement under Rz(pi) — dx alone cannot distinguish a
tile permutation.  A dx scale check then catches constants mismatches
(codex r1 P0-1: an oracle rebuilt with GFDL constants R=6371.0e3 vs
our published-run pin R=6371.2e3 shifts every metric by 3.14e-5).

FAIL-LOUD (codex r1 P0-2): a missing stage, a shape mismatch, an
empty comparison overlap, or a coverage mismatch is FATAL (exit 1) —
a broken dump must never read as a clean stage.

Region masks (Fortran indices, face extent [1..n] per unstaggered
axis): interior / side-halo / corner-wedge; at S3/S4 the corner-wedge
region is EXCLUDED (upstream leaves it undefined until the Lagrange
fill inside cubed_a2d_halo); at S5geo the outermost wedge ring (4) is
excluded (ours keeps AGRID index-copies there, upstream leaves it
undefined; both dead on the faithful path).  Unwritten sentinels:
-99999./-999./-888. (theirs) and NaN (ours).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

SENTINELS = (-99999.0, -999.0, -888.0)


def is_unwritten_fort(a):
    w = ~np.isfinite(a)
    for s in SENTINELS:
        w |= np.abs(a - s) < 0.5
    return w


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


def overlap_crop(ours, lo_oi, lo_oj, fort, ilo, jlo):
    """Crop both arrays to the intersection of their Fortran index
    windows.  Returns (ours_c, fort_c, ilo_c, jlo_c) or None if the
    windows do not overlap."""
    hi_oi, hi_oj = lo_oi + ours.shape[0] - 1, lo_oj + ours.shape[1] - 1
    hi_fi, hi_fj = ilo + fort.shape[0] - 1, jlo + fort.shape[1] - 1
    i0, i1 = max(lo_oi, ilo), min(hi_oi, hi_fi)
    j0, j1 = max(lo_oj, jlo), min(hi_oj, hi_fj)
    if i0 > i1 or j0 > j1:
        return None
    oc = ours[i0 - lo_oi:i1 - lo_oi + 1, j0 - lo_oj:j1 - lo_oj + 1]
    fc = fort[i0 - ilo:i1 - ilo + 1, j0 - jlo:j1 - jlo + 1]
    return oc, fc, i0, j0


def compare(ours, fort, ilo, jlo, n, sign, stag_i=0, stag_j=0,
            exclude=None, floor=1e-6):
    """(max_abs, max_rel, argmax_fort_ij, n_cov_mismatch, n_compared)
    or an error-marker tuple ("EMPTY",)."""
    wf = ~is_unwritten_fort(fort)
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
        return ("EMPTY",)
    d = np.where(both, np.abs(ours - sign * fort), 0.0)
    scale = np.maximum(np.maximum(np.abs(ours), np.abs(fort)), floor)
    rel = np.where(both, d / scale, 0.0)
    k = np.unravel_index(np.argmax(rel), rel.shape)
    return (float(d.max()), float(rel.max()),
            (k[0] + ilo, k[1] + jlo), cov_mismatch, int(both.sum()))


def _xyz(lon, lat):
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fort-dir", required=True)
    ap.add_argument("--ours", required=True)
    args = ap.parse_args(argv)

    fatal = []
    fort = load_fort_dir(Path(args.fort_dir))
    ours = np.load(args.ours, allow_pickle=False)
    n = int(ours["n"])
    ng = int(ours["ng"])
    lo_m = 1 - ng                 # model-lattice numpy origin (Fortran)
    lo_p = -ng                    # p1 (ng+1) lattice origin = 1-(ng+1)
    if not fort:
        print("FATAL: no extvec_*.dat in", args.fort_dir)
        return 1
    print(f"loaded {len(fort)} fortran arrays; ours git_sha="
          f"{ours['git_sha']}")

    # ---- alignment gate 1: agrid under Rz(pi), tile-for-tile ----
    # (dx alone cannot distinguish a tile permutation — codex r1 P1-6)
    worst_ang = 0.0
    for t in range(1, 7):
        for key in (("S0", t, "aglon"), ("S0", t, "aglat")):
            if key not in fort:
                print(f"FATAL: missing fortran {key}")
                return 1
        flon, ilo, jlo = fort[("S0", t, "aglon")]
        flat, _, _ = fort[("S0", t, "aglat")]
        olon = ours[f"S0_t{t}_aglon"]
        olat = ours[f"S0_t{t}_aglat"]
        if olon.shape != flon.shape:
            print(f"FATAL: agrid shape t{t} ours {olon.shape} "
                  f"fort {flon.shape}")
            return 1
        po = _xyz(olon, olat)
        pf = _xyz(flon, flat)
        pf_rot = pf * np.array([-1.0, -1.0, 1.0])      # Rz(pi)
        ang = np.linalg.norm(po - pf_rot, axis=-1)
        worst_ang = max(worst_ang, float(ang.max()))
    print(f"ALIGNMENT GATE agrid Rz(pi): max |dxyz| {worst_ang:.3e}")
    if worst_ang > 1e-12:
        print("FATAL: tile mapping is not identity-under-Rz(pi) — "
              "comparison aborted")
        return 1

    # ---- alignment gate 2: dx scale (constants/radius mismatch) ----
    worst = 0.0
    for t in range(1, 7):
        fa, ilo, jlo = fort[("S0", t, "dx")]
        oa = ours[f"S0_t{t}_dx"]
        if oa.shape != fa.shape:
            print(f"FATAL: dx shape t{t} ours {oa.shape} fort {fa.shape}")
            return 1
        r = np.abs(oa - fa) / np.maximum(np.abs(fa), 1e-9)
        worst = max(worst, float(r.max()))
    print(f"ALIGNMENT GATE dx scale: max rel {worst:.3e}")
    if worst > 1e-11:
        print("FATAL: dx mismatch — wrong constants/radius in one "
              "side (GFDL R=6371.0e3 vs GFS/published R=6371.2e3?) — "
              "comparison aborted")
        return 1

    # ---- sign from S2 interior fit ----
    if ("S2", 1, "ull") not in fort or "S2_t1_ull" not in ours.files:
        print("FATAL: S2 ull t1 missing — cannot fit sign")
        return 1
    fa, ilo, jlo = fort[("S2", 1, "ull")]
    oa = ours["S2_t1_ull"]
    reg, _ = region_mask(fa.shape, ilo, jlo, n)
    both = (~is_unwritten_fort(fa)) & np.isfinite(oa) & (reg == 0)
    if not both.any():
        print("FATAL: empty S2 overlap — cannot fit sign")
        return 1
    dp = float(np.abs(oa[both] - fa[both]).mean())
    dm = float(np.abs(oa[both] + fa[both]).mean())
    sign = 1.0 if dp <= dm else -1.0
    print(f"SIGN (S2 interior): +1 residual {dp:.3e}, -1 residual "
          f"{dm:.3e} -> sign={sign:+.0f}")

    # ---- per-stage compare ----
    # (stage, name, sign, our (lo_i, lo_j), stag_i, stag_j, exclude)
    plan = [
        # a-matrix/sinsg5 fort window is is-1..ie+1: its corner region
        # is exactly the 4 corner-DIAGONAL cells where our sin5
        # recompute is disclosed-different (center_a_matrix docstring)
        ("S0", "a11", 1.0, (lo_m, lo_m), 0, 0, "corners"),
        ("S0", "a12", 1.0, (lo_m, lo_m), 0, 0, "corners"),
        ("S0", "a21", 1.0, (lo_m, lo_m), 0, 0, "corners"),
        ("S0", "a22", 1.0, (lo_m, lo_m), 0, 0, "corners"),
        ("S0", "dy", 1.0, (lo_m, lo_m), 1, 0, None),
        ("S0", "sinsg5", 1.0, (lo_m, lo_m), 0, 0, "corners"),
        ("S1", "uin", sign, (lo_m, lo_m), 0, 1, None),
        ("S1", "vin", sign, (lo_m, lo_m), 1, 0, None),
        ("S2", "ull", sign, (lo_m, lo_m), 0, 0, None),
        ("S2", "vll", sign, (lo_m, lo_m), 0, 0, None),
        ("S3", "ullp1", sign, (lo_p, lo_p), 0, 0, "corners"),
        ("S3", "vllp1", sign, (lo_p, lo_p), 0, 0, "corners"),
        ("S4", "ullp1", sign, (lo_p, lo_p), 0, 0, "corners"),
        ("S4", "vllp1", sign, (lo_p, lo_p), 0, 0, "corners"),
        ("S5", "ullp1", sign, (lo_p, lo_p), 0, 0, "ring4corners"),
        ("S5", "vllp1", sign, (lo_p, lo_p), 0, 0, "ring4corners"),
        # cubed_a2d projections (codex r1 P1-5): ours are inter-center
        # slots — rows at p1 centers (lo_p), columns at nodes whose
        # Fortran index j maps to col j - 2 + ng+1 => lo = 2-(ng+1)
        ("S5", "up1", sign, (lo_p, 1 - ng), 0, 1, "ring4corners"),
        ("S5", "vp1", sign, (1 - ng, lo_p), 1, 0, "ring4corners"),
        ("S6", "uin", sign, (lo_m, lo_m), 0, 1, None),
        ("S6", "vin", sign, (lo_m, lo_m), 1, 0, None),
    ]
    hdr = (f"{'stage':6s} {'name':7s} {'tile':4s} {'max_abs':>12s} "
           f"{'max_rel':>12s} {'argmax(i,j)':>14s} {'cov_mm':>7s} "
           f"{'ncmp':>8s}")
    print(hdr)
    stage_worst = {}
    ourkeys = {("S5", "up1"): "up1", ("S5", "vp1"): "vp1"}
    for st, nm, sg, (loi, loj), sti, stj, excl in plan:
        for t in range(1, 7):
            key = (st, t, nm)
            onm = ourkeys.get((st, nm), nm)
            okey = f"{st}_t{t}_{onm}"
            if key not in fort or okey not in ours.files:
                print(f"{st:6s} {nm:7s} t{t}   MISSING "
                      f"(fort={key in fort} ours={okey in ours.files})")
                fatal.append(f"missing {st} {nm} t{t}")
                continue
            fa, ilo, jlo = fort[key]
            cr = overlap_crop(ours[okey], loi, loj, fa, ilo, jlo)
            if cr is None:
                print(f"{st:6s} {nm:7s} t{t}   NO-OVERLAP")
                fatal.append(f"no-overlap {st} {nm} t{t}")
                continue
            oc, fc, i0, j0 = cr
            r = compare(oc, fc, i0, j0, n, sg, sti, stj, excl)
            if r[0] == "EMPTY":
                print(f"{st:6s} {nm:7s} t{t}   EMPTY-OVERLAP")
                fatal.append(f"empty {st} {nm} t{t}")
                continue
            mabs, mrel, arg, cov, ncmp = r
            print(f"{st:6s} {nm:7s} t{t}  {mabs:12.4e} {mrel:12.4e} "
                  f"{str(arg):>14s} {cov:7d} {ncmp:8d}")
            if cov:
                fatal.append(f"coverage-mismatch {st} {nm} t{t} ({cov})")
            w = stage_worst.setdefault(st, [0.0, 0.0])
            w[0] = max(w[0], mabs)
            w[1] = max(w[1], mrel)

    print("\n=== per-stage worst (abs, rel) ===")
    for st in sorted(stage_worst):
        print(f"  {st}: abs {stage_worst[st][0]:.4e}  "
              f"rel {stage_worst[st][1]:.4e}")
    print("\nREADING: static S0 should sit at ~1e-12 rel; IC-formula "
          "FP noise between the two codes is ~1e-13 rel.  Judge "
          "dynamic stages primarily on max_abs [m/s-scale]: the "
          "first stage whose max_abs jumps orders above its "
          "predecessor is the defect locus.  rel uses a 1e-6 floor "
          "and can look inflated where winds are ~0 (codex r1 P1-7).")
    if fatal:
        print("\nFATAL findings (broken/incomplete instrument — do "
              "NOT trust clean-looking rows):")
        for f in fatal:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
