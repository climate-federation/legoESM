#!/usr/bin/env python
"""Symmetry-quotient state twin: divergence modulo the config symmetry.

The 2026-07-27 mapping retraction: the do_schmidt oracle's tile
correspondence is a per-tile dihedral map, and the case-8 two-burst
config's own symmetry makes the exact representative AMBIGUOUS (their
t2 matches our-t2-mirror_i and our-t5-transpose at identical residual).
The honest divergence measure is therefore the MINIMUM residual over
all (our-tile x 8 dihedral ops) candidates — the quotient by the
symmetry group.  Applied to two mapping-safe SCALAR fields per block:

  delp   — the mass field directly;
  speed  — A-centered wind speed hypot(mean-adjacent u, mean-adjacent
           v), component-free so no covariant transform rules enter
           (the twice-failed hand-derivation trap).

For --control (a second runner npz, e.g. the burst-perturbed chaos
run) the same quotient metric gives the floor; a real defect =
ours-vs-oracle departing the ours-vs-control curve by orders at some
block, with argmax cells localizing it.

Self-test: --self-test seeds a known op + a known violation and must
recover both (instrument-validation mandate).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

OPS = {
    "ident": lambda a: a, "mir_i": lambda a: a[::-1, :],
    "mir_j": lambda a: a[:, ::-1], "rot180": lambda a: a[::-1, ::-1],
    "transp": lambda a: a.T, "anti_T": lambda a: a[::-1, ::-1].T,
    "rot90": lambda a: np.rot90(a), "rot270": lambda a: np.rot90(a, 3),
}


def read_fort(path: Path):
    with open(path) as f:
        ilo, ihi, jlo, jhi = (int(x) for x in f.readline().split())
        a = np.full((ihi - ilo + 1, jhi - jlo + 1), np.nan)
        for line in f:
            i, j, v = line.split()
            a[int(i) - ilo, int(j) - jlo] = float(v)
    return a, ilo, jlo


def crop(a, lo_i, lo_j, n, si=0, sj=0):
    i0, j0 = 1 - lo_i, 1 - lo_j
    return a[i0:i0 + n + si, j0:j0 + n + sj]


def speed_field(u, v, lo, n):
    """A-centered wind speed from D-grid u/v compute+1 windows."""
    uc = crop(u, lo, lo, n, 0, 1)
    vc = crop(v, lo, lo, n, 1, 0)
    ua = 0.5 * (uc[:, :-1] + uc[:, 1:])
    va = 0.5 * (vc[:-1, :] + vc[1:, :])
    return np.hypot(ua, va)


def quotient_residual(oc_fields, fc_field):
    """min over (our tile, op) of mean|oc - op(fc)|; returns
    (mean_res, max_res_under_argmin, argmax_cell, tile, op)."""
    best = None
    for t, oc in enumerate(oc_fields, start=1):
        for nm, op in OPS.items():
            d = np.abs(oc - op(fc_field))
            m = float(d.mean())
            if best is None or m < best[0]:
                k = np.unravel_index(np.argmax(d), d.shape)
                best = (m, float(d.max()),
                        (int(k[0]) + 1, int(k[1]) + 1), t, nm)
    return best


def run(fort_get, ours, blocks, n, ng, label):
    lo = 1 - ng
    print(f"\n=== {label} (quotient residuals) ===")
    print(f"{'blk':>4s} {'field':>6s} {'mean_res':>12s} {'max_res':>12s}"
          f" {'arg(i,j)':>12s} {'their_t':>7s} {'map':>14s}")
    curve = {}
    for b in blocks:
        for fld in ("delp", "speed"):
            worst = (0.0, 0.0, None, None, None, None)
            for T in range(1, 7):
                if fld == "delp":
                    r = fort_get(b, "delp", T)
                    if r is None:
                        return None, f"missing ref b{b} delp t{T}"
                    fa, ilo, jlo = r
                    fc = crop(fa, ilo, jlo, n)
                    ocs = []
                    for t in range(1, 7):
                        key = f"b{b}_delp_t{t}"
                        if key not in ours.files:
                            return None, f"missing ours b{b} delp t{t}"
                        ocs.append(crop(ours[key], lo, lo, n))
                else:
                    ru = fort_get(b, "u", T)
                    rv = fort_get(b, "v", T)
                    if ru is None or rv is None:
                        return None, f"missing ref b{b} u/v t{T}"
                    fu, ilo, jlo = ru
                    fv, _, _ = rv
                    ua = 0.5 * (crop(fu, ilo, jlo, n, 0, 1)[:, :-1]
                                + crop(fu, ilo, jlo, n, 0, 1)[:, 1:])
                    va = 0.5 * (crop(fv, ilo, jlo, n, 1, 0)[:-1, :]
                                + crop(fv, ilo, jlo, n, 1, 0)[1:, :])
                    fc = np.hypot(ua, va)
                    ocs = []
                    for t in range(1, 7):
                        ku, kv = f"b{b}_u_t{t}", f"b{b}_v_t{t}"
                        if ku not in ours.files or kv not in ours.files:
                            return None, f"missing ours b{b} u/v t{t}"
                        ocs.append(speed_field(ours[ku], ours[kv],
                                               lo, n))
                q = quotient_residual(ocs, fc)
                if q[0] > worst[0]:
                    worst = (*q[:3], T, q[3], q[4])
            mres, mxres, arg, T, t, nm = worst
            print(f"{b:4d} {fld:>6s} {mres:12.4e} {mxres:12.4e} "
                  f"{str(arg):>12s} {T:7d} {'t' + str(t) + ':' + nm:>14s}")
            curve.setdefault(fld, []).append((b, mres, mxres))
    return curve, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fort-dir", default=None)
    ap.add_argument("--control", default=None)
    ap.add_argument("--ours", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if not args.self_test and args.ours is None:
        ap.error("--ours is required unless --self-test")

    if args.self_test:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((12, 12))
        q = quotient_residual([a.T.copy()], a)     # seeded transp
        assert q[0] < 1e-15 and q[4] == "transp", q
        b = a.T.copy()
        b[3, 4] += 5.0                             # seeded violation
        q = quotient_residual([b], a)
        assert abs(q[1] - 5.0) < 1e-12 and q[2] == (4, 5), q
        print("self-test OK (op recovery + seeded violation)")
        return 0

    ours = np.load(args.ours, allow_pickle=False)
    n, ng = int(ours["n"]), int(ours["ng"])
    if (args.fort_dir is None) == (args.control is None):
        print("FATAL: exactly one of --fort-dir/--control")
        return 1
    if args.fort_dir:
        fort = {}
        for p in sorted(Path(args.fort_dir).glob("dyncore_b*.dat")):
            m = re.match(r"dyncore_b(\d+)_(\w+)_t(\d)\.dat", p.name)
            if m and int(m.group(1)) < 100:
                fort[(int(m.group(1)), m.group(2),
                      int(m.group(3)))] = read_fort(p)
        blocks = sorted({b for (b, _, _) in fort})

        def fort_get(b, k, t):
            return fort.get((b, k, t))
        label = "OURS vs ORACLE"
    else:
        ctrl = np.load(args.control, allow_pickle=False)
        lo = 1 - ng
        blocks = sorted({int(m.group(1)) for f in ours.files
                         if (m := re.match(r"b(\d+)_delp", f))})

        def fort_get(b, k, t):
            key = f"b{b}_{k}_t{t}"
            return (ctrl[key], lo, lo) if key in ctrl.files else None
        label = "OURS vs CONTROL"
    if not blocks:
        print("FATAL: no blocks found")
        return 1
    curve, err = run(fort_get, ours, blocks, n, ng, label)
    if err:
        print("FATAL:", err)
        return 1
    print("\nREADING: the metric is min over (tile x dihedral op) — "
          "divergence modulo the config symmetry.  Judge ours-vs-"
          "oracle ONLY against the ours-vs-control curve at the same "
          "blocks; the departure block + argmax cells localize the "
          "real defect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
