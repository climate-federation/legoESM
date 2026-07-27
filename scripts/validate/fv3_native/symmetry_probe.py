#!/usr/bin/env python
"""Gauge-free internal-symmetry probe (no cross-model mapping at all).

The case-8 two-burst config is invariant under a discrete group: the
180-degree polar rotation Rz, the meridian-plane reflections through
90E (Mx: x->-x) and 0E (My: y->-y), and the equatorial reflection
(Mz: z->-z) — for the SCALAR delp field all four are exact analytic
symmetries, and the gnomonic cube + A-grid delp discretization is
invariant under all of them.  A faithful solver therefore keeps
delp's symmetry residual at FP-roundoff forever; growth of the
residual is a REAL symmetry-breaking defect in that run, measured
without any reference to the other model (kills the mapping-gauge
problem that forced two instrument retractions on 2026-07-27).

For each run (ours npz / oracle dump dir) and each symmetry G: find
all index maps (t,i,j) -> (t',i',j') with agrid(t') == G(agrid(t))
(unique per (G,t) or FATAL), then per block report
max |delp(s) - delp(G s)| + argmax.

Self-test: symmetric synthetic field -> 0; seeded asymmetry ->
localized.
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
    "transp": lambda a: np.transpose(a), "anti_T": lambda a: a[::-1, ::-1].T,
    "rot90": lambda a: np.rot90(a), "rot270": lambda a: np.rot90(a, 3),
}
GS = {
    "Rz180": np.array([-1.0, -1.0, 1.0]),
    "Mx": np.array([-1.0, 1.0, 1.0]),
    "My": np.array([1.0, -1.0, 1.0]),
    "Mz": np.array([1.0, 1.0, -1.0]),
}


def read_fort(path: Path):
    with open(path) as f:
        ilo, ihi, jlo, jhi = (int(x) for x in f.readline().split())
        a = np.full((ihi - ilo + 1, jhi - jlo + 1), np.nan)
        for line in f:
            i, j, v = line.split()
            a[int(i) - ilo, int(j) - jlo] = float(v)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"hole in {path}")
    return a, ilo, jlo


def crop(a, lo_i, lo_j, n):
    i0, j0 = 1 - lo_i, 1 - lo_j
    return a[i0:i0 + n, j0:j0 + n]


def xyz(lon, lat):
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def derive_selfmap(ag, gvec, tol=1e-10):
    """{t: (t', op)} with ag[t'] == diag(gvec) @ ag[t] index-for-index
    (after applying op to ag[t']); unique per t or ValueError."""
    out = {}
    for t, a in ag.items():
        img = a * gvec
        hits = []
        for t2, b in ag.items():
            for nm, op in OPS.items():
                d = max(float(np.abs(op(b[..., k]) - img[..., k]).max())
                        for k in range(3))
                if d < tol:
                    hits.append((t2, nm))
        if len(hits) != 1:
            raise ValueError(f"G-map t{t}: {len(hits)} hits {hits}")
        out[t] = hits[0]
    return out


def sym_residual(fields, smap):
    """fields: {t: (n,n) delp}; smap {t: (t', op)}.
    max over tiles of |f(t) - op(f(t'))| + argmax."""
    worst = None
    for t, (t2, nm) in smap.items():
        d = np.abs(fields[t] - OPS[nm](fields[t2]))
        mx = float(d.max())
        if worst is None or mx > worst[0]:
            k = np.unravel_index(np.argmax(d), d.shape)
            worst = (mx, (t, int(k[0]) + 1, int(k[1]) + 1))
    return worst


def self_test():
    rng = np.random.default_rng(1)
    lon = rng.uniform(0, 2 * np.pi, (6, 6))
    lat = rng.uniform(-1.0, 1.0, (6, 6))
    ag = {1: xyz(lon, lat), 2: xyz(lon, lat) * np.array([-1.0, 1, 1])}
    sm = derive_selfmap(ag, GS["Mx"])
    assert sm[1] == (2, "ident") and sm[2] == (1, "ident"), sm
    f = {1: rng.standard_normal((6, 6))}
    f[2] = f[1].copy()
    r = sym_residual(f, sm)
    assert r[0] == 0.0
    f[2][1, 4] += 3.0
    r = sym_residual(f, sm)
    assert abs(r[0] - 3.0) < 1e-12 and r[1][1:] == (2, 5), r
    print("self-test OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", help="runner block-dump npz")
    ap.add_argument("--fort-dir", help="oracle dyncore dump dir")
    ap.add_argument("--geom-dir",
                    help="their agrid dumps (extvec S0) for the "
                         "oracle side; ours uses the bounded builder")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.ours and not args.fort_dir:
        ap.error("need --ours and/or --fort-dir")

    sides = []
    if args.ours:
        ours = np.load(args.ours, allow_pickle=False)
        n, ng = int(ours["n"]), int(ours["ng"])
        lo = 1 - ng
        from legoesm.grids.fv3_native_gridstruct import (
            build_fv3_native_gridstruct_bounded,
        )
        ag = {}
        for t in range(1, 7):
            gs = build_fv3_native_gridstruct_bounded(n, ng, tile=t)
            ag[t] = xyz(crop(np.asarray(gs["agrid_lon"]), lo, lo, n),
                        crop(np.asarray(gs["agrid_lat"]), lo, lo, n))
        blocks = sorted({int(m.group(1)) for f in ours.files
                         if (m := re.match(r"b(\d+)_delp", f))
                         and not 101 <= int(m.group(1)) <= 205})

        def get(b, t):
            return crop(ours[f"b{b}_delp_t{t}"], lo, lo, n)
        sides.append(("OURS", ag, blocks, get))
    if args.fort_dir:
        if not args.geom_dir:
            ap.error("--geom-dir required with --fort-dir")
        fd = Path(args.fort_dir)
        gl0, ilo, jlo = read_fort(
            Path(args.geom_dir) / "extvec_S0_aglon_t1.dat")
        # n from ours if given, else from dump shape (data domain =
        # interior + 2*ng with ng=3)
        n = (int(np.load(args.ours)["n"]) if args.ours
             else gl0.shape[0] - 6)
        ag = {}
        for t in range(1, 7):
            gl, ilo, jlo = read_fort(
                Path(args.geom_dir) / f"extvec_S0_aglon_t{t}.dat")
            gb, _, _ = read_fort(
                Path(args.geom_dir) / f"extvec_S0_aglat_t{t}.dat")
            ag[t] = xyz(crop(gl, ilo, jlo, n), crop(gb, ilo, jlo, n))
        avail = {}
        for p in sorted(fd.glob("dyncore_b*_delp_t*.dat")):
            m = re.match(r"dyncore_b(\d+)_delp_t(\d)\.dat", p.name)
            if m and not (101 <= int(m.group(1)) <= 205):
                avail.setdefault(int(m.group(1)), {})[
                    int(m.group(2))] = p
        blocks = sorted(b for b, d in avail.items() if len(d) == 6)

        def get(b, t, avail=avail):
            a, ilo, jlo = read_fort(avail[b][t])
            return crop(a, ilo, jlo, n)
        sides.append(("ORACLE", ag, blocks, get))

    for label, ag, blocks, get in sides:
        smaps = {}
        for gname, gvec in GS.items():
            try:
                smaps[gname] = derive_selfmap(ag, gvec)
            except ValueError as e:
                print(f"{label}: {gname} NOT a grid symmetry ({e}) — "
                      "skipped")
        print(f"\n=== {label}: internal delp symmetry residuals ===")
        print("maps:", {g: {t: f"t{v[0]}:{v[1]}" for t, v in m.items()}
                        for g, m in smaps.items()})
        hdr = f"{'blk':>4s}" + "".join(
            f" {g + '_max':>12s} {g + '_arg':>13s}" for g in smaps)
        print(hdr)
        for b in blocks:
            fields = {t: get(b, t) for t in range(1, 7)}
            row = f"{b:4d}"
            for g, sm in smaps.items():
                mx, arg = sym_residual(fields, sm)
                row += f" {mx:12.4e} {str(arg):>13s}"
            print(row)
    print("\nREADING: a faithful solver keeps every listed residual "
          "at FP-roundoff for all time (the analytic config and the "
          "discretization are both invariant).  Growth = REAL "
          "symmetry-breaking in THAT run, localized by argmax — no "
          "cross-model mapping involved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
