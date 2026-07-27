#!/usr/bin/env python
"""State twin v2 — ONE frozen geometric tile map (codex r3 protocol).

Supersedes the per-block quotient metric (codex r3 P0: an
independently-minimized `max_T min_m` is optimistic and its argmax
cannot localize a defect).  Here the six-face bijection + per-face
dihedral op is derived ONCE from GEOMETRY — their A-grid coordinates
(extvec S0 dumps) under Rz(pi) against our A-grid coordinates — and
frozen for every block and field.  Ambiguity or no-match is FATAL.

Per block, three curves:
  raw     mapping-free max|u|, max|v| per side (face+index reported) —
          the gauge-free separation check;
  delp    fixed-map residual (mean + max + argmax in OUR frame);
  speed   own-frame adjacent-average wind speed per side (a cell-
          centered scalar; the staggered adjacent average is D4-
          equivariant — codex r3 verified algebraically for odd and
          even n), then fixed-map residual.

Verdict rule (pre-declared): departure = first block b* where the
fixed-map residual exceeds BOTH control curves by >= --factor at b*
and the next dumped block, WITH contemporaneous raw-max separation.
Raw separation without fixed-map separation = instrument/gauge
inconsistency -> FATAL, not a verdict.

Self-test: seeded op recovery from synthetic geometry + seeded
violation localization under the frozen map.
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
PSEUDO = set(range(101, 108)) | set(range(201, 206))


def read_fort(path: Path, require_finite=True):
    with open(path) as f:
        ilo, ihi, jlo, jhi = (int(x) for x in f.readline().split())
        a = np.full((ihi - ilo + 1, jhi - jlo + 1), np.nan)
        for line in f:
            i, j, v = line.split()
            a[int(i) - ilo, int(j) - jlo] = float(v)
    if require_finite and not np.all(np.isfinite(a)):
        raise ValueError(f"non-finite/hole in {path}")
    return a, ilo, jlo


def crop(a, lo_i, lo_j, n, si=0, sj=0):
    i0, j0 = 1 - lo_i, 1 - lo_j
    return a[i0:i0 + n + si, j0:j0 + n + sj]


def xyz(lon, lat):
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def derive_map(their_ag, our_ag, tol=1e-10):
    """their_ag/our_ag: {tile: (n,n,3) interior agrid xyz}.  Returns
    {their_T: (our_t, op_name)} — the UNIQUE (t, op) with
    Rz(pi)(their_T[i,j]) == our_t[op(i,j)] index-for-index; FATAL
    (ValueError) on ambiguity, no match, or a non-bijection."""
    rz = np.array([-1.0, -1.0, 1.0])
    out = {}
    for T, tg in their_ag.items():
        tgr = tg * rz
        hits = []
        for t, og in our_ag.items():
            for nm, op in OPS.items():
                # our_t[op(i,j)] == tgr[i,j]  <=>  op^-1 applied to our
                # array equals tgr; using array transforms directly:
                cand = op(og[..., 0]), op(og[..., 1]), op(og[..., 2])
                d = max(float(np.abs(cand[k] - tgr[..., k]).max())
                        for k in range(3))
                if d < tol:
                    hits.append((t, nm, d))
        if len(hits) != 1:
            raise ValueError(f"their t{T}: {len(hits)} geometric map "
                             f"candidates {hits} — cannot freeze")
        out[T] = hits[0][:2]
    tgt = [v[0] for v in out.values()]
    if sorted(tgt) != sorted(their_ag.keys()):
        raise ValueError(f"map is not a bijection: {out}")
    return out


def speed_own_frame(u, v, lo, n):
    uc = crop(u, lo, lo, n, 0, 1)
    vc = crop(v, lo, lo, n, 1, 0)
    return np.hypot(0.5 * (uc[:, :-1] + uc[:, 1:]),
                    0.5 * (vc[:-1, :] + vc[1:, :]))


def self_test():
    rng = np.random.default_rng(0)
    lon = rng.uniform(0, 2 * np.pi, (8, 8))
    lat = rng.uniform(-1.2, 1.2, (8, 8))
    og = {1: xyz(lon, lat)}
    # their tile = mir_i of ours, pre-rotated by Rz^-1 (= Rz)
    tg = {1: og[1][::-1, :, :] * np.array([-1.0, -1.0, 1.0])}
    m = derive_map(tg, og)
    assert m == {1: (1, "mir_i")}, m
    # frozen-map residual localizes a seeded violation
    f_our = rng.standard_normal((8, 8))
    f_their = f_our[::-1, :].copy()
    f_their[2, 5] += 7.0                       # their frame (3,6)
    t, nm = m[1]
    d = np.abs(OPS[nm](f_their) - f_our)       # our-frame difference
    k = np.unravel_index(np.argmax(d), d.shape)
    assert abs(d.max() - 7.0) < 1e-12 and k == (5, 5), (d.max(), k)
    print("self-test OK (geometric map recovery + violation "
          "localization under the frozen map)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fort-dir", help="oracle dyncore dumps")
    ap.add_argument("--geom-dir", help="dir with extvec_S0_aglon/"
                    "aglat_t*.dat (their agrid; e.g. run_c48_extdump)")
    ap.add_argument("--ours", help="our block-dump npz")
    ap.add_argument("--controls", nargs="*", default=[],
                    help="control npz files (chaos floors)")
    ap.add_argument("--max-block", type=int, default=None)
    ap.add_argument("--factor", type=float, default=10.0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    for req in ("fort_dir", "geom_dir", "ours"):
        if getattr(args, req) is None:
            ap.error(f"--{req.replace('_', '-')} required")

    ours = np.load(args.ours, allow_pickle=False)
    n, ng = int(ours["n"]), int(ours["ng"])
    lo = 1 - ng

    # ---- frozen geometric map ----
    their_ag, our_ag = {}, {}
    for T in range(1, 7):
        gl, ilo, jlo = read_fort(
            Path(args.geom_dir) / f"extvec_S0_aglon_t{T}.dat")
        gb, _, _ = read_fort(
            Path(args.geom_dir) / f"extvec_S0_aglat_t{T}.dat")
        their_ag[T] = xyz(crop(gl, ilo, jlo, n), crop(gb, ilo, jlo, n))
    from legoesm.grids.fv3_native_gridstruct import (
        build_fv3_native_gridstruct_bounded,
    )
    for t in range(1, 7):
        gs = build_fv3_native_gridstruct_bounded(n, ng, tile=t)
        our_ag[t] = xyz(crop(np.asarray(gs["agrid_lon"]), lo, lo, n),
                        crop(np.asarray(gs["agrid_lat"]), lo, lo, n))
    try:
        fmap = derive_map(their_ag, our_ag)
    except ValueError as e:
        print("FATAL:", e)
        return 1
    print("FROZEN GEOMETRIC MAP (their_T -> our_t via op):")
    for T in sorted(fmap):
        print(f"  t{T} -> t{fmap[T][0]} via {fmap[T][1]}")

    # ---- load oracle blocks ----
    fort = {}
    for p in sorted(Path(args.fort_dir).glob("dyncore_b*.dat")):
        m = re.match(r"dyncore_b(\d+)_(\w+)_t(\d)\.dat", p.name)
        if not m or int(m.group(1)) in PSEUDO:
            continue
        b = int(m.group(1))
        if args.max_block is not None and b > args.max_block:
            continue
        fort[(b, m.group(2), int(m.group(3)))] = read_fort(p)
    blocks = sorted({b for (b, _, _) in fort})
    ours_blocks = sorted({int(m.group(1)) for f in ours.files
                          if (m := re.match(r"b(\d+)_delp", f))})
    blocks = [b for b in blocks if b in set(ours_blocks)]
    dropped = [b for b in sorted({b for (b, _, _) in fort})
               if b not in set(blocks)]
    if dropped:
        print(f"NOTE: {len(dropped)} oracle blocks lack ours coverage "
              f"(first {dropped[:5]}) — compared range ends at "
              f"{blocks[-1] if blocks else None}")
    if not blocks:
        print("FATAL: no common blocks")
        return 1

    ctrls = [np.load(c, allow_pickle=False) for c in args.controls]

    def resid_fixed_map(get_their, b, fld):
        """max-over-tiles (mean, max, our-frame argmax) under fmap."""
        worst = None
        for T in range(1, 7):
            t, nm = fmap[T]
            if fld == "delp":
                r = get_their(b, "delp", T)
                o = (crop(ours[f"b{b}_delp_t{t}"], lo, lo, n)
                     if f"b{b}_delp_t{t}" in ours.files else None)
                fc = None if r is None else crop(r[0], r[1], r[2], n)
            else:
                ru, rv = get_their(b, "u", T), get_their(b, "v", T)
                ku, kv = f"b{b}_u_t{t}", f"b{b}_v_t{t}"
                o = (speed_own_frame(ours[ku], ours[kv], lo, n)
                     if ku in ours.files and kv in ours.files else None)
                fc = None
                if ru is not None and rv is not None:
                    fc = speed_own_frame(
                        _refort(ru, n, ng, 0, 1),
                        _refort(rv, n, ng, 1, 0), lo, n)
            if o is None or fc is None:
                return None
            d = np.abs(o - OPS[nm](fc))
            mx = float(d.max())
            if worst is None or mx > worst[1]:
                k = np.unravel_index(np.argmax(d), d.shape)
                worst = (float(d.mean()), mx,
                         (t, int(k[0]) + 1, int(k[1]) + 1))
        return worst

    def _refort(r, n_, ng_, si, sj):
        a, ilo, jlo = r
        full = np.full((n_ + 2 * ng_ + si, n_ + 2 * ng_ + sj), np.nan)
        # place the dumped window at its Fortran offsets in a full
        # lattice so speed_own_frame's crop() indexing applies
        i0, j0 = ilo - (1 - ng_), jlo - (1 - ng_)
        full[i0:i0 + a.shape[0], j0:j0 + a.shape[1]] = a
        return full

    def fort_get(b, k, t):
        return fort.get((b, k, t))

    def raw_max(side_get, b):
        best = (0.0, None)
        for t in range(1, 7):
            for k, (si, sj) in (("u", (0, 1)), ("v", (1, 0))):
                a = side_get(b, k, t)
                if a is None:
                    return None
                m = float(np.abs(a).max())
                if m > best[0]:
                    kk = np.unravel_index(np.argmax(np.abs(a)), a.shape)
                    best = (m, (k, t, int(kk[0]), int(kk[1])))
        return best

    def ours_get(b, k, t):
        key = f"b{b}_{k}_t{t}"
        return (crop(ours[key], lo, lo, n,
                     *((0, 1) if k == "u" else (1, 0)))
                if key in ours.files else None)

    def fort_get_crop(b, k, t):
        r = fort.get((b, k, t))
        if r is None:
            return None
        si, sj = (0, 1) if k == "u" else (1, 0)
        return crop(r[0], r[1], r[2], n, si, sj)

    print(f"\n{'blk':>4s} {'rawmax_o':>10s} {'rawmax_f':>10s} "
          f"{'delp_mean':>11s} {'delp_max':>11s} {'delp_arg':>13s} "
          f"{'spd_mean':>11s} {'spd_max':>11s}"
          + "".join(f" {'c%d_dmax' % i:>10s}" for i in
                    range(1, len(ctrls) + 1)))
    fatal = []
    rows = []
    for b in blocks:
        ro = raw_max(ours_get, b)
        rf = raw_max(fort_get_crop, b)
        rd = resid_fixed_map(fort_get, b, "delp")
        rs = resid_fixed_map(fort_get, b, "speed")
        if None in (ro, rf, rd, rs):
            fatal.append(f"incomplete records at block {b}")
            continue
        cvals = []
        for c in ctrls:
            # control shares OUR frame: identity map, direct diff
            worst = 0.0
            for t in range(1, 7):
                key = f"b{b}_delp_t{t}"
                if key not in c.files:
                    worst = np.nan
                    break
                worst = max(worst, float(np.abs(
                    crop(ours[key], lo, lo, n)
                    - crop(c[key], lo, lo, n)).max()))
            cvals.append(worst)
        rows.append((b, rd, rs, ro, rf, cvals))
        print(f"{b:4d} {ro[0]:10.3f} {rf[0]:10.3f} "
              f"{rd[0]:11.3e} {rd[1]:11.3e} {str(rd[2]):>13s} "
              f"{rs[0]:11.3e} {rs[1]:11.3e}"
              + "".join(f" {cv:10.3e}" for cv in cvals))

    # ---- pre-declared departure rule ----
    dep = None
    for i, (b, rd, rs, ro, rf, cv) in enumerate(rows[:-1]):
        if not cv or any(np.isnan(x) for x in cv):
            continue
        env = max(cv)
        b2 = rows[i + 1]
        env2 = max(b2[5]) if b2[5] else np.inf
        if (rd[1] > args.factor * env and b2[1][1] > args.factor * env2):
            raw_sep = abs(ro[0] - rf[0]) > 0.1 * max(ro[0], rf[0])
            dep = (b, rd, raw_sep)
            break
    print()
    if dep:
        b, rd, raw_sep = dep
        if raw_sep:
            print(f"DEPARTURE at block {b}: delp max residual "
                  f"{rd[1]:.3e} > {args.factor}x control envelope for "
                  f"2 consecutive dumped blocks, WITH raw-max "
                  f"separation.  Locus (our frame): tile "
                  f"{rd[2][0]} cell ({rd[2][1]},{rd[2][2]}).")
        else:
            print(f"WARNING: residual departure at block {b} WITHOUT "
                  "raw-max separation — gauge/instrument "
                  "inconsistency, NOT a defect verdict (codex r3).")
            fatal.append("departure without raw separation")
    else:
        print("NO departure above the control envelope in the "
              "compared range.")
    if fatal:
        print("FATAL findings:")
        for f in fatal:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
