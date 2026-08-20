#!/usr/bin/env python
"""BARRIER-2 partner identification: measure, from data, which neighbour
value each model's B-grid blend actually used at every cube edge.

Follow-up to compare_dyncore_stages.py, which localised the substep-1
injection to S12_kee's EDGE strips (1.44e-07) while every input row it
could check (direct faces) sat at 1e-14.  kee's edge rows are exactly
the BARRIER-2-blended B-grid perimeter, so the suspect is the blend's
partner convention (which neighbour array / line / orientation / sign
is averaged in) on the edges the direct-face rows cannot see.

METHOD (label-safe: no face map, no sign derivation, no staggering
assumptions):

  oracle side: S10 dumps are the PRE-blend ubb/vbbtemp, S11 the POST.
    At every blended slot the implied partner is P = 2*S11 - S10_own.
    Search ALL candidates (5 neighbour tiles x {ubb, vbbtemp} x 4
    perimeter lines x 2 orientations x 2 signs) for the one matching P
    at ~1e-13.  The match IS the oracle's effective exchange
    convention, measured per (tile, edge).

  port side: run the port's own `average_shared_edge_bgrid` on
    SYNTHETIC uniquely-coded inputs (value encodes face/array/line/
    index), decode which partner each slot averaged in.  No physics.

  compare: translate the port's measured convention through the frozen
    face map and diff against the oracle's, edge by edge.  Any edge
    whose conventions differ names the formulation difference exactly.

  control: also recompute kee from the oracle's OWN pre/post fields
    with the port's assembly formula and diff against the oracle's
    S12 dump -- separates "blend differs" from "kee assembly differs".
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CMP = _load("compare_dyncore_stages")
N, NG, KM = CMP.N, CMP.NG, CMP.KM
NPX = N + 1


def bwin(arr):
    """Padded-or-not B array -> compute window (49, 49, km)."""
    return CMP.window(arr, "bscalar")


LINES = {  # perimeter lines of a (49,49) B window, each a (49, km) slab
    "W": lambda a: a[0, :, :],
    "E": lambda a: a[NPX - 1, :, :],
    "S": lambda a: a[:, 0, :],
    "N": lambda a: a[:, NPX - 1, :],
}


def implied_partner(pre, post, edge):
    """P = 2*post - pre along one blended perimeter line."""
    return 2.0 * LINES[edge](post) - LINES[edge](pre)


def identify(P, pres, self_tile):
    """Find (tile, array, line, orient, sign) whose pre-blend line
    matches P.  `pres` = {tile: {"ubb": arr, "vbbtemp": arr}}."""
    best = (np.inf, None)
    scale = float(np.abs(P).max()) or 1.0
    for t, d in pres.items():
        if t == self_tile:
            continue
        for nm, arr in d.items():
            for ln in LINES:
                cand0 = LINES[ln](arr)
                for orient, cand in (("+", cand0), ("-", cand0[::-1, :])):
                    for sgn in (1.0, -1.0):
                        r = float(np.abs(P - sgn * cand).max()) / scale
                        if r < best[0]:
                            best = (r, (t, nm, ln, orient,
                                        "+" if sgn > 0 else "-"))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True)
    args = ap.parse_args(argv)

    dumps = []
    for t in range(1, 7):
        f, _, certs = CMP.read_tile(args.dump_dir, t)
        dumps.append(f)
        for k, v in certs.items():
            if v != 0.0:
                raise SystemExit(f"CERT {k} != 0 on tile {t}; dumps "
                                 f"untrusted")

    pre = {t: {"ubb": bwin(dumps[t]["S10_dsw23_ubb"]),
               "vbbtemp": bwin(dumps[t]["S10_dsw23_vbbtemp"])}
           for t in range(6)}
    post = {t: {"ubb": bwin(dumps[t]["S11_b2_ubb"]),
                "vbbtemp": bwin(dumps[t]["S11_b2_vbbtemp"])}
            for t in range(6)}

    # ---------------- oracle convention, measured ----------------
    # Oracle blends: vbbtemp on S/N rows, ubb on W/E cols
    # (dyn_core.F90:989-997).
    print("ORACLE effective BARRIER-2 partner per (tile, array, edge)")
    print(f"{'tile':>4s} {'array':>8s} {'edge':>4s}   "
          f"{'partner (tile, array, line, orient, sign)':44s} "
          f"{'resid':>10s} {'blend|d|max':>11s}")
    oracle_conv = {}
    for t in range(6):
        for nm, edges in (("vbbtemp", ("S", "N")), ("ubb", ("W", "E"))):
            for e in edges:
                moved = float(np.abs(LINES[e](post[t][nm])
                                     - LINES[e](pre[t][nm])).max())
                P = implied_partner(pre[t][nm], post[t][nm], e)
                r, ident = identify(P, pre, t)
                oracle_conv[(t + 1, nm, e)] = (ident, r)
                it, inm, iln, iori, isg = ident
                print(f"{t+1:4d} {nm:>8s} {e:>4s}   "
                      f"tile{it+1} {inm:>8s} {iln} {iori}{isg}"
                      f"{'':20s} {r:10.2e} {moved:11.3e}")
        # untouched arrays control: ubb S/N rows and vbbtemp W/E cols
        # must NOT move except at the two corner slots the other
        # array's blend shares... they are different arrays, so they
        # must not move at all.
        for nm, edges in (("ubb", ("S", "N")), ("vbbtemp", ("W", "E"))):
            for e in edges:
                interior = slice(1, NPX - 1)
                d = float(np.abs(LINES[e](post[t][nm])[interior]
                                 - LINES[e](pre[t][nm])[interior]).max())
                if d != 0.0:
                    print(f"  NOTE tile{t+1} {nm} {e}: non-blended line "
                          f"moved by {d:.3e} (interior slots)")

    # ---------------- port convention, decoded ----------------
    # Unique integer codes (exact in f64): value = (face+1)*1e6 +
    # arr_id*1e5 + i*100 + j.  A blended slot becomes
    # 0.5*(own + sign*partner); 2*post - own restores sign*partner
    # EXACTLY (0.5 and 2 are exact scalings of an integer sum).
    print("\nPORT effective BARRIER-2 partner per (face, array, edge)")
    from legoesm.grids.fv3_native_gridstruct import (  # noqa: E402
        average_shared_edge_bgrid,
    )

    def code(face, arr_id):
        i = np.arange(NPX)[:, None]
        j = np.arange(NPX)[None, :]
        return ((face + 1) * 1e6 + arr_id * 1e5 + i * 100.0 + j) * 1.0

    xb6 = [code(f, 1) for f in range(6)]
    yb6 = [code(f, 2) for f in range(6)]
    xb6_pre = [a.copy() for a in xb6]
    yb6_pre = [a.copy() for a in yb6]
    average_shared_edge_bgrid(xb6, yb6, N, NG)

    def decode(v):
        s = "+" if v >= 0 else "-"
        v = int(round(abs(v)))
        face = v // 10**6 - 1
        arr_id = (v % 10**6) // 10**5
        rem = v % 10**5
        return (face + 1, ("ubb" if arr_id == 1 else "vbbtemp"),
                rem // 100, rem % 100, s)

    for f in range(6):
        for nm, arrs, arrs_pre, edges in (
                ("ubb", xb6, xb6_pre, ("W", "E")),
                ("vbbtemp", yb6, yb6_pre, ("S", "N"))):
            for e in edges:
                lp = LINES[e](arrs[f][:, :, None])[:, 0]
                l0 = LINES[e](arrs_pre[f][:, :, None])[:, 0]
                partners = 2.0 * lp - l0
                ids = [decode(v) for v in partners]
                probes = {k: ids[k] for k in (0, 2, 24, 46, NPX - 1)}
                print(f"face{f+1} {nm:>8s} {e}: partner(slot->"
                      f"(face,arr,i,j,sign)) {probes}")

    # ---------------- kee assembly control ----------------
    # Recompute kee from the ORACLE's own pre/post fields with the
    # port's formula and diff against the oracle's S12 dump.
    print("\nKEE ASSEMBLY CONTROL (port formula on oracle fields vs "
          "oracle S12):")
    worst = 0.0
    for t in range(6):
        ubbtemp = bwin(dumps[t]["S10_dsw23_ubbtemp"])
        vbb = bwin(dumps[t]["S10_dsw23_vbb"])
        kee_port = 0.5 * (ubbtemp * post[t]["vbbtemp"]
                          + post[t]["ubb"] * vbb)
        kee_orc = bwin(dumps[t]["S12_kee_kee"])
        d = float(np.abs(kee_port - kee_orc).max())
        worst = max(worst, d)
        print(f"  tile{t+1}: max|d| = {d:.3e}  "
              f"(scale {float(np.abs(kee_orc).max()):.3g})")
    print(f"  worst: {worst:.3e} -- if ~0, the assembly is identical "
          f"and ONLY the blend inputs can differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
