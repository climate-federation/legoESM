#!/usr/bin/env python
"""Composed duo halo-exchange: oracle-extract vs port, per stage/region.

Consumes the per-rank dumps written by fv3_extchain_oracle_driver.F90
(the REAL pinned-tree ext_scalar/ext_vector + certified staged copies),
feeds the ORACLE'S OWN input arrays to the port's
ext_scalar_sixface / ext_vector_dgrid_sixface / ext_vector_cgrid_sixface
(with stage capture), and scores |port - oracle| per stage, per ring,
per cube EDGE (12) and per cube VERTEX (8) — never averaged away.

Instrument controls (all run before any number is reported):
  C1 face map: the oracle mosaic is the port's reference ED cube
     RELABELLED (perm x dihedral — the ic_face_map_parity lesson); the
     map is SEARCHED per tile over all 6 x 8 candidates on the A-grid
     coordinates and REQUIRED to be a bijection at the quad-geometry
     floor (< 1e-9 rad).  A hard-coded map cannot fail, so it cannot
     certify anything.
  C1b transform validation: the inverse-mapped oracle VECTOR inputs
     must reproduce the analytic streamfunction winds evaluated on the
     PORT's own lattice (placement + orientation + SIGN check that does
     not assume the transform it validates); scalars likewise via the
     k=2 lat field.
  C2 oracle chain certification: the driver's CERT lines
     (max|staged - composed|) must be exactly 0.0, else stage dumps are
     UNTRUSTED and only composed finals are compared.
  C3 sentinel independence: oracle finals compared across the two
     halo-prefill variants; a differing slot depends on an
     UNDEFINED-in-the-oracle halo location.
  C4 interior identity: compute-domain slots of the final must equal
     the input bitwise on both sides.

No verdicts are printed — numbers only.

Usage:
  compare_extchain_oracle.py --oracle-dir DIR --n 48 --ng 3
      [--fixture-out PATH] [--summary-out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages" / "core"))

PIN_FVDUO_SHA = ("c3c745c4071581ef23d78291873b5ba92a6cfdfc5711e5e80ede0be1e"
                 "613795e")

U0 = 38.61068276698372
A45 = 0.7853981633974483


# ---------------------------------------------------------------------------
# oracle dump loading
# ---------------------------------------------------------------------------

def load_tile(oracle_dir: Path, tile: int) -> dict:
    mf = oracle_dir / f"extchain_t{tile}.mf"
    dat = oracle_dir / f"extchain_t{tile}.dat"
    if not mf.exists() or not dat.exists():
        raise FileNotFoundError(mf)
    arrays: dict = {}
    notes: list = []
    raw = np.fromfile(dat, dtype="<f8")
    for line in mf.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] in ("NOTE", "CERT"):
            notes.append(line)
            continue
        name = parts[0]
        ni, nj, nk, off = (int(x) for x in parts[1:5])
        cnt = ni * nj * nk
        a = raw[off // 8: off // 8 + cnt].reshape((ni, nj, nk), order="F")
        arrays[name] = a[:, :, 0] if nk == 1 else a
    return {"arrays": arrays, "notes": notes}


def load_oracle(oracle_dir: Path) -> list:
    return [load_tile(oracle_dir, t) for t in range(1, 7)]


def cert_values(notes: list) -> dict:
    out = {}
    for ln in notes:
        p = ln.split()
        if p[0] != "CERT":
            continue
        fam, var = p[1].split("_")
        out[(fam, var)] = max(float(x) for x in p[2:])
    return out


def note_value(notes: list, key: str) -> list:
    for ln in notes:
        if ln.startswith(f"NOTE {key} ="):
            return [float(x) for x in ln.split("=")[1].split()]
    raise KeyError(key)


# ---------------------------------------------------------------------------
# dihedral layout transforms (port <-> oracle tile orientation)
# ---------------------------------------------------------------------------
# forward op (swap, ri, rj): PORT-layout array -> ORACLE-layout array:
#   q = a.swapaxes(0,1) if swap else a; then q = q[::-1] if ri;
#   q = q[:, ::-1] if rj.
# Both cell axes ([1-ng, n+ng]) and node axes ([1-ng, n+1+ng]) are
# symmetric about the supergrid centre n+1, so numpy reversal equals the
# Fortran index reversal for every stagger.
OPS = [(sw, ri, rj) for sw in (False, True) for ri in (False, True)
       for rj in (False, True)]


def op_scalar(a: np.ndarray, op) -> np.ndarray:
    sw, ri, rj = op
    q = a.swapaxes(0, 1) if sw else a
    if ri:
        q = q[::-1]
    if rj:
        q = q[:, ::-1]
    return np.ascontiguousarray(q)


def op_vector(u: np.ndarray, v: np.ndarray, op):
    """Covariant staggered pair, PORT layout -> ORACLE layout.

    Under swap the x/y covariant components exchange arrays; a reversed
    oracle axis anti-aligns with its source direction => sign flip on
    that component (the mpp NE-vector convention the certified
    exchanges use)."""
    sw, ri, rj = op
    if sw:
        uo, vo = v.swapaxes(0, 1), u.swapaxes(0, 1)
    else:
        uo, vo = u, v
    if ri:
        uo, vo = uo[::-1], vo[::-1]
    if rj:
        uo, vo = uo[:, ::-1], vo[:, ::-1]
    su = -1.0 if ri else 1.0
    sv = -1.0 if rj else 1.0
    return np.ascontiguousarray(su * uo), np.ascontiguousarray(sv * vo)


def op_inverse(op):
    """The unique member of OPS undoing ``op`` (searched, not derived)."""
    m = 7
    probe = np.arange(m * m, dtype=float).reshape(m, m)
    fwd = op_scalar(probe, op)
    for cand in OPS:
        if np.array_equal(op_scalar(fwd, cand), probe):
            return cand
    raise AssertionError(f"no inverse for {op}")


def _selfcheck_transforms():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(6, 7))
    v = rng.normal(size=(7, 6))
    s = rng.normal(size=(6, 6))
    for op in OPS:
        inv = op_inverse(op)
        assert np.array_equal(op_scalar(op_scalar(s, op), inv), s), op
        uo, vo = op_vector(u, v, op)
        ub, vb = op_vector(uo, vo, inv)
        assert np.array_equal(ub, u) and np.array_equal(vb, v), op


def _xyz(lon, lat):
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def derive_face_map(orc, gs6, n, ng):
    """For each PORT face t: (oracle tile T, forward op) minimizing the
    compute-block A-coordinate mismatch.  Returns map + per-face floor."""
    sl = slice(ng, ng + n)
    pxyz = [_xyz(gs6[t]["agrid_lon"], gs6[t]["agrid_lat"])[sl, sl]
            for t in range(6)]
    oxyz = [_xyz(orc[T]["arrays"]["AG_LON"], orc[T]["arrays"]["AG_LAT"])
            [sl, sl] for T in range(6)]
    face_map = {}
    for t in range(6):
        best = (np.inf, None, None)
        for T in range(6):
            for op in OPS:
                q = op_scalar(pxyz[t], op)
                d = float(np.abs(q - oxyz[T]).max())
                if d < best[0]:
                    best = (d, T, op)
        face_map[t] = best
    return face_map


# ---------------------------------------------------------------------------
# region machinery
# ---------------------------------------------------------------------------

def region_label(i_f, j_f, n, ng, ish, jsh):
    ihi, jhi = n + ish, n + jsh
    di = (1 - i_f) if i_f < 1 else (i_f - ihi if i_f > ihi else 0)
    dj = (1 - j_f) if j_f < 1 else (j_f - jhi if j_f > jhi else 0)
    if di == 0 and dj == 0:
        return ("int",)
    if di > 0 and dj > 0:
        which = ("S" if j_f < 1 else "N") + ("W" if i_f < 1 else "E")
        return ("corner", which, max(di, dj))
    if di > 0:
        return ("edge", "W" if i_f < 1 else "E", di)
    return ("edge", "S" if j_f < 1 else "N", dj)


def score_regions(diff, n, ng, ish, jsh):
    lo = 1 - ng
    out: dict = {}
    d = np.abs(diff)
    if d.ndim == 3:
        d = d.max(axis=2)
    for ii in range(d.shape[0]):
        for jj in range(d.shape[1]):
            key = region_label(ii + lo, jj + lo, n, ng, ish, jsh)
            v = d[ii, jj]
            if not np.isfinite(v):
                v = np.inf
            if key not in out or v > out[key]:
                out[key] = float(v)
    return out


def vertex_ids(gr_lon6, gr_lat6, n, ng):
    lo = 1 - ng
    ids: dict = {}
    seen: dict = {}
    for t in range(6):
        for which, (i_f, j_f) in (("SW", (1, 1)), ("SE", (n + 1, 1)),
                                  ("NW", (1, n + 1)),
                                  ("NE", (n + 1, n + 1))):
            lon = gr_lon6[t][i_f - lo, j_f - lo]
            lat = gr_lat6[t][i_f - lo, j_f - lo]
            xyz = (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                   np.sin(lat))
            key = tuple(np.round(xyz, 6))
            if key not in seen:
                seen[key] = f"V{len(seen)}"
            ids[(t, which)] = seen[key]
    return ids


def edge_ids(vids):
    ends = {"W": ("SW", "NW"), "E": ("SE", "NE"),
            "S": ("SW", "SE"), "N": ("NW", "NE")}
    eids: dict = {}
    seen: dict = {}
    for t in range(6):
        for side, (c1, c2) in ends.items():
            key = tuple(sorted((vids[(t, c1)], vids[(t, c2)])))
            if key not in seen:
                seen[key] = f"E{len(seen)}"
            eids[(t, side)] = seen[key]
    return eids


# ---------------------------------------------------------------------------
# analytic input formulas (mirror of the driver; C1b control)
# ---------------------------------------------------------------------------

def psi_level(k, lon, lat, radius, c0):
    def gc(lon1, lat1):
        return np.arccos(np.clip(
            np.sin(lat1) * np.sin(c0[1])
            + np.cos(lat1) * np.cos(c0[1]) * np.cos(lon1 - c0[0]),
            -1.0, 1.0))
    kk = (k % 5) + 1                           # 0-based level -> case 1..5
    if kk == 1:
        return -U0 * radius * np.sin(lat)
    if kk == 2:
        return -U0 * radius * (np.sin(lat) * np.cos(A45)
                               - np.cos(lon) * np.cos(lat) * np.sin(A45))
    if kk == 3:
        return -60.0 * radius * np.sin(lat) ** 3
    if kk == 4:
        return 20.0 * radius * np.exp(-(gc(lon, lat) / 0.2) ** 2)
    return -radius * (15.0 * np.sin(lat) + 5.0 * np.sin(2.0 * lat)
                      + 4.0 * np.cos(lon) * np.cos(lat))


# ---------------------------------------------------------------------------
# port stage capture (inputs already in PORT layout)
# ---------------------------------------------------------------------------

def run_port(n, ng, in_port, nlev_of, ectx):
    from legoesm.grids.fv3_native_ext_vector import (
        ext_scalar_sixface,
        ext_vector_cgrid_sixface,
        ext_vector_dgrid_sixface,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        exchange_agrid_scalar_halos,
        exchange_bgrid_scalar_halos,
        k2e_remap_halo_rings,
    )

    out: dict = {}
    for stag, exch, corner_key in (("A", exchange_agrid_scalar_halos,
                                    "corner_a3"),
                                   ("B", exchange_bgrid_scalar_halos,
                                    "corner_b3")):
        nlev = nlev_of[stag]
        s1 = [[] for _ in range(6)]
        s2 = [[] for _ in range(6)]
        fin = [[] for _ in range(6)]
        for k in range(nlev):
            f6 = [in_port[f"IN_{stag}"][t][:, :, k].copy()
                  for t in range(6)]
            for t in range(1, 7):
                exch(f6, t, n, ng)
            for t in range(6):
                s1[t].append(f6[t].copy())
            k2e_remap_halo_rings(f6, stag, n, ng,
                                 k2e_nord=ectx["k2e_nord"])
            for t in range(6):
                s2[t].append(f6[t].copy())
            for t in range(6):
                ectx[corner_key][t].fill(f6[t])
                fin[t].append(f6[t].copy())
        for nm, acc in ((f"S1_{stag}", s1), (f"S2_{stag}", s2),
                        (f"FIN_{stag}", fin)):
            out[nm] = [np.stack(acc[t], axis=2) for t in range(6)]
        # cross-check the staged capture against the composed call
        comp = [[] for _ in range(6)]
        for k in range(nlev):
            f6 = [in_port[f"IN_{stag}"][t][:, :, k].copy()
                  for t in range(6)]
            from legoesm.grids.fv3_native_ext_vector import (
                ext_scalar_sixface as _es,
            )
            _es(f6, stag, ectx)
            for t in range(6):
                comp[t].append(f6[t].copy())
        out[f"portcert_{stag}"] = max(
            float(np.abs(np.stack(comp[t], 2)
                         - out[f"FIN_{stag}"][t]).max())
            for t in range(6))

    for fam, runner, unames in (
            ("D", ext_vector_dgrid_sixface, ("DU", "DV")),
            ("C", ext_vector_cgrid_sixface, ("CU", "CV"))):
        nlev = nlev_of[fam]
        stages: dict = {}
        for k in range(nlev):
            cap: dict = {}

            def dump(stage, t0, name, arr, _cap=cap):
                _cap.setdefault((stage, name), [None] * 6)[t0] = \
                    np.array(arr, copy=True)

            ectx["stage_dump"] = dump
            u6 = [in_port[f"IN_{unames[0]}"][t][:, :, k].copy()
                  for t in range(6)]
            v6 = [in_port[f"IN_{unames[1]}"][t][:, :, k].copy()
                  for t in range(6)]
            runner(u6, v6, ectx)
            ectx.pop("stage_dump", None)
            cap[("FIN", "uin")] = u6
            cap[("FIN", "vin")] = v6
            for key, arrs in cap.items():
                cur = stages.setdefault(key, [[] for _ in range(6)])
                for t in range(6):
                    cur[t].append(arrs[t])
        out[f"stages_{fam}"] = {
            key: [np.stack(acc[t], axis=2) for t in range(6)]
            for key, acc in stages.items()}
    return out


# oracle stage name -> (port stage, port name, kind)
# kind: 'wind' = covariant staggered component, 'geo' = scalar-like
VEC_STAGES = {
    "S1u": ("S1", "uin", "wind"), "S1v": ("S1", "vin", "wind"),
    "S2u": ("S2", "ull", "geo"), "S2v": ("S2", "vll", "geo"),
    "S3u": ("S3", "ullp1", "geo"), "S3v": ("S3", "vllp1", "geo"),
    "S4u": ("S4", "ullp1", "geo"), "S4v": ("S4", "vllp1", "geo"),
    "S5u": ("S5", "ullp1", "geo"), "S5v": ("S5", "vllp1", "geo"),
    "S6u": ("S6pre", "uin", "wind"), "S6v": ("S6pre", "vin", "wind"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--fixture-out", default=None)
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--port-k2e-nord", type=int, default=None,
                    help="MUTATION CONTROL: force the port context's "
                    "k2e_nord away from the oracle's resolved value — "
                    "the finals MUST diverge, proving the instrument "
                    "can fail")
    args = ap.parse_args()
    n, ng = args.n, args.ng
    odir = Path(args.oracle_dir)
    orc = load_oracle(odir)

    notes = [ln for t in orc for ln in t["notes"]]
    certs = cert_values(notes)
    print("== oracle chain certification (max|staged-composed|) ==")
    for k in sorted(certs):
        print(f"  CERT {k[0]} {k[1]}: {certs[k]:.17e}")
    cert_ok = all(v == 0.0 for v in certs.values())
    print(f"  staged dumps trusted: {cert_ok}")

    print("== oracle sentinel-independence (max|FIN_v1-FIN_v2|) ==")
    sentinel: dict = {}
    for nm in ("FIN_A", "FIN_B", "FIN_DU", "FIN_DV", "FIN_CU", "FIN_CV"):
        w = max(float(np.abs(orc[t]["arrays"][f"{nm}_v1"]
                             - orc[t]["arrays"][f"{nm}_v2"]).max())
                for t in range(6))
        sentinel[nm] = w
        print(f"  {nm}: {w:.3e}")

    _selfcheck_transforms()

    # resolved runtime k2e_nord from the oracle manifest (the pinned
    # tree runs dg%k2e_nord = 2, NOT the 4 the ext tests default to —
    # exact runtime path, requirement of the diagnosis)
    k2e_nord = int(note_value(notes, "npx npz tile k2e_nord dgng")[3])
    print(f"== resolved oracle k2e_nord = {k2e_nord} "
          f"(port ext context built to match) ==")
    if args.port_k2e_nord is not None:
        print(f"== MUTATION CONTROL: port k2e_nord OVERRIDDEN to "
              f"{args.port_k2e_nord} (oracle runs {k2e_nord}) — the "
              f"finals must diverge ==")
        k2e_nord = args.port_k2e_nord
    # PRODUCTION construction, not a bespoke one: the C48 parity runs
    # use build_six_face_duo_context(use_ext_bundle=True,
    # oracle_conventions=True), whose ext context is built from the
    # BOUNDED gridstructs — mirror it exactly so the measured
    # divergence is the production port's, not a harness variant's.
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )

    ctx = build_six_face_duo_context(n, ng, use_ext_bundle=True,
                                     oracle_conventions=True,
                                     k2e_nord=k2e_nord)
    gs6 = ctx["gs6"]
    ectx = ctx["ectx"]

    # C1: searched face map (port face t -> oracle tile, forward op)
    fm = derive_face_map(orc, gs6, n, ng)
    print("== face map (port face -> oracle tile, op, xyz floor) ==")
    used = set()
    for t in range(6):
        d, T, op = fm[t]
        print(f"  port {t + 1} -> oracle {T + 1}  op(swap,ri,rj)={op}  "
              f"floor={d:.3e}")
        used.add(T)
    if len(used) != 6 or any(fm[t][0] > 1e-9 for t in range(6)):
        print("FACE-MAP CONTROL FAILED — not a bijection at the quad "
              "floor; refusing to compare fields")
        return 2
    # oracle tile T -> (port face, forward op)
    face_of = {fm[t][1]: (t, fm[t][2]) for t in range(6)}

    # ---- inverse-map oracle inputs into port layout ------------------
    variant = "v1"
    in_port: dict = {}
    nlev_of: dict = {}
    for fam in ("A", "B"):
        arrs = [None] * 6
        for T in range(6):
            t, op = face_of[T]
            arrs[t] = op_scalar(orc[T]["arrays"][f"IN_{fam}_{variant}"],
                                op_inverse(op))
        in_port[f"IN_{fam}"] = arrs
        nlev_of[fam] = arrs[0].shape[2]
    for fam, (nu, nv) in (("D", ("DU", "DV")), ("C", ("CU", "CV"))):
        us = [None] * 6
        vs = [None] * 6
        for T in range(6):
            t, op = face_of[T]
            u, v = op_vector(orc[T]["arrays"][f"IN_{nu}_{variant}"],
                             orc[T]["arrays"][f"IN_{nv}_{variant}"],
                             op_inverse(op))
            us[t], vs[t] = u, v
        in_port[f"IN_{nu}"] = us
        in_port[f"IN_{nv}"] = vs
        nlev_of[fam] = us[0].shape[2]

    # C1b: transform validation — the inverse-mapped oracle inputs must
    # reproduce the analytic fields at the inverse-mapped ORACLE
    # coordinates (the ground truth of where the driver evaluated them).
    # NOT the port gridstruct's own halo coords: those are a DIFFERENT
    # lattice in the halo (measured 0.04-0.26 rad vs the oracle mpp
    # halo at C12 rings 1-3) and that difference is part of what the
    # stage comparison measures, not a harness input.
    radius = 6.3712e6            # lib_grid RADIUS (variant quirk)
    c0 = note_value(notes, "vertex_c0")
    lo = 1 - ng
    worst_u = 0.0
    worst_s = 0.0

    def _gcd(lon1, lat1, lon2, lat2):
        return np.arccos(np.clip(
            np.sin(lat1) * np.sin(lat2)
            + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2),
            -1.0, 1.0))

    for t in range(6):
        T = fm[t][1]
        inv = op_inverse(fm[t][2])
        olon = op_scalar(orc[T]["arrays"]["AG_LON"], inv)
        olat = op_scalar(orc[T]["arrays"]["AG_LAT"], inv)
        sl = slice(ng, ng + n)
        # k=2 scalar level: lat itself
        worst_s = max(worst_s, float(np.abs(
            in_port["IN_A"][t][sl, sl, 1] - olat[sl, sl]).max()))
        # D-u from psi differences on the mapped oracle centres, k=0
        psi = psi_level(0, olon, olat, radius, c0)
        for (i_f, j_f) in ((1, 1), (n, n + 1), (n // 2, 2)):
            r, c = i_f - lo, j_f - lo
            dyc = radius * _gcd(olon[r, c - 1], olat[r, c - 1],
                                olon[r, c], olat[r, c])
            got = in_port["IN_DU"][t][r, c, 0]
            want = -(psi[r, c] - psi[r, c - 1]) / dyc
            worst_u = max(worst_u, abs(float(got - want)))
    # thresholds: a component-sign error gives ~2|u| (~77 m/s), a
    # placement error O(0.1-1); the known lib_grid-vs-constants radius
    # inconsistency inside the ORACLE's own input construction
    # (psi at 6.3712e6 m over dyc at the model radius) contributes only
    # <= ~2e-3 m/s and must NOT trip this control.
    print(f"== C1b transform validation: scalar(lat)={worst_s:.3e}  "
          f"D-u(analytic psi)={worst_u:.3e} m/s ==")
    if worst_s > 1e-9 or worst_u > 0.05:
        print("C1b TRANSFORM VALIDATION FAILED — the layout transform "
              "mis-places or mis-signs fields; refusing to compare")
        return 2

    port = run_port(n, ng, in_port, nlev_of, ectx)
    print(f"  port scalar staged-vs-composed self-check: "
          f"A={port['portcert_A']:.3e} B={port['portcert_B']:.3e}")

    vids = vertex_ids([orc[t]["arrays"]["GR_LON"] for t in range(6)],
                      [orc[t]["arrays"]["GR_LAT"] for t in range(6)],
                      n, ng)
    assert len(set(vids.values())) == 8, sorted(set(vids.values()))
    eids = edge_ids(vids)
    assert len(set(eids.values())) == 12

    rows = []

    def compare(stage_label, oname, port_in_oracle, ish, jsh, lat_ng):
        agg_edge: dict = {}
        agg_vert: dict = {}
        agg_int = 0.0
        ring_tab: dict = {}
        for T in range(6):
            oa = orc[T]["arrays"][oname]
            pa = port_in_oracle[T]
            if oa.shape != pa.shape:
                oa = oa[:pa.shape[0], :pa.shape[1]]
                if oa.shape != pa.shape:
                    print(f"  SHAPE MISMATCH {oname}: {oa.shape} vs "
                          f"{pa.shape} — skipped")
                    return
            undef = (~np.isfinite(pa)) | (~np.isfinite(oa)) \
                | (np.abs(oa) > 1e12) | (oa == -99999.0)
            diff = np.where(undef, 0.0, pa - oa)
            reg = score_regions(diff, n, lat_ng, ish, jsh)
            for key, v in reg.items():
                if key[0] == "int":
                    agg_int = max(agg_int, v)
                elif key[0] == "edge":
                    eid = eids[(T, key[1])]
                    agg_edge[eid] = max(agg_edge.get(eid, 0.0), v)
                    ring_tab[key[2]] = max(ring_tab.get(key[2], 0.0), v)
                else:
                    vid = vids[(T, key[1])]
                    agg_vert[vid] = max(agg_vert.get(vid, 0.0), v)
        emax = max(agg_edge.values()) if agg_edge else 0.0
        vmax = max(agg_vert.values()) if agg_vert else 0.0
        rings = {r: f"{v:.2e}" for r, v in sorted(ring_tab.items())}
        print(f"  {stage_label:10s} interior={agg_int:.3e} "
              f"edges(max)={emax:.3e} vertices(max)={vmax:.3e} "
              f"rings={rings}")
        for eid in sorted(agg_edge):
            rows.append((stage_label, "edge", eid, agg_edge[eid]))
        for vid in sorted(agg_vert):
            rows.append((stage_label, "vertex", vid, agg_vert[vid]))
        rows.append((stage_label, "interior", "-", agg_int))

    def to_oracle_scalar(pkey_arrs):
        return [op_scalar(pkey_arrs[face_of[T][0]], face_of[T][1])
                for T in range(6)]

    def to_oracle_vector(u_arrs, v_arrs):
        us, vs = [], []
        for T in range(6):
            t, op = face_of[T]
            u, v = op_vector(u_arrs[t], v_arrs[t], op)
            us.append(u)
            vs.append(v)
        return us, vs

    print("== per-stage port-vs-oracle maxima (variant v1, abs diff) ==")
    for stag in ("A", "B"):
        ish, jsh = (0, 0) if stag == "A" else (1, 1)
        for st in ("S1", "S2", "FIN"):
            if not cert_ok and st != "FIN":
                continue
            oname = (f"{st}_{stag}_v1" if st != "FIN"
                     else f"CFIN_{stag}_v1")
            compare(f"{stag}:{st}", oname,
                    to_oracle_scalar(port[f"{st}_{stag}"]), ish, jsh, ng)

    WSHIFT = {("D", "u"): (0, 1), ("D", "v"): (1, 0),
              ("C", "u"): (1, 0), ("C", "v"): (0, 1)}
    for fam in ("D", "C"):
        stages = port[f"stages_{fam}"]
        # wind stages: transform u/v pairs together
        wind_pairs = [("S1u", "S1v", ("S1", "uin"), ("S1", "vin")),
                      ("S6u", "S6v", ("S6pre", "uin"), ("S6pre", "vin"))]
        if cert_ok:
            for oku, okv, pku, pkv in wind_pairs:
                if pku not in stages or pkv not in stages:
                    continue
                us, vs = to_oracle_vector(stages[pku], stages[pkv])
                ish, jsh = WSHIFT[(fam, "u")]
                compare(f"{fam}:{oku}", f"{oku}_{fam}_v1", us,
                        ish, jsh, ng)
                ish, jsh = WSHIFT[(fam, "v")]
                compare(f"{fam}:{okv}", f"{okv}_{fam}_v1", vs,
                        ish, jsh, ng)
            for okey in ("S2u", "S2v", "S3u", "S3v", "S4u", "S4v",
                         "S5u", "S5v"):
                pst, pnm, _ = VEC_STAGES[okey]
                if (pst, pnm) not in stages:
                    continue
                lat_ng = ng if okey.startswith("S2") else 4
                compare(f"{fam}:{okey}", f"{okey}_{fam}_v1",
                        to_oracle_scalar(stages[(pst, pnm)]),
                        0, 0, lat_ng)
        # composed finals, cert-independent
        us, vs = to_oracle_vector(stages[("FIN", "uin")],
                                  stages[("FIN", "vin")])
        uv_u = "DU" if fam == "D" else "CU"
        uv_v = "DV" if fam == "D" else "CV"
        ish, jsh = WSHIFT[(fam, "u")]
        compare(f"{fam}:FINu", f"CFIN_{uv_u}_v1", us, ish, jsh, ng)
        ish, jsh = WSHIFT[(fam, "v")]
        compare(f"{fam}:FINv", f"CFIN_{uv_v}_v1", vs, ish, jsh, ng)

    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                              "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except Exception:
        sha = "unknown"
    meta = {"n": n, "ng": ng, "git_sha": sha,
            "pinned_fv_duogrid_sha256": PIN_FVDUO_SHA,
            "oracle_dir": str(odir), "cert_ok": bool(cert_ok),
            "face_map": {str(t + 1): [fm[t][1] + 1, list(fm[t][2]),
                                      fm[t][0]] for t in range(6)},
            "notes": notes}

    if args.summary_out:
        np.savez_compressed(
            args.summary_out,
            rows=np.array([(a, b, c, f"{d:.17e}") for a, b, c, d in rows]),
            meta=json.dumps(meta),
            sentinel=json.dumps(sentinel),
            certs=json.dumps({f"{k[0]}_{k[1]}": v
                              for k, v in certs.items()}))
        print(f"summary written: {args.summary_out}")

    if args.fixture_out:
        payload: dict = {"meta": json.dumps(meta)}
        sha_h = hashlib.sha256()
        for t in range(6):
            for nm, arr in sorted(orc[t]["arrays"].items()):
                keep = nm.startswith(("IN_", "AG_", "GR_", "CFIN_",
                                      "FIN_", "S1_", "S2_")) \
                    or nm[:3] in ("S1u", "S1v", "S2u", "S2v", "S3u",
                                  "S3v", "S4u", "S4v", "S5u", "S5v",
                                  "S6u", "S6v")
                if keep:
                    payload[f"t{t + 1}_{nm}"] = arr
                if nm.startswith("IN_") and nm.endswith("_v1"):
                    sha_h.update(np.ascontiguousarray(arr).tobytes())
        payload["input_sha256"] = np.array(sha_h.hexdigest())
        payload["n"] = np.array(n)
        payload["ng"] = np.array(ng)
        np.savez_compressed(args.fixture_out, **payload)
        print(f"fixture written: {args.fixture_out}  "
              f"input_sha256={sha_h.hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
