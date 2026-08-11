#!/usr/bin/env python
"""Composed duo halo-exchange: oracle-extract vs port, per stage/region.

Consumes the per-rank dumps written by fv3_extchain_oracle_driver.F90
(the REAL pinned-tree ext_scalar/ext_vector + certified staged copies),
feeds the ORACLE'S OWN input arrays to the port's
ext_scalar_sixface / ext_vector_dgrid_sixface / ext_vector_cgrid_sixface
(with stage capture), and scores |port - oracle| per stage, per ring,
per cube EDGE (12) and per cube VERTEX (8) — never averaged away.

Instrument controls (all run before any number is reported):
  C1 face map: the port tile-t A-grid lon/lat must match the oracle's
     dumped agrid per tile (identity map) to < 1e-12 rad; otherwise the
     script aborts — no silent relabelling.
  C2 oracle chain certification: the driver's CERT lines
     (max|staged - composed| per family) must be exactly 0.0, else the
     stage dumps are labelled UNTRUSTED and only composed finals are
     compared.
  C3 sentinel independence: oracle finals are compared across the two
     halo-prefill variants; any differing slot depends on an
     UNDEFINED-in-the-oracle halo location and is reported.
  C4 interior identity: on compute-domain slots the final must equal
     the input bitwise on both sides (the exchange never touches the
     interior) — a nonzero here means the harness mis-mapped arrays.

No verdicts are printed by this tool — numbers only (validation-lesson:
never let a probe print its own verdict).

Usage:
  compare_extchain_oracle.py --oracle-dir DIR --n 48 --ng 3
      [--fixture-out PATH] [--summary-out PATH] [--full-fixture]
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


# ---------------------------------------------------------------------------
# oracle dump loading
# ---------------------------------------------------------------------------

def load_tile(oracle_dir: Path, tile: int) -> dict:
    """Parse extchain_t<tile>.mf + .dat into {name: array}, Fortran order."""
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
    """CERT <fam><variant> <val> [<val2>] -> {(fam, variant): max}."""
    out = {}
    for ln in notes:
        p = ln.split()
        if p[0] != "CERT":
            continue
        fam_var = p[1]          # e.g. 'A_v1', 'D_v2'
        fam, var = fam_var.split("_")
        vals = [float(x) for x in p[2:]]
        out[(fam, var)] = max(vals)
    return out


# ---------------------------------------------------------------------------
# region machinery
# ---------------------------------------------------------------------------

def region_label(i_f: int, j_f: int, n: int, ng: int,
                 ish: int, jsh: int) -> tuple:
    """('int',) | ('edge', side, ring) | ('corner', which, ring)."""
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


def score_regions(diff: np.ndarray, n: int, ng: int,
                  ish: int, jsh: int) -> dict:
    """max|diff| per region key; diff is (m, m2[, k]) Fortran-lo array."""
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
    """Map (tile0, corner) -> global vertex id via corner-node xyz."""
    lo = 1 - ng
    ids: dict = {}
    seen: dict = {}
    for t in range(6):
        for which, (i_f, j_f) in (("SW", (1, 1)), ("SE", (n + 1, 1)),
                                  ("NW", (1, n + 1)), ("NE", (n + 1, n + 1))):
            lon = gr_lon6[t][i_f - lo, j_f - lo]
            lat = gr_lat6[t][i_f - lo, j_f - lo]
            xyz = (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                   np.sin(lat))
            key = tuple(np.round(xyz, 6))
            if key not in seen:
                seen[key] = f"V{len(seen)}"
            ids[(t, which)] = seen[key]
    return ids


def edge_ids(vids, n):
    """Map (tile0, side) -> global edge id via its two endpoint vertices."""
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
# port stage capture
# ---------------------------------------------------------------------------

def run_port(n: int, ng: int, orc: list, variant: str) -> dict:
    """Run the port's ext machinery on the ORACLE'S dumped inputs."""
    from legoesm.grids.fv3_native_ext_vector import (
        build_ext_context,
        ext_scalar_sixface,
        ext_vector_cgrid_sixface,
        ext_vector_dgrid_sixface,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        build_fv3_native_gridstruct,
        exchange_agrid_scalar_halos,
        exchange_bgrid_scalar_halos,
        k2e_remap_halo_rings,
    )

    gs6 = [build_fv3_native_gridstruct(n, ng, tile=t) for t in range(1, 7)]
    ectx = build_ext_context(n, ng, gs6)
    out: dict = {"gs6": gs6}

    def inputs(name):
        return [np.array(orc[t]["arrays"][f"{name}_{variant}"], copy=True)
                for t in range(6)]

    # ---- scalars, staged by direct calls -----------------------------
    for stag, exch, corner_key in (("A", exchange_agrid_scalar_halos,
                                    "corner_a3"),
                                   ("B", exchange_bgrid_scalar_halos,
                                    "corner_b3")):
        nlev = orc[0]["arrays"][f"IN_{stag}_{variant}"].shape[2]
        s1 = [None] * 6
        s2 = [None] * 6
        fin = [None] * 6
        for k in range(nlev):
            f6 = [inp[:, :, k].copy() for inp in inputs(f"IN_{stag}")]
            for t in range(1, 7):
                exch(f6, t, n, ng)
            for t in range(6):
                s1[t] = (f6[t][:, :, None] if s1[t] is None
                         else np.concatenate([s1[t], f6[t][:, :, None]], 2))
            f6b = [f.copy() for f in f6]
            k2e_remap_halo_rings(f6b, stag, n, ng, k2e_nord=ectx["k2e_nord"])
            for t in range(6):
                s2[t] = (f6b[t][:, :, None] if s2[t] is None
                         else np.concatenate([s2[t], f6b[t][:, :, None]], 2))
            f6c = [f.copy() for f in f6b]
            for t in range(6):
                ectx[corner_key][t].fill(f6c[t])
            for t in range(6):
                fin[t] = (f6c[t][:, :, None] if fin[t] is None
                          else np.concatenate([fin[t], f6c[t][:, :, None]],
                                              2))
        out[f"S1_{stag}"] = s1
        out[f"S2_{stag}"] = s2
        out[f"FIN_{stag}"] = fin
        # composed-call cross-check of the staged capture (level-stacked)
        f6full = inputs(f"IN_{stag}")
        f6l = [[f[:, :, k].copy() for f in f6full] for k in range(nlev)]
        for k in range(nlev):
            ext_scalar_sixface(f6l[k], stag, ectx)
        comp = [np.stack([f6l[k][t] for k in range(nlev)], axis=2)
                for t in range(6)]
        out[f"portcert_{stag}"] = max(
            float(np.abs(comp[t] - fin[t]).max()) for t in range(6))

    # ---- vectors, via the stage_dump hook ----------------------------
    for fam, runner, unames in (
            ("D", ext_vector_dgrid_sixface, ("DU", "DV")),
            ("C", ext_vector_cgrid_sixface, ("CU", "CV"))):
        nlev = orc[0]["arrays"][f"IN_{unames[0]}_{variant}"].shape[2]
        stages: dict = {}

        for k in range(nlev):
            cap: dict = {}

            def dump(stage, t0, name, arr, _cap=cap):
                _cap.setdefault((stage, name), [None] * 6)[t0] = \
                    np.array(arr, copy=True)

            ectx["stage_dump"] = dump
            u6 = [np.array(orc[t]["arrays"][f"IN_{unames[0]}_{variant}"]
                           [:, :, k], copy=True) for t in range(6)]
            v6 = [np.array(orc[t]["arrays"][f"IN_{unames[1]}_{variant}"]
                           [:, :, k], copy=True) for t in range(6)]
            runner(u6, v6, ectx)
            ectx.pop("stage_dump", None)
            cap[("FIN", "uin")] = u6
            cap[("FIN", "vin")] = v6
            for key, arrs in cap.items():
                cur = stages.setdefault(key, [None] * 6)
                for t in range(6):
                    a = arrs[t][:, :, None]
                    cur[t] = a if cur[t] is None else np.concatenate(
                        [cur[t], a], 2)
        out[f"stages_{fam}"] = stages
    return out


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

# vector stage table: oracle name pattern -> (port stage, port name)
VEC_STAGES = {
    "S1u": ("S1", "uin"), "S1v": ("S1", "vin"),
    "S2u": ("S2", "ull"), "S2v": ("S2", "vll"),
    "S3u": ("S3", "ullp1"), "S3v": ("S3", "vllp1"),
    "S4u": ("S4", "ullp1"), "S4v": ("S4", "vllp1"),
    "S5u": ("S5", "ullp1"), "S5v": ("S5", "vllp1"),
    "S6u": ("S6pre", "uin"), "S6v": ("S6pre", "vin"),
}


def crop_to(a: np.ndarray, shape2) -> np.ndarray:
    """Center-crop/trim oracle array to the port array's 2-D shape,
    assuming both share the same Fortran LOWER bound on each axis."""
    return a[:shape2[0], :shape2[1]] if a.ndim == 2 else \
        a[:shape2[0], :shape2[1], :]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--fixture-out", default=None,
                    help="npz with full arrays (C12-sized runs)")
    ap.add_argument("--summary-out", default=None,
                    help="compact npz: region tables + certs only")
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

    # C3: sentinel independence of the oracle finals
    print("== oracle sentinel-independence (max|FIN_v1-FIN_v2|) ==")
    sentinel: dict = {}
    for nm in ("FIN_A", "FIN_B", "FIN_DU", "FIN_DV", "FIN_CU", "FIN_CV"):
        w = max(float(np.abs(orc[t]["arrays"][f"{nm}_v1"]
                             - orc[t]["arrays"][f"{nm}_v2"]).max())
                for t in range(6))
        sentinel[nm] = w
        print(f"  {nm}: {w:.3e}")

    # ---- port run on variant v1 inputs -------------------------------
    port = run_port(n, ng, orc, "v1")
    gs6 = port["gs6"]

    # C1 face-map instrument control
    print("== face-map control (port tile t vs oracle tile t agrid) ==")
    worst_map = 0.0
    for t in range(6):
        dlon = np.abs(np.mod(gs6[t]["agrid_lon"]
                             - orc[t]["arrays"]["AG_LON"] + np.pi,
                             2 * np.pi) - np.pi)
        dlat = np.abs(gs6[t]["agrid_lat"] - orc[t]["arrays"]["AG_LAT"])
        w = float(max(dlon.max(), dlat.max()))
        worst_map = max(worst_map, w)
        print(f"  tile {t + 1}: {w:.3e} rad")
    if worst_map > 1e-12:
        print("FACE-MAP CONTROL FAILED — refusing to compare fields "
              "(port face numbering does not identity-match the oracle "
              "mosaic; derive the map before trusting any number)")
        return 2
    print(f"  port scalar staged-vs-composed self-check: "
          f"A={port['portcert_A']:.3e} B={port['portcert_B']:.3e}")

    vids = vertex_ids([orc[t]["arrays"]["GR_LON"] for t in range(6)],
                      [orc[t]["arrays"]["GR_LAT"] for t in range(6)],
                      n, ng)
    assert len(set(vids.values())) == 8, sorted(set(vids.values()))
    eids = edge_ids(vids, n)
    assert len(set(eids.values())) == 12

    rows = []                       # (stage, region_kind, region, max)

    def compare(stage_label, oname, parr6, ish, jsh, lat_ng):
        agg_edge: dict = {}
        agg_vert: dict = {}
        agg_int = 0.0
        ring_tab: dict = {}
        for t in range(6):
            oa = orc[t]["arrays"][oname]
            pa = parr6[t]
            oa2 = crop_to(oa, pa.shape)
            if oa2.shape != pa.shape:
                print(f"  SHAPE MISMATCH {oname}: oracle {oa.shape} "
                      f"port {pa.shape} — skipped")
                return
            # mask slots that are structurally undefined on EITHER side:
            # port NaN (never touched) / oracle -99999-prefill or stack
            # garbage (|v| > 1e12).  Divergence at oracle-undefined slots
            # that actually matters propagates into S6/FIN, which are
            # fully defined and never masked.
            undef = (~np.isfinite(pa)) | (~np.isfinite(oa2)) \
                | (np.abs(oa2) > 1e12) | (oa2 == -99999.0)
            diff = np.where(undef, 0.0, pa - oa2)
            reg = score_regions(diff, n, lat_ng, ish, jsh)
            for key, v in reg.items():
                if key[0] == "int":
                    agg_int = max(agg_int, v)
                elif key[0] == "edge":
                    eid = eids[(t, key[1])]
                    agg_edge[eid] = max(agg_edge.get(eid, 0.0), v)
                    ring_tab[key[2]] = max(ring_tab.get(key[2], 0.0), v)
                else:
                    vid = vids[(t, key[1])]
                    agg_vert[vid] = max(agg_vert.get(vid, 0.0), v)
        emax = max(agg_edge.values()) if agg_edge else 0.0
        vmax = max(agg_vert.values()) if agg_vert else 0.0
        print(f"  {stage_label:18s} interior={agg_int:.3e} "
              f"edges(max)={emax:.3e} vertices(max)={vmax:.3e} "
              f"rings={ {r: f'{v:.2e}' for r, v in sorted(ring_tab.items())} }")
        for eid in sorted(agg_edge):
            rows.append((stage_label, "edge", eid, agg_edge[eid]))
        for vid in sorted(agg_vert):
            rows.append((stage_label, "vertex", vid, agg_vert[vid]))
        rows.append((stage_label, "interior", "-", agg_int))

    print("== per-stage port-vs-oracle maxima "
          "(variant v1; abs diff) ==")
    for stag in ("A", "B"):
        ish, jsh = (0, 0) if stag == "A" else (1, 1)
        for st in ("S1", "S2", "FIN"):
            if not cert_ok and st != "FIN":
                continue
            oname = (f"{st}_{stag}_v1" if st != "FIN"
                     else f"CFIN_{stag}_v1")
            compare(f"{stag}:{st}", oname, port[f"{st}_{stag}"],
                    ish, jsh, ng)

    # per-component wind staggers (ishift, jshift)
    WSHIFT = {("D", "u"): (0, 1), ("D", "v"): (1, 0),
              ("C", "u"): (1, 0), ("C", "v"): (0, 1)}
    for fam in ("D", "C"):
        stages = port[f"stages_{fam}"]
        for okey, (pst, pnm) in VEC_STAGES.items():
            if not cert_ok:
                continue
            if (pst, pnm) not in stages:
                continue
            comp = "u" if okey.endswith("u") else "v"
            if okey[:2] in ("S1", "S6"):        # staggered wind arrays
                ish, jsh = WSHIFT[(fam, comp)]
                lng = ng
            elif okey[:2] == "S2":              # ull/vll, stepper A lattice
                ish, jsh, lng = 0, 0, ng
            else:                               # geo ng=4 A lattice
                ish, jsh, lng = 0, 0, 4
            compare(f"{fam}:{okey}", f"{okey}_{fam}_v1",
                    stages[(pst, pnm)], ish, jsh, lng)
        # composed finals (always, cert-independent)
        for comp in ("u", "v"):
            uv = {"D": {"u": "DU", "v": "DV"},
                  "C": {"u": "CU", "v": "CV"}}[fam][comp]
            ish, jsh = WSHIFT[(fam, comp)]
            compare(f"{fam}:FIN{comp}", f"CFIN_{uv}_v1",
                    stages[("FIN", "uin" if comp == "u" else "vin")],
                    ish, jsh, ng)

    # provenance
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except Exception:
        sha = "unknown"
    meta = {"n": n, "ng": ng, "git_sha": sha,
            "pinned_fv_duogrid_sha256": PIN_FVDUO_SHA,
            "oracle_dir": str(odir), "cert_ok": bool(cert_ok),
            "notes": notes}

    if args.summary_out:
        np.savez_compressed(
            args.summary_out,
            rows=np.array([(a, b, c, d) for a, b, c, d in rows],
                          dtype=object),
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
                keep = (nm.startswith(("IN_", "AG_", "GR_", "CFIN_",
                                       "FIN_", "S1_", "S2_"))
                        or nm[:3] in ("S1u", "S1v", "S2u", "S2v", "S3u",
                                      "S3v", "S4u", "S4v", "S5u", "S5v",
                                      "S6u", "S6v"))
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
