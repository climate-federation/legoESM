#!/usr/bin/env python
"""Per-family gridstruct-metric comparison: oracle runtime vs port builders.

THE QUESTION THIS ANSWERS (halo-lattice diagnosis, 2026-08-11): which
metric families, at which halo rings, differ between the oracle's OWN
runtime gridstruct (dumped by ``fv3_extchain_oracle_driver.F90`` from a
real ``fv_control_init`` on the duo deck) and the port's two builders --
the KINKED gs6 the stencil kernels consume today, and the EXTENDED
variant ``extend_gridstruct`` produces.

fv_grid_tools.F90:749-835 shows the duo oracle builds its model grid
FROM ``dg%b_pt`` with every mpp/fill_corners/get_symmetry step skipped,
so these dumps ARE what dyn_core multiplies by at halo cells.  The
partial-extension parity probe moved the one-step residual by only 0.7%
(9.7882e-06 -> 9.8553e-06), so the length/area families look
insufficient; this measurement names the binding families instead of
guessing further.

Regions per family: compute interior, compute edge (outermost compute
ring), halo rings 1..ng split into SIDE strips vs CORNER wedges.
The face map comes from the dumped AG/GR coordinates via the same
searched-bijection machinery as the extchain certificate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from compare_extchain_oracle import (  # noqa: E402
    derive_face_map,
    load_oracle,
    op_scalar,
)

# family -> (dump tag, port gs key, i-extra, j-extra)  [extents beyond
# the (m_a, m_a) cell plane; matches fv_arrays.F90 allocations]
FAMILIES = {
    "dx":     ("M_DX",     "dx",     0, 1),
    "dy":     ("M_DY",     "dy",     1, 0),
    "dxa":    ("M_DXA",    "dxa",    0, 0),
    "dya":    ("M_DYA",    "dya",    0, 0),
    "dxc":    ("M_DXC",    "dxc",    1, 0),
    "dyc":    ("M_DYC",    "dyc",    0, 1),
    "area":   ("M_AREA",   "area",   0, 0),
    "area_c": ("M_AREAC",  "area_c", 1, 1),
    "cosa_u": ("M_COSA_U", "cosa_u", 1, 0),
    "sina_u": ("M_SINA_U", "sina_u", 1, 0),
    "rsin_u": ("M_RSIN_U", "rsin_u", 1, 0),
    "cosa_v": ("M_COSA_V", "cosa_v", 0, 1),
    "rsin_v": ("M_RSIN_V", "rsin_v", 0, 1),
    "sina_v": ("M_SINA_V", "sina_v", 0, 1),
    "cosa_s": ("M_COSA_S", "cosa_s", 0, 0),
    "rsin2":  ("M_RSIN2",  "rsin2",  0, 0),
    # rsina is allocated COMPUTE-B only (fv_arrays.F90:1346) -- no halo
    # to probe; cosa AND sina are the padded B-plane siblings.
    "cosa":   ("M_COSA",   "cosa",   1, 1),
    "sina":   ("M_SINA",   "sina",   1, 1),
    "divg_u": ("M_DIVG_U", "divg_u", 0, 1),
    "divg_v": ("M_DIVG_V", "divg_v", 1, 0),
    "del6_u": ("M_DEL6_U", "del6_u", 0, 1),
    "del6_v": ("M_DEL6_V", "del6_v", 1, 0),
    # reciprocal + remaining static families (codex retro-review
    # finding 5: rdxc/rdyc feed p_grad_c, rdx/rdy the d_sw KE ranges,
    # rarea the vorticity, f0 the d_sw5 absolute vorticity -- all
    # consumed directly, none previously scored).
    "rarea":   ("M_RAREA",  "rarea",   0, 0),
    "rarea_c": ("M_RAREAC", "rarea_c", 1, 1),
    "rdx":     ("M_RDX",    "rdx",     0, 1),
    "rdy":     ("M_RDY",    "rdy",     1, 0),
    "rdxa":    ("M_RDXA",   "rdxa",    0, 0),
    "rdya":    ("M_RDYA",   "rdya",    0, 0),
    "rdxc":    ("M_RDXC",   "rdxc",    1, 0),
    "rdyc":    ("M_RDYC",   "rdyc",    0, 1),
    "f0":      ("M_F0",     "f0",      0, 0),
}
# 3-D staggered-sine family handled separately (9 slots).
FAMILIES_3D = {"sin_sg": ("M_SIN_SG", "sin_sg"),
               "cos_sg": ("M_COS_SG", "cos_sg")}

# Under a TRANSPOSE-type face map the port's x-staggered family lands on
# the oracle tile's y-staggered SIBLING (the same u<->v pairing the
# winds obey).  Self-paired families map to themselves.
SWAP_PARTNER = {
    "dx": "dy", "dy": "dx", "dxa": "dya", "dya": "dxa",
    "dxc": "dyc", "dyc": "dxc",
    "rdx": "rdy", "rdy": "rdx", "rdxa": "rdya", "rdya": "rdxa",
    "rdxc": "rdyc", "rdyc": "rdxc",
    "cosa_u": "cosa_v", "cosa_v": "cosa_u",
    "rsin_u": "rsin_v", "rsin_v": "rsin_u",
    "sina_u": "sina_v", "sina_v": "sina_u",
    "divg_u": "divg_v", "divg_v": "divg_u",
    "del6_u": "del6_v", "del6_v": "del6_u",
}

# Convention-sentinel magnitudes, PER FAMILY CLASS: upstream leaves 1e30
# in wedge slots it never initialises, and BOTH sides carry O(1e7-1e8)
# vertex-convention values in the trig families (STATE.md: port cosa
# vertices 5e7, rsina 1e8; upstream fill_ghost tiny-floor flavours).
# Cells where EITHER side exceeds the class threshold are counted
# separately, never scored in the headline.  Length/area families have
# REAL values up to ~4e10 (cell areas), so their threshold is only the
# 1e30 uninit sentinel.
_TRIG = {"cosa_u", "cosa_v", "rsin_u", "rsin_v", "sina_u", "sina_v",
         "cosa_s", "rsin2", "cosa", "sina", "divg_u", "divg_v",
         "del6_u", "del6_v", "f0"}
# Reciprocal families: the poisoned slots are 1/(+-big_number) ~=
# +-1e-30 -- a TINY sentinel, invisible to a magnitude ceiling.  Real
# reciprocals at C48 are >= 1/(4e10 m^2) ~ 2.5e-11, so cells where
# EITHER side is below 1e-20 are sentinel-class for these families.
_RECIP = {"rarea", "rarea_c", "rdx", "rdy", "rdxa", "rdya",
          "rdxc", "rdyc"}


def sent_mag(fam: str) -> float:
    return 1.0e6 if fam in _TRIG else 1.0e29


def sent_lo(fam: str) -> float:
    return 1.0e-20 if fam in _RECIP else 0.0


# cos-of-intersection-angle families are ORIENTATION-ODD under a
# reflection (cos(theta) -> cos(pi - theta) = -cos): the first
# instrument pass scored them at exactly 2*max|cosa| (1.00, 0.96, 0.97
# -- the pure-sign-flip signature).  Codex retro-review finding 1: the
# earlier PER-CELL fold min(|o-p|, |o+p|) accepted ANY spatially
# varying sign field (checkerboards, side-specific flips, wrong
# orientation masks).  Now the sign is resolved GLOBALLY per
# (face, family): pick s in {+1,-1} minimizing the face-wide masked
# max|o - s*p|, score |o - s*p| everywhere, and REPORT s per face --
# a spatially varying flip can no longer hide (no single s fixes it,
# the residual jumps to O(2*max|cosa|)).
_ODD = {"cosa", "cosa_u", "cosa_v", "cosa_s"}


def resolve_sign(orc2d, port2d, sent) -> tuple:
    """Global per-face sign: s minimizing masked max|o - s*p|.

    Returns (s, e_chosen, e_other, determined).  ``determined`` is
    False when the two candidates are within 10x of each other (a
    near-zero field cannot vote); such faces are excluded from the
    sign-coherence check rather than failing it spuriously.
    """
    live = ~sent
    if not live.any():
        return 1.0, 0.0, 0.0, False
    ep = float(np.abs(orc2d - port2d)[live].max())
    em = float(np.abs(orc2d + port2d)[live].max())
    s = 1.0 if ep <= em else -1.0
    e_chosen, e_other = (ep, em) if s > 0 else (em, ep)
    determined = e_other >= 10.0 * max(e_chosen, 1.0e-300)
    return s, e_chosen, e_other, determined


def sentinel_mask(orc2d, port2d, fam: str):
    hi = sent_mag(fam)
    lo = sent_lo(fam)
    sent = (np.abs(orc2d) >= hi) | (np.abs(port2d) >= hi)
    if lo > 0.0:
        sent |= (np.abs(orc2d) <= lo) | (np.abs(port2d) <= lo)
    return sent


def region_masks(n: int, ng: int, ish: int, jsh: int) -> dict:
    """Masks over an (n+2ng+ish, n+2ng+jsh) padded plane."""
    mi = n + 2 * ng + ish
    mj = n + 2 * ng + jsh
    ii = np.arange(mi)[:, None] * np.ones((1, mj), dtype=int)
    jj = np.ones((mi, 1), dtype=int) * np.arange(mj)[None, :]
    # distance OUTSIDE the compute box (0 inside)
    di = np.maximum(np.maximum(ng - ii, ii - (ng + n - 1 + ish)), 0)
    dj = np.maximum(np.maximum(ng - jj, jj - (ng + n - 1 + jsh)), 0)
    ring = np.maximum(di, dj)
    corner = (di > 0) & (dj > 0)
    inner = (ring == 0)
    # compute EDGE cells (inside, touching the boundary of the box) --
    # where the builders apply their panel-edge sg/metric SPECIALS, so
    # they must never contaminate the interior row (first instrument
    # pass scored "interior 0.49" that lived entirely on these rows).
    edge = inner & ((ii == ng) | (ii == ng + n - 1 + ish)
                    | (jj == ng) | (jj == ng + n - 1 + jsh))
    out = {"interior": inner & ~edge}
    out["compute_edge"] = edge
    for r in range(1, ng + 1):
        out[f"ring{r}_side"] = (ring == r) & ~corner
        out[f"ring{r}_corner"] = (ring == r) & corner
    return out


def score(orc2d, port2d, masks, sent, sign: float = 1.0) -> tuple:
    """Per-region max |o - sign*p| over non-sentinel cells + sent count."""
    d = np.abs(orc2d - sign * port2d)
    d = np.where(sent, 0.0, d)
    out = {k: float(d[m].max()) if m.any() else 0.0
           for k, m in masks.items()}
    return out, int(sent.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-dir", required=True,
                    help="directory with extchain_t{1..6}.dat dumps")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--ext-metrics", action="store_true",
                    help="build the P column with use_ext_metrics=True, "
                         "mirroring the parity gate's --ext-metrics flag "
                         "(the P column must score the exact gs6 the "
                         "gate integrates)")
    args = ap.parse_args()

    from legoesm.grids.fv3_native_gridstruct import (
        build_fv3_native_gridstruct,
        extend_gridstruct,
        FV3_OMEGA,
        FV3_RADIUS_M,
    )

    n, ng = args.n, args.ng
    orc = load_oracle(Path(args.oracle_dir))

    kink6 = [dict(build_fv3_native_gridstruct(n, ng, tile=t,
                                              radius=FV3_RADIUS_M,
                                              omega=FV3_OMEGA))
             for t in range(1, 7)]
    ext6 = [extend_gridstruct(kink6[t], n, ng, tile=t + 1)
            for t in range(6)]
    # P = the PRODUCTION parity lane's gridstructs: the
    # oracle_conventions=True bounded context, built with the SAME
    # context options as the full-step parity gate invocation (codex
    # retro-review finding 4: the P column used to hard-code the
    # default context while the gate takes --ext-metrics; P must score
    # the exact gs6 the gate integrates).
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    prod6 = build_six_face_duo_context(
        n, ng, use_ext_bundle=True,
        use_ext_metrics=args.ext_metrics,
        oracle_conventions=True)["gs6"]
    print(f"P context: use_ext_bundle=True "
          f"use_ext_metrics={args.ext_metrics} oracle_conventions=True")

    # face map from the dumped coordinates.  derive_face_map returns
    # {port_face: (dist, oracle_tile, op)} with op mapping PORT layout
    # -> ORACLE layout; enforce the bijection + geometry floor here
    # (the callee does not).
    fmap = derive_face_map(orc, kink6, n, ng)
    tiles = [fmap[t][1] for t in range(6)]
    if sorted(tiles) != list(range(6)):
        raise SystemExit(f"face map is not a bijection: {tiles}")
    worst_floor = max(fmap[t][0] for t in range(6))
    if worst_floor > 1.0e-12:
        raise SystemExit(
            f"INSTRUMENT CONTROL FAILED: face-map coordinate floor "
            f"{worst_floor:.3e} > 1e-12")
    print(f"face map bijection OK, coordinate floor {worst_floor:.3e}\n")

    print(f"per-family max |oracle - port| by region (C{n}, ng={ng});")
    print("K = kinked builder, E = extended builder\n")
    hdr = ["interior", "compute_edge"] + \
        [f"ring{r}_{k}" for r in range(1, ng + 1)
         for k in ("side", "corner")]
    print(f"{'family':10s} src " + "  ".join(f"{h:>12s}" for h in hdr))

    worst = {}
    # resolved global signs per (src, face, family) for the _ODD
    # families -- reported AND cross-checked for coherence below.
    signs = {}
    for fam, (tag, key, ish, jsh) in sorted(FAMILIES.items()):
        if key not in kink6[0]:
            print(f"{fam:10s}  --  (port gs6 has no {key!r}; skipped)")
            continue
        rows = {}
        for lbl, gs6 in (("K", kink6), ("E", ext6), ("P", prod6)):
            per = {h: 0.0 for h in hdr}
            nsent = 0
            fam_signs = []
            for pf in range(6):
                _dist, ot, op = fmap[pf]
                sw = op[0]
                # Under a transpose-type op the port's x-staggered
                # family lands on the oracle's y-staggered SIBLING
                # (u<->v pairing), so pick the partner's dump tag.
                fam_t = SWAP_PARTNER.get(fam, fam) if sw else fam
                tag_t = FAMILIES[fam_t][0] if fam_t in FAMILIES else tag
                o2 = orc[ot]["arrays"].get(tag_t)
                if o2 is None:
                    raise SystemExit(
                        f"dump tag {tag_t} missing from tile {ot + 1} -- "
                        f"regenerate the oracle dumps with the metric-"
                        f"family driver")
                p2 = op_scalar(np.asarray(gs6[pf][key], dtype=np.float64),
                               op)
                o2 = np.asarray(o2, dtype=np.float64)
                if o2.shape != p2.shape:
                    raise SystemExit(
                        f"{fam}: oracle[{fam_t}] shape {o2.shape} vs "
                        f"mapped port {p2.shape} (op={op})")
                masks = region_masks(n, ng,
                                     jsh if sw else ish,
                                     ish if sw else jsh)
                sent = sentinel_mask(o2, p2, fam)
                if fam in _ODD:
                    sgn, _ec, _eo, det = resolve_sign(o2, p2, sent)
                    fam_signs.append((pf, sgn, det))
                    signs[(lbl, pf, fam)] = (sgn, det)
                else:
                    sgn = 1.0
                s, ns = score(o2, p2, masks, sent, sign=sgn)
                nsent += ns
                for h in hdr:
                    per[h] = max(per[h], s[h])
            rows[lbl] = per
            sgn_note = ""
            if fam in _ODD:
                sgn_note = "   s=[" + " ".join(
                    ("+" if sg > 0 else "-") + ("" if det else "?")
                    for _pf, sg, det in fam_signs) + "]"
            print(f"{fam:10s}  {lbl}  "
                  + "  ".join(f"{per[h]:12.4e}" for h in hdr)
                  + f"   sent={nsent}" + sgn_note)
        worst[fam] = rows

    # ---- sign-coherence cross-check (codex retro finding 1) ----
    # All cos-of-intersection-angle families on one face flip TOGETHER
    # under a reflection-type face op; a per-face disagreement between
    # the resolved signs is exactly the side-specific/checkerboard
    # class the old per-cell fold could not see.  Indeterminate faces
    # ('?' above, near-zero field) abstain.
    coherent = True
    for lbl in ("K", "E", "P"):
        for pf in range(6):
            det_signs = {fam: signs[(lbl, pf, fam)][0]
                         for fam in sorted(_ODD)
                         if (lbl, pf, fam) in signs
                         and signs[(lbl, pf, fam)][1]}
            if len(set(det_signs.values())) > 1:
                coherent = False
                print(f"SIGN-COHERENCE FAIL src={lbl} face={pf + 1}: "
                      f"{ {k: ('+' if v > 0 else '-') for k, v in det_signs.items()} }")
    if coherent:
        print("\nsign coherence OK: every face's cosa-family global "
              "signs agree (indeterminate faces abstain)")

    # ---- cos^2+sin^2 magnitude coherence, per SIDE (no face map) ----
    # Validates that the sina magnitudes belong to the same angles as
    # the cosa magnitudes on EACH side independently; with the global-
    # sign report above this closes the finding-1 gap (a wrong-sign
    # sina cannot hide behind |sin| symmetry AND a folded cosa).
    print("\ncos^2+sin^2-1 residual (max over non-sentinel cells):")
    pairs = [("cosa_u", "sina_u"), ("cosa_v", "sina_v"),
             ("cosa", "sina")]
    for cf, sf in pairs:
        ctag, _k, _i, _j = FAMILIES[cf]
        stag = FAMILIES[sf][0]
        r_orc = 0.0
        for t in range(6):
            c2 = np.asarray(orc[t]["arrays"][ctag], dtype=np.float64)
            s2 = np.asarray(orc[t]["arrays"][stag], dtype=np.float64)
            live = (np.abs(c2) < sent_mag(cf)) & (np.abs(s2) < sent_mag(sf))
            if live.any():
                r_orc = max(r_orc, float(
                    np.abs(c2 * c2 + s2 * s2 - 1.0)[live].max()))
        print(f"  {cf}/{sf}: oracle {r_orc:.3e}", end="")
        for lbl, gs6 in (("K", kink6), ("E", ext6), ("P", prod6)):
            r = 0.0
            for t in range(6):
                c2 = np.asarray(gs6[t][cf], dtype=np.float64)
                s2 = np.asarray(gs6[t][sf], dtype=np.float64)
                live = ((np.abs(c2) < sent_mag(cf))
                        & (np.abs(s2) < sent_mag(sf)))
                if live.any():
                    r = max(r, float(
                        np.abs(c2 * c2 + s2 * s2 - 1.0)[live].max()))
            print(f"  {lbl} {r:.3e}", end="")
        print()

    print("\nNOTE sin_sg/cos_sg (9-slot) not scored here -- their slot "
          "semantics need the per-edge orientation map; a family that "
          "already discriminates above localizes the fix first.")
    return 0 if coherent else 1


if __name__ == "__main__":
    raise SystemExit(main())
