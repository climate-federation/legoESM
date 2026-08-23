#!/usr/bin/env python
"""Physics-coupling wind vectors: port vs the pinned-Fortran gridstruct.

``update_dwinds_phys`` rotates the A-grid physics tendency into a Cartesian
3-vector with ``gridstruct%vlon`` / ``vlat`` and projects it onto the D-grid
edges with ``gridstruct%es(:,:,:,1)`` / ``ew(:,:,:,2)``.  The port never carried
any of these -- the dycore works in grid-relative components -- so physics
coupling is their first consumer and they need a parity certificate before a
line of ``update_dwinds_phys`` is ported on top of them.

This scores the port's ``compute_fv3_native_wind_vectors`` against the four
fields dumped from the pinned tree's OWN runtime gridstruct
(``fv3_extchain_oracle_driver.F90`` -> ``M_VLON{1,2,3}`` / ``M_VLAT{1,2,3}`` /
``M_ES{1,2,3}`` / ``M_EW{1,2,3}``, the three Cartesian components each), through
the same searched face map the extchain metric certificate uses.

WHY THIS IS NOT ``compare_gs_metrics``.  These are not gridstruct scalar keys;
they are 3-component Cartesian families computed from ``grid_lon``/``agrid_lon``,
and their oracle windows differ from the metric families (``vlon``/``vlat`` are
dumped on ``is-2:ie+2`` per ``fv_arrays.F90:1396``, ``es``/``ew`` on the full
``isd:ied`` data domain).  The port builds on the full data domain, so ``es``/
``ew`` already match their oracle shape and ``vlon``/``vlat`` are trimmed one
ring to the narrower oracle window; then ``op_scalar`` (index-only; the
Cartesian component axis is Earth-fixed and never transformed) maps port layout
-> oracle layout exactly as for a scalar.

SIGN IS MEASURED, NOT ASSUMED.  ``vlon``/``vlat`` are absolute east/north and
must not flip under any face op; ``es``/``ew`` are edge directions tied to the
grid orientation and may flip under an axis reversal.  Rather than hand a sign
convention to the instrument, a single global sign per (face, family) is
resolved from the data (all three components jointly), reported, and required to
come back +1 for ``vlon``/``vlat`` (a determined -1 there is a finding, not
something to absorb).  Under a transposing op ``es`` and ``ew`` swap roles
(south-edge normal <-> west-edge normal), exactly the u<->v pairing the vector
families already use.

SCOPE: interior compute cells, C48 and C12, ng=3, do_schmidt identity, grid_type
0, bounded (duo) domain -- the configuration the extchain certificate covers.
The vectors' own halo rings are reported but not gated: the A-grid tendency halo
``update_dwinds_phys`` reads is filled by a separate domain update whose duo
behaviour is a distinct, still-open seam.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_extchain_oracle import (  # noqa: E402
    load_oracle,
    derive_face_map,
    op_scalar,
)

# family -> (oracle tag stem, port key, oracle window halo)
# window halo = rings the ORACLE dump carries around the compute box: vlon/vlat
# are dumped on is-2:ie+2 (halo 2), es/ew on the full isd:ied data domain (3).
# The staggered edge count is read from the oracle shape, not tabulated here.
FAMILIES = {
    "vlon": ("M_VLON", "vlon", 2),
    "vlat": ("M_VLAT", "vlat", 2),
    "es1":  ("M_ES",   "es1",  3),
    "ew2":  ("M_EW",   "ew2",  3),
}
# under a transposing op the port family lands on its partner's oracle tag
SWAP_PARTNER = {"es1": "ew2", "ew2": "es1"}
# vlon/vlat are absolute (Earth-frame) directions: expected sign is always +1
NO_FLIP = {"vlon", "vlat"}

# es/ew are oriented by the grid index direction, so the op's expected sign is
# the orientation half of the certified covariant NE-vector transform used for
# divg_u/divg_v (compare_extchain_oracle.op_vector): es plays the i/u role, ew
# the j/v role.  The sign is DERIVED from the op and the data's best fit must
# agree -- a negated port es/ew then fails instead of being absorbed.
PARITY_FLOOR = 1.0e-13  # compute-block gate; ~3x the observed C48 float64 floor


def edge_expected_sign(fam: str, op) -> float:
    sw, ri, rj = op
    if fam == "es1":                 # i-oriented; under swap it lands in ew slot
        return -1.0 if (rj if sw else ri) else 1.0
    return -1.0 if (ri if sw else rj) else 1.0   # ew2, j-oriented


def resolve_sign_vec(orc: np.ndarray, port: np.ndarray, live: np.ndarray):
    """One global sign over ALL cartesian components of a face.

    ``orc``/``port`` are (..., 3); ``live`` is the (...) validity mask.  Returns
    (sign, err_chosen, err_other, determined) with the same 10x-separation
    determinacy rule as ``compare_gs_metrics.resolve_sign``."""
    if not live.any():
        return 1.0, 0.0, 0.0, False
    o = orc[live]
    p = port[live]
    ep = float(np.abs(o - p).max())
    em = float(np.abs(o + p).max())
    s = 1.0 if ep <= em else -1.0
    e_ch, e_ot = (ep, em) if s > 0 else (em, ep)
    determined = e_ot >= 10.0 * max(e_ch, 1e-300) and e_ot > 1e-13
    return s, e_ch, e_ot, determined


def crop_to_oracle(port2d: np.ndarray, oshape) -> np.ndarray:
    """Trim the port plane's outer rings so it matches the oracle field.

    The builder writes on the FULL data domain (ng=3 halo), so es/ew already
    match their oracle shape while vlon/vlat are one ring wider than the oracle
    dump (is-2:ie+2, halo 2).  The trim is symmetric: (port - oracle)/2 rings off
    each side.  A non-symmetric or negative gap is refused loudly."""
    di = (port2d.shape[0] - oshape[0])
    dj = (port2d.shape[1] - oshape[1])
    if di < 0 or dj < 0 or di % 2 or dj % 2:
        raise SystemExit(
            f"port {port2d.shape} vs oracle {oshape}: gap ({di},{dj}) is not "
            f"a symmetric non-negative ring trim")
    di //= 2
    dj //= 2
    hi = port2d.shape[0] - di
    hj = port2d.shape[1] - dj
    return port2d[di:hi, dj:hj]


def verify_manifest(oracle_dir: Path, manifest: Path) -> int:
    """In-process provenance gate: the oracle dir's dumps must match the committed
    sha256 manifest for this resolution.  Lives HERE, not only in the sbatch, so
    the pass token cannot be emitted from an unpinned or fabricated dir even when
    the comparator is invoked directly.  Returns the number of files verified."""
    import hashlib
    base = oracle_dir.name  # run_c12 / run_c48
    want = {}
    for line in Path(manifest).read_text().splitlines():
        h, _, name = line.partition("  ")
        if name.startswith(base + "/"):
            want[name.split("/", 1)[1]] = h
    if not want:
        raise SystemExit(f"provenance: manifest {manifest} has no '{base}/' "
                         f"entries -- cannot verify this dump")
    for fn, exp in want.items():
        got = hashlib.sha256((oracle_dir / fn).read_bytes()).hexdigest()
        if got != exp:
            raise SystemExit(f"provenance FAIL: {base}/{fn} sha256 {got[:12]} "
                             f"!= committed {exp[:12]} -- refuse to certify")
    return len(want)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-dir", required=True,
                    help="dir with extchain_t{1..6}.dat/.mf (vector dump)")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--ng", type=int, default=3)
    args = ap.parse_args()
    n, ng = args.n, args.ng
    odir = Path(args.oracle_dir)

    # provenance is pinned to the GIT-COMMITTED manifest, resolved from this
    # script's own location -- not a caller-supplied path -- so no fabricated
    # manifest can be trusted.  The residual (a commit could change the committed
    # manifest itself) is the irreducible authority of any in-repo test: the
    # certificate means "true for the committed comparator + manifest at the
    # reviewed SHA", which no self-check can supersede.
    manifest = (Path(__file__).resolve().parents[3]
                / "tests" / "grids" / "fixtures"
                / "fv3_wind_vectors_oracle.sha256")
    if not manifest.exists():
        raise SystemExit(f"committed manifest not found at {manifest} -- "
                         f"refuse to certify without pinned provenance")
    nfiles = verify_manifest(odir, manifest)
    print(f"provenance OK: {nfiles} dump files match committed {manifest.name}")

    from legoesm.grids.fv3_native_metrics import (
        compute_fv3_native_wind_vectors,
    )
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )

    orc = load_oracle(odir)

    # provenance: refuse an unprovenanced dump.  The norm/orthogonality controls
    # cannot tell a genuine pinned-Fortran dump from a substituted or stale one
    # that happens to hold valid unit vectors, so require the physics-vector NOTE
    # lines only the patched Fortran driver (fv3_extchain_oracle_driver.F90)
    # writes.  The sbatch additionally records the dump hashes and asserts the
    # driver's own EXTCHAIN_DRIVER_DONE marker in run_out.txt.
    for t in range(6):
        notes = " ".join(orc[t].get("notes", []))
        if "VLON_BOUNDS" not in notes or "gridstruct%es/ew" not in notes:
            raise SystemExit(
                f"tile {t+1}: dump lacks the wind-vector provenance NOTE lines "
                f"(VLON_BOUNDS / gridstruct%es/ew) the patched Fortran driver "
                f"writes -- refuse to certify from an unprovenanced dump")

    # production gridstructs: update_dwinds_phys reads gridstruct%vlon on the
    # oracle_conventions bounded context, so build the vectors from THAT grid.
    gs6 = build_six_face_duo_context(
        n, ng, use_ext_bundle=True, use_ext_metrics=False,
        oracle_conventions=True)["gs6"]

    fmap = derive_face_map(orc, gs6, n, ng)
    tiles = [fmap[t][1] for t in range(6)]
    if sorted(tiles) != list(range(6)):
        raise SystemExit(f"face map is not a bijection: {tiles}")
    floor = max(fmap[t][0] for t in range(6))
    if floor > 1.0e-12:
        raise SystemExit(f"INSTRUMENT CONTROL FAILED: face-map floor "
                         f"{floor:.3e} > 1e-12")
    print(f"C{n} ng={ng}: face map bijection OK, coord floor {floor:.3e}")

    # ---- build port vectors per face ----
    port = []
    for t in range(6):
        g = gs6[t]
        pv = compute_fv3_native_wind_vectors(
            g["grid_lon"], g["grid_lat"], g["agrid_lon"], g["agrid_lat"])
        port.append(pv)

    # ---- instrument self-controls on the ORACLE (known answers) ----
    # The oracle ZERO-FILLS unwritten edge slots (not NaN), so a finite value is
    # NOT proof of a written cell.  Run the unit/orthogonality checks on the
    # STRICT interior mask (a subset of the interior|edge block the parity gate
    # scores) and require those cells finite.
    def _ovec_int(t, stem, halo):
        v = np.stack([np.asarray(orc[t]["arrays"][f"{stem}{c}"],
                                 dtype=np.float64) for c in (1, 2, 3)], -1)
        return v, _interior_mask(v.shape[:2], halo)
    ctl = {}
    for stem, lab, halo in (("M_VLON", "vlon", 2), ("M_VLAT", "vlat", 2),
                            ("M_ES", "es", 3), ("M_EW", "ew", 3)):
        worst_norm = 0.0
        for t in range(6):
            v, im = _ovec_int(t, stem, halo)
            vi = v[im]
            if not np.isfinite(vi).all():
                raise SystemExit(f"INSTRUMENT CONTROL FAILED: oracle {stem} has "
                                 f"a NaN in the scored interior (tile {t+1})")
            nrm = np.sqrt((vi ** 2).sum(-1))
            worst_norm = max(worst_norm, float(np.abs(nrm - 1.0).max()))
        ctl[lab] = worst_norm
    orth = 0.0
    for t in range(6):
        vlo, im = _ovec_int(t, "M_VLON", 2)
        vla, _ = _ovec_int(t, "M_VLAT", 2)
        orth = max(orth, float(np.abs((vlo * vla).sum(-1)[im]).max()))
    print(f"oracle self-control (interior): |v|-1 max "
          f"vlon={ctl['vlon']:.2e} vlat={ctl['vlat']:.2e} "
          f"es={ctl['es']:.2e} ew={ctl['ew']:.2e}; vlon.vlat={orth:.2e}")
    if max(ctl.values()) > 1e-10 or orth > 1e-10:
        raise SystemExit("INSTRUMENT CONTROL FAILED: oracle vectors are not "
                         "orthonormal on the scored interior -- dump/reader bug")

    # ---- per family, per face: map port -> oracle, score at the OP's sign ----
    # The sign is DERIVED from the face op (edge_expected_sign / +1 for the
    # absolute pair), not fit to the data; the data's best-fit sign is computed
    # too and REQUIRED to agree, so a negated port es/ew or vlon/vlat fails.
    print(f"\n{'family':7s} {'interior':>12s} {'edge':>12s} {'GATE':>12s} "
          f"{'ring(max)':>12s}  signs[f1..f6] (=agree !disagree ?abstain)")
    worst_gate = {}
    ok = True
    for fam, (stem, key, halo) in FAMILIES.items():
        per_int = per_edge = per_gate = per_ring = 0.0
        n_live_cmp = 0
        signs = []
        for pf in range(6):
            _d, ot, op = fmap[pf]
            sw = op[0]
            fam_o = SWAP_PARTNER[fam] if (sw and fam in SWAP_PARTNER) else fam
            stem_o = FAMILIES[fam_o][0]
            # port plane -> crop to THIS family's oracle shape -> op to oracle
            mapped = np.stack(
                [op_scalar(crop_to_oracle(port[pf][key][..., c - 1],
                                          _oshape_for(fam, orc, ot)), op)
                 for c in (1, 2, 3)], -1)                       # oracle layout
            ovec = np.stack([np.asarray(orc[ot]["arrays"][f"{stem_o}{c}"],
                                        dtype=np.float64) for c in (1, 2, 3)],
                            -1)
            if mapped.shape != ovec.shape:
                raise SystemExit(f"{fam} face{pf+1}: mapped {mapped.shape} vs "
                                 f"oracle[{stem_o}] {ovec.shape} (op={op})")
            oshape_o = ovec.shape[:2]
            imask = _interior_mask(oshape_o, halo)
            emask = _edge_mask(oshape_o, halo)
            rmask = _ring_mask(oshape_o, halo)
            # GATE THE FULL COMPUTE BLOCK, not just the interior: the wind
            # builder applies no panel-edge special (bounded domain -> one
            # formula covers the face), and update_dwinds_phys reads every
            # compute cell/edge including the boundary row/col.  The halo rings
            # stay reported-only (a separate tendency-halo seam).
            cmask = imask | emask
            # the whole compute block must be finite on BOTH sides -- a defect
            # must show as error, never be silently masked to zero by a NaN.
            if not (np.isfinite(mapped[cmask]).all()
                    and np.isfinite(ovec[cmask]).all()):
                raise SystemExit(
                    f"{fam} face{pf+1}: NaN in the gated compute block -- a "
                    f"defect would be masked to zero, not scored")
            if not cmask.any():
                raise SystemExit(f"{fam} face{pf+1}: empty compute mask")
            # expected sign from the op; data's best fit must agree
            s_exp = 1.0 if fam in NO_FLIP else edge_expected_sign(fam, op)
            live = np.isfinite(mapped).all(-1) & np.isfinite(ovec).all(-1)
            s_dat, _ec, _eo, det = resolve_sign_vec(ovec, mapped, live)
            signs.append(("+" if s_exp > 0 else "-")
                         + ("=" if (det and s_dat == s_exp)
                            else ("!" if det else "?")))
            if det and s_dat != s_exp:
                ok = False
                print(f"  SIGN FINDING {fam} face{pf+1}: data sign "
                      f"{s_dat:+.0f} != expected {s_exp:+.0f} from op {op}")
            diff = np.abs(ovec - s_exp * mapped).max(-1)
            n_live_cmp += int(cmask.sum())
            per_int = max(per_int, float(diff[imask].max()))
            per_edge = max(per_edge, float(diff[emask].max())
                           if emask.any() else per_edge)
            per_gate = max(per_gate, float(diff[cmask].max()))
            per_ring = max(per_ring, float(diff[rmask & live].max()
                                           if (rmask & live).any() else 0.0))
        # non-vacuous gate: the compute block must actually be mostly live.
        # Expected ~= 6 faces x (n-1)^2 cells; require at least half so a masking
        # regression cannot turn the gate green by emptying it.
        expect = 6 * (n - 1) * (n - 1) // 2
        if n_live_cmp < expect:
            raise SystemExit(
                f"{fam}: only {n_live_cmp} gated compute cells (< {expect}); "
                f"the gate is not scoring a full compute block")
        worst_gate[fam] = per_gate
        print(f"{fam:7s} {per_int:12.4e} {per_edge:12.4e} {per_gate:12.4e} "
              f"{per_ring:12.4e}  [{' '.join(signs)}]  cells={n_live_cmp}")
        if per_gate > PARITY_FLOOR:
            ok = False

    print(f"\nworst gated (interior+edge) over all families: "
          f"{max(worst_gate.values()):.4e} (gate {PARITY_FLOOR:.1e})")
    # provenance against the committed manifest is a hard precondition above
    # (verify_manifest raises on any mismatch), so reaching here means the dumps
    # are the pinned bytes; the token then reflects parity alone.
    tag = "OK" if ok else "FAIL"
    print(f"WIND_VECTOR_PARITY_{tag} n={n}")
    return 0 if ok else 1


def _oshape_for(fam, orc, ot):
    """Oracle shape the PORT plane must pad to BEFORE op_scalar.

    op_scalar transposes under a swap op, so the port pads to the pre-transpose
    shape whose op image equals the destination oracle field.  For a swapped
    es/ew that destination is the partner; the pre-image shape is the partner's
    shape transposed == this family's own oracle shape.  So the port always pads
    to its OWN family's oracle shape on this tile."""
    stem = FAMILIES[fam][0]
    return np.asarray(orc[ot]["arrays"][f"{stem}1"], dtype=np.float64).shape


def _interior_mask(oshape, halo):
    n_i, n_j = oshape[0] - 2 * halo, oshape[1] - 2 * halo
    m = np.zeros(oshape, dtype=bool)
    m[halo + 1:halo + n_i - 1, halo + 1:halo + n_j - 1] = True
    return m


def _edge_mask(oshape, halo):
    n_i, n_j = oshape[0] - 2 * halo, oshape[1] - 2 * halo
    inner = np.zeros(oshape, dtype=bool)
    inner[halo:halo + n_i, halo:halo + n_j] = True
    interior = _interior_mask(oshape, halo)
    return inner & ~interior


def _ring_mask(oshape, halo):
    n_i, n_j = oshape[0] - 2 * halo, oshape[1] - 2 * halo
    inner = np.zeros(oshape, dtype=bool)
    inner[halo:halo + n_i, halo:halo + n_j] = True
    return ~inner


if __name__ == "__main__":
    raise SystemExit(main())
