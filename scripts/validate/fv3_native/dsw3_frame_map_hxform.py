#!/usr/bin/env python
"""H-XFORM decisive tests for the S10 ubbtemp/vbb edge residual.

Two mechanisms for the compare_dyncore_stages.py 5e-7-rel edge lines on
the S10 d_sw3 m/s-advected pair (ubbtemp/vbb, oracle tiles 1/2/5, all
"transposed" faces) were ALREADY refuted before this probe:

  - bitwise oracle-frame replay (dsw3_oracle_frame_replay.py): the port
    operator, fed the oracle's OWN inputs in the oracle's OWN tile
    frame (no face mapping at all), reproduces the oracle's own
    ubbtemp/vbbtemp/ubb/vbb EXACTLY 0.0 on every field -- so the
    OPERATOR's formulation is not the defect.
  - additive Courant-branch sweep (dsw3_replay_sweep.json): the
    response/eps ratio across eps in 1e-20..1e-8 is a perfectly
    constant 8.494e-3 on the flagged tile vs 2.028e1 on the control --
    no step, no branch flip, and the flagged strips are LESS sensitive
    to input perturbation than the control, not more.

H-XFORM is the third candidate: the 5e-7 enters through the SCORING /
frame-mapping transform used ONLY to compare a port face against a
DIFFERENT, rotated oracle tile ("transposed" faces), not through the
operator or its inputs. compare_dyncore_stages.py's frozen transform
for this pair is basis-derived (codex r2 HIGH#1): ubbtemp is ytp_v's
advected D-grid v (y-like -> kind "bv", takes the dihedral's v-sign,
pairs under transpose with vbb's tile) and vbb is xtp_u's advected
D-grid u (x-like -> kind "bu", u-sign, pairs with ubbtemp's tile). This
probe runs three checks against that specific hypothesis, all offline
on the existing dynstage dump (no Fortran rebuild):

  TEST 1 -- FRAME-FREE INVARIANT.  dyn_core.F90:1017-1018 computes
      kee(i,j) = 0.5*(ubbtemp*vbbtemp + ubb*vbb)
  at the SAME (i,j) on each side -- a genuine scalar (kinetic energy),
  not a vector component, so its port-vs-oracle comparison ("bscalar"
  kind: axis-swap only under a transposed face, NO sign choice and NO
  partner-swap ambiguity) cannot inherit a sign/pairing mistake from
  the ubbtemp/vbb scoring. kee is ALREADY a row in the stage ladder
  (S12_kee); this probe reproduces it face-by-face (not just the
  ladder's flagged-only printout) as a spot-check, and prints its
  baseline scale before any ratio. NOTE: u^2+v^2 is explicitly NOT
  used as the invariant -- ubbtemp and vbb are not the two components
  of one vector at one point (ubbtemp is ytp_v's y-direction advected
  *v*, vbb is xtp_u's x-direction advected *u*: different stencils,
  different physical construction), so a naive rotation invariant
  would not be meaningful even if their global scales happen to match
  (~20 m/s each). kee is the quantity the oracle itself treats as
  frame-free (dyn_core.F90 multiplies same-tile same-index fields).

  TEST 2 -- TRANSFORM-VARIANT ABLATION.  On each face's EDGE-STRIP
  region (width 3, the flagged region) only, evaluate the 4 candidate
  assignments the frozen map's ingredients could have taken: {read the
  oracle's direct field for this quantity, or its rotated partner} x
  {+1, -1}. (The DIHEDRAL maps in use -- id/fi/fj/r180 -- are each
  their own inverse, i.e. pure axis reflections, so "transpose vs
  inverse" collapse to the same operation here; that is stated, not
  assumed -- see full_step_oracle_parity.DIHEDRAL.) If any candidate
  collapses the edge max|diff| to the ~1e-14 floor while the frozen
  candidate stays at ~5e-7, that names the mislabelling. This mirrors
  compare_dyncore_stages.map_bm_best's candidate construction exactly
  (reused, not re-derived) but reduces over the edge-strip mask
  instead of the full window, and reports ALL 4 values instead of only
  the argmin.

  TEST 3 -- TRANSPOSED-vs-NOT PARTITION.  Print, for all 6 faces (not
  only the ones the ladder's threshold printed), the GROUND-TRUTH
  transpose flag from the frozen S00 face map alongside the measured
  edge rel for ubbtemp, vbb and kee, so "flagged == transposed" can be
  read off the actual per-face numbers rather than the log's printed
  labels (which only showed faces above a 1e-9 threshold).

No verdict is printed here -- only baseline magnitudes, the ablation
table and the per-face partition. NaN anywhere is fatal.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(
        modname, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fatal_if_nan(arr, where: str):
    """Fatal on NaN OR inf -- codex r1 LOW#2: a plain isnan() check lets
    an infinity (e.g. a genuine div-by-zero) pass silently through every
    downstream max()/comparison. Not masked, not nan*-hidden."""
    a = np.asarray(arr)
    if not np.isfinite(a).all():
        n_nan = int(np.isnan(a).sum())
        n_inf = int(np.isinf(a).sum())
        raise SystemExit(f"non-finite values in {where} -- fatal, not "
                         f"masked (nan={n_nan} inf={n_inf})")


def masked_edge_max(d: np.ndarray, mask: np.ndarray) -> float:
    """max|d| over the SAME width-3 edge mask (corners excluded) used by
    the Test-2 ablation (region_masks()["edge"]) -- codex r1 MEDIUM#1: a
    separate W/E/S/N-strip reduction includes the corner wedges, so its
    'self-check' against the mask-based ablation number was comparing
    two different domains. This is now the ONE edge reduction used for
    every printed edge_max/edge_rel in this probe."""
    sel = d[mask] if d.ndim == 2 else d[mask, ...]
    return float(sel.max()) if sel.size else 0.0


def safe_rel(edge_max: float, scale: float) -> float:
    """edge_max / scale, with a LOUD zero-denominator (never a silent
    NaN): scale == 0 and edge_max == 0 is a genuine 0/0 -> 0.0 (both
    sides agree exactly on an all-zero window); scale == 0 and
    edge_max != 0 is a real defect divided by a void baseline -> +inf,
    printed loudly rather than swallowed as NaN."""
    if scale == 0.0:
        if edge_max == 0.0:
            return 0.0
        print(f"      ZERO-SCALE WARNING: edge_max={edge_max:.6e} but "
              f"the comparison baseline (scale) is exactly 0.0 -- "
              f"reporting +inf, not a ratio")
        return float("inf")
    return edge_max / scale


def region_edge_mask(cmp_mod, ni: int, nj: int, width: int = 3):
    return cmp_mod.region_masks(ni, nj, width)["edge"]


def ablation_candidates(cmp_mod, fsp_mod, p_arr, o_dir, o_par, kind, meta,
                        edge_width: int = 3):
    """4 candidates -- {direct oracle field, dihedral-swapped partner
    field} x {+1, -1} -- reduced over the edge-strip mask only.
    Mirrors compare_dyncore_stages.map_bm_best's candidate set exactly
    (same window(), same FSP.DIHEDRAL, same partner lookup); the only
    difference is the reduction domain (edge strips, not the full
    compute window) and that ALL 4 values are returned, not just the
    argmin.
    """
    transposed, nm, su, sv = meta
    (ei, ej), partner, sgn_ix = cmp_mod.KINDS[kind]
    _fatal_if_nan(p_arr, f"ablation input p_arr kind={kind}")
    _fatal_if_nan(o_dir, f"ablation input o_dir kind={kind}")
    _fatal_if_nan(o_par, f"ablation input o_par kind={kind}")
    p0 = fsp_mod.DIHEDRAL[nm](cmp_mod.window(np.asarray(p_arr), kind))
    cand_fields = [
        ("direct", cmp_mod.window(np.asarray(o_dir), kind)),
        ("partnerT", np.swapaxes(cmp_mod.window(np.asarray(o_par), partner),
                                 0, 1)),
    ]
    ni, nj = p0.shape[0], p0.shape[1]
    mask = region_edge_mask(cmp_mod, ni, nj, edge_width)
    n_edge_cells = int(mask.sum())
    out = {}
    for tag, o in cand_fields:
        if o.shape != p0.shape:
            out[tag] = {"+1": None, "-1": None}
            continue
        row = {}
        for s in (1.0, -1.0):
            d = np.abs(p0 * s - o)
            _fatal_if_nan(d, f"ablation candidate {kind}/{tag}/{s:+.0f}")
            sel = d[mask] if d.ndim == 2 else d[mask, ...]
            row[f"{s:+.0f}"] = float(sel.max()) if sel.size else float("nan")
        out[tag] = row
    frozen_tag = "partnerT" if transposed else "direct"
    frozen_sign_val = (su if sgn_ix == +1 else
                       (sv if sgn_ix == -1 else 1.0))
    frozen_sign_key = f"{frozen_sign_val:+.0f}"
    return out, frozen_tag, frozen_sign_key, n_edge_cells


def edge_strips(d: np.ndarray, width: int = 3) -> dict:
    ni, nj = d.shape[0], d.shape[1]
    return {"W": float(np.abs(d[:width]).max()),
            "E": float(np.abs(d[ni - width:]).max()),
            "S": float(np.abs(d[:, :width]).max()),
            "N": float(np.abs(d[:, nj - width:]).max())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--step-run", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    CMP = _load("compare_dyncore_stages", "compare_dyncore_stages.py")
    FSP = CMP.FSP
    N, NG, KM = CMP.N, CMP.NG, CMP.KM
    step_run = args.step_run or f"{FSP.ORACLE_ROOT}/run_hydro_1step_gfs"

    # ---- rungs 1-2: same instrument controls as compare_dyncore_stages ----
    dumps, certs_by_tile, notes0 = [], [], None
    for t in range(1, 7):
        f, notes, certs = CMP.read_tile(args.dump_dir, t)
        dumps.append(f)
        certs_by_tile.append(certs)
        if notes0 is None:
            notes0 = notes
    CMP.require_certs(certs_by_tile)
    CMP.external_control(dumps, step_run)

    # ---- port setup (identical to compare_dyncore_stages.main) ----
    from legoesm.core.fv3_native_acoustic_3d import acoustic_substep_3d
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.core.fv3_native_dynamics import (
        p_var_hydrostatic,
        pt_to_theta_v,
    )
    from legoesm.core.fv3_native_eta import set_eta_analytic
    from legoesm.grids.fv3_native_gridstruct import FV3_CP_AIR, FV3_KAPPA

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    ak, bk, ptop, ks = set_eta_analytic(KM)
    ptop = float(ptop)
    state = FSP.build_port_ic(ctx, ak, bk)
    press = [p_var_hydrostatic(f["delp"], ptop=ptop, akap=FV3_KAPPA,
                               n=N, ng=NG, km=KM) for f in state]
    for t in range(6):
        pt_to_theta_v(state[t]["pt"], press[t]["pkz"], n=N, ng=NG)

    p_ic = FSP.port_window(state, ctx)

    def orc_kji(t, name, stag):
        arr = dumps[t][name]
        ei = 48 + (1 if stag == "v" else 0)
        ej = 48 + (1 if stag == "u" else 0)
        win = arr[NG:NG + ei, NG:NG + ej, :]
        return win.transpose(2, 1, 0)

    orc_ic = [{"u": orc_kji(t, "S00_entry_u", "u"),
               "v": orc_kji(t, "S00_entry_v", "v"),
               "pt": orc_kji(t, "S00_entry_pt", "a"),
               "delp": orc_kji(t, "S00_entry_delp", "a")}
              for t in range(6)]
    (cost, meta, perm, worst, _pf, _wo) = FSP.derive_face_map(p_ic, orc_ic)
    print(f"S00 face-map control: worst rel {worst:.3e} "
          f"(floor {FSP.IC_CONTROL_MAX_REL:.0e})")
    if worst > FSP.IC_CONTROL_MAX_REL:
        raise SystemExit(
            f"S00 INSTRUMENT CONTROL FAILED ({worst:.3e}): refuse to "
            f"compare on an unverified face map.")
    _FROZEN = {0: {3}, 1: {4}, 2: {2}, 3: {0, 1}, 4: {0, 1}, 5: {5}}
    drift = [(pf + 1, perm[pf] + 1) for pf in range(6)
             if perm[pf] not in _FROZEN[pf]]
    if drift:
        raise SystemExit(f"FACE-MAP DRIFT {drift}: refuse to compare.")

    print("\nGROUND-TRUTH per-face transpose flag (from the frozen S00 "
          "map, NOT log labels):")
    for pf in range(6):
        ot = perm[pf]
        tr, nm, su, sv = meta[pf][ot]
        print(f"  face{pf + 1}->tile{ot + 1}: transposed={tr} "
              f"dihedral={nm} s_u={su:+.0f} s_v={sv:+.0f}")

    stages: dict = {}

    def hook(name, payload):
        stages[name] = [{k: np.array(v, copy=True) for k, v in d.items()}
                        for d in payload]

    dt_sub = 1920.0 / 1.0 / 8.0
    acoustic_substep_3d(ctx, state, dt_sub, KM, first_substep=True,
                        ptop=ptop, akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
                        remap_step=False, remap_follows=True,
                        stage_hook=hook)
    print(f"port substep 1 replayed; stages captured: {sorted(stages)}\n")

    # ---- pull the EXACT rows the validated ladder uses (no re-derived
    # field-name strings) ----
    rows_by_key = {(r[0], r[2]): r for r in CMP.stage_rows()}
    row_ubbtemp = rows_by_key[("S10_dsw23", "ubbtemp")]
    row_vbb = rows_by_key[("S10_dsw23", "vbb")]
    row_kee = rows_by_key[("S12_kee", "kee")]

    per_face = []
    print("PER-FACE: baseline scale printed BEFORE any ratio; "
          "edge = max|port-oracle| over the width-3 edge-strip mask "
          "(frozen transform); ablation = same 4-candidate table as "
          "map_bm_best but edge-masked, all 4 values shown")
    for pf in range(6):
        ot = perm[pf]
        m = meta[pf][ot]
        transposed = m[0]
        face_rec = {"port_face": pf + 1, "oracle_tile": ot + 1,
                    "transposed_gt": bool(transposed)}

        for tag, row, kind in (("ubbtemp", row_ubbtemp, "bv"),
                               ("vbb", row_vbb, "bu")):
            _stage, pkey, pf_name, od_name, op_name, kind_row, halo, note = row
            assert kind_row == kind, (kind_row, kind)
            p_arr = CMP.get_port_field(stages, pkey, pf_name, pf)
            o_dir = CMP.get_oracle_field(dumps[ot], od_name)
            o_par = CMP.get_oracle_field(dumps[ot], op_name)
            _fatal_if_nan(p_arr, f"port {pf_name} face{pf+1}")
            _fatal_if_nan(o_dir, f"oracle {od_name} tile{ot+1}")
            _fatal_if_nan(o_par, f"oracle {op_name} tile{ot+1}")

            # frozen mapping (identical call to compare_dyncore_stages)
            p_mapped, o_mapped = CMP.map_stage_field(p_arr, o_dir, o_par,
                                                     kind, m)
            scale = float(np.abs(o_mapped).max())
            d = np.abs(p_mapped - o_mapped)
            strips = edge_strips(d)                    # informational only
            edge_mask = region_edge_mask(CMP, d.shape[0], d.shape[1])
            edge_max = masked_edge_max(d, edge_mask)    # SAME domain as
                                                        # the ablation below
            interior = float(np.abs(
                d[3:d.shape[0] - 3, 3:d.shape[1] - 3]).max())
            edge_rel = safe_rel(edge_max, scale)

            ablation, frozen_tag, frozen_sign, n_edge = ablation_candidates(
                CMP, FSP, p_arr, o_dir, o_par, kind, m)
            # self-consistency: the frozen combo inside `ablation` is
            # computed by a SEPARATELY coded path (ablation_candidates)
            # but reduces over the identical region_masks()["edge"]
            # mask, so it must equal edge_max exactly.
            frozen_ablation_val = ablation[frozen_tag][frozen_sign]

            print(f"  face{pf+1}->tile{ot+1} [{pf_name:8s} kind={kind} "
                  f"transposed={transposed}] scale={scale:.6g} "
                  f"edge_max={edge_max:.6e} edge_rel={edge_rel:.6e} "
                  f"interior={interior:.6e} strips(incl.corners,info-only)="
                  f"{strips}")
            print(f"      ablation (edge-mask, n={n_edge} cells): "
                  f"direct[+1]={ablation['direct']['+1']:.6e} "
                  f"direct[-1]={ablation['direct']['-1']:.6e} "
                  f"partnerT[+1]={ablation['partnerT']['+1']:.6e} "
                  f"partnerT[-1]={ablation['partnerT']['-1']:.6e}  "
                  f"frozen={frozen_tag}[{frozen_sign}]="
                  f"{frozen_ablation_val:.6e} (same-domain self-check vs "
                  f"edge_max={edge_max:.6e}; must be exactly equal)")
            if frozen_ablation_val != edge_max:
                raise SystemExit(
                    f"SELF-CHECK FAILED face{pf+1} {pf_name}: ablation's "
                    f"frozen candidate {frozen_ablation_val!r} != "
                    f"edge_max {edge_max!r} -- the two independently "
                    f"coded paths disagree, refusing to report")

            face_rec[f"{pf_name}_scale"] = scale
            face_rec[f"{pf_name}_edge_max"] = edge_max
            face_rec[f"{pf_name}_edge_rel"] = edge_rel
            face_rec[f"{pf_name}_interior"] = interior
            face_rec[f"{pf_name}_ablation"] = ablation
            face_rec[f"{pf_name}_frozen_combo"] = f"{frozen_tag}[{frozen_sign}]"

        # TEST 1: kee -- bscalar, no sign/partner ambiguity
        _stage, pkey, pf_name, od_name, op_name, kind, halo, note = row_kee
        assert kind == "bscalar"
        p_kee = CMP.get_port_field(stages, pkey, pf_name, pf)
        o_kee_dir = CMP.get_oracle_field(dumps[ot], od_name)
        _fatal_if_nan(p_kee, f"port kee face{pf+1}")
        _fatal_if_nan(o_kee_dir, f"oracle kee tile{ot+1}")
        p_mapped, o_mapped = CMP.map_stage_field(p_kee, o_kee_dir, o_kee_dir,
                                                 kind, m)
        scale = float(np.abs(o_mapped).max())
        d = np.abs(p_mapped - o_mapped)
        strips = edge_strips(d)                        # informational only
        edge_mask = region_edge_mask(CMP, d.shape[0], d.shape[1])
        edge_max = masked_edge_max(d, edge_mask)        # SAME domain as
                                                        # the ubbtemp/vbb rows
        interior = float(np.abs(
            d[3:d.shape[0] - 3, 3:d.shape[1] - 3]).max())
        edge_rel = safe_rel(edge_max, scale)
        print(f"  face{pf+1}->tile{ot+1} [kee      kind=bscalar "
              f"transposed={transposed}] scale={scale:.6g} "
              f"edge_max={edge_max:.6e} edge_rel={edge_rel:.6e} "
              f"interior={interior:.6e} strips(incl.corners,info-only)="
              f"{strips}")
        face_rec["kee_scale"] = scale
        face_rec["kee_edge_max"] = edge_max
        face_rec["kee_edge_rel"] = edge_rel
        face_rec["kee_interior"] = interior
        per_face.append(face_rec)

    FLAG_THRESH = 1e-8
    print("\nTEST-3 PARTITION TABLE (flagged = edge_rel > "
          f"{FLAG_THRESH:.0e}):")
    hdr = (f"{'face':5s} {'tile':5s} {'transposed_gt':>13s} "
           f"{'ubbtemp_rel':>12s} {'flag':>5s} {'vbb_rel':>12s} "
           f"{'flag':>5s} {'kee_rel':>12s} {'flag':>5s}")
    print(hdr)
    for fr in per_face:
        ub_flag = fr["ubbtemp_edge_rel"] > FLAG_THRESH
        vb_flag = fr["vbb_edge_rel"] > FLAG_THRESH
        ke_flag = fr["kee_edge_rel"] > FLAG_THRESH
        print(f"{fr['port_face']:<5d} {fr['oracle_tile']:<5d} "
              f"{str(fr['transposed_gt']):>13s} "
              f"{fr['ubbtemp_edge_rel']:>12.3e} {str(ub_flag):>5s} "
              f"{fr['vbb_edge_rel']:>12.3e} {str(vb_flag):>5s} "
              f"{fr['kee_edge_rel']:>12.3e} {str(ke_flag):>5s}")
        fr["ubbtemp_flag"] = ub_flag
        fr["vbb_flag"] = vb_flag
        fr["kee_flag"] = ke_flag

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"face_map_worst": worst, "per_face": per_face,
                       "flag_threshold": FLAG_THRESH}, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
