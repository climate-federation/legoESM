#!/usr/bin/env python
"""NH w-floor metric transplant + N=0 w control (Experiment 2, 2026-08-19).

The dual review (codex+GLM) downgraded the metric family from "refuted"
to "unsupported by a scaling probe": the coherent 1e-12/1e-10 boundary
perturbation samples ONE RAY of metric-difference space and cannot
excite a DISCRETE representation difference.  The decisive test is a
BITWISE TRANSPLANT: import the pinned oracle's own runtime gridstruct
metric values (dumped at C48 by ``fv3_extchain_oracle_driver.F90`` from
a real ``fv_control_init`` on the parity deck) into the NumPy lane's
gridstructs, then run the SAME --nh full-step gate.

PRE-REGISTERED:
  (a) N=0 control FIRST: w must be bitwise identical between the port
      IC and the oracle zerostep restart under the frozen face map.
      Not bitwise => everything downstream is misdirected; report and
      STOP (no transplant arm).
  (b) transplant: w one-step residual collapses under oracle metrics
      => metric-seeded after all (the scaling probe was blind);
      unchanged (baseline 6.6116e-04 rel, 1.3003e-07 m/s abs)
      => the transplanted metric families are genuinely closed.

Scope of the transplant (stated, not implied):
  * every 2-D family in ``compare_gs_metrics.FAMILIES`` (33 families:
    lengths, areas, reciprocals, intersection trig, divg/del6, f0),
    oracle values imported at every non-sentinel cell of the full
    padded plane, orientation via the extchain face map
    (``op_scalar``/``op_inverse``) and the globally-resolved sign for
    the orientation-odd cosa families;
  * rsina (compute-B extent, dumped M_RSINA);
  * sin_sg/cos_sg via a SELF-VALIDATING per-face slot assignment: for
    each port slot the oracle slot+sign minimizing the masked residual
    must be a unique winner below 1e-11 and the assignment a
    permutation, else that face/family is NOT transplanted and said so
    (the slot semantics under the dihedral are derived-by-match, never
    assumed);
  * da_min/da_max/da_min_c/da_max_c recomputed from the transplanted
    areas; ectx dx6/dy6 snapshots refreshed where they matched the
    pre-transplant gs values (``_replace_where_held`` analog).
  NOT covered (stays port-computed, same boundary as the perturbation
  probe): ectx amat6 and the ext-vector bases (coordinate-derived),
  and the vertical ak/bk -- checked separately against the oracle
  restart's own ak/bk if present.

Instrument controls (refuse loudly, no verdict printed):
  * extchain face map bijective, coordinate floor <= 1e-12;
  * per-family transplant receipt (cells changed, pre max|d|); zero
    total changed cells => the arm is a no-op control => REFUSE;
  * post-transplant re-comparison: masked max|oracle - s*mapped(port)|
    must be exactly 0.0 for every transplanted family/face.

``--widen-corners`` (OPT-IN, off by default; the default arms keep their
exact behaviour and JSON tags so their numbers stay reproducible).
THE FIRST ELIMINATION WAS BLIND WHERE THE ERROR LIVES.
``sentinel_mask`` (compare_gs_metrics.py) masks a cell when EITHER side
is sentinel, and the port's single-tile builder leaves 1e30 in exactly
the corner-diagonal halo cells -- so those were dropped on the PORT side
and never transplanted, while the oracle holds real, mirror-symmetric
values there.  Those cells ARE read on this lane: ``del6_vt_flux`` at
nord=2 fills its work array over the whole padded box including the
corner diagonals (sw_core.F90:2051-2089), and ``copy_corners`` is skipped
on a bounded domain on BOTH sides (tp_core.F90:139-141/160-162), so no
corner repair hides them.  This arm transplants wherever the ORACLE is
real, i.e. it additionally imports the cells the port marks sentinel.

Controls specific to this arm:
  * per-family and total count of NEWLY covered cells (port sentinel,
    oracle real), with the pre-transplant disagreement at those cells,
    printed BEFORE the physics re-runs.  Total zero => REFUSE: an arm
    that quietly changed nothing would report "unchanged", which reads
    as an elimination -- the same mistake this arm exists to correct;
  * cells where the ORACLE side is sentinel or non-finite are NEVER
    written (they carry no information) and are counted as refused
    rather than silently skipped;
  * the exact post-transplant equality check is re-evaluated over the
    WIDENED set, with only oracle-sentinel cells excused.

NOT covered by ``--widen-corners`` either, stated rather than implied:
oracle-sentinel/non-finite cells; the sin_sg/cos_sg slot ASSIGNMENT,
which stays derived on the both-sides-live cells because a port sentinel
cannot vote on a permutation (only the WRITE widens); ectx amat6, the
ext-vector bases and the vertical ak/bk, which keep the base arm's
boundary; and no corner-rotation repair is performed or emulated.

usage: python scripts/validate/fv3_native/nh_metric_transplant.py
       python scripts/validate/fv3_native/nh_metric_transplant.py \
           --widen-corners
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

_spec = importlib.util.spec_from_file_location(
    "full_step_oracle_parity",
    os.path.join(_HERE, "full_step_oracle_parity.py"))
fsp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fsp)

from compare_extchain_oracle import (  # noqa: E402
    derive_face_map as derive_metric_face_map,
    load_oracle as load_extchain,
    op_inverse,
    op_scalar,
)
from compare_gs_metrics import (  # noqa: E402
    FAMILIES,
    FAMILIES_3D,
    SWAP_PARTNER,
    _ODD,
    resolve_sign,
    sentinel_mask,
)

# the RETRO dump set is the complete 34-family one (M_F0 + reciprocals
# added by the codex retro-review driver build, 2026-08-11 10:33); the
# older extchain/run_c48 predates the metric families entirely.
EXTCHAIN_C48 = Path(os.environ.get(
    "EXTCHAIN_DIR",
    "/burg-archive/glab/users/pg2328/fv3_duo_gaps/"
    "retro_gsmetrics/run_c48"))
N, NG, KM = fsp.N, fsp.NG, fsp.KM
OUT_DIR = os.environ.get(
    "TRANSPLANT_OUT", "/burg-archive/glab/users/pg2328/fv3_duo_gaps")


def _sha() -> str:
    try:
        return subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------
# (a) N=0 control: w bitwise at the IC, before any step
# ---------------------------------------------------------------------

def n0_control() -> bool:
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.core.fv3_native_eta import set_eta_analytic

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    ak, bk, _ptop, _ks = set_eta_analytic(KM)
    ic_run = f"{fsp.ORACLE_ROOT}/run_nh_zerostep_gfs"
    orc_ic = fsp.load_oracle(ic_run, nh=True)
    state = fsp.build_port_ic(ctx, ak, bk, nh=True)
    p_ic = fsp.port_window(state, ctx)
    (cost, meta, perm, worst, _pf, _wo) = fsp.derive_face_map(p_ic, orc_ic)
    print(f"\nA. N=0 CONTROL (port IC vs oracle {ic_run}):")
    print(f"   face-map worst IC rel = {worst:.4e} "
          f"(gate floor {fsp.IC_CONTROL_MAX_REL:.0e})")
    if worst > fsp.IC_CONTROL_MAX_REL:
        raise SystemExit("REFUSING: IC face map above the gate's own "
                         "floor -- N=0 comparison meaningless")
    ok = True
    for pf in range(6):
        ot = perm[pf]
        pairs, _ws = fsp.apply_map(p_ic[pf], orc_ic[ot], meta[pf][ot])
        for fld in ("w", "delz"):
            pw, ow = pairs[fld]
            nbit = int((pw != ow).sum())
            d = float(np.abs(pw - ow).max())
            omax = float(np.abs(ow).max())
            print(f"   face {pf + 1} -> tile {ot + 1}  {fld}: "
                  f"max|port-oracle| {d:.6e}  cells differing {nbit}"
                  f"/{pw.size}  oracle max|{fld}| {omax:.6e}")
            if fld == "w" and nbit:
                ok = False
    return ok


def akbk_check() -> None:
    from legoesm.core.fv3_native_eta import set_eta_analytic
    import netCDF4 as nc

    p = f"{fsp.ORACLE_ROOT}/run_nh_zerostep_gfs/RESTART/fv_core.res.nc"
    print("\nB. VERTICAL METRIC (ak/bk) vs oracle restart:")
    if not os.path.exists(p):
        print(f"   {p} MISSING -- vertical metric not checkable from "
              f"restarts")
        return
    d = nc.Dataset(p)
    names = list(d.variables)
    print(f"   {p}: variables {names}")
    ak, bk, ptop, _ks = set_eta_analytic(KM)
    for nm, mine in (("ak", np.asarray(ak, np.float64)),
                     ("bk", np.asarray(bk, np.float64))):
        if nm not in d.variables:
            print(f"   no {nm!r} in restart -- not checkable")
            continue
        oa = np.array(d[nm][:], dtype=np.float64).ravel()
        if oa.size != mine.size:
            print(f"   {nm}: size {oa.size} vs port {mine.size} -- "
                  f"MISMATCH, not comparable")
            continue
        nbit = int((oa != mine).sum())
        print(f"   {nm}: max|d| {float(np.abs(oa - mine).max()):.6e}  "
              f"entries differing {nbit}/{oa.size}")
    d.close()


# ---------------------------------------------------------------------
# (b) the transplant
# ---------------------------------------------------------------------

def _map_to_port(o2, op):
    return op_scalar(np.ascontiguousarray(o2), op_inverse(op))


def _sent_mask_single(a, fam: str):
    """Sentinel/non-finite mask for ONE side only.

    The both-sides ``sentinel_mask`` is what blinded the first
    elimination: a port-side sentinel removed the cell from the
    transplant even where the oracle held a real value.  This is used on
    the oracle side to DEFINE the widened transplant set, and on the port
    side only to COUNT which cells are newly covered.
    """
    a = np.asarray(a, dtype=np.float64)
    m = (np.abs(a) >= sent_mag(fam)) | ~np.isfinite(a)
    lo = sent_lo(fam)
    if lo > 0.0:
        m |= np.abs(a) <= lo
    return m


def transplant_metrics(ctx, widen: bool = False) -> None:
    orcm = load_extchain(EXTCHAIN_C48)
    gs6 = ctx["gs6"]
    fmap = derive_metric_face_map(orcm, gs6, N, NG)
    tiles = [fmap[t][1] for t in range(6)]
    if sorted(tiles) != list(range(6)):
        raise SystemExit(f"REFUSING: metric face map not a bijection: "
                         f"{tiles}")
    floor = max(fmap[t][0] for t in range(6))
    if floor > 1.0e-12:
        raise SystemExit(f"REFUSING: metric face-map coordinate floor "
                         f"{floor:.3e} > 1e-12")
    print(f"TRANSPLANT: extchain C48 face map bijective, floor "
          f"{floor:.3e}")
    if all(not np.asarray(orcm[t]["arrays"]["M_F0"]).any()
           for t in range(6)):
        raise SystemExit(
            "REFUSING: M_F0 dump is ALL ZEROS -- stale oracle dumps "
            "predating the f0-init replication (compare_gs_metrics "
            "carries the same guard); regenerate before transplanting.")

    # keep pre-transplant dx/dy for the ectx where-held refresh
    old_dxdy = [{k: np.asarray(gs6[t][k], np.float64).copy()
                 for k in ("dx", "dy")} for t in range(6)]

    total_changed = 0
    total_widen_new = 0
    total_widen_refused = 0
    skipped = []
    for fam in sorted(FAMILIES):
        tag, key, ish, jsh = FAMILIES[fam]
        if key not in gs6[0]:
            skipped.append(f"{fam} (no port key)")
            continue
        fam_changed, fam_pre = 0, 0.0
        # -inf, not nan: max(nan, x) returns nan in Python, so a nan
        # seed would swallow every real reading it is compared against.
        fam_widen_total, fam_widen_pre = 0, float("-inf")
        for pf in range(6):
            _d, ot, op = fmap[pf]
            sw = op[0]
            fam_t = SWAP_PARTNER.get(fam, fam) if sw else fam
            tag_t = FAMILIES[fam_t][0]
            o2 = np.asarray(orcm[ot]["arrays"][tag_t], np.float64)
            p_old = np.asarray(gs6[pf][key], np.float64)
            p2o = op_scalar(p_old, op)
            if o2.shape != p2o.shape:
                raise SystemExit(
                    f"REFUSING: {fam} oracle {o2.shape} vs mapped port "
                    f"{p2o.shape} (op={op})")
            sent_both = sentinel_mask(o2, p2o, fam)
            sgn = 1.0
            if fam in _ODD:
                # The sign stays resolved on the BOTH-SIDES-live set even
                # when widening: at a widened cell the port holds its
                # sentinel, which swamps both candidates equally and
                # cannot vote on the sign (resolve_sign's contract).
                sgn, _ec, _eo, _det = resolve_sign(o2, p2o, sent_both)
            sent_o = _sent_mask_single(o2, fam) if widen else sent_both
            new_cells_o = ((~sent_o) & _sent_mask_single(p2o, fam)
                           if widen else None)
            live = ~sent_o
            if live.any():
                fam_pre = max(fam_pre, float(
                    np.abs(o2 - sgn * p2o)[live].max()))
            o_port = sgn * _map_to_port(o2, op)
            sent_p = _map_to_port(sent_o.astype(np.float64), op) > 0.5
            new = np.where(sent_p, p_old, o_port)
            fam_widen_new = 0
            if widen:
                if new_cells_o.any():
                    # The disagreement AT THE NEWLY COVERED CELLS, before
                    # any write. The port holds a sentinel there, so this
                    # is a receipt that the arm had something to change --
                    # reported as a number rather than inferred from a
                    # downstream "unchanged".
                    fam_widen_pre = max(fam_widen_pre, float(
                        np.abs(o2 - sgn * p2o)[new_cells_o].max()))
                new_cells_p = _map_to_port(
                    new_cells_o.astype(np.float64), op) > 0.5
                fam_widen_new = int((new_cells_p & (new != p_old)).sum())
                total_widen_new += fam_widen_new
                total_widen_refused += int(_sent_mask_single(o2, fam).sum())
            fam_changed += int((new != p_old).sum())
            gs6[pf][key] = new
            # post-check: transplanted values match the oracle EXACTLY,
            # now over the WIDENED set (only oracle-sentinel excused)
            resid = np.abs(o2 - sgn * op_scalar(new, op))
            resid = np.where(sent_o, 0.0, resid)
            if float(resid.max()) != 0.0:
                raise SystemExit(
                    f"REFUSING: {fam} face {pf + 1} post-transplant "
                    f"masked residual {float(resid.max()):.3e} != 0.0")
            fam_widen_total += fam_widen_new
        total_changed += fam_changed
        line = (f"  {fam:10s} pre max|d| {fam_pre:.4e}  cells changed "
                f"{fam_changed}")
        if widen:
            pre_txt = ("none" if fam_widen_pre == float("-inf")
                       else f"{fam_widen_pre:.4e}")
            line += (f"  WIDEN newly covered {fam_widen_total} "
                     f"pre max|d| there {pre_txt}")
        print(line)

    # rsina: compute-B extent (M_RSINA dumped is:ie+1, js:je+1)
    if "rsina" in gs6[0]:
        fam_changed, fam_pre = 0, 0.0
        rs_widen, rs_widen_pre = 0, float("-inf")
        for pf in range(6):
            _d, ot, op = fmap[pf]
            o2 = np.asarray(orcm[ot]["arrays"]["M_RSINA"], np.float64)
            p_old = np.asarray(gs6[pf]["rsina"], np.float64)
            if p_old.shape == o2.shape:
                window = None
                p_win = p_old
            elif p_old.shape == (N + 2 * NG + 1, N + 2 * NG + 1):
                window = (slice(NG, NG + N + 1), slice(NG, NG + N + 1))
                p_win = p_old[window]
            else:
                skipped.append(f"rsina (port shape {p_old.shape})")
                p_win = None
            if p_win is None:
                break
            p2o = op_scalar(p_win, op)
            sent_o = (_sent_mask_single(o2, "rsin2") if widen
                      else sentinel_mask(o2, p2o, "rsin2"))
            live = ~sent_o
            if live.any():
                fam_pre = max(fam_pre, float(
                    np.abs(o2 - p2o)[live].max()))
            o_port = _map_to_port(o2, op)
            sent_p = _map_to_port(sent_o.astype(np.float64), op) > 0.5
            new_win = np.where(sent_p, p_win, o_port)
            if widen:
                new_cells = (~sent_o) & _sent_mask_single(p2o, "rsin2")
                if new_cells.any():
                    rs_widen_pre = max(rs_widen_pre, float(
                        np.abs(o2 - p2o)[new_cells].max()))
                new_cells_p = _map_to_port(
                    new_cells.astype(np.float64), op) > 0.5
                rs_widen += int((new_cells_p & (new_win != p_win)).sum())
            fam_changed += int((new_win != p_win).sum())
            if window is None:
                gs6[pf]["rsina"] = new_win
            else:
                p_new = p_old.copy()
                p_new[window] = new_win
                gs6[pf]["rsina"] = p_new
        else:
            total_changed += fam_changed
            line = (f"  {'rsina':10s} pre max|d| {fam_pre:.4e}  cells "
                    f"changed {fam_changed}")
            if widen:
                total_widen_new += rs_widen
                pre_txt = ("none" if rs_widen_pre == float("-inf")
                           else f"{rs_widen_pre:.4e}")
                line += (f"  WIDEN newly covered {rs_widen} "
                         f"pre max|d| there {pre_txt}")
            print(line)

    # sin_sg / cos_sg: self-validating slot assignment per face
    for fam in sorted(FAMILIES_3D):
        tag, key = FAMILIES_3D[fam]
        if key not in gs6[0]:
            skipped.append(f"{fam} (no port key)")
            continue
        fam_changed = 0
        sg_widen = 0
        for pf in range(6):
            _d, ot, op = fmap[pf]
            sw = op[0]
            fam_t = ({"sin_sg": "sin_sg", "cos_sg": "cos_sg"}[fam])
            # under transpose sin/cos_sg stay in-family; only slots
            # permute -- derived BY MATCH below, never assumed
            o3 = np.asarray(orcm[ot]["arrays"][FAMILIES_3D[fam_t][0]],
                            np.float64)
            p3 = np.asarray(gs6[pf][key], np.float64)
            if o3.shape[2] != 9 or p3.shape[2] != 9:
                raise SystemExit(f"REFUSING: {fam} slot count "
                                 f"{o3.shape} / {p3.shape}")
            o3p = np.stack([_map_to_port(o3[:, :, s], op)
                            for s in range(9)], axis=2)
            if o3p.shape != p3.shape:
                raise SystemExit(f"REFUSING: {fam} mapped {o3p.shape} "
                                 f"vs port {p3.shape}")
            assign = {}
            ok_face = True
            for sp in range(9):
                best = (np.inf, None, 1.0)
                second = np.inf
                for so in range(9):
                    for sgn in (1.0, -1.0):
                        a, b = o3p[:, :, so], p3[:, :, sp]
                        sent = (np.abs(a) >= 1.0e6) | (np.abs(b) >= 1.0e6)
                        live = ~sent
                        if not live.any():
                            continue
                        e = float(np.abs(sgn * a - b)[live].max())
                        if e < best[0]:
                            second = best[0]
                            best = (e, so, sgn)
                        elif e < second:
                            second = e
                if (best[1] is None or best[0] > 1.0e-11
                        or second < 10.0 * max(best[0], 1.0e-300)):
                    ok_face = False
                    break
                assign[sp] = best
            if not ok_face or sorted(v[1] for v in assign.values()) \
                    != list(range(9)):
                skipped.append(f"{fam} face {pf + 1} (no unique slot "
                               f"permutation below 1e-11)")
                continue
            new3 = p3.copy()
            for sp, (_e, so, sgn) in assign.items():
                a = o3p[:, :, so]
                if widen:
                    # THE WRITE widens; the slot ASSIGNMENT above does
                    # NOT (a port sentinel cannot vote on which oracle
                    # slot a port slot corresponds to, so widening there
                    # would let garbage decide a permutation).
                    sent = (~np.isfinite(a)) | (np.abs(a) >= 1.0e6)
                else:
                    sent = (np.abs(a) >= 1.0e6) | (np.abs(p3[:, :, sp])
                                                   >= 1.0e6)
                was_sent_p = np.abs(p3[:, :, sp]) >= 1.0e6
                new3[:, :, sp] = np.where(sent, p3[:, :, sp], sgn * a)
                if widen:
                    sg_widen += int(((~sent) & was_sent_p
                                     & (new3[:, :, sp]
                                        != p3[:, :, sp])).sum())
            fam_changed += int((new3 != p3).sum())
            gs6[pf][key] = new3
        total_changed += fam_changed
        if widen:
            total_widen_new += sg_widen
        print(f"  {fam:10s} cells changed {fam_changed}"
              + (f"  WIDEN newly covered {sg_widen}" if widen else ""))

    # da_min/da_max scalars from the transplanted areas
    sl = slice(NG, NG + N)
    for t in range(6):
        gs = gs6[t]
        for sk, pk, red in (("da_min", "area", np.min),
                            ("da_max", "area", np.max),
                            ("da_min_c", "area_c", np.min),
                            ("da_max_c", "area_c", np.max)):
            if sk not in gs:
                continue
            old = float(gs[sk])
            new = float(red(np.asarray(gs[pk])[sl, sl]))
            if new != old:
                gs[sk] = new
                total_changed += 1
                print(f"  face {t + 1} {sk}: {old!r} -> {new!r}")

    # ectx dx6/dy6 snapshots: where-held refresh against pre-transplant
    if ctx.get("ectx") is not None:
        for t in range(6):
            for ek, pk in (("dx6", "dx"), ("dy6", "dy")):
                snap = np.asarray(ctx["ectx"][ek][t], np.float64)
                held = snap == old_dxdy[t][pk]
                new = np.where(held, np.asarray(gs6[t][pk], np.float64),
                               snap)
                nc_ = int((new != snap).sum())
                total_changed += nc_
                ctx["ectx"][ek][t] = new
            print(f"  face {t + 1} ectx dx6/dy6 refreshed")

    if skipped:
        print(f"  NOT transplanted: {skipped}")
    if widen:
        # Printed BEFORE any physics re-runs: this function is called by
        # the context builder, ahead of the gate.
        print(f"WIDEN RECEIPT: newly covered cells TOTAL "
              f"{total_widen_new}; oracle-sentinel cells left alone "
              f"{total_widen_refused}")
        if total_widen_new == 0:
            raise SystemExit(
                "WIDEN CONTROL FAILED: zero newly covered cells, so this "
                "arm is a no-op and its result would be indistinguishable "
                "from the narrow arm's. Reporting 'unchanged' from a "
                "no-op would read as an elimination -- which is exactly "
                "the blind spot this arm exists to correct. REFUSING.")
    if total_changed == 0:
        raise SystemExit(
            "TRANSPLANT CONTROL FAILED: zero cells changed -- the arm "
            "would be a no-op control (perturb-a-zero class)")
    print(f"TRANSPLANT COMPLETE: {total_changed} cells/scalars changed "
          f"across all faces/families")


# ---------------------------------------------------------------------
# (c) drive the unchanged gate, baseline then transplanted
# ---------------------------------------------------------------------

def run_gate(tag: str, transplant: bool,
             widen: bool = False) -> dict | None:
    import legoesm.core.fv3_native_duo_stepper as ds

    out_json = os.path.join(OUT_DIR, f"nh_transplant_{tag}.json")
    real = ds.build_six_face_duo_context
    if transplant:
        def wrapper(*a, **kw):
            ctx = real(*a, **kw)
            transplant_metrics(ctx, widen=widen)
            return ctx
        ds.build_six_face_duo_context = wrapper
    print(f"\n===== GATE ARM: {tag} =====")
    try:
        rc = fsp.main(["--nh", "--json", out_json])
    except SystemExit as e:
        rc = e.code
    finally:
        ds.build_six_face_duo_context = real
    print(f"===== GATE ARM {tag} rc={rc}")
    if not os.path.exists(out_json):
        return None
    with open(out_json) as fh:
        return json.load(fh)


def main() -> int:
    argv = sys.argv[1:]
    widen = "--widen-corners" in argv
    unknown = [a for a in argv if a != "--widen-corners"]
    if unknown:
        # A mistyped flag that is silently ignored is how an arm gets
        # reported under the wrong name.
        raise SystemExit(f"unknown argument(s): {unknown}. The only flag "
                         f"is --widen-corners.")
    print(f"REPO_SHA={_sha()}  n={N} ng={NG} km={KM}  "
          f"extchain={EXTCHAIN_C48}  widen_corners={widen}")
    if not (EXTCHAIN_C48 / "extchain_t1.mf").exists():
        raise SystemExit(f"missing extchain C48 dumps at {EXTCHAIN_C48}")

    w_bitwise = n0_control()
    akbk_check()
    if not w_bitwise:
        print("\nN=0 CONTROL: w is NOT bitwise at the IC -- per the "
              "pre-registration everything downstream is misdirected. "
              "STOPPING before the transplant arm.")
        print("\nNH_METRIC_TRANSPLANT_DONE (stopped at N=0)")
        return 0
    print("\nN=0 CONTROL: w bitwise at the IC on all six faces.")

    base = run_gate("baseline", transplant=False)
    trans = run_gate("transplanted", transplant=True)
    arms = [("baseline", base), ("transplanted", trans)]
    if widen:
        # Its own tag, so the two default arms' JSON files stay
        # byte-stable and the earlier numbers remain reproducible.
        arms.append(("widened",
                     run_gate("widened", transplant=True, widen=True)))

    print("\nSUMMARY (numbers only; verdict belongs to the analysis):")
    for tag, j in arms:
        if j is None:
            print(f"  {tag}: gate did not produce a JSON (arm FAILED)")
            continue
        print(f"  {tag}: step_worst_rel {j['step_worst_rel']:.6e}")
        for face, row in sorted(j.get("residuals", {}).items()):
            w = row.get("w")
            if isinstance(w, dict):
                print(f"    {face}  w rel {w['rel']:.6e}  |d|max "
                      f"{w['max_abs_diff']:.6e}  |d|/tend "
                      f"{w['frac_of_tendency']:.6e}  "
                      f"argmax {w.get('argmax_ijk')}")
    print("\nNH_METRIC_TRANSPLANT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
