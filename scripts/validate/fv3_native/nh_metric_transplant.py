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

usage: python scripts/validate/fv3_native/nh_metric_transplant.py
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


def transplant_metrics(ctx) -> None:
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
    skipped = []
    for fam in sorted(FAMILIES):
        tag, key, ish, jsh = FAMILIES[fam]
        if key not in gs6[0]:
            skipped.append(f"{fam} (no port key)")
            continue
        fam_changed, fam_pre = 0, 0.0
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
            sent_o = sentinel_mask(o2, p2o, fam)
            sgn = 1.0
            if fam in _ODD:
                sgn, _ec, _eo, _det = resolve_sign(o2, p2o, sent_o)
            live = ~sent_o
            if live.any():
                fam_pre = max(fam_pre, float(
                    np.abs(o2 - sgn * p2o)[live].max()))
            o_port = sgn * _map_to_port(o2, op)
            sent_p = _map_to_port(sent_o.astype(np.float64), op) > 0.5
            new = np.where(sent_p, p_old, o_port)
            fam_changed += int((new != p_old).sum())
            gs6[pf][key] = new
            # post-check: transplanted values match the oracle EXACTLY
            resid = np.abs(o2 - sgn * op_scalar(new, op))
            resid = np.where(sent_o, 0.0, resid)
            if float(resid.max()) != 0.0:
                raise SystemExit(
                    f"REFUSING: {fam} face {pf + 1} post-transplant "
                    f"masked residual {float(resid.max()):.3e} != 0.0")
        total_changed += fam_changed
        print(f"  {fam:10s} pre max|d| {fam_pre:.4e}  cells changed "
              f"{fam_changed}")

    # rsina: compute-B extent (M_RSINA dumped is:ie+1, js:je+1)
    if "rsina" in gs6[0]:
        fam_changed, fam_pre = 0, 0.0
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
            sent_o = sentinel_mask(o2, p2o, "rsin2")
            live = ~sent_o
            if live.any():
                fam_pre = max(fam_pre, float(
                    np.abs(o2 - p2o)[live].max()))
            o_port = _map_to_port(o2, op)
            sent_p = _map_to_port(sent_o.astype(np.float64), op) > 0.5
            new_win = np.where(sent_p, p_win, o_port)
            fam_changed += int((new_win != p_win).sum())
            if window is None:
                gs6[pf]["rsina"] = new_win
            else:
                p_new = p_old.copy()
                p_new[window] = new_win
                gs6[pf]["rsina"] = p_new
        else:
            total_changed += fam_changed
            print(f"  {'rsina':10s} pre max|d| {fam_pre:.4e}  cells "
                  f"changed {fam_changed}")

    # sin_sg / cos_sg: self-validating slot assignment per face
    for fam in sorted(FAMILIES_3D):
        tag, key = FAMILIES_3D[fam]
        if key not in gs6[0]:
            skipped.append(f"{fam} (no port key)")
            continue
        fam_changed = 0
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
                sent = (np.abs(a) >= 1.0e6) | (np.abs(p3[:, :, sp])
                                               >= 1.0e6)
                new3[:, :, sp] = np.where(sent, p3[:, :, sp], sgn * a)
            fam_changed += int((new3 != p3).sum())
            gs6[pf][key] = new3
        total_changed += fam_changed
        print(f"  {fam:10s} cells changed {fam_changed}")

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
    if total_changed == 0:
        raise SystemExit(
            "TRANSPLANT CONTROL FAILED: zero cells changed -- the arm "
            "would be a no-op control (perturb-a-zero class)")
    print(f"TRANSPLANT COMPLETE: {total_changed} cells/scalars changed "
          f"across all faces/families")


# ---------------------------------------------------------------------
# (c) drive the unchanged gate, baseline then transplanted
# ---------------------------------------------------------------------

def run_gate(tag: str, transplant: bool) -> dict | None:
    import legoesm.core.fv3_native_duo_stepper as ds

    out_json = os.path.join(OUT_DIR, f"nh_transplant_{tag}.json")
    real = ds.build_six_face_duo_context
    if transplant:
        def wrapper(*a, **kw):
            ctx = real(*a, **kw)
            transplant_metrics(ctx)
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
    print(f"REPO_SHA={_sha()}  n={N} ng={NG} km={KM}  "
          f"extchain={EXTCHAIN_C48}")
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

    print("\nSUMMARY (numbers only; verdict belongs to the analysis):")
    for tag, j in (("baseline", base), ("transplanted", trans)):
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
