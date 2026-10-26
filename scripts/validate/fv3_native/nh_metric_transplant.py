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
``sentinel_mask`` (compare_gs_metrics.py) masks a cell when EITHER side
is sentinel, so a PORT sentinel dropped the cell even where the oracle
held a real value.  This arm transplants wherever the ORACLE is real,
i.e. it additionally imports those cells.

HOW MUCH THAT ACTUALLY BUYS -- MEASURED, because the first version of
this docstring got it wrong and would have oversold a null result.  The
port's sentinel is ``BIG_NUMBER = 1.0e8`` (fv3_native_gridstruct.py:72),
NOT 1e30; the single 1e30 array (``area_c``) is fully overwritten by the
outermost-ends replication, so it has no sentinel cells left.  At C48,
tile 1, the cells this arm newly covers are:

    cosa_s   36 per face  (the 3x3 corner wedge, four corners)
    rsin2     4 per face
    cosa      2 per face
    every length / area / reciprocal family:  NONE

So the earlier experiment DID cover the corner diagonals of the length,
area and reciprocal families -- the widening adds ~42 cells per face in
three trigonometric families and nothing else.  ``del6_u/del6_v/divg_u/
divg_v`` carry ``BIG_NUMBER``-derived values on BOTH sides at those
cells, so neither arm can touch them; if those two families matter, they
need a comparison, not a transplant.

Controls specific to this arm:
  * per-family counts of NEWLY covered cells (port sentinel, oracle
    real), SPLIT by whether the cell lies in a corner region, printed
    BEFORE the physics re-runs.  Zero corner-region cells => REFUSE.
    The refusal is keyed to the CORNER count, not the total: a nonzero
    total sourced from an unrelated family would otherwise let a no-op
    pass, and "unchanged" from a no-op reads as an elimination -- the
    same mistake this arm exists to correct.  A skipped family or face
    is fatal here too, for the same reason;
  * the printed "pre max|d| there" is, by construction, about the
    sentinel magnitude (the port holds BIG_NUMBER at those cells); it
    is a receipt that the arm had something to write, not a physical
    disagreement;
  * cells where the ORACLE side is sentinel or non-finite are NEVER
    written (they carry no information) and are counted as refused
    rather than silently skipped;
  * the exact post-transplant equality check is re-evaluated over the
    WIDENED set, with only oracle-sentinel cells excused -- and it now
    also covers ``rsina`` and the staggered-grid families, which had no
    such check at all.

NOT covered by ``--widen-corners`` either, stated rather than implied:
oracle-sentinel/non-finite cells; the sin_sg/cos_sg slot ASSIGNMENT,
which stays derived on the both-sides-live cells because a port sentinel
cannot vote on a permutation (only the WRITE widens); ectx amat6, the
ext-vector bases and the vertical ak/bk, which keep the base arm's
boundary; and no corner-rotation repair is performed or emulated.

``--transplant-sentinel-damping`` (OPT-IN, off by default, independent
of ``--widen-corners``).  The sentinel-region comparison
(``corner_metric_compare.py``, job 9455470, face map at 8.9e-16) found
``del6_u``/``del6_v``/``divg_u``/``divg_v`` DISAGREEING between port and
oracle at 79-99 of the 108 both-sides-sentinel cells per face (7-12 of
them inside corner regions, |d| 2.5e-06 to 7.0e-06), while every
trigonometric family is BITWISE identical there.  Both sides hold
sentinel-magnitude values (~1e8, so the relative disagreement is
~1e-14), which is exactly why no transplant arm could reach them: a mask
that drops a cell when either side is sentinel drops all of them.  This
arm writes the oracle's value at those cells, for those four families
ONLY, and re-runs the same gate.

THIS ARM DOES NOT RUN, AND THE REASON IS THE RESULT. Both reviewers
returned a BLOCKER and the decisive one is an index-bounds argument that
costs no compute: all 108 cells sit in the OUTERMOST staggered row or
column, and at this deck's damping order the stencils reach one ring
short of them. So the disagreement these four arrays carry is real and
is at cells nothing dereferences. The gate below refuses on that, rather
than spending a gate arm to obtain an "unchanged" that was already
determined. A second, independent argument agrees: 2.5e-06 absolute on
coefficients of ~1e8 is 2.5e-14 relative, which propagates to ~5e-18 m/s
of w -- eleven orders below the 1.3e-07 under investigation. Only
catastrophic cancellation could rescue it, and nobody has shown that
regime exists here.

The pre-registration is kept below because it is what the arm WOULD have
tested, and because the "confirms" branch is itself unreachable (see the
caveat under it) -- a fact worth keeping next to the arm rather than
rediscovering.

PRE-REGISTERED READING, against the stated baseline (step_worst_rel
6.611558e-04; |d|max on w 1.3003e-07 m/s):
  * CONFIRMS these coefficients are the seat: w becomes bitwise
    identical on all six faces.  Anything short of bitwise is NOT a
    confirmation -- the coefficients start only ~1e-14 apart in
    relative terms, so a partial improvement is consistent with them
    contributing without being the seat;
  * REFUTES, at the instrument's resolution and NOT more strongly than
    that: step_worst_rel and the per-face |d|max on w unchanged. The
    summary rounds to six digits, so a null bounds the effect below that
    resolution rather than proving it zero (codex MAJOR);
  * ESTABLISHES NEITHER: improved but not bitwise -- that is
    participation, not seat.

CAVEAT ON THE "CONFIRMS" BRANCH, from review: it is unreachable anyway.
The base transplant leaves ``area_c``/``rarea_c`` differing by 1e30 at
one cell per face (the deliberately poisoned slot), and ``rarea_c`` is
read; ``amat6``, the ext-vector bases and ``ak``/``bk`` are untransplanted
by design. So bitwise ``w`` cannot be reached by any arm here, and this
instrument could only ever have refuted. An arm whose confirming branch
cannot fire should say so.

NOT covered by this arm: any family outside the four; cells where the
oracle side is non-finite (the arm refuses rather than partially write);
and the code PATHS that read these arrays -- transplanting a value
cannot repair a different computation.

usage: python scripts/validate/fv3_native/nh_metric_transplant.py
       python scripts/validate/fv3_native/nh_metric_transplant.py \
           --widen-corners
       python scripts/validate/fv3_native/nh_metric_transplant.py \
           --transplant-sentinel-damping
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
    region_masks,
    resolve_sign,
    sent_lo,
    sent_mag,
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
    # build_port_ic returns (state, sphum) since the moist arm landed;
    # passing the TUPLE to port_window died with "list indices must be
    # integers" and had made this whole script unrunnable -- the failure
    # was invisible because the runner reported success on it.
    state, _sphum = fsp.build_port_ic(ctx, ak, bk, nh=True)
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
    total_widen_corner = 0
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
        fam_widen_chg, fam_widen_corner = 0, 0
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
                # COVERAGE and CHANGE are different questions: a newly
                # eligible cell whose oracle value already equals the
                # port's is covered but not changed, and the refusal must
                # key off coverage (codex MINOR).
                fam_widen_new = int(new_cells_p.sum())
                fam_widen_chg += int((new_cells_p & (new != p_old)).sum())
                # AND KEY IT TO THE REGION THE ARM IS ABOUT. A nonzero
                # global count can come entirely from a family that has
                # nothing to do with the corner diagonals, and would then
                # satisfy the refusal while the cells under test stayed
                # untouched (codex MAJOR + the third reviewer's M3).
                ish, jsh = FAMILIES[fam][2], FAMILIES[fam][3]
                rm = region_masks(N, NG, ish, jsh)
                corner = np.zeros(new_cells_p.shape, dtype=bool)
                for rk, rv in rm.items():
                    if rk.endswith("_corner"):
                        corner |= rv
                fam_widen_corner += int((new_cells_p & corner).sum())
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
            line += (f"  WIDEN covered {fam_widen_total} "
                     f"(corner-region {fam_widen_corner}, changed "
                     f"{fam_widen_chg}) pre max|d| there {pre_txt}")
            total_widen_corner += fam_widen_corner
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
            # Post-check, which this family never had: the written cells
            # must equal the oracle exactly (codex MAJOR).
            rs_resid = np.where(sent_o, 0.0,
                                np.abs(o2 - op_scalar(new_win, op)))
            if float(rs_resid.max()) != 0.0:
                raise SystemExit(
                    f"REFUSING: rsina face {pf + 1} post-transplant "
                    f"masked residual {float(rs_resid.max()):.3e} != 0.0")
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
                # abs(nan) >= 1e6 is False, so a non-finite port value
                # would not have counted as sentinel (codex MINOR).
                was_sent_p = ((np.abs(p3[:, :, sp]) >= 1.0e6)
                              | ~np.isfinite(p3[:, :, sp]))
                new3[:, :, sp] = np.where(sent, p3[:, :, sp], sgn * a)
                if widen:
                    sg_widen += int(((~sent) & was_sent_p).sum())
                # Post-check, which this family never had either.
                sg_resid = np.where(sent, 0.0,
                                    np.abs(sgn * a - new3[:, :, sp]))
                if float(sg_resid.max()) != 0.0:
                    raise SystemExit(
                        f"REFUSING: {fam} face {pf + 1} slot {sp} "
                        f"post-transplant residual "
                        f"{float(sg_resid.max()):.3e} != 0.0")
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
              f"{total_widen_new}, of which in a CORNER region "
              f"{total_widen_corner}; oracle-sentinel cells left alone "
              f"in the 2-D families {total_widen_refused}")
        if skipped:
            # A family or face that was skipped is a family this arm did
            # not cover, and a null result would then be read as covering
            # it. Under --widen-corners that is fatal, not a note.
            raise SystemExit(
                f"WIDEN CONTROL FAILED: {len(skipped)} family/face "
                f"entries were skipped, so a null result would claim "
                f"coverage this run does not have: {skipped}")
        if total_widen_corner == 0:
            raise SystemExit(
                "WIDEN CONTROL FAILED: zero newly covered cells in any "
                "corner region. The whole point of this arm is the "
                "corner diagonals; a nonzero total elsewhere would let it "
                "pass while the cells under test stayed untouched, and "
                "reporting 'unchanged' from that would read as an "
                "elimination -- the blind spot this arm exists to "
                "correct. REFUSING.")
    if total_changed == 0:
        raise SystemExit(
            "TRANSPLANT CONTROL FAILED: zero cells changed -- the arm "
            "would be a no-op control (perturb-a-zero class)")
    print(f"TRANSPLANT COMPLETE: {total_changed} cells/scalars changed "
          f"across all faces/families")


# ---------------------------------------------------------------------
# (--transplant-sentinel-damping) write the oracle's values at the
# BOTH-SIDES-SENTINEL cells of the four damping families only
# ---------------------------------------------------------------------

# Restricted to these four BY NAME. Widening to every family that
# differs at sentinel cells would also rewrite the deliberately poisoned
# area_c / rarea_c slot (1e30 by construction, not a representation
# difference), and several families moving at once means no single
# family's effect can be read out of the gate.
DAMPING_FAMILIES = frozenset(("del6_u", "del6_v", "divg_u", "divg_v"))

#: The deck's resolved damping order. dyn_core.F90:757 derives
#: ``nord_v(k) = min(2, flagstruct%nord)`` and the duo stepper passes
#: that through (fv3_native_duo_stepper.py:678); ``nord_w`` stays at its
#: 0 default (fv3_native_d_sw.py:2758). The damping stencils read
#: ``is-nord .. ie+nord+1``, so this sets how far into the halo any of
#: these four arrays is ever dereferenced.
_NORD_DECK = 2


def transplant_sentinel_damping(ctx) -> None:
    """Oracle values at the cells every other arm is blind to."""
    orcm = load_extchain(EXTCHAIN_C48)
    gs6 = ctx["gs6"]
    fmap = derive_metric_face_map(orcm, gs6, N, NG)
    tiles = [fmap[t][1] for t in range(6)]
    if sorted(tiles) != list(range(6)):
        raise SystemExit(f"REFUSING: face map not a bijection: {tiles}")
    floor = max(fmap[t][0] for t in range(6))
    if floor > 1.0e-12:
        raise SystemExit(f"REFUSING: face-map coordinate floor "
                         f"{floor:.3e} > 1e-12")

    total_corner, total_written, total_read = 0, 0, 0
    missing = []
    for fam in sorted(DAMPING_FAMILIES):
        _tag, key, ish, jsh = FAMILIES[fam]
        if key not in gs6[0]:
            missing.append(f"{fam} (no port key)")
            continue
        rm = region_masks(N, NG, ish, jsh)
        corner = np.zeros((N + 2 * NG + ish, N + 2 * NG + jsh), dtype=bool)
        for rk, rv in rm.items():
            if rk.endswith("_corner"):
                corner |= rv
        for pf in range(6):
            _d, ot, op = fmap[pf]
            fam_t = SWAP_PARTNER.get(fam, fam) if op[0] else fam
            o2 = np.asarray(orcm[ot]["arrays"][FAMILIES[fam_t][0]],
                            np.float64)
            p_old = np.asarray(gs6[pf][key], np.float64)
            p2o = op_scalar(p_old, op)
            if o2.shape != p2o.shape:
                raise SystemExit(
                    f"REFUSING: {fam} oracle {o2.shape} vs mapped port "
                    f"{p2o.shape} (op={op})")
            if not np.isfinite(o2).all():
                raise SystemExit(
                    f"REFUSING: {fam} face {pf + 1} oracle holds "
                    f"non-finite values; there is nothing to transplant "
                    f"at those cells and a partial write would be "
                    f"reported as coverage")
            # BOTH sides sentinel, spelled out.
            #
            # THE MOTIVATING COMPARISON USED THE EITHER-SIDE SET, so
            # these are formally different populations and the "108 per
            # face" it reported is a union count (codex BLOCKER). They
            # coincide for these four families, and here is why, since
            # the code cannot show it: the widen arm measured ZERO cells
            # that are port-sentinel and oracle-real, so port-sentinel is
            # a subset of oracle-sentinel; and the largest disagreement
            # over the union is 7.0e-06, far too small for one side to
            # sit at ~1e8 while the other is O(1). Hence union equals
            # intersection here. That reasoning does NOT generalise to
            # another family, so do not copy this mask elsewhere without
            # redoing it.
            write_o = (_sent_mask_single(o2, fam)
                       & _sent_mask_single(p2o, fam))
            if not write_o.any():
                raise SystemExit(
                    f"REFUSING: {fam} face {pf + 1} has ZERO "
                    f"both-sides-sentinel cells, but the comparison "
                    f"measured 108 per face -- the instrument and its "
                    f"premise have diverged, and a null result here "
                    f"would be a no-op dressed as a refutation")
            # The disagreement AT THE WRITE SET, before the write, so a
            # null gate result is read against a measured starting
            # disagreement rather than an assumed one.
            pre = float(np.abs(o2 - p2o)[write_o].max())
            n_write = int(write_o.sum())
            corner_o = op_scalar(corner.astype(np.float64), op) > 0.5
            n_corner = int((write_o & corner_o).sum())
            # The window every consumer of these arrays actually reads,
            # in ORACLE index space (the same space write_o lives in).
            lo, hi = NG - _NORD_DECK, NG + N + _NORD_DECK
            read = np.zeros_like(write_o)
            read[max(lo, 0):hi + 1, max(lo, 0):hi + 1] = True
            n_read = int((write_o & read).sum())
            total_read += n_read

            write_p = _map_to_port(write_o.astype(np.float64), op) > 0.5
            gs6[pf][key] = np.where(write_p, _map_to_port(o2, op), p_old)

            # Post-transplant exactness over the WRITTEN set.
            new = np.asarray(gs6[pf][key], np.float64)
            resid = np.where(write_o,
                             np.abs(o2 - op_scalar(new, op)), 0.0)
            if float(resid.max()) != 0.0:
                raise SystemExit(
                    f"REFUSING: {fam} face {pf + 1} post-transplant "
                    f"write-set residual {float(resid.max()):.3e} != 0.0")
            total_corner += n_corner
            total_written += n_write
            print(f"  DAMPING {fam:8s} face {pf + 1}: wrote {n_write} "
                  f"both-sides-sentinel cells, {n_corner} of them in a "
                  f"corner region, pre max|d| there {pre:.6e}")
    if missing:
        raise SystemExit(
            f"DAMPING CONTROL FAILED: families absent from the port "
            f"gridstruct, so this arm would be partial: {missing}")
    print(f"DAMPING RECEIPT: wrote {total_written} cells, "
          f"{total_corner} of them in corner regions, {total_read} of "
          f"them inside an operator's read window at nord={_NORD_DECK}")
    # KEYED TO THE READ WINDOW, not the corner region. Being in a corner
    # region is not enough: a cell that no operator ever dereferences
    # cannot move the answer, so an arm that writes only such cells
    # returns "unchanged" by construction and that null means nothing.
    #
    # MEASURED, and it is why this arm does not run: all 108 cells sit
    # in the OUTERMOST staggered row/column (j in {0, 54} for the
    # u-families, i in {0, 54} for the v-families, on the 54x55 / 55x54
    # planes). The damping reads del6_u/del6_v over
    # ``is-nord .. ie+nord+1`` and divg_u/divg_v over a window one
    # narrower, and the resolved deck is nord = nord_v = 2 with
    # nord_w = 0 (fv3_native_d_sw.py:2758 default, and the duo stepper
    # passes only nord_v/nord_t at :678). That reaches indices 1..53.
    # Index 0 and 54 would need nord >= 3, which nothing here uses.
    if total_read == 0:
        raise SystemExit(
            f"DAMPING CONTROL FAILED: none of the {total_written} written "
            f"cells lies inside any operator's read window at this deck's "
            f"nord={_NORD_DECK}. They are all in the outermost halo ring, "
            f"which the damping stencil reaches only at nord >= 3. This "
            f"arm would return 'unchanged' by construction, and that null "
            f"would be worth nothing -- the index bounds already answer "
            f"it, at no compute. REFUSING.")


# ---------------------------------------------------------------------
# (c) drive the unchanged gate, baseline then transplanted
# ---------------------------------------------------------------------

def run_gate(tag: str, transplant: bool, widen: bool = False,
             damping: bool = False) -> dict | None:
    import legoesm.core.fv3_native_duo_stepper as ds

    out_json = os.path.join(OUT_DIR, f"nh_transplant_{tag}.json")
    # An arm that raises before writing would otherwise have its
    # PREVIOUS run's JSON read back and reported as this run's result
    # (codex MAJOR).
    if os.path.exists(out_json):
        os.unlink(out_json)
    real = ds.build_six_face_duo_context
    if transplant:
        def wrapper(*a, **kw):
            ctx = real(*a, **kw)
            transplant_metrics(ctx, widen=widen)
            if damping:
                # AFTER the base transplant, at cells that arm provably
                # leaves alone -- so the only difference from the
                # "transplanted" arm is these four families' sentinel
                # cells.
                transplant_sentinel_damping(ctx)
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
    damping = "--transplant-sentinel-damping" in argv
    unknown = [a for a in argv
               if a not in ("--widen-corners",
                            "--transplant-sentinel-damping")]
    if unknown:
        # A mistyped flag that is silently ignored is how an arm gets
        # reported under the wrong name.
        raise SystemExit(f"unknown argument(s): {unknown}. The flags are "
                         f"--widen-corners and "
                         f"--transplant-sentinel-damping.")
    print(f"REPO_SHA={_sha()}  n={N} ng={NG} km={KM}  "
          f"extchain={EXTCHAIN_C48}  widen_corners={widen}  "
          f"sentinel_damping={damping}")
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
    if damping:
        arms.append(("sentinel_damping",
                     run_gate("sentinel_damping", transplant=True,
                              damping=True)))

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
