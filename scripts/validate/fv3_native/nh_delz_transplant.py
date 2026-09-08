#!/usr/bin/env python
"""NH w-floor delz-IC transplant + N=0 IC-field bitwise census.

Next discriminator after the metric transplant (jobs 9435749/50/78)
left the 1.3003e-07 m/s w floor unchanged: the port's IC delz is NOT
bitwise vs the pinned oracle (2.7e-11 abs, ~1.9e-14 rel, ~6600/11520
cells per face at N=0, job 9435750).  Since Riem_Solver3 overwrites w
every substep, a standing floor implies a persistent difference in the
solver's INPUTS -- delz is the only IC field measured non-bitwise so
far.  This probe transplants the ORACLE's IC delz (run_nh_zerostep_gfs,
the same restarts the N=0 comparison read) bitwise into the port IC and
re-runs the --nh one-step gate, baseline vs transplanted, in one job.

PRE-REGISTERED:
  (a) w floor collapses (>= 10x drop in w |d|max) => the delz IC seed
      is the (dominant) cause; then localise WHERE build_port_ic's
      delz derivation diverges from init_hydro.F90:147-158 (read, no
      fix);
  (b) unchanged => the 2.7e-11 delz seed is NOT the cause (would need
      ~7e3x amplification, 1.3e-07 / 2e-11 -- note plausibility);
  (c) partial => report the split.

Also: a full N=0 bitwise census of ALL IC fields (u, v, pt, delp, w,
delz) under the frozen face map -- closes "which inputs carry seeds".

Instrument controls (refuse loudly, no verdict printed):
  * face map must be bijective at the gate's own floor (1e-12) and is
    derived from u/v/pt/delp only, so the transplant cannot move it;
  * per-face transplant receipt (cells changed, pre max|d|); ZERO total
    changed cells => no-op control (perturb-a-zero class) => REFUSE;
  * post-transplant re-comparison via the gate's own apply_map: delz
    max|port - oracle| must be exactly 0.0 and 0 differing cells on
    every face (the dihedral ops are involutions -- pure reindexing,
    so f(f(ow)) == ow bitwise).

usage: python scripts/validate/fv3_native/nh_delz_transplant.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

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

N, NG, KM = fsp.N, fsp.NG, fsp.KM
OUT_DIR = os.environ.get(
    "TRANSPLANT_OUT", "/burg-archive/glab/users/pg2328/fv3_duo_gaps")
IC_RUN = f"{fsp.ORACLE_ROOT}/run_nh_zerostep_gfs"
CENSUS_FIELDS = ("u", "v", "pt", "delp", "w", "delz")


def _sha() -> str:
    try:
        return subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def _derive_map(state, ctx, orc_ic):
    """Frozen-map derivation on u/v/pt/delp -- delz plays no role, so
    the SAME map comes out before and after the transplant."""
    p_ic = fsp.port_window(state, ctx)
    (cost, meta, perm, worst, _pf, _wo) = fsp.derive_face_map(p_ic, orc_ic)
    if worst > fsp.IC_CONTROL_MAX_REL:
        raise SystemExit(f"REFUSING: IC face map worst rel {worst:.3e} "
                         f"above the gate floor "
                         f"{fsp.IC_CONTROL_MAX_REL:.0e}")
    if sorted(perm) != list(range(6)):
        raise SystemExit(f"REFUSING: face map not a bijection: {perm}")
    return p_ic, meta, perm, worst


# ---------------------------------------------------------------------
# A. N=0 bitwise census of every IC field
# ---------------------------------------------------------------------

def n0_census() -> None:
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.core.fv3_native_eta import set_eta_analytic

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    ak, bk, _ptop, _ks = set_eta_analytic(KM)
    orc_ic = fsp.load_oracle(IC_RUN, nh=True)
    state = fsp.build_port_ic(ctx, ak, bk, nh=True)
    p_ic, meta, perm, worst = _derive_map(state, ctx, orc_ic)
    print(f"\nA. N=0 IC-FIELD BITWISE CENSUS (port IC vs oracle "
          f"{IC_RUN}; face-map worst rel {worst:.4e}):")
    print(f"   {'field':6s} " + " ".join(
        f"{'f' + str(pf + 1) + '->t' + str(perm[pf] + 1):>16s}"
        for pf in range(6)) + "   (cells differing / total, max|d|)")
    for fld in CENSUS_FIELDS:
        cells, dmax = [], []
        for pf in range(6):
            pairs, _ws = fsp.apply_map(p_ic[pf], orc_ic[perm[pf]],
                                       meta[pf][perm[pf]])
            pw, ow = pairs[fld]
            cells.append(int((pw != ow).sum()))
            dmax.append(float(np.abs(pw - ow).max()))
        tot = pw.size
        line = " ".join(f"{c:6d} {d:9.2e}" for c, d in zip(cells, dmax))
        tag = "BITWISE" if max(cells) == 0 else "DIFFERS"
        print(f"   {fld:6s} {line}   /{tot} per face  => {tag}")


# ---------------------------------------------------------------------
# B. the delz transplant (installed via build_port_ic monkeypatch)
# ---------------------------------------------------------------------

def transplant_delz(state, ctx) -> None:
    orc_ic = fsp.load_oracle(IC_RUN, nh=True)
    _p_ic, meta, perm, worst = _derive_map(state, ctx, orc_ic)
    print(f"TRANSPLANT: face map bijective, worst rel {worst:.4e}")
    total_changed = 0
    for pf in range(6):
        ot = perm[pf]
        transposed, nm, _su, _sv = meta[pf][ot]
        ow = fsp.oracle_ij(orc_ic[ot]["delz"], transposed)
        # apply_map compares f(port_delz) vs ow; the dihedral f is an
        # involution, so the port array that maps ONTO ow is f(ow).
        new = np.array(fsp.DIHEDRAL[nm](ow), dtype=np.float64,
                       copy=True, order="C")
        old = np.asarray(state[pf]["delz"], np.float64)
        if new.shape != old.shape:
            raise SystemExit(f"REFUSING: face {pf + 1} delz shape "
                             f"{old.shape} vs mapped oracle {new.shape}")
        nch = int((new != old).sum())
        pre = float(np.abs(new - old).max())
        state[pf]["delz"] = new
        total_changed += nch
        print(f"  face {pf + 1} -> tile {ot + 1}: cells changed "
              f"{nch}/{old.size}  pre max|d| {pre:.4e}")
    if total_changed == 0:
        raise SystemExit(
            "TRANSPLANT CONTROL FAILED: zero cells changed -- the arm "
            "would be a no-op control (perturb-a-zero class)")
    # post-check: bitwise 0.0 under the gate's own comparison path
    p2 = fsp.port_window(state, ctx)
    for pf in range(6):
        ot = perm[pf]
        pairs, _ws = fsp.apply_map(p2[pf], orc_ic[ot], meta[pf][ot])
        pw, ow = pairs["delz"]
        d = float(np.abs(pw - ow).max())
        nbit = int((pw != ow).sum())
        if d != 0.0 or nbit != 0:
            raise SystemExit(f"REFUSING: face {pf + 1} post-transplant "
                             f"delz max|d| {d:.3e}, {nbit} cells differ "
                             f"(must be exactly 0.0 / 0)")
    print(f"TRANSPLANT COMPLETE: {total_changed} cells changed; "
          f"post-check max|d| exactly 0.0 on all 6 faces")


# ---------------------------------------------------------------------
# C. drive the unchanged gate, baseline then transplanted
# ---------------------------------------------------------------------

def run_gate(tag: str, transplant: bool) -> dict | None:
    out_json = os.path.join(OUT_DIR, f"nh_delz_transplant_{tag}.json")
    real = fsp.build_port_ic
    if transplant:
        def wrapper(ctx, ak, bk, nh=False):
            st = real(ctx, ak, bk, nh=nh)
            if nh:
                transplant_delz(st, ctx)
            return st
        fsp.build_port_ic = wrapper
    print(f"\n===== GATE ARM: {tag} =====")
    try:
        rc = fsp.main(["--nh", "--json", out_json])
    except SystemExit as e:
        rc = e.code
    finally:
        fsp.build_port_ic = real
    print(f"===== GATE ARM {tag} rc={rc}")
    if not os.path.exists(out_json):
        return None
    with open(out_json) as fh:
        return json.load(fh)


def main() -> int:
    print(f"REPO_SHA={_sha()}  n={N} ng={NG} km={KM}  ic_run={IC_RUN}")
    n0_census()
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
    print("\nNH_DELZ_TRANSPLANT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
