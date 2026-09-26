#!/usr/bin/env python
"""GATE: the standalone NEMO-faithful DINO run must take NEMO'S FIRST STEP.

Companion to ``nemo_dino_fromrest_gate.py``.  That gate certifies where the
run STARTS (the initial state and the surface forcing, both bit-exact against
NEMO's own record).  This one certifies the first STEP: it scores every
prognostic legoESM carries after exactly one step against NEMO's kt=1
now-level, cell by cell.

Why the same record answers both questions: ``nn_it000 = 1`` with
``ln_rstart = .false.`` sets ``l_1st_euler = .true.``, and NEMO rotates its
time-level indices (stpmlf.f90:577-580) BEFORE writing the restart
(stpmlf.f90:591).  So the file's before-level (``tb``/``sb``/``sshb``/``ub``/
``vb``) is the pre-step now-state and its now-level (``tn``/``sn``/``sshn``/
``un``/``vn``) is the state AFTER the Euler step -- exactly what one
``model.step`` from the certified initial state must reproduce.

Rule 10 (score through the model's own path): the legoESM side is NOT rebuilt
here.  It is read from a ``run_dino.py`` snapshot directory produced by the
shipped card, so the numbers scored are the ones the production driver
actually wrote.

Rule 3 (calibrate the instrument): every row prints the SIZE OF NEMO'S OWN
STEP for that field (``|now - before|``) next to the residual, so a residual
is always read against the motion it is a fraction of.  A residual that is a
large fraction of the step is a different finding from one that is a small
one, and neither can be judged without the floor.

THE BAR IS EXACT: zero cells unequal, per field.  Rows that are not at the
bar say DEBT, never "matched"/"close"/"good enough" (Rule 1b).  Quantities
the snapshot does not carry are listed as UNMEASURED with the measurement
that would close them -- they are never silently absent (Rule 1).

Usage
-----
    JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/dino_1226/\\
        nemo_dino_step1_gate.py --run-dir DIR [--restart-glob PAT] [--plant]

``DIR`` is a ``run_dino.py --output-dir`` whose ``snapshots/`` holds
``snapshot_00001.npz`` written after ONE step (``--days 0.03125
--snapshot-every-days 0.03125`` at dt=2700 s).

``--plant`` moves one wet cell of the loaded after-state temperature and one
wet U-point of the after-state velocity by 1 ulp; the gate MUST then exit
non-zero -- the non-vacuity check for the gate itself.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # scripts/validate/ocean_fidelity
sys.path.insert(0, _HERE)                           # this directory
from rebuild_nemo_restart import rebuild            # noqa: E402
# ONE Table/ulp implementation for both from-rest gates -- extended, not
# re-derived (repo rule: search before you build).
from nemo_dino_fromrest_gate import Table, _is_post_step  # noqa: E402

DEFAULT_RESTART = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
                   "RUN_FROMREST_KT1/DINO_00000001_restart_*.nc")
RN_DT = 2700.0                       # namdom rn_Dt, cfgs/DINO/*/namelist_cfg:116

# Rule 1 dispositions for every restart variable that is part of the kt=1
# AFTER state.  Anything else the file carries is either the initial state
# (certified by nemo_dino_fromrest_gate.py) or #1226 trend instrumentation,
# and is filtered out rather than waived one by one.
WAIVED = {
    "tb": "Kbb level = the INITIAL state; certified by nemo_dino_fromrest_gate",
    "sb": "Kbb level = the INITIAL state; certified by nemo_dino_fromrest_gate",
    "sshb": "Kbb level = the INITIAL state; certified by nemo_dino_fromrest_gate",
    "utau_b": "kt=1 surface forcing; certified by nemo_dino_fromrest_gate",
    "vtau_b": "kt=1 surface forcing; certified by nemo_dino_fromrest_gate",
    "emp_b": "kt=1 surface forcing; certified by nemo_dino_fromrest_gate",
    "qns_b": "kt=1 surface forcing. Certified ORACLE-SIDE ONLY by "
             "nemo_dino_fromrest_gate, and DEBT there (5304 cells, 5.7e-14, "
             "last bit of the solar term). The temperature row above is "
             "downstream of this flux, so that DEBT is not independent of "
             "it -- read the two together",
    "sfx_b": "kt=1 surface forcing; certified by nemo_dino_fromrest_gate",
    "qsr_hc_b": "traqsr.F90 per-level solar heat content bookkeeping",
    "sbc_hc_b": "trasbc.F90 applied-heat bookkeeping",
    "sbc_sc_b": "trasbc.F90 applied-salt bookkeeping",
    "fraqsr_1lev": "traqsr.F90 level-1 absorbed fraction; a consequence of "
                   "the Jerlov coefficients, not a prognostic",
    "rhd": "in-situ density anomaly diagnosed during the step, not carried "
           "in legoESM's state",
    "kt": "file bookkeeping", "ndastp": "file bookkeeping",
    "adatrj": "file bookkeeping", "ntime": "file bookkeeping",
    "rdt": "file bookkeeping", "time_counter": "file bookkeeping",
    "nav_lat": "coordinate copy", "nav_lon": "coordinate copy",
    "nav_lev": "coordinate copy",
}

# Carried by the restart AND by legoESM's state, but NOT written into
# run_dino.py's snapshot -- so this gate cannot see them.  Listed loudly with
# the measurement that would close each, never silently dropped.
# RETRACTION, 2026-09-10.  Three of these first said legoESM had no
# counterpart -- for `dissl`, `avt_k` and `avm_k` that was simply WRONG
# (state.py:556-564 carries tke_dissl / tke_avm / tke_avt as closure memory,
# and tke.py reads dissl back the next step), and for `ub`/`vb` it asserted a
# non-cancellation the oracle refutes.  A waiver's REASON is a claim; a
# plausible wrong one hides the defect permanently, which is exactly what the
# ub/vb row was doing.
UNMEASURED = {
    "ub": "MEASURED ELSEWHERE, not here. NEMO's ub/vb at kt=1 are EXACTLY "
          "zero over all 16 tiles (max|ub| = max|vb| = 0.0): from rest, "
          "mlf_baro_corr's ln_bt_fw=F tail (stpmlf.f90:720) removes exactly "
          "what dynspg_ts.f90:1003 installed. legoESM DOES build this "
          "quantity since #1729 (the Euler tail's kmm cycle), but it lives "
          "on state.u_before, which run_dino's snapshot does not write. "
          "Scored by step1_euler_term_attribution.py, which holds the state "
          "object",
    "vb": "as ub",
    "en": "legoESM carries prognostic TKE on state.tke, but _save_snapshot "
          "(scripts/run/run_dino.py) writes only eta/T/S/u/v. Closing it "
          "needs tke in the snapshot, which changes every DINO run's "
          "artifacts -- not taken unasked",
    "avt_k": "legoESM carries this as state.tke_avt (state.py:556-560, "
             "'NEMO TKE-closure coefficient memory (avm_k/avt_k)') -- it is "
             "closure memory, not merely a within-step diagnostic. Not in "
             "the snapshot; the same one-line change as en would close it",
    "avm_k": "as avt_k (state.tke_avm)",
    "dissl": "legoESM carries this as state.tke_dissl (state.py:564, NEMO's "
             "SAVE'd dissl), read back as dissl_old by the TKE closure on "
             "the next step. Not in the snapshot; same as en",
}

# NEMO writes no barotropic-transport array in this restart, so the
# depth-mean the reconciliation installs cannot be scored directly here.
NO_ORACLE_ARRAY = {
    "uu_b/vv_b (barotropic transports)":
        "dyn_atf_qco recomputes them (dynatf_qco.f90:252-265) but restart.f90 "
        "does not write them: absent from all 131 variables of this record. "
        "The depth mean is scored INDIRECTLY through un/vn, whose column "
        "average IS uu_b(Kaa) after mlf_baro_corr",
}


def _load_run(run_dir: str, mesh, index: int = 1) -> dict:
    """The legoESM side, as run_dino.py wrote it (Rule 10) -- BOUND to the run.

    An adversarial review broke the first version of this by hand-writing an
    npz containing NEMO's own kt=1 answer over an all-land domain: every row
    read AT BAR and the gate exited zero, under a header claiming it had
    scored the production driver.  A gate that certifies a FILE certifies
    nothing, so the file is now tied to the run that must have produced it:

      * ``run_metadata.json`` must exist beside ``snapshots/`` and must say
        this was the NEMO-faithful lat-lon card at dt = 2700 s;
      * the snapshot index must be step ONE, computed from that run's own dt
        and snapshot stride -- ``--dt 1350`` reaches t = 2700 s after TWO
        steps, and the npz carries no step count of its own;
      * the domain in the npz must BE NEMO's DINO domain: the land mask is
        compared bit-for-bit against the analytic mesh, which is what refuses
        the all-land forgery.
    """
    p = os.path.join(run_dir, "snapshots", f"snapshot_{index:05d}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"no snapshot at {p}")
    mpath = os.path.join(run_dir, "run_metadata.json")
    if not os.path.exists(mpath):
        raise SystemExit(
            f"no run_metadata.json in {run_dir}: a snapshot with no run "
            f"behind it cannot certify the model")
    with open(mpath) as fh:
        meta = json.load(fh)
    a = meta.get("args", {})
    dt = float(a.get("dt") or 0.0)
    if dt != RN_DT:
        raise SystemExit(
            f"this run used dt={dt} s; NEMO's kt=1 record is one {RN_DT} s "
            f"step, so its state is not comparable")
    stride = max(1, int(round(float(a.get("snapshot_every_days") or 0.0)
                              * 86400.0 / dt)))
    n_steps = index * stride
    if n_steps != 1:
        raise SystemExit(
            f"snapshot_{index:05d} is step {n_steps} of this run (stride "
            f"{stride}); this gate scores STEP 1 only")
    if not a.get("nemo_faithful_grid") or a.get("grid") != "latlon":
        raise SystemExit(
            "this run is not the NEMO-faithful lat-lon card "
            f"(nemo_faithful_grid={a.get('nemo_faithful_grid')!r}, "
            f"grid={a.get('grid')!r})")
    d = np.load(p)
    out = {k: np.asarray(d[k], dtype=np.float64) for k in d.files}
    surf = np.asarray(mesh.tmask[:, :, 0], dtype=np.float64)
    lm = out.get("land_mask")
    if lm is None or lm.shape != surf.shape or not np.array_equal(lm, surf):
        raise SystemExit(
            "the snapshot's land mask is not NEMO's DINO domain: this file "
            "was not produced by the card it claims")
    with open(p, "rb") as fh:
        out["_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    out["_path"] = p
    out["_n_steps"] = n_steps
    out["_recipe"] = a.get("recipe")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="run_dino.py --output-dir holding snapshots/"
                         "snapshot_00001.npz written after ONE step. Required "
                         "unless --oracle-self-test, which scores no legoESM "
                         "number and therefore needs no run")
    ap.add_argument("--restart-glob", default=DEFAULT_RESTART)
    ap.add_argument("--oracle-self-test", action="store_true",
                    help="replace the legoESM side with NEMO's own kt=1 "
                         "now-level; every row MUST then be AT BAR and the "
                         "gate MUST exit zero (proves the gate can pass)")
    ap.add_argument("--plant", action="store_true",
                    help="move one wet after-state T cell and one wet "
                         "U-point by 1 ulp; the gate MUST then exit non-zero "
                         "(with --oracle-self-test this is the non-vacuity "
                         "pair: pass on the right answer, fail on a 1-ulp lie)")
    args = ap.parse_args()

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64())                       # Rule 1c
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm

    tiles = sorted(glob.glob(args.restart_glob))
    if not tiles:
        raise SystemExit(f"no restart tiles match {args.restart_glob}")
    h = hashlib.sha256()
    for t in tiles:
        with open(t, "rb") as fh:
            h.update(fh.read())
    print(f"{len(tiles)} tiles, concatenated sha256[:16] = {h.hexdigest()[:16]}"
          f"   precision = {get_policy().storage.__name__}")

    import netCDF4 as nc
    ds = nc.Dataset(tiles[0])
    all_vars = set(ds.variables)
    ds.close()

    want = ["tn", "sn", "sshn", "un", "vn", "tb", "sb", "sshb"]
    R = rebuild(args.restart_glob, want)
    for k in want:
        if k not in R or np.isnan(R[k]).any():
            raise SystemExit(f"restart field {k} missing or has gaps")

    def O3(k):       # (z,y,x) -> (y,x,z), the nemo_io array order
        return np.moveaxis(R[k], 0, -1)

    g = ndm.nemo_dino_mesh()
    if args.run_dir is None:
        if not args.oracle_self_test:
            raise SystemExit("--run-dir is required (or --oracle-self-test)")
        T = S = eta = u = v = None
    else:
        run = _load_run(args.run_dir, g)
        print(f"legoESM after-state: {run['_path']}"
              f"\n  sha256 = {run['_sha256']}")
        print(f"  recipe = {run['_recipe']!r}  step = {run['_n_steps']}  "
              f"day = {float(run['time_days']):.6f} "
              f"(NEMO kt=1 = {RN_DT / 86400.0:.6f})")
        if abs(float(run["time_seconds"]) - RN_DT) > 1e-9:
            raise SystemExit(
                f"snapshot stamps t={float(run['time_seconds'])} s but its "
                f"run says step 1 of dt={RN_DT} s -- inconsistent artifact")
        T, S = run["T"].copy(), run["S"].copy()
        eta = run["eta"].copy()
        # legoESM's u/v carry one REDUNDANT face column/row (the periodic
        # image west of cell 0, and the closed southern wall); NEMO's un/vn
        # are the east/north faces.  nemo_state_bridge._u_east_to_face_periodic
        # / _v_north_to_face define the mapping, and this is its inverse.
        u = run["u"][:, 1:, :].copy()
        v = run["v"][1:, :, :].copy()

    wet3 = g.tmask > 0.5
    uwet = g.umask > 0.5
    vwet = g.vmask > 0.5

    if args.oracle_self_test:
        # NON-VACUITY, half one: feed the gate the oracle's OWN after-state.
        # Every row must read AT BAR.  Without this the gate's --plant proves
        # nothing, because the real model is not yet at the bar and the gate
        # fails either way.  This arm scores NO legoESM number and must never
        # be reported as a fidelity result.
        print("\nSELF-TEST: legoESM side REPLACED by NEMO's own kt=1 "
              "now-level. This measures the GATE, not the model.")
        T, S = O3("tn").copy(), O3("sn").copy()
        eta = np.asarray(R["sshn"], dtype=np.float64).copy()
        u, v = O3("un").copy(), O3("vn").copy()

    if args.plant:
        j, i, k = 100, 25, 3
        assert wet3[j, i, k], "plant cell must be wet"
        run_T_ref = float(T[j, i, k])
        T[j, i, k] = np.nextafter(T[j, i, k], np.inf)
        ju, iu, ku = 100, 25, 0
        assert uwet[ju, iu, ku], "plant U-point must be wet"
        run_u_ref = float(u[ju, iu, ku])
        u[ju, iu, ku] = np.nextafter(u[ju, iu, ku], np.inf)
        assert T[j, i, k] != run_T_ref, "plant on T was a no-op"
        assert u[ju, iu, ku] != run_u_ref, "plant on u was a no-op"
        print("PLANT ACTIVE: one wet after-T cell and one wet after-U point "
              "moved by 1 ulp; the gate MUST fail")

    # ---- Rule 3: the floor.  How far NEMO ITSELF moved in this one step.
    print("\nINSTRUMENT FLOOR -- the size of NEMO's own first step "
          "(|now - before| over wet cells)")
    print(f"{'field':10s} {'rms':>12s} {'max':>12s}")
    for name, now, bef, msk in (
            ("T", O3("tn"), O3("tb"), wet3),
            ("S", O3("sn"), O3("sb"), wet3),
            ("eta", R["sshn"], R["sshb"], g.tmask[:, :, 0] > 0.5)):
        d = (np.asarray(now) - np.asarray(bef))[msk]
        print(f"{name:10s} {float(np.sqrt(np.mean(d**2))):12.4e} "
              f"{float(np.max(np.abs(d))):12.4e}")

    t = Table()
    print("\nSTEP 1 -- legoESM after one step vs NEMO kt=1 now-level")
    t.check("T (all cells)", T, O3("tn"), oracle_name="tn")
    t.check("T (wet only)", T, O3("tn"), mask=wet3, oracle_name="tn")
    t.check("S (all cells)", S, O3("sn"), oracle_name="sn")
    t.check("S (wet only)", S, O3("sn"), mask=wet3, oracle_name="sn")
    t.check("eta", eta, R["sshn"], oracle_name="sshn")
    t.check("u (wet U-points)", u, O3("un"), mask=uwet, oracle_name="un")
    t.check("v (wet V-points)", v, O3("vn"), mask=vwet, oracle_name="vn")
    t.report()

    # The residual next to the floor it must be read against.  NOT a verdict:
    # a row is AT BAR or DEBT above; this is the size of the DEBT.
    print("\nRESIDUAL vs FLOOR (wet cells; DEBT rows only)")
    print(f"{'field':10s} {'rms residual':>14s} {'rms step':>12s} "
          f"{'fraction':>10s}")
    for name, built, now, bef, msk in (
            ("T", T, O3("tn"), O3("tb"), wet3),
            ("S", S, O3("sn"), O3("sb"), wet3),
            ("eta", eta, R["sshn"], R["sshb"], g.tmask[:, :, 0] > 0.5),
            ("u", u, O3("un"), None, uwet),
            ("v", v, O3("vn"), None, vwet)):
        r = (np.asarray(built) - np.asarray(now))[msk]
        rms_r = float(np.sqrt(np.mean(r ** 2)))
        if bef is None:
            # From rest the before-level velocity is identically zero, so the
            # step size IS |now|.
            s = np.asarray(now)[msk]
        else:
            s = (np.asarray(now) - np.asarray(bef))[msk]
        rms_s = float(np.sqrt(np.mean(s ** 2)))
        frac = rms_r / rms_s if rms_s > 0 else float("nan")
        print(f"{name:10s} {rms_r:14.4e} {rms_s:12.4e} {frac:10.4f}")

    # ---- Rule 1 coverage -------------------------------------------------
    print("\nCOVERAGE (oracle-fidelity Rule 1)")
    # The shared post-step filter matches on substrings, and "uu_"/"vv_" are
    # among them -- so a record that DID carry the barotropic transports
    # uu_b/vv_b would drop them silently instead of reporting them
    # unaccounted (review finding). Assert their absence instead of relying
    # on the filter to be right about it.
    _bt = sorted(v for v in all_vars if v in ("uu_b", "vv_b", "ub_b", "vb_b"))
    if _bt:
        raise SystemExit(
            f"this record carries barotropic transports {_bt}; the coverage "
            f"filter would hide them. Add explicit rows before re-running.")
    after = {v for v in all_vars if not _is_post_step(v)}
    verified = t.checked
    debt = t.debt
    unaccounted = sorted(after - verified - debt - set(WAIVED)
                         - set(UNMEASURED))
    print(f"  {len(after)} state / forcing variables in the restart "
          f"({len(all_vars)} total, {len(all_vars) - len(after)} post-step "
          f"trend and staggered dumps)")
    print(f"  {len(verified & after)} scored, {len(debt & after)} DEBT, "
          f"{len(set(WAIVED) & after)} waived, "
          f"{len(set(UNMEASURED) & after)} UNMEASURED, "
          f"{len(unaccounted)} unaccounted")
    for k in sorted(debt & after):
        print(f"  DEBT       {k}: measured, NOT at bar -- see the row above")
    for k in sorted(set(UNMEASURED) & after):
        print(f"  UNMEASURED {k}: {UNMEASURED[k]}")
    for k, why in NO_ORACLE_ARRAY.items():
        print(f"  UNMEASURED {k}: {why}")
    for k in sorted(set(WAIVED) & after):
        print(f"  WAIVED     {k}: {WAIVED[k]}")
    if unaccounted:
        print(f"  UNACCOUNTED (hard failure): {unaccounted}")
        t.failed = True

    print("\nGATE " + ("FAIL" if t.failed else "PASS"))
    return 1 if t.failed else 0


if __name__ == "__main__":
    sys.exit(main())
