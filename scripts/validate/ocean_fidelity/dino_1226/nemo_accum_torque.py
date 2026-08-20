#!/usr/bin/env python
"""#1455 SG-B: NEMO's OWN depth-integrated momentum deposit, ACCUMULATED over
every step of the 90-day twin, reduced to the SAME band, rows and units as
legoESM's stage table (``southern_term_torque_accum.py``, commit 55de03e71).

WHY THIS RUN EXISTS.  legoESM's side of the southern-band circulation budget is
measured to fp64 roundoff.  NEMO's was not: its per-term momentum trends existed
only as INSTANTANEOUS samples in the 10-day restarts, and the barotropic-solve
row had to be reconstructed from a 595:1 cancellation whose systematic part
(~1 m3/s2 per row) does not average down over 9 samples and is larger than the
0.50 m3/s2 residual the campaign is chasing.  So the NEMO column could never
settle the question, and 55de03e71 said so and specified this run.

WHAT CHANGED IN THE ORACLE (oracle repo cfgs/DINO, MY_SRC/trddump.F90 +
MY_SRC/stpmlf.F90).  Additive, write-only, all REAL(dp): every per-term momentum
trend summed over EVERY step; ``uu_b(:,:,Naa)*r1_Dt`` summed right after
``dyn_spg``; and the 3-D velocity at three intra-step seams (the before level,
what ``dyn_zdf`` leaves, and what survives ``mlf_baro_corr`` + ``finalize_lbc``).
The cancellation is now EXACT model-side arithmetic rather than a 9-sample
estimate -- note EXACT, not absent: the recovered vertical-mixing row is still
formed as +3889.74 - 3871.83, a 3400:1 cancellation, but both operands are fp64
accumulators from inside the model and the closure below holds at 1e-15.

===========================================================================
THE HEADLINE STRUCTURAL RESULT, and the reason this file does NOT print a
five-row NEMO decomposition:  NEMO HAS ONLY ONE ROW.
===========================================================================
Three separate EXACT identities, each read from the oracle and each CHECKED
below rather than asserted, collapse NEMO's step to a single number:

  I1  explicit RHS row  ==  barotropic-solve row.
      ``dynspg_ts.F90:337`` forms the RHS's e3u_0-weighted column mean and :351
      subtracts it, but :1126 then adds ``(puu_b(Kaa)-puu_b(Kbb))*r1_Dt`` back
      in, and ``dynspg.F90:185-187`` diagnoses ``jpdyn_spg`` as the WHOLE change
      across ``dyn_spg_ts`` -- so the addition is inside the bucket.  The
      post-spg RHS's column mean is therefore exactly the barotropic solve's
      increment.  RETRACTED, and the retraction matters: an earlier revision of
      this file claimed on the strength of :351 alone that this row is "~0 by
      construction -- a prediction this probe checks rather than assumes."  It
      is not ~0, it equals the barotropic row, no check existed, and the
      sentence was written from a partial read of one routine.  Both reviewers
      caught it independently.
  I2  post-``mlf_baro_corr`` deposit  ==  MINUS the dumped vertical-mixing row.
      ``dynzdf.F90:168`` strips ``uu_b(Kaa)`` from ``puu(Kaa)``; ``mlf_baro_corr``
      (stpmlf.F90:763) re-installs it, discarding whatever column mean the
      implicit solve produced.
  I3  realized row  ==  barotropic-solve row.  Follows from I2 plus the closure.

So NEMO's depth-integrated circulation after a step is set ENTIRELY by the
barotropic solve, and the vertical solve's own barotropic deposit -- which is
essentially the wind torque, +17.9 m3/s2/row -- is THROWN AWAY.  legoESM keeps
part of it.  That is a structural difference between the two models, and it is
the most useful thing this probe produces.

WHAT THAT COSTS THE PRE-REGISTERED TEST, said plainly.  The prediction recorded
before this run was about the BARO row.  On NEMO the BARO row IS the realized
rate.  On legoESM it is not: its recorded realized (+0.886) minus its BARO
(+0.578) is +0.308, and that +0.308 is a METRIC ARTIFACT, not physics -- its
``ZDF bt`` (+0.339) and its ``BCLIN``/``ZDF bc`` pair (-2.129/+2.098, which
nearly cancel to -0.031) arise from splitting on the live h_u while reducing
with e3u_0.  Half the 0.61 deficit is therefore already inside the difference
between legoESM's own two rows.  CONSEQUENCE: the clean cross-model pairing is
REALIZED vs REALIZED -- both sides compute it as the e3u_0 row integral of an
actual state difference.  The BARO-vs-BARO pairing is still printed because it
was pre-registered, but it is labelled CONTAMINATED and no attribution is drawn
from it.

CONTROLS, each of which CAN fail (``--controls``):
  C0   BIT-IDENTITY: instrumented vs certified binary, 4 steps, every shared
       variable of every tile.
  C0b  TWIN TRAJECTORY: the instrumented run's day-10 restart against the
       RECORDED RUN_90D_TWIN's own -- the run legoESM's table was measured
       against.  C0 alone would still allow a different-but-valid trajectory.
  C1   CLOSURE, and it is NOT a telescoping identity: ``dynzdf.F90:597`` defines
       utrd_zdf against the pre-zdf RHS, so
           (uacc_zdf - uacc_bb)*r1_Dt == SUM(pre-zdf buckets) + utrd_zdf
       holds only if NOTHING outside that bucket list writes uu(:,:,:,Naa)
       between the seams.  Run at 4 steps AND at 1 day.
  C2   SYNTHETIC VIOLATION: a run with DINO_ACC_PLANT set injects a constant
       into uu(Naa) between the seams; no bucket sees it, so C1 must fail by
       exactly plant*r1_Dt*nsteps.  The plant is read from the RESTART
       (``rn_acc_plant``, written per rank), never from a Python constant.
  C3   SLOT COVERAGE -- NOT a physics waiver.  utrd_bfr/tau/bfri have no
       producer on this build (bfr is emitted only when ln_drgimp=.FALSE.;
       tau and bfri are only ever read by trdglo's printout), so their being
       zero is structural.  It is checked so the budget's term list is
       complete, and it is NOT evidence that the surface stress is accounted
       for -- the wind is +17.8 m3/s2/row and lives inside jpdyn_zdf.
  C4   CALL-COUNT: nacc_trdchk == 0 on EVERY rank => every live trend fired
       exactly once per accumulated step.
  C5   TIME STEP: nacc_r1dt/nacc_steps must reproduce 1/(2*rdt) exactly; an
       Euler first step would rescale every rate in the table.
  C6   FORCING STATE (standing rule, printed at the day-0 gate): the analytic
       DINO wind torque per row against NEMO's own dumped ``utau_b``.
  C7   CIRCULAR BY DESIGN, and labelled so: realized/BARO == 1 is guaranteed by
       algebra under key_qco (e3u(k,Kaa) = e3u_0(k)*(1+r3u), r3u depth-uniform,
       so the two column means coincide).  What it tests is the READING of
       mlf_baro_corr and that the probe's grid arrays are NEMO's -- it can never
       test physics.  Measured live-metric vs e3u_0 row difference: 0.04% of the
       0.61 deficit, so the weighting choice cannot fake or mask the signal.
  C8   TELESCOPING: the per-step leapfrog rate is NOT the trajectory's drift --
       ``dyn_atf_qco`` (stpmlf.F90:613) runs AFTER seam 3 and rewrites the NOW
       level, which becomes the next step's before level.  Compared against the
       restarts' own ub/un.  Measured at ~0.5% on the 1-day run; measured again
       over 90 days rather than assumed to stay there.
  I1/I2/I3 above are printed with their residuals by ``--table``.

WHAT THIS PROBE CANNOT SEE (the coverage statement, not a disclaimer).  It
reduces zonal momentum to ROW INTEGRALS, so intra-row compensation is invisible,
as is all of v and ssh.  Three NEMO quantities are NOT accumulated and could
still carry the residual: ``un_adv``/``ub2_b``/``un_bf`` (dynspg_ts.F90:1100-1120,
the two barotropic sub-step averages that 55de03e71 named as the partial owner),
and ``mlf_baro_corr``'s ``.NOT.ln_bt_fw`` branch, which rewrites uu(:,:,:,Kmm)
-- the NOW level, i.e. the next step's before level -- and is invisible to all
three seams.

Run (fp64 throughout; the E3T ladder gate is inherited from
``southern_circulation_budget``):
  JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=off .venv/bin/python \\
    scripts/validate/ocean_fidelity/dino_1226/nemo_accum_torque.py --selftest
  ... --controls
  ... --table
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import netCDF4 as nc
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

import southern_circulation_budget as B  # noqa: E402  (recorded band/rows/reducers)
from rebuild_nemo_restart import rebuild  # noqa: E402

DINO = os.path.dirname(B.G.RUN_90D_TWIN)
RUN_ACC90 = f"{DINO}/RUN_ACC90"
RUN_V4 = f"{DINO}/RUN_ACC_V4"
RUN_CERT4 = f"{DINO}/RUN_ACC_CERT4"
RUN_V1D = f"{DINO}/RUN_ACC_V1D"
RUN_PLANT = f"{DINO}/RUN_ACC_PLANT"
RUN_TWINCHK = f"{DINO}/RUN_ACC_TWINCHK"

TILES = "restart_*.nc"          # per-rank tiles ONLY: a stitched
#   DINO_<kt>_restart.nc would sort BEFORE the tiles under '.' < '_' and would
#   then be used as the scalar source and stitched on top of them.

PRE_ZDF = ("hpg", "spg", "keg", "rvo", "pvo", "zad", "ldf")
LIVE = PRE_ZDF + ("zdf", "atf")
UNPRODUCED = ("bfr", "tau", "bfri")     # no producer on this build -- see C3
ACC_2D = ("ubtacc_aa",)
ACC_3D = ("uacc_bb", "uacc_zdf", "uacc_fin")
SCALARS = ("nacc_steps", "nacc_r1dt", "nacc_trdchk", "rn_acc_plant", "rdt")

ROWS = list(B.ROWS)
H_U = np.sum(np.where(B.umask, B.e3u0, 0.0), axis=2)      # column depth at u-pts [m]
DAYS_90 = 90

# legoESM's recorded stage table (commit 55de03e71), band-mean m3/s2 per row,
# for the three arms of the SAME 90-day twin.  Transcribed constants: nothing
# here re-derives them, and that is a real limitation of the comparison.
LEGO_ARMS = ("off/transport_avg", "off/velocity_avg", "trueT/transport_avg")
LEGO_BARO = (+0.578, +0.707, +0.368)
LEGO_REALIZED = (+0.886, +0.996, +0.663)
LEGO_ZDF_BT_PLUS_BC = (+0.339 + 2.098, +0.319 + 2.118, +0.324 + 2.101)


# ------------------------------------------------------------------ reducers --
def _yxz(a):
    """(k,y,x) as written by NEMO -> (y,x,k), the recorded harness's layout."""
    return np.moveaxis(np.asarray(a, np.float64), 0, -1)


def row_int_3d(field_kyx):
    """sum_i e1u * sum_k e3u_0 * field, per u-row.  The RECORDED reducer
    (``southern_circulation_budget.row_int_trend``), not a re-derivation."""
    return B.row_int_trend(_yxz(field_kyx))


def row_int_2d(field_yx):
    """Row integral of a DEPTH-UNIFORM 2-D field: sum_i e1u * H_u * field, with
    H_u from e3u_0 so it is commensurable with ``row_int_3d``."""
    f = np.asarray(field_yx, np.float64)
    return np.sum(np.where(H_U > 0, f * H_U, 0.0) * B.e1u, axis=1)


# -------------------------------------------------------------------- loading --
def _scalars_all_tiles(paths):
    """Read the bookkeeping scalars from EVERY rank and require agreement.
    nacc_trdchk in particular is per-rank: a trend firing on a subset of steps
    on ONE rank would pass a rank-0-only read."""
    vals = {}
    for p in paths:
        ds = nc.Dataset(p)
        try:
            for s in SCALARS:
                if s not in ds.variables:
                    raise SystemExit(f"FATAL: scalar {s} absent from {p}")
                vals.setdefault(s, []).append(float(np.asarray(ds[s][:]).squeeze()))
        finally:
            ds.close()
    out = {}
    for s, v in vals.items():
        if len(set(v)) != 1:
            raise SystemExit(f"FATAL: scalar {s} disagrees across ranks: "
                             f"min {min(v)} max {max(v)} -- the accumulators are "
                             "not in step across the decomposition")
        out[s] = v[0]
    return out


def load_acc(run_dir, kt, fields=None):
    """Stitch one accumulated restart.  ONLY accumulators are loaded: ``diff_acc``
    differences every array in the dict, which is right for a cumulative
    accumulator and meaningless for a prognostic state.  NaN anywhere is FATAL."""
    want = list(fields) if fields is not None else (
        [f"utrdacc_{t}" for t in LIVE]
        + [f"{c}trdacc_{t}" for t in UNPRODUCED for c in "uv"]
        + list(ACC_3D) + list(ACC_2D))
    pat = f"{run_dir}/DINO_{kt:08d}_{TILES}"
    paths = sorted(glob.glob(pat))
    if not paths:
        raise SystemExit(f"FATAL: no restart tiles match {pat}")
    out = rebuild(pat, want)
    missing = [f for f in want if f not in out]
    if missing:
        raise SystemExit(f"FATAL: {pat} lacks {missing} -- stale binary?")
    out.update(_scalars_all_tiles(paths))
    for k, v in out.items():
        if isinstance(v, np.ndarray) and not np.isfinite(v).all():
            raise SystemExit(f"FATAL: non-finite in {k} of {pat}")
    return out


def load_state(run_dir, kt, fields=("ub", "un")):
    """Prognostic velocity, for the telescoping control C8 only."""
    pat = f"{run_dir}/DINO_{kt:08d}_restart*.nc"
    d = rebuild(pat, list(fields))
    for f in fields:
        if f not in d:
            raise SystemExit(f"FATAL: {pat} lacks {f}")
    return d


def r1_dt(d):
    return d["nacc_r1dt"] / d["nacc_steps"]


def diff_acc(d_hi, d_lo):
    """Accumulator difference => the mean over the interval between two dumps,
    fed through the SAME ``stage_rows`` as the whole-run mean."""
    n = d_hi["nacc_steps"] - d_lo["nacc_steps"]
    if n <= 0:
        raise SystemExit("FATAL: non-positive step count between dumps -- the "
                         "accumulators were reset (a chained job?)")
    out = {"nacc_steps": n,
           "nacc_r1dt": d_hi["nacc_r1dt"] - d_lo["nacc_r1dt"],
           "nacc_trdchk": d_hi["nacc_trdchk"],          # cumulative, not differenced
           "rn_acc_plant": d_hi["rn_acc_plant"],
           "rdt": d_hi["rdt"]}
    for k, v in d_hi.items():
        if isinstance(v, np.ndarray):
            out[k] = v - d_lo[k]
    return out


# ------------------------------------------------------------------- the rows --
def stage_rows(d):
    """NEMO's rows, m3/s2 per u-row, as means over the accumulated steps.

    Every accumulator is a SUM over ``nacc_steps``, so dividing by nacc_steps
    gives that quantity's mean and the rows are ALREADY tendencies -- the same
    normalisation as NEMO's own ``utrd_*``.  Nothing is rescaled by rDt/T_int
    (the factor-2 error retracted at 55de03e71 came from exactly such a rescale).

    Only ``BARO``, ``ZDF`` and ``realized`` are rows.  ``expl`` and ``post`` are
    returned for the identity checks I1/I2 and are NOT independent measurements:
    printing them as budget rows was the defect both reviewers caught.
    """
    n, r1 = d["nacc_steps"], r1_dt(d)
    ubt = row_int_2d(d["ubtacc_aa"] / n)                       # already * r1_Dt
    ubb = row_int_3d(d["uacc_bb"] / n) * r1
    zdf_dumped = row_int_3d(d["utrdacc_zdf"] / n)
    return {
        "BARO": ubt - ubb,
        "ZDF_recovered": zdf_dumped + ubt,
        "realized": row_int_3d((d["uacc_fin"] - d["uacc_bb"]) / n) * r1,
        "_expl": row_int_3d(sum(d[f"utrdacc_{t}"] for t in PRE_ZDF) / n),
        "_post": row_int_3d((d["uacc_fin"] - d["uacc_zdf"]) / n) * r1,
        "_zdf_dumped": zdf_dumped,
        "_atf": row_int_3d(d["utrdacc_atf"] / n),
    }


def identities(r, label=""):
    """I1/I2/I3, printed with residuals.  These are the checks that replaced the
    three duplicate 'rows'.  Each CAN fail: I1 fails if dynspg_ts stops putting
    the barotropic increment back into the RHS, I2 if mlf_baro_corr stops
    discarding the vertical solve's column mean, I3 if either does."""
    sc = max(float(np.max(np.abs(r["BARO"][ROWS]))), 1e-30)
    out = []
    for name, a, b, why in (
        ("I1  explicit RHS == BARO", r["_expl"], r["BARO"],
         "dynspg_ts.F90:337/351 remove the column mean, :1126 puts the "
         "barotropic increment back"),
        ("I2  post-fixer == -ZDF(dumped)", r["_post"], -r["_zdf_dumped"],
         "mlf_baro_corr (stpmlf.F90:763) discards the implicit solve's "
         "column mean"),
        ("I3  realized == BARO", r["realized"], r["BARO"],
         "I2 plus the closure => the barotropic solve sets the whole row"),
    ):
        e = float(np.max(np.abs((a - b)[ROWS])))
        out.append((name, e, e / sc, why))
        print(f"  {name:34s} max|diff| {e:10.3e}  rel {e / sc:9.2e}   {why}")
    if label:
        print(f"     ({label})")
    for name, e, rel, _ in out:
        if rel > 1e-9:
            raise SystemExit(f"FATAL {name}: the oracle no longer behaves as this "
                             "probe's reading of it says -- re-read the source "
                             "before quoting any number")
    return out


# ------------------------------------------------------------------- selftest --
def selftest():
    """The one runnable check on the row arithmetic itself, with a planted
    double-count -- the exact defect that shipped in the first revision."""
    ny, nx, nz = B.umask.shape
    n, rdt = 7.0, 5400.0
    z = np.zeros((nz, ny, nx))
    d = {f"utrdacc_{t}": z.copy() for t in LIVE}
    d.update({f"{c}trdacc_{t}": z.copy() for t in UNPRODUCED for c in "uv"})
    d.update({k: z.copy() for k in ACC_3D})
    d["ubtacc_aa"] = np.zeros((ny, nx))
    d.update(nacc_steps=n, nacc_r1dt=n / rdt, nacc_trdchk=0.0,
             rn_acc_plant=0.0, rdt=rdt / 2)

    # a pure barotropic increment: u_b(Naa) = U, u(Nbb) = 0
    U = 3.0e-4
    d["ubtacc_aa"][:] = n * U / rdt
    # the after state must then carry that same depth-uniform velocity
    d["uacc_fin"][:] = np.where(_yxz(np.ones_like(z)).any(-1)[None], 0.0, 0.0)
    d["uacc_fin"] = np.moveaxis(np.where(B.umask, n * U, 0.0), -1, 0)
    r = stage_rows(d)
    want = row_int_2d(np.full((ny, nx), U / rdt))
    e = float(np.max(np.abs((r["BARO"] - want)[ROWS])))
    assert e < 1e-9 * max(np.max(np.abs(want[ROWS])), 1e-30), f"BARO reducer off by {e}"
    e = float(np.max(np.abs((r["realized"] - want)[ROWS])))
    assert e < 1e-9 * max(np.max(np.abs(want[ROWS])), 1e-30), f"realized off by {e}"

    # PLANT: credit the barotropic correction twice, as the first revision did.
    # ZDF must move by exactly +ubt and the check must SEE it.
    bad = dict(r)
    bad["ZDF_recovered"] = r["ZDF_recovered"] + row_int_2d(d["ubtacc_aa"] / n)
    moved = float(np.max(np.abs((bad["ZDF_recovered"] - r["ZDF_recovered"])[ROWS])))
    assert moved > 1e-6, "the double-count plant did not move the row -- vacuous"
    print(f"  selftest OK: BARO and realized reproduce a known pure-barotropic "
          f"increment to <1e-9 relative;\n     the double-count plant moves the ZDF "
          f"row by {moved:.4f} m3/s2 (it must, and the first revision shipped it)")


# ------------------------------------------------------------------- controls --
def _tile_diff(dir_a, dir_b, kt, tag):
    fa = sorted(glob.glob(f"{dir_a}/DINO_{kt:08d}_{TILES}"))
    fb = sorted(glob.glob(f"{dir_b}/DINO_{kt:08d}_{TILES}"))
    if not fa or len(fa) != len(fb):
        raise SystemExit(f"FATAL {tag}: {len(fa)} vs {len(fb)} tiles at kt={kt}")
    npair = ndiff = 0
    worst = (0.0, None)
    for pa, pb in zip(fa, fb):
        A = nc.Dataset(pa)
        C = nc.Dataset(pb)
        try:
            sh = [v for v in A.variables if v in C.variables and A[v].dtype.kind in "fi"]
            for v in sh:
                x, y = np.asarray(A[v][:]), np.asarray(C[v][:])
                npair += 1
                m = float(np.max(np.abs(x - y))) if x.size else 0.0
                if m != 0.0:
                    ndiff += 1
                    worst = max(worst, (m, v))
        finally:
            A.close()
            C.close()
    print(f"  {tag}: {npair} (variable,tile) pairs over {len(fa)} tiles, {ndiff} differ,"
          f" worst {worst[0]:.3e} ({worst[1]})")
    return ndiff


def _closure(d, label):
    r1 = r1_dt(d)
    lhs = (d["uacc_zdf"] - d["uacc_bb"]) * r1
    rhs = sum(d[f"utrdacc_{t}"] for t in PRE_ZDF) + d["utrdacc_zdf"]
    e = float(np.max(np.abs(lhs - rhs)))
    sc = float(np.max(np.abs(rhs)))
    print(f"  C1 closure [{label}]: max|lhs-rhs| = {e:.3e} m/s2, scale {sc:.3e},"
          f" rel {e / max(sc, 1e-300):.3e}  ({int(d['nacc_steps'])} steps)")
    return e, sc


def _c8(run_dir, kt_lo, kt_hi, rows, tag):
    """C8: the per-step leapfrog rate vs the trajectory's ACTUAL drift.
    dyn_atf_qco runs after seam 3 and rewrites the NOW level, which becomes the
    next step's before level -- so the two need not agree, and the gap is the
    filter's contribution.  Measured, never assumed."""
    lo, hi = load_state(run_dir, kt_lo), load_state(run_dir, kt_hi)
    secs = (kt_hi - kt_lo) * 2.0 * B.SEC_PER_DAY / (2.0 * B.G.STEPS_PER_DAY)
    for fld in ("ub", "un"):
        drift = (B.row_circulation(_yxz(hi[fld])) - B.row_circulation(_yxz(lo[fld]))) / secs
        d = float(np.mean((rows["realized"] - drift)[ROWS]))
        rel = abs(d) / max(abs(float(np.mean(rows["realized"][ROWS]))), 1e-30)
        print(f"  C8 telescoping [{tag}, via {fld}]: per-step rate"
              f" {float(np.mean(rows['realized'][ROWS])):+.4f} vs trajectory drift"
              f" {float(np.mean(drift[ROWS])):+.4f} m3/s2/row,  gap {d:+.4f} ({rel:.2%})")
    print(f"     accumulated Asselin-filter row (utrd_atf), band mean:"
          f" {float(np.mean(rows['_atf'][ROWS])):+.4f} m3/s2/row")


def controls():
    print("=" * 104)
    print("CONTROLS -- the licence for the table.  Every one can fail.")
    print("=" * 104)
    selftest()
    kt = B.G.KT_RESTART + 4
    if _tile_diff(RUN_V4, RUN_CERT4, kt, "C0  bit-identity vs the CERTIFIED binary"):
        raise SystemExit("FATAL C0: the instrumentation moved the trajectory")
    kt10 = B.G.KT_RESTART + 10 * B.G.STEPS_PER_DAY
    if _tile_diff(RUN_TWINCHK, B.G.RUN_90D_TWIN, kt10,
                  "C0b TWIN trajectory vs the RECORDED RUN_90D_TWIN, day 10"):
        raise SystemExit("FATAL C0b: NOT the twin legoESM's table was measured against")

    d4 = load_acc(RUN_V4, kt)
    d1d = load_acc(RUN_V1D, B.G.KT_RESTART + B.G.STEPS_PER_DAY)
    e4, sc4 = _closure(d4, "4 steps")
    e1, sc1 = _closure(d1d, "1 day / 32 steps")
    if e4 / sc4 > 1e-12 or e1 / sc1 > 1e-12:
        raise SystemExit("FATAL C1: the accumulated trends do NOT sum to the state change")

    dp_ = load_acc(RUN_PLANT, kt)
    plant = dp_["rn_acc_plant"]                      # from the DATA, not a constant
    if plant == 0.0:
        raise SystemExit("FATAL C2: the plant run records rn_acc_plant = 0 -- the "
                         "control never fired")
    ep, _ = _closure(dp_, f"PLANTED {plant:g} m/s (read from the restart)")
    pred = plant * r1_dt(dp_) * dp_["nacc_steps"]
    print(f"  C2 predicted violation = plant*r1_Dt*nsteps = {pred:.6e} m/s2;"
          f" measured {ep:.6e}; ratio {ep / pred:.8f}")
    if not (0.99 < ep / pred < 1.01):
        raise SystemExit("FATAL C2: the closure does not respond to a planted "
                         "violation as predicted -- it cannot be trusted")
    if e4 >= ep * 1e-6:
        raise SystemExit("FATAL C2: the un-planted closure error is not far below the "
                         "planted one -- the control is vacuous")

    kt90 = B.G.KT_RESTART + DAYS_90 * B.G.STEPS_PER_DAY
    have90 = bool(sorted(glob.glob(f"{RUN_ACC90}/DINO_{kt90:08d}_{TILES}")))
    d = load_acc(RUN_ACC90, kt90) if have90 else d1d
    run, ktlo, kthi = ((RUN_ACC90, B.G.KT_RESTART, kt90) if have90
                       else (RUN_V1D, B.G.KT_RESTART, B.G.KT_RESTART + B.G.STEPS_PER_DAY))
    tag = "90-day run" if have90 else "1-day validation run (90-day not yet present)"
    if d["rn_acc_plant"] != 0.0:
        raise SystemExit(f"FATAL: {tag} was PLANTED (rn_acc_plant="
                         f"{d['rn_acc_plant']}) -- it is not a production run")
    for t in UNPRODUCED:
        for c in "uv":
            m = float(np.max(np.abs(d[f"{c}trdacc_{t}"])))
            if m != 0.0:
                raise SystemExit(f"FATAL C3: {c}trd_{t} has no producer on this build "
                                 f"and is not zero ({m:.3e}) -- the term list is wrong")
    print(f"  C3 slot coverage [{tag}]: utrd/vtrd_{{{','.join(UNPRODUCED)}}} all exactly 0"
          " (structural: no producer on this build).\n     NOT a statement about the wind"
          " -- the surface stress is inside jpdyn_zdf.")
    print(f"  C4 call-count nacc_trdchk = {d['nacc_trdchk']:.0f} on all ranks"
          " (0 => every live trend fired once per step)")
    if d["nacc_trdchk"] != 0.0:
        raise SystemExit("FATAL C4: a trend term did not fire once per step")
    # C5 tolerance is STEP-COUNT-AWARE on purpose.  racc_r1dt is a naive sum of
    # nacc_steps identical doubles, so its own roundoff grows ~n*eps: 1.5e-16 at
    # 32 steps, 3.0e-14 at 2880.  A fixed 1e-14 gate passes the short runs and
    # fails the long one for no physical reason.  What C5 must catch is a
    # DIFFERENT rDt on some step; a single Euler first step alone shows up at
    # 1/(2n) = 1.7e-4 relative, eight orders of magnitude above this bound, so
    # widening to n*eps costs the control nothing.
    n_ = d["nacc_steps"]
    r1, expect = r1_dt(d), 1.0 / (2.0 * d["rdt"])
    tol = max(1.0e-14, 8.0 * n_ * np.finfo(np.float64).eps)
    print(f"  C5 time step: nacc_r1dt/nacc_steps = {r1:.12e}, 1/(2*rdt) = {expect:.12e},"
          f" rel {abs(r1 - expect) / expect:.3e} vs tol {tol:.3e}"
          f"  (steps {int(n_)}; one Euler step would read 1/(2n) = {0.5 / n_:.2e})")
    if abs(r1 - expect) / expect > tol:
        raise SystemExit("FATAL C5: rDt is not 2*rdt on every step (Euler start?)")

    raw = rebuild(f"{B.G.RUN_90D_TWIN}/DINO_{kt10:08d}_restart*.nc", ["utau_b"])
    tau = np.asarray(raw["utau_b"], np.float64)
    phi_n = np.sum(np.where(B.umask[:, :, 0], tau / B.RHO0 * B.e1u, 0.0), axis=1)
    phi_a = B.phi_wind()
    dmax = float(np.max(np.abs((phi_n - phi_a)[ROWS])))
    print(f"  C6 FORCING STATE at the day-0 gate: analytic DINO wind torque vs NEMO's own"
          f" dumped utau_b,\n     rows {ROWS[0]}-{ROWS[-1]}: max|diff| {dmax:.3e} m3/s2,"
          f" band mean torque {float(np.mean(phi_a[ROWS])):+.4f} m3/s2/row,"
          f" max stress {float(np.max(np.abs(tau))):.4f} Pa")
    if dmax > 1e-10:
        raise SystemExit("FATAL C6: the wind the twin applied is not the analytic one")

    rows = stage_rows(d)
    ratio = rows["realized"][ROWS] / np.where(np.abs(rows["BARO"][ROWS]) > 0,
                                              rows["BARO"][ROWS], np.nan)
    print(f"  C7 realized/BARO = [{np.nanmin(ratio):.9f}, {np.nanmax(ratio):.9f}]"
          f"  ({tag}).  1 is EXPECTED BY\n     ALGEBRA under key_qco (r3u depth-uniform =>"
          " the e3u(Kaa) and e3u_0 column means coincide);\n     this tests the READING of"
          " mlf_baro_corr and the probe's grid arrays, never physics.")
    if not (abs(np.nanmedian(ratio) - 1.0) < 1e-6):
        raise SystemExit("FATAL C7: realized != BARO -- the reading of mlf_baro_corr "
                         "is wrong, or something after it moves an interior column mean")
    _c8(run, ktlo, kthi, rows, tag)
    print()
    return d, have90


# ---------------------------------------------------------------------- table --
def table():
    kt0 = B.G.KT_RESTART
    kts = [kt0 + day * B.G.STEPS_PER_DAY for day in range(10, DAYS_90 + 1, 10)]
    dumps = {}
    for kt in kts:
        if not sorted(glob.glob(f"{RUN_ACC90}/DINO_{kt:08d}_{TILES}")):
            raise SystemExit(f"FATAL: {RUN_ACC90} has no dump at kt={kt}; run the "
                             "90-day accumulation first")
        dumps[kt] = load_acc(RUN_ACC90, kt)
    full = dumps[kts[-1]]
    want_steps = DAYS_90 * B.G.STEPS_PER_DAY
    if int(full["nacc_steps"]) != want_steps:
        raise SystemExit(f"FATAL: the final dump accumulated {int(full['nacc_steps'])} "
                         f"steps, not {want_steps} -- this is NOT a 90-day mean (the "
                         "accumulators are SAVE-initialised and a chained job restarts "
                         "them at zero)")
    if full["rn_acc_plant"] != 0.0:
        raise SystemExit("FATAL: the 90-day run was PLANTED")
    rows = stage_rows(full)

    print("=" * 104)
    print(f"T0  THE THREE IDENTITIES that collapse NEMO's step to ONE row"
          f" ({int(full['nacc_steps'])} steps, fp64)")
    print("=" * 104)
    identities(rows, "90-day accumulation")

    print("\n" + "=" * 104)
    print(f"T1  NEMO ACCUMULATED ROWS, 90-day mean, band rows {ROWS[0]}-{ROWS[-1]}"
          " [m3/s2 per u-row].\n    Only two independent numbers: the circulation the"
          " barotropic solve deposits, and the\n    vertical-mixing deposit that"
          " mlf_baro_corr then DISCARDS.")
    print("=" * 104)
    print(f"{'row':>5s}{'BARO = realized':>18s}{'ZDF (discarded)':>18s}")
    for j in ROWS:
        print(f"{j:5d}{rows['realized'][j]:+18.3f}{rows['ZDF_recovered'][j]:+18.3f}")
    print(f"{'band':>5s}{float(np.mean(rows['realized'][ROWS])):+18.3f}"
          f"{float(np.mean(rows['ZDF_recovered'][ROWS])):+18.3f}")

    print("\n" + "=" * 104)
    print("T2  THE CLEAN CROSS-MODEL COMPARISON: realized vs realized.\n"
          "    Both sides are the e3u_0 row integral of an actual state difference over"
          " the SAME\n    90-day twin, same band, same rows, same units.  This is the"
          " spin-up rate itself.")
    print("=" * 104)
    nemo = float(np.mean(rows["realized"][ROWS]))
    print(f"{'arm':>24s}{'legoESM':>12s}{'NEMO':>10s}{'NEMO-lego':>12s}")
    for i, a in enumerate(LEGO_ARMS):
        print(f"{a:>24s}{LEGO_REALIZED[i]:+12.3f}{nemo:+10.3f}"
              f"{nemo - LEGO_REALIZED[i]:+12.3f}")

    print("\n" + "=" * 104)
    print("T3  THE PRE-REGISTERED BARO-vs-BARO TEST -- band means only, and labelled twice\n"
          "    over.  (1) CONTAMINATED: on NEMO the BARO row IS the realized rate (T0/I3);\n"
          "    on legoESM realized-minus-BARO is +0.308, a metric artifact of splitting on the\n"
          "    live h_u while reducing with e3u_0, so half the 0.61 deficit already sits\n"
          "    between legoESM's own two rows.  (2) BAND MEAN ONLY: legoESM's PER-ROW array is\n"
          "    not recorded anywhere -- commit 55de03e71 published band means -- so the\n"
          "    'broad and single-signed ACROSS ROWS' half of the pre-registered prediction is\n"
          "    UNTESTABLE from this lane.  An earlier revision printed NEMO's per-row values\n"
          "    against legoESM's band-mean SCALAR and reported the resulting spread as a\n"
          "    per-row difference; that was NEMO's own row structure wearing a cross-model\n"
          "    label, and it is retracted.  To test the shape: re-run\n"
          "    southern_term_torque_accum.py --out-npz and difference the per-row arrays.")
    print("=" * 104)
    nb = float(np.mean(rows["BARO"][ROWS]))
    print(f"{'arm':>24s}{'legoESM BARO':>14s}{'NEMO BARO':>11s}{'NEMO-lego':>11s}")
    for i, a in enumerate(LEGO_ARMS):
        print(f"{a:>24s}{LEGO_BARO[i]:+14.3f}{nb:+11.3f}{nb - LEGO_BARO[i]:+11.3f}")
    print(f"\n    NEMO's own per-row BARO spread (NOT a cross-model difference):"
          f" min {float(np.min(rows['BARO'][ROWS])):+.3f},"
          f" max {float(np.max(rows['BARO'][ROWS])):+.3f}, sd"
          f" {float(np.std(rows['BARO'][ROWS])):.3f} -- strongly structured in latitude,"
          f"\n    which is why a band mean is the only honest summary until legoESM's per-row"
          " array exists.")
    print(f"\n    for reference, NEMO's DISCARDED vertical-mixing row"
          f" {float(np.mean(rows['ZDF_recovered'][ROWS])):+.3f} vs legoESM's RETAINED"
          f" (ZDF bt + ZDF bc)\n    {LEGO_ZDF_BT_PLUS_BC[0]:+.3f}/"
          f"{LEGO_ZDF_BT_PLUS_BC[1]:+.3f}/{LEGO_ZDF_BT_PLUS_BC[2]:+.3f} -- NOT a"
          " discrepancy in vertical mixing: NEMO's never reaches the\n    circulation,"
          " and legoESM's ZDF bc is nonzero only through the same metric mismatch.")

    print("\n" + "=" * 104)
    print("T4  TIME FLATNESS -- the same rows on each 10-day interval (accumulator"
          " differences).\n    A circulation deficit LINEAR in time requires a rate"
          " deficit FLAT in time, so flat is\n    the CONFIRMING shape here, not the"
          " disqualifying one.")
    print("    NOTE the last three columns are NEMO's INTERVAL mean minus legoESM's"
          " 90-DAY mean.\n    legoESM's own per-interval series is not recorded, so ALL the"
          " time structure in those\n    columns is NEMO's.  They show whether NEMO alone is"
          " flat, not whether the DIFFERENCE is.")
    print("=" * 104)
    print(f"{'interval [d]':>14s}{'BARO = realized':>18s}{'ZDF (discarded)':>18s}"
          + "".join(f"{'-lego90 ' + a[:7]:>16s}" for a in LEGO_ARMS))
    ser = []
    prev = None
    for i, kt in enumerate(kts):
        d = dumps[kt]
        seg = stage_rows(d if prev is None else diff_acc(d, prev))
        prev = d
        m = float(np.mean(seg["realized"][ROWS]))
        ser.append(m)
        print(f"{(i * 10):>6d}-{(i + 1) * 10:<7d}{m:+18.3f}"
              f"{float(np.mean(seg['ZDF_recovered'][ROWS])):+18.3f}"
              + "".join(f"{m - LEGO_REALIZED[k]:+16.3f}" for k in range(3)))
    print(f"{'mean+-sd':>14s}{np.mean(ser):+13.3f}+-{np.std(ser):<4.2f}"
          f"   range {max(ser) - min(ser):+.3f} ({max(ser) / min(ser):.2f}x)")
    print("\n    NEMO's rate is NOT flat in time: it rises to day 30-40 and decays"
          " thereafter.  The\n    pre-registered prediction asked for a FLAT difference;"
          " with legoESM's per-interval\n    series unrecorded, flatness OF THE DIFFERENCE"
          " remains unmeasured.")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="row arithmetic only")
    ap.add_argument("--controls", action="store_true", help="C0-C8 + selftest")
    ap.add_argument("--table", action="store_true", help="the 90-day comparison")
    a = ap.parse_args(argv)
    if not (a.selftest or a.controls or a.table):
        ap.error("nothing to do: pass --selftest, --controls and/or --table")
    print(f"[protocol] band = T-rows {ROWS[0]}..{ROWS[-1]} (row 0 dry), reducer ="
          " southern_circulation_budget.row_int_trend, units m3/s2 per u-row, fp64")
    print(f"[protocol] oracle runs under {DINO}")
    if a.selftest and not a.controls:
        selftest()
    if a.controls:
        controls()
    if a.table:
        table()


if __name__ == "__main__":
    main()
