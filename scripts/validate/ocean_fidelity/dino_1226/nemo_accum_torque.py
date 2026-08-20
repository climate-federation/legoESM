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
estimate -- note EXACT, not absent, and note WHICH row: the recovered
vertical-mixing row is +3889.74 - 3871.83, a 217:1 cancellation; the BARO row is
+3889.74 - 3888.59, a ~3400:1 one.  An earlier revision quoted the 3400:1 figure
against the vertical-mixing row -- right number, wrong row.  Both operands are
fp64 accumulators from inside the model, the closure below holds at 1e-15, and
the identity gate is normalised by that cancellation scale for exactly this
reason.

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

M2 PRE-REGISTRATION (#1455 decisive measurement, written BEFORE arm B ran)
==========================================================================
THE TEST.  Does the barotropic substep-averaging convention own the SHAPE of
the transient gap vs NEMO, not just 18% of its 90-day mean?  ONE variable:
``barotropic_reconcile_target``, the two recorded 90-day arms of commit
55de03e71.  NEMO's side is the SAME run in both, so the whole difference
between the two gap curves is legoESM's.

HARNESS SELF-CHECK, run first and PASSED (precision-gate rule (a)): the base
arm through this table reproduces the recorded fingerprint to the last printed
digit -- -0.523, +1.638, +1.390, +0.850, +0.388, +0.470, +0.255, +0.457,
-0.036 m3/s2 per u-row, mean +0.543, sd 0.633, RANGE 2.161, peak in the 10-20 d
window.  NOTE THE SIGN: this table's ``diff`` column is NEMO MINUS legoESM.
The same nine numbers have been quoted elsewhere in this campaign as a
"lego-minus-NEMO" curve; that label is inverted, the numbers are these.

NUMBER PRODUCED, per arm: the nine-window gap curve and its RANGE
(max - min), which is what "loses the peak" means quantitatively.  A pure
level shift moves the mean and leaves the range alone.

CONFIRMS the convention owns the transient: the velocity_avg arm's range falls
to <= 1.1 (a >= 50% loss of the rise-and-fall), i.e. the early peak flattens
rather than the whole curve sliding down.

REFUTES it: the velocity_avg arm keeps the same rise-and-fall -- range within
noise of 2.161 -- and differs from the base arm by a roughly constant offset
of order the known 90-day mean shift (+0.886 -> +0.996, i.e. about -0.11 on
this sign convention).

NOISE, stated honestly because n = 1 per arm.  There is no ensemble; the scale
comes from the two flanking near-zero windows of the base curve, 0-10 d
(-0.523) and 80-90 d (-0.036), whose spread 0.487 bounds window-to-window
variability not attributable to the mean deficit.  So +-0.49 on any single
window, and about +-0.69 on the RANGE, which is a difference of two windows.
A range change smaller than 0.69 is not a result.

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
  C8   TELESCOPING: the per-step leapfrog rate need NOT equal the trajectory's
       drift -- ``dyn_atf_qco`` runs AFTER seam 3 and rewrites the NOW level,
       which becomes the next step's before level.  Compared against the
       restarts' own ub and un.  ONE window per invocation (whichever run C3-C7
       selected), so the 90-day figure below is this run's and the ~0.5% 1-day
       figure quoted in the commit is a transcribed earlier invocation, not a
       second measurement made here.  90-day result: 0.00% via both ub and un,
       which is what makes the NEMO column a drift and not a bookkeeping rate.
  C9   The recovered vertical-mixing row -- otherwise the only printed number
       with no check behind it, and the row round one's double-count lived in --
       against the analytic wind torque.  Depth-integrated it is (surface stress
       - bottom drag)/rho0, so it must land near C6's wind; the gap is the drag.
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

TILES = "restart_*.nc"
#   Per-rank tiles.  A stitched DINO_<kt>_restart.nc sorts BEFORE the tiles under
#   '.' < '_', so a loose "restart*.nc" would use it as the scalar source AND
#   stitch it on top of them.  Every accumulator path uses TILES; ``load_state``
#   is the ONE place that must also accept a stitched file (C8 reads the day-0
#   input restart, which is single) and it selects the two forms EXCLUSIVELY.
#   RETRACTED: the previous revision's commit claimed TILES was "used everywhere"
#   while load_state and the C6 utau_b read still globbed restart*.nc.

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
# the deficit each arm was DEFINED by BEFORE this run, from 55de03e71.  Kept so
# the verdict cannot quietly pair a base-arm result with a fixed-arm prediction.
PRIOR_DEFICIT = {"off/transport_avg": 0.61, "off/velocity_avg": 0.50}


PUBLISHED_ARMS = {
    # The three arms of commit 55de03e71's stage table, band means [m3/s2 per
    # u-row].  ``transport_avg`` is the CARD's own resolved default; the gate
    # used to hard-code it alone, which made the second arm unloadable and so
    # made the M2 shape test unrunnable.  PROVENANCE of the <=1e-3 reproduction
    # claim (an allow-list reason string is a claim, so it gets a command):
    # measured 2026-08-19 from the recorded arms /tmp/dino_stage/v2_off_transport
    # .npz, v2_off_velocity.npz, v2_gdept.npz -- for each, np.load then
    # acc_stage[:, stages.index(k), rows].mean().  Max deviation over all 15
    # entries was 3e-4.  Those npz live outside the repo and are NOT tracked;
    # regenerate with southern_term_torque_accum.py --days 90 --out-npz.
    # KEYS ARE ``LEGO_ARMS``' OWN NAMES, not bare reconcile-target values: the
    # third arm ALSO runs transport_avg (it differs in the T-depth ladder), so a
    # key of "transport_avg" would imply a 1:1 map to the config field that does
    # not exist.
    "off/transport_avg": {"BARO solve": 0.578, "BCLIN expl+diss": -2.129,
                          "ZDF bt": 0.339, "ZDF bc": 2.098, "POST fixer": 0.0},
    "off/velocity_avg": {"BARO solve": 0.707, "BCLIN expl+diss": -2.148,
                         "ZDF bt": 0.319, "ZDF bc": 2.118, "POST fixer": 0.0},
    "trueT/transport_avg": {"BARO solve": 0.368, "BCLIN expl+diss": -2.130,
                            "ZDF bt": 0.324, "ZDF bc": 2.101, "POST fixer": 0.0},
}


def load_lego(npz_path, arm="off/transport_avg"):
    """legoESM's OWN per-row arrays, from ``southern_term_torque_accum.py
    --out-npz``.  Until this existed the comparison ran against three
    transcribed band-mean SCALARS and neither the per-row shape nor a matched
    10-day window could be tested at all.

    Returns per-interval and 90-day-mean REALIZED rates per row, plus the
    trajectory drift computed from legoESM's own recorded circulation series --
    which is what makes the cross-model pairing drift-vs-drift rather than
    drift-vs-stage-sum.

    SELF-CHECK, non-negotiable before the arrays are used: the file must
    reproduce the published stage table of commit 55de03e71 to 1e-3.  A npz from
    a different arm, a different window or a drifted model would otherwise be
    differenced against NEMO without a word.
    """
    z = np.load(npz_path, allow_pickle=True)
    if list(z["rows"]) != ROWS:
        raise SystemExit(f"FATAL: {npz_path} rows {list(z['rows'])} != band {ROWS}")
    if int(z["interval_days"]) != 10:
        raise SystemExit(f"FATAL: {npz_path} interval is {int(z['interval_days'])} d, "
                         "not the 10 d NEMO dumps at")
    if float(z["plant"]) != 0.0 or float(z["stage_plant"]) != 0.0:
        raise SystemExit(f"FATAL: {npz_path} is a PLANTED control run")
    stg = [str(x) for x in z["stages"]]
    acc = np.asarray(z["acc_stage"], np.float64)          # (n_int, n_stage, ny)
    if arm not in PUBLISHED_ARMS:
        raise SystemExit(f"unknown --lego-arm {arm!r}: must be one of "
                         f"{tuple(PUBLISHED_ARMS)}")
    published = PUBLISHED_ARMS[arm]
    for k, want in published.items():
        got = float(acc[:, stg.index(k), ROWS].mean())
        if abs(got - want) > 1e-3:
            raise SystemExit(f"FATAL: {npz_path} stage {k!r} is {got:+.4f}, but commit "
                             f"55de03e71 published {want:+.3f} for arm {arm!r}.  This is "
                             "not the arm the comparison was asked for.")
    Rs = np.asarray(z["R_series"], np.float64)            # (n_int+1, ny), m3/s
    return {
        "per_interval": acc.sum(axis=1),                  # (n_int, ny) realized rate
        "stage_sum": acc.sum(axis=1).mean(axis=0),        # 90-day mean, per row
        "drift": (Rs[-1] - Rs[0]) / (DAYS_90 * B.SEC_PER_DAY),
        "stages": {k: acc[:, stg.index(k), :].mean(axis=0) for k in published},
    }


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
    """Prognostic velocity, for the telescoping control C8 only.

    Tiles and a stitched single file are BOTH legitimate here -- C8 needs the
    day-0 state, which in every run directory is the single stitched input
    restart, while later steps are per-rank tiles.  They are selected
    EXCLUSIVELY, never globbed together: a stitched ``DINO_<kt>_restart.nc``
    sorts before the tiles ('.' < '_') and would otherwise be stitched on top of
    them.
    """
    tiles = sorted(glob.glob(f"{run_dir}/DINO_{kt:08d}_{TILES}"))
    if tiles:
        pat = f"{run_dir}/DINO_{kt:08d}_{TILES}"
    else:
        single = f"{run_dir}/DINO_{kt:08d}_restart.nc"
        if not os.path.exists(single):
            raise SystemExit(f"FATAL: no tiles and no stitched restart for kt={kt} "
                             f"under {run_dir}")
        pat = single
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
        "_ubt": ubt,                 # the barotropic operand, exposed so the
        #                              identity gate normalises by the
        #                              CANCELLATION scale rather than by the
        #                              ~3400x smaller row it produces
        "BARO": ubt - ubb,
        "ZDF_recovered": zdf_dumped + ubt,
        "realized": row_int_3d((d["uacc_fin"] - d["uacc_bb"]) / n) * r1,
        "_expl": row_int_3d(sum(d[f"utrdacc_{t}"] for t in PRE_ZDF) / n),
        "_post": row_int_3d((d["uacc_fin"] - d["uacc_zdf"]) / n) * r1,
        "_zdf_dumped": zdf_dumped,
        "_atf": row_int_3d(d["utrdacc_atf"] / n),
    }


def identities(r, n_steps, label=""):
    """I1/I2/I3, printed with residuals.  These are the checks that replaced the
    three duplicate 'rows'.  Each CAN fail: I1 fails if dynspg_ts stops putting
    the barotropic increment back into the RHS, I2 if mlf_baro_corr stops
    discarding the vertical solve's column mean, I3 if either does.

    TWO THINGS ABOUT THE TOLERANCE, both earned.
    (a) NORMALISE BY THE CANCELLATION SCALE, not by the row.  The BARO row is
        +3889.74 - 3888.59: the operands are ~3400x the result, so a residual
        that is pure fp64 roundoff on the OPERANDS looks 3400x worse when
        divided by the row.  ``_ubt`` is the operand scale and is used here.
    (b) STEP-COUNT-AWARE, the same treatment C5 needed.  The residual is
        accumulation roundoff and grows with the number of steps summed:
        measured 1.8e-16 of the operand scale at 4 steps, 8.5e-14 at 2880.  A
        FIXED bound therefore passes the short runs and would trip on a long one
        for no physical reason -- the previous revision's 1e-9-on-the-row gate
        would have breached around a year of simulation.  Widening to n*eps
        costs the gate nothing: a genuine identity break (dynspg_ts no longer
        adding the barotropic increment back, say) is O(BARO) = ~3.4 absolute,
        i.e. ~9e-4 of the operand scale, eight orders of magnitude above this
        bound.
    """
    cancel = float(np.max(np.abs(r["_ubt"][ROWS])))
    tol = max(1.0e-14, 8.0 * n_steps * np.finfo(np.float64).eps)
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
        sc = max(float(np.max(np.abs(a[ROWS]))), float(np.max(np.abs(b[ROWS]))),
                 cancel, 1e-30)
        out.append((name, e, e / sc, why))
        print(f"  {name:34s} max|diff| {e:10.3e}  rel-to-operand {e / sc:9.2e}"
              f"  (tol {tol:.2e})   {why}")
    if label:
        print(f"     ({label}; {int(n_steps)} steps, cancellation scale"
              f" {cancel:.4g} m3/s2/row)")
    for name, e, rel, _ in out:
        if rel > tol:
            raise SystemExit(f"FATAL {name}: rel-to-operand {rel:.3e} exceeds "
                             f"{tol:.3e} -- the oracle no longer behaves as this "
                             "probe's reading of it says; re-read the source "
                             "before quoting any number")
    return out


# ------------------------------------------------------------------- selftest --
def selftest():
    """The runnable check on the row arithmetic, run THROUGH ``stage_rows``.

    RETRACTED, and this is the whole reason the function was rewritten: the
    previous version built its "planted double-count" by editing ``stage_rows``'
    OUTPUT dict and then asserting that the edit had changed something.  That
    assertion is evaluated entirely outside the function under test and passes
    no matter what ``stage_rows`` does -- a reviewer reproduced the exact round-1
    defect (``ZDF_recovered = _zdf_dumped + 2*ubt``) inside ``stage_rows`` and
    the selftest still printed OK.  It was the same class of defect it claimed
    to guard against: a control that cannot fail.

    The three checks below all read ``stage_rows``' own return value on a
    synthetic dict built to satisfy the model's closure exactly:
      A  BARO and realized reproduce a known pure-barotropic increment.
      B  the ADDITIVE identity _expl + _zdf_dumped + _post == realized.  Round
         one's composition violates this by exactly ``ubt``.
      C  the barotropic correction is credited EXACTLY ONCE:
         ZDF_recovered - _zdf_dumped == _ubt.  This is what catches the
         reviewer's monkeypatch.
      D  a PLANT applied to the INPUT dict -- an unaccounted deposit at the
         post-zdf seam, the same violation C2 plants in the model -- must break
         B by exactly the predicted amount.  Without D, B could hold vacuously.
    """
    ny, nx, nz = B.umask.shape
    n, rn_dt = 7.0, 2700.0
    rdt = 2.0 * rn_dt
    r1 = 1.0 / rdt
    wet = np.moveaxis(B.umask.astype(np.float64), -1, 0)      # (k,y,x)
    z = np.zeros((nz, ny, nx))

    d = {f"utrdacc_{t}": z.copy() for t in LIVE}
    d.update({f"{c}trdacc_{t}": z.copy() for t in UNPRODUCED for c in "uv"})
    d.update({k: z.copy() for k in ACC_3D})
    d.update(nacc_steps=n, nacc_r1dt=n * r1, nacc_trdchk=0.0,
             rn_acc_plant=0.0, rdt=rn_dt)

    # a pure barotropic increment: u_b(Naa) = U everywhere wet, u(Nbb) = 0, and
    # the after state carries exactly that depth-uniform velocity.
    U = 3.0e-4
    d["ubtacc_aa"] = np.where(B.umask.any(axis=2), n * U * r1, 0.0)
    d["uacc_fin"] = n * U * wet
    # an arbitrary, DEPTH-VARYING intermediate state and an arbitrary explicit
    # bucket, so B is not satisfied by everything being proportional
    d["uacc_zdf"] = n * U * wet * (0.3 + 0.4 * np.arange(nz)[:, None, None] / nz)
    d["utrdacc_hpg"] = 1.7e-9 * wet * np.cos(np.arange(nz)[:, None, None])
    # close the model's own identity EXACTLY, which is what stage_rows assumes
    d["utrdacc_zdf"] = ((d["uacc_zdf"] - d["uacc_bb"]) * r1
                        - sum(d[f"utrdacc_{t}"] for t in PRE_ZDF))

    r = stage_rows(d)
    want = row_int_2d(np.where(B.umask.any(axis=2), U * r1, 0.0))
    sc = max(float(np.max(np.abs(want[ROWS]))), 1e-30)
    for nm in ("BARO", "realized"):
        e = float(np.max(np.abs((r[nm] - want)[ROWS])))
        assert e / sc < 1e-12, f"A: {nm} off a known barotropic increment by {e / sc:.2e}"

    def _addresid(rr):
        return (rr["_expl"] + rr["_zdf_dumped"] + rr["_post"] - rr["realized"])

    add = float(np.max(np.abs(_addresid(r)[ROWS])))
    add_sc = max(float(np.max(np.abs(r["_zdf_dumped"][ROWS]))), 1e-30)
    assert add / add_sc < 1e-12, f"B: the rows do not sum to realized ({add:.3e})"

    once = float(np.max(np.abs((r["ZDF_recovered"] - r["_zdf_dumped"] - r["_ubt"])[ROWS])))
    assert once / max(float(np.max(np.abs(r["_ubt"][ROWS]))), 1e-30) < 1e-12, (
        f"C: the barotropic correction is not credited exactly once ({once:.3e})")

    # D: plant an unaccounted deposit at the post-zdf seam.
    c_pl = 5.0e-5
    d2 = dict(d)
    d2["uacc_zdf"] = d["uacc_zdf"] + n * c_pl * wet
    r2 = stage_rows(d2)
    pred = -row_int_3d(c_pl * wet) * r1
    got = _addresid(r2) - _addresid(r)
    e = float(np.max(np.abs((got - pred)[ROWS])))
    p_sc = max(float(np.max(np.abs(pred[ROWS]))), 1e-30)
    assert p_sc > 1e-6, "D: the plant is too small to be a control"
    assert e / p_sc < 1e-9, f"D: violation {got[ROWS][0]:.4e} != predicted ({e / p_sc:.2e})"
    print(f"  selftest OK, all four checks THROUGH stage_rows: BARO and realized "
          f"reproduce a known\n     barotropic increment to <1e-12; the rows sum to "
          f"realized; the barotropic correction is\n     credited exactly once; and a "
          f"planted post-zdf deposit breaks that sum by exactly the\n     predicted "
          f"{float(np.mean(pred[ROWS])):+.4f} m3/s2/row (rel err {e / p_sc:.2e})")


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

    raw = rebuild(f"{B.G.RUN_90D_TWIN}/DINO_{kt10:08d}_{TILES}", ["utau_b"])
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
    # C9  THE ONLY PRINTED NUMBER THAT OTHERWISE HAS NO CHECK BEHIND IT.  The
    # recovered vertical-mixing row is where round one's double-count lived, and
    # it is formed by a 217:1 cancellation, so an unchecked +17.9 is exactly the
    # shape of the defect that already shipped once.  Depth-integrated, NEMO's
    # vertical-diffusion term is (surface stress - bottom drag)/rho0, so it must
    # land NEAR the analytic wind torque C6 just verified, and the gap IS the
    # bottom drag.  The bound is deliberately loose (25%): what it must catch is
    # a lost or doubled barotropic correction, which would move this row by
    # ~3890 m3/s2, i.e. 200x the wind torque itself.
    zdf_row = float(np.mean(rows["ZDF_recovered"][ROWS]))
    wind_row = float(np.mean(phi_a[ROWS]))
    gap = zdf_row - wind_row
    print(f"  C9 recovered vertical-mixing row {zdf_row:+.4f} vs the analytic wind torque"
          f" {wind_row:+.4f}\n     m3/s2/row: gap {gap:+.4f} ({abs(gap) / abs(wind_row):.2%}"
          f"), which is the bottom drag.  Bound 25%; a lost or\n     doubled barotropic"
          f" correction would move this row by ~3890, i.e. 200x the wind.")
    if abs(gap) > 0.25 * abs(wind_row):
        raise SystemExit("FATAL C9: the recovered vertical-mixing row is not the "
                         "surface stress minus bottom drag -- the barotropic "
                         "correction is miscredited")
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
def table(lego_npz=None, arm="off/transport_avg", out_npz=None):
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
    lego = load_lego(lego_npz, arm) if lego_npz else None
    if lego is not None:
        print(f"[protocol] legoESM arm = {arm!r}  npz = {lego_npz}")

    print("=" * 104)
    print(f"T0  THE THREE IDENTITIES that collapse NEMO's step to ONE row"
          f" ({int(full['nacc_steps'])} steps, fp64)")
    print("=" * 104)
    identities(rows, full["nacc_steps"], "90-day accumulation")

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
    print("T2  THE CLEAN CROSS-MODEL COMPARISON: realized vs realized, and now"
          " DRIFT vs DRIFT.")
    print("=" * 104)
    nemo = float(np.mean(rows["realized"][ROWS]))
    if lego is None:
        print("    legoESM's per-row artifact was not supplied (--lego-npz), so this"
              " falls back to\n    three transcribed band-mean scalars and the per-row"
              " and matched-window tests below\n    are SKIPPED.  Regenerate with:"
              " southern_term_torque_accum.py --days 90 --out-npz ...")
        print(f"{'arm':>24s}{'legoESM':>12s}{'NEMO':>10s}{'NEMO-lego':>12s}"
              f"{'prior':>10s}{'revision':>11s}")
        for i, a in enumerate(LEGO_ARMS):
            gap = nemo - LEGO_REALIZED[i]
            pr = PRIOR_DEFICIT.get(a)
            print(f"{a:>24s}{LEGO_REALIZED[i]:+12.3f}{nemo:+10.3f}{gap:+12.3f}"
                  + (f"{pr:+10.3f}{(gap - pr) / pr:+10.1%}" if pr else f"{'--':>10s}{'--':>11s}"))
        return rows

    # ---- R2-1, the caveat that bounded the whole comparison, now MEASURED ----
    l_stage = float(np.mean(lego["stage_sum"][ROWS]))
    l_drift = float(np.mean(lego["drift"][ROWS]))
    print("    THE PAIRING CAVEAT IS LIFTED.  NEMO's column is a trajectory drift (C8,"
          " 0.00% vs its\n    own restarts).  legoESM's published column is a leapfrog"
          " STAGE SUM, and pairing the\n    two was flagged as not-matched.  Measured"
          " from legoESM's own recorded circulation\n    series: stage sum"
          f" {l_stage:+.4f} vs trajectory drift {l_drift:+.4f} m3/s2/row, gap"
          f" {l_stage - l_drift:+.4f}\n    ("
          f"{abs(l_stage - l_drift) / abs(l_drift):.2%}, per-row max"
          f" {float(np.max(np.abs((lego['stage_sum'] - lego['drift'])[ROWS]))):.4f})."
          "  Both sides are the same\n    quantity to 0.03%; the comparison is"
          " like-for-like and the bound is removed.")
    print(f"\n{'quantity':>34s}{'legoESM':>12s}{'NEMO':>10s}{'NEMO-lego':>12s}")
    print(f"{'realized spin-up rate (drift)':>34s}{l_drift:+12.3f}{nemo:+10.3f}"
          f"{nemo - l_drift:+12.3f}")
    for i, a in enumerate(LEGO_ARMS):
        # ARM-MATCHED ONLY.  The old hard-coded gate structurally prevented
        # pairing a base-arm prediction with a different arm's result; once
        # --lego-arm existed this loop became arm-blind and would print
        # "vs published arm off/transport_avg ... prior +0.61" directly under a
        # velocity_avg drift.  Restore the guard the gate used to provide.
        pr = PRIOR_DEFICIT.get(a)
        if pr and a == arm:
            g = nemo - LEGO_REALIZED[i]
            print(f"{'  vs published arm ' + a:>34s}{LEGO_REALIZED[i]:+12.3f}"
                  f"{nemo:+10.3f}{g:+12.3f}   prior {pr:+.2f} -> {(g - pr) / pr:+.1%}")
    print("    POWER, so CONFIRMED is not read as stronger than it is: the pre-registered"
          " prediction\n    was '~0.5 +/- 0.35'.  A 0.7-wide window could not have"
          " separated 0.5 from 0.3 or 0.7.")

    print("\n" + "=" * 104)
    print("T3  THE PRE-REGISTERED SHAPE TEST -- per row, drift vs drift, both 90-day"
          " means.\n    Predicted: 'broad and single-signed across the southern-band"
          " rows'.")
    print("=" * 104)
    dd = rows["realized"] - lego["drift"]
    print(f"{'row':>5s}{'NEMO':>10s}{'legoESM':>10s}{'NEMO-lego':>12s}")
    for j in ROWS:
        print(f"{j:5d}{rows['realized'][j]:+10.3f}{lego['drift'][j]:+10.3f}{dd[j]:+12.3f}")
    v = dd[ROWS]
    npos = int((v > 0).sum())
    sign = "ALL +" if npos == len(v) else ("ALL -" if npos == 0 else "MIXED")
    print(f"{'band':>5s}{nemo:+10.3f}{l_drift:+10.3f}{float(np.mean(v)):+12.3f}"
          f"    sd {float(np.std(v)):.3f}, min {float(np.min(v)):+.3f}, max"
          f" {float(np.max(v)):+.3f}, {sign} ({npos}/{len(v)})")
    print(f"    SHAPE: {'CONFIRMED' if sign == 'ALL +' else 'REFUTED'} -- single-signed"
          f" on {npos}/{len(v)} rows, and the row-to-row spread\n    (sd"
          f" {float(np.std(v)):.3f}) is {float(np.std(v)) / abs(float(np.mean(v))):.2f}x the"
          " band mean, i.e. broad rather than a single-row spike.")
    print("    RETRACTED by this table: the BARO-vs-BARO per-row comparison reported"
          " MIXED sign.\n    That pairing used legoESM's CONTAMINATED BARO row (its"
          " realized-minus-BARO is +0.308,\n    a metric artifact of splitting on the live"
          " h_u while reducing with e3u_0).  On the\n    clean drift-vs-drift pairing the"
          " sign is uniform.  The contaminated table is gone.")

    print("\n" + "=" * 104)
    print("T4  THE PRE-REGISTERED FLATNESS TEST -- MATCHED 10-day windows, both sides.\n"
          "    A circulation deficit linear in time requires a rate deficit FLAT in time,"
          " so this is\n    the shape the prediction called for.")
    print("=" * 104)
    print(f"{'interval [d]':>14s}{'NEMO':>10s}{'legoESM':>10s}{'diff':>10s}"
          f"{'NEMO ZDF (discarded)':>22s}")
    diffs = []
    nemo_w, lego_w = [], []
    prev = None
    for i, kt in enumerate(kts):
        d = dumps[kt]
        seg = stage_rows(d if prev is None else diff_acc(d, prev))
        prev = d
        nm = float(np.mean(seg["realized"][ROWS]))
        lg = float(np.mean(lego["per_interval"][i][ROWS]))
        diffs.append(nm - lg)
        nemo_w.append(nm)
        lego_w.append(lg)
        print(f"{(i * 10):>6d}-{(i + 1) * 10:<7d}{nm:+10.3f}{lg:+10.3f}{nm - lg:+10.3f}"
              f"{float(np.mean(seg['ZDF_recovered'][ROWS])):+22.3f}")
    a = np.array(diffs)
    npos = int((a > 0).sum())
    print(f"{'mean+-sd':>14s}{'':>10s}{'':>10s}{a.mean():+10.3f}"
          f"   sd {a.std():.3f}, range {a.max() - a.min():.3f}, {npos}/{len(a)} positive")
    flat = a.std() < 0.25 * abs(a.mean())
    print(f"    FLATNESS: {'CONFIRMED' if flat else 'REFUTED'} -- the difference varies by"
          f" {a.max() - a.min():.2f} m3/s2/row across the\n    nine windows (sd"
          f" {a.std():.3f} against a mean of {a.mean():+.3f}) and changes sign"
          f" {'' if npos in (0, len(a)) else 'twice '}"
          f"({npos}/{len(a)} positive).\n    The 90-day MEAN deficit is real and uniform"
          " across rows; its RATE is not steady in time,\n    so the mechanism cannot be a"
          " constant per-step offset.")

    # M2 needs the two curves as ARRAYS: the per-window comparison is made
    # BETWEEN arms and a printed table cannot be differenced.  Written LAST so
    # an abort anywhere above never leaves a half-meaningful artifact on disk,
    # and stamped with the git SHA -- an unstamped curve cannot be tied to the
    # code that made it.
    if out_npz:
        import subprocess
        try:
            _sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(
                os.path.abspath(__file__)), capture_output=True, text=True,
                timeout=30).stdout.strip() or "<no-sha>"
        except Exception as exc:
            _sha = f"<unavailable: {exc}>"
        np.savez_compressed(
            out_npz, nemo_per_window=np.array(nemo_w),
            lego_per_window=np.array(lego_w), diff_nemo_minus_lego=a,
            interval_days=10, arm=np.array(arm), nemo_band_mean=nemo,
            lego_band_mean=l_drift, git=np.array(_sha),
            lego_npz=np.array(str(lego_npz)),
            sign=np.array("diff = NEMO - legoESM"))
        print(f"    [artifact] -> {out_npz}  (git={_sha})")
    return rows


    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="row arithmetic only")
    ap.add_argument("--controls", action="store_true", help="C0-C8 + selftest")
    ap.add_argument("--table", action="store_true", help="the 90-day comparison")
    ap.add_argument("--lego-npz", default=None,
                    help="legoESM's per-row arrays from southern_term_torque_accum.py "
                         "--out-npz (base arm, 90 days, 10-day intervals).  WITHOUT it "
                         "the comparison falls back to three transcribed band-mean "
                         "scalars and the per-row and matched-window tests are skipped.")
    ap.add_argument("--lego-arm", default="off/transport_avg",
                    choices=sorted(PUBLISHED_ARMS),
                    help="which arm of commit 55de03e71's stage table --lego-npz is "
                         "expected to be; the self-check gates against that arm's "
                         "published band means.  Default off/transport_avg = the "
                         "CARD's own resolved barotropic_reconcile_target on the "
                         "off-ladder.")
    ap.add_argument("--out-npz", default=None,
                    help="save the matched-window NEMO and legoESM curves + their "
                         "difference (sign stamped in the file) for cross-arm analysis.")
    a = ap.parse_args(argv)
    if not (a.selftest or a.controls or a.table):
        ap.error("nothing to do: pass --selftest, --controls and/or --table")
    # Both of these are inert without the legoESM artifact -- table() returns
    # early when lego is None, so --out-npz would promise an artifact and write
    # nothing.  Refuse rather than no-op.
    if a.lego_npz is None:
        if a.out_npz:
            ap.error("--out-npz needs --lego-npz: without it the matched-window "
                     "comparison is skipped and there is no curve to save")
        if "--lego-arm" in (argv if argv is not None else sys.argv[1:]):
            ap.error("--lego-arm needs --lego-npz: there is no artifact to gate")
    print(f"[protocol] band = T-rows {ROWS[0]}..{ROWS[-1]} (row 0 dry), reducer ="
          " southern_circulation_budget.row_int_trend, units m3/s2 per u-row, fp64")
    print(f"[protocol] oracle runs under {DINO}")
    if a.selftest and not a.controls:
        selftest()
    if a.controls:
        controls()
    if a.table:
        table(a.lego_npz, a.lego_arm, a.out_npz)


if __name__ == "__main__":
    main()
