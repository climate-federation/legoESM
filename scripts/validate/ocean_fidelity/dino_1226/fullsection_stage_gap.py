#!/usr/bin/env python
"""#1455 SG-C: the ACC gap decomposed by STAGE x BAND, over the FULL section.

WHAT THIS IS.  ``southern_term_torque_accum.py`` measures legoESM's per-stage
momentum deposit accumulated over the 90-day twin; ``nemo_accum_torque.py``
measures NEMO's, from accumulators compiled into the oracle.  Both reduce
ZONALLY (row circulation, ``sum_i e1u * sum_k e3u_0``).  The acceptance gate's
ACC number reduces MERIDIONALLY (``acc_thermal_wind.acc_full``: per longitude,
``sum_j sum_k u * e3t_1d * e2u``, median over longitudes 2..-2).  Those two
reductions are ORTHOGONAL, so no row-circulation table has ever been able to
decompose an ACC gap -- it could only bound a band-mean torque next to it.

This probe closes that.  ``southern_term_torque_accum.py`` now reduces the SAME
five stage increments a second time with the gate metric's own weights (its
``W_FULL`` / ``W_CHAN``, imported here, never re-derived), giving each stage's
CUMULATIVE contribution to the ACC number in Sv, per row.  This file builds the
matching NEMO-side rows from the SG-B accumulators and prints the difference.

THE PAIRING, and why it is the one it is.  legoESM's five stages come from the
leap-frog combine's own seams.  NEMO's step, read off the oracle (the sources
are cited in ``nemo_accum_torque``'s docstring), has the same four seams:

    lego "BARO solve"       <->  NEMO  e3u_0-column-mean part of
    lego "BCLIN expl+diss"  <->  NEMO  deviation part      } of (pre-spg trends
                                                             + utrd_spg)
    lego "ZDF bt" / "ZDF bc" <-> NEMO  the two parts of utrd_zdf
    lego "POST fixer"        <-> NEMO  uu after mlf_baro_corr + lbc, minus uu
                                        after dyn_zdf

THE ZDF ROW IS NOT THE SAME BUCKET ON THE TWO SIDES, and the individual
BCLIN and ZDF-bc rows must NOT be read as a cross-model attribution.  The
shipped legoESM card resolves ``surface_stress_implicit=False``, so its wind
stress and bottom drag enter the EXPLICIT leap-frog RHS and land in
BARO/BCLIN.  NEMO applies the wind as the surface boundary condition of the
``dyn_zdf`` tridiagonal and the drag through ``ln_drgimp``, so both land in
``utrd_zdf`` (this campaign's own sibling says so: nemo_accum_torque control
C3, "the wind is +17.8 m3/s2/row and lives inside jpdyn_zdf").  The BAROTROPIC
halves still pair, because on BOTH models the wind's column mean reaches the
depth-integrated circulation only through the barotropic solve.  The BAROCLINIC
halves are SWAPPED between two rows.  That is why the table and the spectrum
print ``(BCLIN + ZDF bc)`` as its own row: the PAIR is the comparable object,
each half alone is not.  Found by adversarial review, 2026-08-21.

NEMO's ``jpdyn_spg`` bucket is the WHOLE change across ``dyn_spg_ts``, which
both removes the pre-spg RHS's column mean and adds the barotropic solve's
increment back (dynspg_ts.F90:337/351/1126, dynspg.F90:185-187).  So
(pre-spg trends + utrd_spg) is exactly "explicit deviation + barotropic
increment", the same object lego's combine forms -- which is what makes the
column-mean split the SAME split on both sides rather than an analogy.

THE THICKNESS THE SPLIT USES IS NOT THE SAME SYMBOL ON THE TWO SIDES, and an
earlier revision of this docstring asserted that it was.  NEMO splits on the
STATIC ``e3u_0`` (dynspg_ts.F90:330-333 under key_qco); legoESM's combine
splits on the LIVE ``h_u``.  Those are the same OPERATOR only if h_u is a
depth-uniform multiple of e3u_0, which is a property of the vertical
coordinate, not of either model's dynamics.  It is therefore MEASURED, by the
M3 gate in ``southern_term_torque_accum.py``, which prints the per-column
spread of h_u/e3u_0 across levels and refuses to run if it is not zero.  Read
that line before reading any BARO/BCLIN partition here.

WHICH ROWS SURVIVE THIS REDUCER, MEASURED RATHER THAN PREDICTED.  I expected
the ``ZDF bt`` + ``POST fixer`` cancellation to be e3u_0-specific -- i.e. that
under the gate metric's e3t_1d weights the discarded vertical deposit would
reach the ACC number.  IT DOES NOT.  On both models the pair cancels as FIELDS,
to fp64 (measured: max|POST + ZDF bt| = 4.1e-13 m/s against a 39 m/s field on
NEMO's side), so it cancels under EVERY weighting.  NEMO's ``mlf_baro_corr``
and legoESM's second reconciliation each delete the implicit solve's whole
barotropic deposit, not merely its e3u_0 column mean.  The prediction is
RETRACTED -- and the reason it was wrong is worth writing down, because the
same inference will be tempting again: BOTH of those rows are DEPTH-UNIFORM
per column (``POST`` is uu_b(Kaa) minus a column mean, ``ZDF bt`` is a
broadcast column mean), and a depth-uniform field's reduction is proportional
to sum_k w_k for ANY weights w.  Equal-and-opposite column values therefore
cancel under e3u_0, e3t_1d, e3t_0, or anything else.  Surviving the change of
weighting was never evidence.  The pair row is printed so the reader sees the
cancellation rather than a claim about it.

AND THE MAGNITUDE OF THAT PAIR IS BOOKKEEPING, NOT PHYSICS.  On NEMO's side
the two halves run to ~1.7e5 Sv before cancelling; that is ``dynzdf.F90:168``
stripping uu_b(Kaa) and ``mlf_baro_corr`` re-installing it, a seam 3e4 times
the gap.  legoESM's halves are the residual of a zero-flux solve and are ~1e4
times smaller.  The two are NOT the same object and no difference of the
individual halves means anything; only the (zero) pair does.

CONTROLS, with an HONEST statement of what each one covers.  Round 1 of the
adversarial review found that the obvious reading of two of them was wrong, so
the coverage is written down next to each rather than left to be inferred.
  G0  the imported weights, applied on the host, against the npz's own
      device-computed day-0 rows.  NOT "two independent implementations": both
      sides use the SAME weight arrays, so this covers the reduction, the
      slice and the axis order -- not the weights.  The weights are covered by
      M0/M0b in ``southern_term_torque_accum``, which check them against
      ``acc_full``'s and ``acc_band``'s own integrands.
  G1  day-0 lego minus NEMO must vanish -- the bridge is bit-identical, so a
      non-zero day-0 gap means the two sides are read on different arrays or
      different time levels.
  G2  NEMO's accumulator run (RUN_ACC90) must sit on the SAME trajectory as the
      RUN_90D_TWIN the lego side is scored against: their day-90 ``un`` are
      compared cell by cell.  C0/C0b in ``nemo_accum_torque`` test the binary
      and the day-10 restart; this tests the endpoint the gap is quoted at.
  G2b the accumulator bookkeeping: exactly 320 accumulated steps per 10-day
      window (so window 0's cumulative-from-zero read is right and no dump was
      reset or chained), ``nacc_trdchk == 0`` (every live trend fired on every
      step), and ``rn_acc_plant == 0`` (not a planted control run).
  G3  NEMO's five stage increments must sum to (uacc_fin - uacc_bb) per window,
      which is the C1 closure carried through this reducer.  COVERAGE: the
      bt/bc split CANCELS out of this sum, so G3 cannot see ``split_bt``.
  G3b the ONE control on ``split_bt``, the only new NEMO-side numeric here:
      its barotropic part, reduced with the RECORDED row-circulation reducer,
      must reproduce ``nemo_accum_torque.stage_rows``'s already-validated
      ``BARO`` row -- which is built by a completely different route (the 2-D
      ``ubtacc_aa`` accumulator minus the before-level, not a column mean of
      the trend sum).  A wrong column-mean weighting moves one and not the
      other.
  G4  the stage x band table's grand total must reproduce the DIRECTLY measured
      day-90 gap.  COVERAGE, and this is the correction that matters: because
      ``nemo_off`` is defined as a residual, every NEMO stage term CANCELS out
      of this identity algebraically.  G4 therefore tests the legoESM
      telescoping and G1, and is BLIND to the NEMO side -- the NEMO side is
      covered by G3 and G3b, not by G4.
  G4b the artifact's own stamps: weightings, band row lists, stage order,
      interval length, window count and the vertical-ladder mode the lego run
      integrated on, each compared against what this analysis assumes.
  G5  a PLANTED violation: with ``--plant-stage N --plant-sv X`` a constant Sv
      is injected into stage N's lego row.  G4 must go red by exactly X (G3 is
      NEMO-only and is correctly unaffected).  Measured: a +0.2500 Sv plant
      moved the table total from -0.401354 to -0.151354 and tripped G4.

CANCELLATION (the SG-B caveat, carried forward and RE-SIZED).  NEMO's
barotropic row is a difference of two large operands (``utrd_spg`` against the
pre-spg trends).  SG-B labelled that row PLAUSIBLE because it was recovered
from NINE INSTANTANEOUS SAMPLES, where the cancellation multiplied a SAMPLING
error of order 1 m3/s2 per row.  Here the operands are fp64 accumulators summed
inside the model, so the same cancellation multiplies fp64 ROUNDOFF instead.
The ratio is still printed per band, but the LABEL keys off the resulting
bound, ``eps*sqrt(nsteps)*|operand|``, not off the ratio -- a warning without a
number behind it is not a caveat, it is a superstition.  Measured on the 90-day
run: worst band 4.3e-09 Sv, i.e. 1.1e-06 % of the gap.

This probe prints measurements.  It does not print a verdict.

Run:
  JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python \
    scripts/validate/ocean_fidelity/dino_1226/fullsection_stage_gap.py \
    --lego-npz /tmp/dino_fs/accum90.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

import acc_thermal_wind as A                       # noqa: E402
import southern_circulation_budget as B            # noqa: E402
import nemo_accum_torque as N                      # noqa: E402
from rebuild_nemo_restart import rebuild           # noqa: E402
from southern_term_torque_accum import (           # noqa: E402  (never re-derived)
    W_FULL, W_CHAN, MET_ILON, MET_NLON, BANDS, FULL_ROWS, CHAN_ROWS,
)

STAGES = ("BARO solve", "BCLIN expl+diss", "ZDF bt", "ZDF bc", "POST fixer")
I_BARO = STAGES.index("BARO solve")
I_BCLIN = STAGES.index("BCLIN expl+diss")
I_ZBT = STAGES.index("ZDF bt")
I_ZBC = STAGES.index("ZDF bc")
I_POST = STAGES.index("POST fixer")
METW = ("FULL e3t_1d", "CHAN e3t_0")
_WEIGHTS = (W_FULL, W_CHAN)
KT0 = 5760                                          # the day-180 twin IC
STEPS_PER_DAY = 32
# The campaign's quoted day-90 gaps, for orientation ONLY.  They were measured
# on the "both" ladder with the card as it stood at d180_step_walk.py:249; the
# shipped card has since changed (velocity_avg + nemo_mlf_baro_corr, commit
# 1d5b19a1d), so a fresh artifact is NOT required to reproduce them and a
# difference is a finding, not a failure.  Never gate on these.
QUOTED = {"FULL e3t_1d": -0.5965, "CHAN e3t_0": +0.2874}


# ------------------------------------------------------------------ reducers --
def met_rows(u_yxz, w):
    """The gate metric's per-row integrand [Sv], MEAN-reduced over longitudes
    2..-2.  ``w`` is one of the imported weight arrays; nothing about the
    weighting is decided here."""
    return np.einsum("jik,jik->ji", np.asarray(u_yxz, np.float64), w
                     )[:, MET_ILON].sum(axis=1) / (MET_NLON * 1.0e6)


_H_E3U0 = np.sum(np.where(B.umask, B.e3u0, 0.0), axis=2)     # column depth [m]


def split_bt(x_yxz):
    """(barotropic, baroclinic) parts of a u field, split on the e3u_0-weighted
    column mean -- NEMO's own ``r1_hu_0`` weighting (dynspg_ts.F90:330-333) and,
    at LEGOESM_NEMO_E3T=both, the same thickness legoESM's combine splits on."""
    x = np.where(B.umask, np.asarray(x_yxz, np.float64), 0.0)
    col = np.sum(x * B.e3u0, axis=2) / np.maximum(_H_E3U0, 1.0e-10)
    bt = np.where(B.umask, col[:, :, None], 0.0)
    return bt, x - bt


# --------------------------------------------------------------- NEMO's side --
def _yxz(a):
    return np.moveaxis(np.asarray(a, np.float64), 0, -1)


def nemo_stage_increments(run_dir, kt_lo, kt_hi):
    """NEMO's five accumulated stage INCREMENTS [m/s, summed over the window],
    in legoESM's stage order.  Built from the SG-B accumulators; the pairing and
    its source lines are in this module's docstring."""
    hi = N.load_acc(run_dir, kt_hi)
    if kt_lo is None:
        d = hi
    else:
        d = N.diff_acc(hi, N.load_acc(run_dir, kt_lo))
    r1 = N.r1_dt(d)                                   # = r1_Dt = 1/(2*rdt)
    expl = _yxz(sum(d[f"utrdacc_{t}"] for t in N.PRE_ZDF)) / r1
    zdf = _yxz(d["utrdacc_zdf"]) / r1
    post = _yxz(d["uacc_fin"] - d["uacc_zdf"])
    realized = _yxz(d["uacc_fin"] - d["uacc_bb"])
    e_bt, e_bc = split_bt(expl)
    z_bt, z_bc = split_bt(zdf)
    # BOTH operands of the cancellation, so the ratio below is
    # max(|spg|, |pre-spg|)/|net| -- the same normalisation nemo_accum_torque's
    # identity gate uses.  Quoting only one operand understates it.
    spg_bt, _ = split_bt(_yxz(d["utrdacc_spg"]) / r1)
    pre_bt, _ = split_bt(_yxz(sum(d[f"utrdacc_{t}"] for t in N.PRE_ZDF
                                  if t != "spg")) / r1)
    # ---- G3b: the ONE control on split_bt, and it is not a tautology.
    # ``nemo_accum_torque.stage_rows`` builds the SAME barotropic row by a
    # completely different route -- the 2-D ``ubtacc_aa`` accumulator (NEMO's
    # own uu_b(Kaa)*r1_Dt, summed in the model) minus the before-level column
    # mean -- with no reference to the trend sum or to any column-mean operator
    # written here.  If ``split_bt`` used the wrong weights, this route would
    # not move and the two would part company.  Normalised by the CANCELLATION
    # scale, for the reason nemo_accum_torque's identity gate gives.
    n = d["nacc_steps"]
    ref = N.stage_rows(d)
    got = B.row_int_trend(e_bt / n) * r1
    resid = float(np.max(np.abs((got - ref["BARO"])[N.ROWS])))
    scale = max(float(np.max(np.abs(ref["_ubt"][N.ROWS]))), 1.0e-30)
    return {
        "stages": np.stack([e_bt, e_bc, z_bt, z_bc,
                            np.where(B.umask, post, 0.0)]),
        "realized": np.where(B.umask, realized, 0.0),
        "cancel_operand": np.stack([spg_bt, pre_bt]),   # cancellation ratio
        "n_steps": int(n),
        "trdchk": float(d["nacc_trdchk"]),
        "plant": float(d["rn_acc_plant"]),
        "g3b_resid": resid,
        "g3b_scale": scale,
    }


def nemo_un(run_dir, kt):
    """NEMO's NOW-level velocity (y,x,z), tiles or stitched, NaN -> 0 on land."""
    tiled = f"{run_dir}/DINO_{kt:08d}_restart_*.nc"
    if glob.glob(tiled):
        un = rebuild(tiled, ["un"])["un"]
    else:
        single = f"{run_dir}/DINO_{kt:08d}_restart.nc"
        if not os.path.exists(single):
            raise SystemExit(f"FATAL: no NEMO restart for kt={kt} under {run_dir}")
        import netCDF4 as nc
        with nc.Dataset(single) as ds:
            un = np.asarray(ds["un"][0], dtype=np.float64)
    return np.nan_to_num(_yxz(un), nan=0.0)


# ------------------------------------------------------------------- helpers --
def _fmt(v):
    return f"{v:14.4f}"


def _band_table(title, per_stage, extra_rows, bands):
    """per_stage: (n_stage, NY).  extra_rows: list of (label, (NY,) array).

    The two CANCELLING PAIRS are printed as their own rows, because neither
    half is separately readable: ``ZDF bt`` + ``POST fixer`` is zero to fp64
    on BOTH sides (mlf_baro_corr on NEMO's, the second reconciliation on
    legoESM's, both measured), and ``BCLIN`` + ``ZDF bc`` is a near-cancelling
    pair whose halves run ~25x the gap on legoESM's side and ~200x on NEMO's --
    a split of the same baroclinic adjustment between two seams, not two
    independent deposits.  A per-half number from either pair is a difference
    of two large quantities and is reported for completeness only.
    """
    print("\n" + "-" * 104)
    print(f"  {title}")
    print("-" * 104)
    print(f"  {'row':24s}" + "".join(f"{b:>14s}" for b, _ in bands))
    for t, name in enumerate(STAGES):
        print(f"  {name:24s}" + "".join(_fmt(per_stage[t, r].sum())
                                        for _, r in bands))
    for lab, ta, tb in (("  (ZDF bt + POST)", I_ZBT, I_POST),
                        ("  (BCLIN + ZDF bc)", I_BCLIN, I_ZBC)):
        pair = per_stage[ta] + per_stage[tb]
        print(f"  {lab:24s}" + "".join(_fmt(pair[r].sum()) for _, r in bands))
    print(f"  {'STAGE SUM':24s}"
          + "".join(_fmt(per_stage[:, r].sum()) for _, r in bands))
    for label, arr in extra_rows:
        print(f"  {label:24s}" + "".join(_fmt(arr[r].sum()) for _, r in bands))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lego-npz", required=True)
    ap.add_argument("--nemo-run", default=N.RUN_ACC90)
    ap.add_argument("--twin-run", default=None,
                    help="the RUN_90D_TWIN the lego side is scored against "
                         "(default: acceptance_gate_90d's)")
    ap.add_argument("--plant-stage", type=int, default=-1,
                    help="G5: inject --plant-sv into this lego stage index")
    ap.add_argument("--plant-sv", type=float, default=0.0)
    ap.add_argument("--out-npz", default=None)
    args = ap.parse_args(argv)
    twin_run = args.twin_run or B.G.RUN_90D_TWIN

    sha = subprocess.run(["git", "-C", _DIR, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", _DIR, "status", "--porcelain",
                            "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    print(f"[provenance] git {sha}  tracked-dirty={'YES' if dirty else 'no'}  "
          f"LEGOESM_NEMO_E3T={os.environ.get('LEGOESM_NEMO_E3T')!r}")
    print(f"[provenance] lego {args.lego_npz}")
    print(f"[provenance] NEMO accumulators {args.nemo_run}; twin {twin_run}")

    # ------------------------------------------------------------- lego side --
    z = np.load(args.lego_npz, allow_pickle=True)
    for k in ("acc_met", "acc_moffs", "Mnow_series", "acc_med_series"):
        if k not in z.files:
            raise SystemExit(f"FATAL: {args.lego_npz} has no {k!r} -- it predates "
                             "the gate-metric extension; re-run "
                             "southern_term_torque_accum.py")
    if float(z["plant"]) or float(z["stage_plant"]):
        raise SystemExit("FATAL: the lego artifact is a PLANTED control run")
    if int(z["interval_days"]) != 10:
        raise SystemExit("FATAL: lego intervals are not the 10 d NEMO dumps at")
    if [str(x) for x in z["stages"]] != list(STAGES):
        raise SystemExit(f"FATAL: lego stage order {list(z['stages'])} != {STAGES}")
    # ---- G4b: everything this analysis ASSUMES about the artifact, checked
    #      against what the artifact says about itself.  A band list or a
    #      weighting that moved between the producing run and this analysis
    #      would otherwise mismatch in silence.
    for key, want, what in (
            ("met_weightings", list(METW), "metric weightings"),
            ("full_rows", FULL_ROWS, "full-section row list"),
            ("chan_rows", CHAN_ROWS, "channel row list")):
        if key not in z.files:
            raise SystemExit(f"FATAL G4b: artifact has no {key!r}")
        if [int(x) if key != "met_weightings" else str(x)
                for x in z[key]] != want:
            raise SystemExit(f"FATAL G4b: the artifact's {what} is not this "
                             f"analysis's ({list(z[key])} vs {want})")
    for key in ("e3t_mode", "days", "n_int", "git_sha"):
        if key not in z.files:
            raise SystemExit(f"FATAL G4b: artifact has no {key!r} stamp -- it "
                             "predates the provenance stamp; re-run "
                             "southern_term_torque_accum.py")
    _e3t = str(z["e3t_mode"])
    _env_e3t = os.environ.get("LEGOESM_NEMO_E3T")
    print(f"  [G4b stamps] lego run: e3t ladder {_e3t!r}, days "
          f"{int(z['days'])}, {int(z['n_int'])} windows, git "
          f"{str(z['git_sha'])[:9]}; this analysis LEGOESM_NEMO_E3T="
          f"{_env_e3t!r}")
    if _env_e3t is not None and _e3t != _env_e3t:
        raise SystemExit(f"FATAL G4b: the lego run integrated on ladder "
                         f"{_e3t!r} but this analysis is configured for "
                         f"{_env_e3t!r} -- the geometry the rows are reduced "
                         "with would not be the geometry they were made on")
    if int(z["days"]) != 90:
        raise SystemExit(f"FATAL G4b: the artifact covers {int(z['days'])} "
                         "days; every label in this table says 90 and the "
                         "NEMO side is read at kt(day 90)")
    acc_met = np.asarray(z["acc_met"], np.float64)          # (n_int,2,5,NY) Sv
    acc_moffs = np.asarray(z["acc_moffs"], np.float64)      # (n_int,2,NY)  Sv
    Mnow = np.asarray(z["Mnow_series"], np.float64)         # (n_int+1,2,NY)
    med = np.asarray(z["acc_med_series"], np.float64)
    n_int = acc_met.shape[0]
    if args.plant_stage >= len(STAGES):
        raise SystemExit(f"--plant-stage must be < {len(STAGES)} (stages "
                         f"{STAGES})")
    if args.plant_stage >= 0:
        acc_met[:, :, args.plant_stage, FULL_ROWS[0]] += (
            args.plant_sv / n_int)
        print(f"  [G5 PLANT] {args.plant_sv:+.4f} Sv injected into lego stage "
              f"{STAGES[args.plant_stage]!r} at row {FULL_ROWS[0]} -- G3 is "
              "unaffected (NEMO side) and G4 MUST go red by exactly this")

    # ---- G0: the imported weights, applied on the host, must reproduce the
    #      npz's own device-computed day-0 rows.
    u0 = nemo_un(twin_run, KT0)
    for w, name in enumerate(METW):
        e = float(np.max(np.abs(met_rows(u0, _WEIGHTS[w]) - Mnow[0, w])))
        print(f"  [G0 reduction gate {name}] host reducer on NEMO's day-0 un "
              f"vs the npz's device rows on legoESM's bridged day-0 u, "
              f"max|diff| = {e:.3e} Sv/row\n      (covers the reduction, the "
              f"slice and the axis order; NOT the weights -- both sides import "
              f"the same\n      weight arrays.  The weights are M0/M0b/M0d in "
              f"southern_term_torque_accum.)")
        if e > 1.0e-10:
            raise SystemExit("FATAL G0: the two metric reducers disagree")

    # ---- G2c: nemo_met[0] is read from the twin run and every later window
    #      from the accumulator run.  G2 gates them equal at day 90; this gates
    #      the other end, so the NEMO series cannot straddle two runs.
    _u0_acc = nemo_un(args.nemo_run, KT0)
    _d0 = float(np.max(np.abs(_u0_acc - u0)))
    print(f"  [G2c day-0 gate] RUN_ACC90 vs RUN_90D_TWIN day-0 un, max|diff| = "
          f"{_d0:.3e} m/s")
    if _d0 != 0.0:
        raise SystemExit("FATAL G2c: the two NEMO runs do not share their "
                         "day-0 state")

    # ---- G1: the bridge is bit-identical, so day 0 must cancel.
    for w, name in enumerate(METW):
        g0 = Mnow[0, w] - met_rows(u0, _WEIGHTS[w])
        print(f"  [G1 day-0 gate {name}] lego-minus-NEMO at day 0, full-section "
              f"sum = {g0[FULL_ROWS].sum():.3e} Sv")
        if abs(float(g0[FULL_ROWS].sum())) > 1.0e-9:
            raise SystemExit("FATAL G1: the twin does not start from NEMO's state")

    # ---- G2: the accumulator run must BE the twin the gap is quoted against.
    kt90 = KT0 + 90 * STEPS_PER_DAY
    un_acc, un_twin = nemo_un(args.nemo_run, kt90), nemo_un(twin_run, kt90)
    d90 = float(np.max(np.abs(un_acc - un_twin)))
    print(f"  [G2 trajectory gate] RUN_ACC90 vs RUN_90D_TWIN day-90 un, "
          f"max|diff| = {d90:.3e} m/s against max|un| = "
          f"{float(np.max(np.abs(un_twin))):.4f}")
    if d90 > 1.0e-12:
        raise SystemExit("FATAL G2: the accumulator run is a DIFFERENT "
                         "trajectory from the twin the lego side is scored "
                         "against -- the difference table would straddle two runs")

    # ------------------------------------------------------------- NEMO side --
    nem_stage = np.zeros((n_int, 2, len(STAGES), B.NY))
    nem_real = np.zeros((n_int, 2, B.NY))
    nem_cancel = np.zeros((n_int, 2, 2, B.NY))
    nemo_met = np.zeros((n_int + 1, 2, B.NY))
    for w in range(2):
        nemo_met[0, w] = met_rows(u0, _WEIGHTS[w])
    for i in range(n_int):
        kt_lo = KT0 + i * 10 * STEPS_PER_DAY
        kt_hi = kt_lo + 10 * STEPS_PER_DAY
        r = nemo_stage_increments(args.nemo_run, None if i == 0 else kt_lo, kt_hi)
        # ---- G2b: the accumulator bookkeeping.  Window 0 reads a
        #      cumulative-from-zero dump because there is no accumulator at the
        #      run's own start; that is only right if the run started AT kt0 and
        #      no dump was reset or chained, which is exactly what the step
        #      count says.  Nothing else here can see a wrong step count.
        _want = 10 * STEPS_PER_DAY
        if r["n_steps"] != _want:
            raise SystemExit(f"FATAL G2b: window {i} accumulated "
                             f"{r['n_steps']} steps, not {_want} -- the "
                             "accumulators were reset, chained, or the run does "
                             "not start at kt0")
        if r["trdchk"] != 0.0:
            raise SystemExit(f"FATAL G2b: nacc_trdchk = {r['trdchk']} at window "
                             f"{i} -- a live momentum trend did not fire on "
                             "every accumulated step")
        if r["plant"] != 0.0:
            raise SystemExit(f"FATAL G2b: rn_acc_plant = {r['plant']} -- this is "
                             "a PLANTED NEMO control run")
        # ---- G3: the five stage increments must sum to the realized increment.
        res = float(np.max(np.abs(r["stages"].sum(axis=0) - r["realized"])))
        sc = float(np.max(np.abs(r["realized"])))
        if sc > 0 and res / sc > 1.0e-9:
            raise SystemExit(f"FATAL G3: NEMO's stage increments do not close on "
                             f"the realized increment in window {i} "
                             f"(residual {res:.3e} on {sc:.3e})")
        # ---- G3b: split_bt against the independent ubtacc_aa route.
        if r["g3b_resid"] / r["g3b_scale"] > 1.0e-12:
            raise SystemExit(
                f"FATAL G3b: the column-mean split disagrees with "
                f"nemo_accum_torque's BARO row in window {i} "
                f"(residual {r['g3b_resid']:.3e} on a cancellation scale of "
                f"{r['g3b_scale']:.3e}) -- split_bt is not NEMO's own split")
        un_i = nemo_un(args.nemo_run, kt_hi)
        for w in range(2):
            nem_stage[i, w] = np.stack([met_rows(x, _WEIGHTS[w])
                                        for x in r["stages"]])
            nem_real[i, w] = met_rows(r["realized"], _WEIGHTS[w])
            nem_cancel[i, w] = np.stack(
                [met_rows(x, _WEIGHTS[w]) for x in r["cancel_operand"]])
            nemo_met[i + 1, w] = met_rows(un_i, _WEIGHTS[w])
        print(f"  window {i} (days {10*i}-{10*(i+1)}): NEMO steps="
              f"{r['n_steps']}, G3 residual {res:.2e} m/s on {sc:.2e}, "
              f"G3b split-check {r['g3b_resid']:.2e} on {r['g3b_scale']:.2e} "
              f"(rel {r['g3b_resid'] / r['g3b_scale']:.1e})")

    # =========================================================== the tables ===
    for w, wname in enumerate(METW):
        bands = BANDS if w == 0 else (("channel", CHAN_ROWS),)
        lego_cum = acc_met[:, w].sum(axis=0)                  # (5,NY)
        lego_off = acc_moffs[:, w].sum(axis=0)                # (NY,)
        lego_real = Mnow[-1, w] - Mnow[0, w]
        nemo_cum = nem_stage[:, w].sum(axis=0)
        nemo_real_states = nemo_met[-1, w] - nemo_met[0, w]
        nemo_off = nem_real[:, w].sum(axis=0) - nemo_real_states

        print("\n" + "=" * 100)
        print(f"CUMULATIVE leap-frog stage INCREMENTS over 90 days [Sv], "
              f"{wname} weighting")
        print("=" * 100)
        print("  UNITS.  A leap-frog increment spans 2*dt while the realized "
              "change of the NOW level\n  advances by dt, so STAGE SUM is ~2x "
              "the realized d(ACC) and the offset row takes the\n  other half "
              "back.  A stage row is NOT 'its share of the ACC change'; the "
              "decomposition\n  that means something is of the STAGE SUM, and "
              "the spectrum below uses that denominator.")
        _band_table("legoESM", lego_cum,
                    [("- leapfrog offset", -lego_off),
                     ("= realized d(ACC)", lego_real)], bands)
        _band_table("NEMO", nemo_cum,
                    [("- leapfrog offset+atf", -nemo_off),
                     ("= realized d(ACC)", nemo_real_states)], bands)
        _band_table("legoESM MINUS NEMO", lego_cum - nemo_cum,
                    [("- offset difference", -(lego_off - nemo_off)),
                     ("= GAP", lego_real - nemo_real_states)], bands)

        # ---- G4: the table's total must be the directly measured gap.
        direct = Mnow[-1, w] - nemo_met[-1, w]
        built = (lego_cum - nemo_cum).sum(axis=0) - (lego_off - nemo_off)
        e = float(np.max(np.abs(direct - built)[FULL_ROWS]))
        tot_d = float(direct[FULL_ROWS].sum())
        tot_b = float(built[FULL_ROWS].sum())
        # TOLERANCE, and why it is not absolute.  The two cancelling stage
        # pairs run to ~1.7e5 Sv on NEMO's side before they cancel, so the
        # residual here is fp64 roundoff ON THOSE OPERANDS and is meaningless
        # divided by the ~0.4 Sv row it produces.  Normalise by the operand
        # scale, exactly as nemo_accum_torque's identity gate does, and make it
        # step-count-aware for the same reason.
        _scale = max(float(np.max(np.abs(lego_cum))),
                     float(np.max(np.abs(nemo_cum))), 1.0)
        # A FIXED relative bound.  An earlier revision wrote this as
        # max(1e-12, 64*n_int*eps); the floor wins for every n_int under ~70,
        # so the "step-count-aware" half was dead code and the comment claiming
        # it was live was false.  1e-12 of the operand scale sits ~2 decades
        # above the measured residual and many below any real defect.
        _tol = 1.0e-12
        print(f"\n  [G4 total gate {wname}] (covers the legoESM telescoping "
              "and G1; NEMO's stage rows cancel\n      out of this identity "
              "algebraically and are covered by G3/G3b, not here.  It is "
              "always gated\n      over the FULL section, including for the "
              "channel weighting whose table shows one band --\n      so a "
              "planted violation outside the channel still trips it.)")
        print(f"  [G4 total gate {wname}] table total {tot_b:+.6f} Sv vs "
              f"directly measured day-90 gap {tot_d:+.6f} Sv "
              f"(per-row max|diff| {e:.3e}, operand scale {_scale:.3e}, "
              f"rel {e / _scale:.2e}, tol {_tol:.2e})")
        if e / _scale > _tol:
            raise SystemExit("FATAL G4: the stage x band table does not sum to "
                             "the gap it claims to decompose")

        # ---- the campaign's quoted number, for orientation.
        band_rows = FULL_ROWS if w == 0 else CHAN_ROWS
        print(f"  measured day-90 gap on THIS card, {wname}: "
              f"{float(direct[band_rows].sum()):+.4f} Sv "
              f"(MEAN over longitudes).  Campaign quoted {QUOTED[wname]:+.4f} "
              "on the card as it stood before the faithful pair shipped.")
        if w == 0:
            # THE SURROGATE, BOUNDED ON THE RIGHT SIDE.  The producing probe's
            # M0c prints legoESM's own median-minus-mean drift; that is the
            # WRONG bound, because the decomposed quantity is the GAP and
            # NEMO's median-minus-mean drifts too, largely in the same
            # direction.  Both series are built here and DIFFERENCED (found by
            # adversarial review: NEMO's own drift is ~-0.11 Sv, so quoting
            # legoESM's -0.12 Sv alone overstates the surrogate ~13x).
            _lego_mm = med - np.array(
                [Mnow[i, 0].sum() for i in range(n_int + 1)])
            _nemo_med = np.array(
                [A.acc_full(nemo_un(args.nemo_run if i else twin_run,
                                    KT0 + i * 10 * STEPS_PER_DAY), A.umask)
                 for i in range(n_int + 1)])
            _nemo_mm = _nemo_med - np.array(
                [nemo_met[i, 0].sum() for i in range(n_int + 1)])
            _dl = float(_lego_mm[-1] - _lego_mm[0])
            _dn = float(_nemo_mm[-1] - _nemo_mm[0])
            print(f"\n  MEAN-for-MEDIAN surrogate, drift over the run: "
                  f"legoESM {_dl:+.4f} Sv, NEMO {_dn:+.4f} Sv,\n  "
                  f"DIFFERENCE {_dl - _dn:+.4f} Sv -- the difference is the "
                  f"error the surrogate puts on the GAP.")
            nm = _nemo_med[-1]
            print(f"  MEDIAN-reduced (acc_full's own reduction): lego "
                  f"{med[-1]:.4f} - NEMO {nm:.4f} = {med[-1] - nm:+.4f} Sv -- "
                  "the size of the\n  mean-for-median surrogate ON THE GAP, "
                  "which is the only place it has to be small.")
        else:
            print("  no surrogate here: the channel number the campaign quotes "
                  "IS the MEAN over the same\n  longitudes "
                  "(floor90_ensemble.band_transport_campaign), so this row is "
                  "the recorded reduction itself.")

        # ---- the spectrum: each stage's share of the gap.
        gap = float((lego_real - nemo_real_states)[band_rows].sum())
        _ssum_pre = float((lego_cum - nemo_cum)[:, band_rows].sum())
        # Percentages against a gap at the measured 90-day noise floor
        # (1.15e-05 Sv single-run, 1.62e-05 for a difference, commit
        # 1dba0b733) are meaningless.  Below 100x that, print Sv only.
        _pct_ok = abs(gap) > 1.0e-3 and abs(_ssum_pre) > 1.0e-3
        print(f"\n  SPECTRUM over "
              f"{'the FULL section' if w == 0 else 'the channel'}.  The "
              f"denominator is the STAGE-SUM\n  difference "
              f"{_ssum_pre:+.4f} Sv, NOT the realized gap {gap:+.4f} Sv: the "
              f"stage rows are 2*dt increments\n  and the leap-frog offset "
              f"row, which is SLAVED to them, halves the sum.  Quoting a "
              f"stage\n  against the gap inflates every share ~2x -- the shape "
              f"of this module's own factor-of-2\n  retraction.  Rows marked "
              f"[pair] are the comparable objects; their halves are not (the\n"
              f"  wind sits in different buckets on the two models -- see the "
              f"docstring).")
        dif = lego_cum - nemo_cum
        offd = -(lego_off - nemo_off)
        ssum = float(dif[:, band_rows].sum())
        _tag = {I_BARO: "", I_BCLIN: "  [half, not comparable]",
                I_ZBT: "  [half, not comparable]",
                I_ZBC: "  [half, not comparable]",
                I_POST: "  [half, not comparable]"}
        shares = [(n + _tag[t], float(dif[t][band_rows].sum()))
                  for t, n in enumerate(STAGES)]
        shares += [("[pair] ZDF bt + POST",
                    float((dif[I_ZBT] + dif[I_POST])[band_rows].sum())),
                   ("[pair] BCLIN + ZDF bc",
                    float((dif[I_BCLIN] + dif[I_ZBC])[band_rows].sum())),
                   ("STAGE SUM difference", ssum),
                   ("offset diff (RESIDUAL)", float(offd[band_rows].sum()))]
        for name, v in shares:
            _p = (f"{100.0 * v / ssum:8.1f} % of stage-sum diff" if _pct_ok
                  else "  (below 1e-3 Sv; no %)")
            print(f"    {name:38s}{v:+12.4f} Sv   {_p}")
        # The leap-frog offset is SLAVED, not a lever: D settles at rDt*S/2 for
        # any Asselin gamma (the homogeneous mode decays as (2g-1)^n), so the
        # offset DIFFERENCE is predicted to be minus half the stage-sum
        # difference.  Printed as a prediction with its residual so a reader
        # can see whether the offset row is independent information or
        # bookkeeping.  It is NOT gated: gamma, the two models' filters and the
        # atf lumped into NEMO's residual all move it.
        off_got = float(offd[band_rows].sum())
        print(f"    slaved-offset prediction: offset difference should be "
              f"-(stage-sum difference)/2 = {-ssum / 2.0:+.4f} Sv; "
              f"measured {off_got:+.4f} Sv (residual {off_got + ssum / 2.0:+.4f})")

        # ---- cancellation, per band (the SG-B caveat carried forward).
        print("\n  NEMO barotropic-row CANCELLATION (|spg operand| / |BARO row|).")
        print("    The RATIO is a precision warning, so it is printed WITH the "
              "size of the error it\n    implies: the operands are fp64 "
              "accumulators summed inside the model over "
              f"{n_int * 10 * STEPS_PER_DAY} steps, so the\n    accumulated "
              "rounding bound is eps*sqrt(nsteps)*|operand|.  A ratio alone "
              "says a row\n    COULD be unreadable; the bound says whether it "
              "IS.")
        _nst = n_int * 10 * STEPS_PER_DAY
        print("    The LARGEST cancellation in this table is NOT the "
              "barotropic row: it is the\n    ZDF-bt/POST pair, whose halves "
              "run to the values below before cancelling to zero.")
        for bname, rws in bands:
            for who, cum in (("lego", lego_cum), ("NEMO", nemo_cum)):
                _h = abs(float(cum[I_ZBT][rws].sum()))
                _pr = abs(float((cum[I_ZBT] + cum[I_POST])[rws].sum()))
                print(f"      ZDF bt/POST {who} {bname:14s} half "
                      f"{_h:14.4f} Sv, pair {_pr:.3e} Sv, "
                      f"bound {np.finfo(np.float64).eps * np.sqrt(_nst) * _h:.2e} Sv")
        for bname, rws in bands:
            _ops = nem_cancel[:, w].sum(axis=0)          # (2, NY)
            op = max(abs(float(_ops[0][rws].sum())),
                     abs(float(_ops[1][rws].sum())))
            row = abs(float(nemo_cum[I_BARO][rws].sum()))
            rat = op / max(row, 1e-30)
            bound = np.finfo(np.float64).eps * np.sqrt(_nst) * op
            # The LABEL keys off the BOUND, not the ratio.  SG-B's caveat was
            # written for a NINE-SAMPLE reconstruction of this row, where the
            # cancellation multiplied a sampling error; with fp64 accumulators
            # summed inside the model it multiplies fp64 roundoff instead.
            # Labelling on the ratio alone would carry a warning that no longer
            # has a number behind it.
            frac = 100.0 * bound / max(abs(gap), 1e-30)
            print(f"    {bname:16s}{rat:12.1f}:1   rounding bound "
                  f"{bound:.2e} Sv = {frac:.1e} % of the gap   "
                  f"{'PLAUSIBLE (bound > 1% of gap)' if frac > 1.0 else 'readable'}")

        # ---- the per-window curve of every stage's difference.
        print("\n  PER-WINDOW difference curve [Sv per 10-day window], "
              f"{'FULL section' if w == 0 else 'channel'}:")
        print(f"  {'stage':22s}" + "".join(f"{10*i:>9d}" for i in range(n_int)))
        for t, name in enumerate(STAGES):
            v = (acc_met[:, w, t] - nem_stage[:, w, t])[:, band_rows].sum(axis=1)
            print(f"  {name:22s}" + "".join(f"{x:9.4f}" for x in v))
        gapc = np.array([float((Mnow[i + 1, w] - Mnow[i, w]
                                - (nemo_met[i + 1, w] - nemo_met[i, w])
                                )[band_rows].sum()) for i in range(n_int)])
        print(f"  {'GAP (realized)':22s}" + "".join(f"{x:9.4f}" for x in gapc))
        print(f"  {'GAP cumulative':22s}"
              + "".join(f"{x:9.4f}" for x in np.cumsum(gapc)))
        # The barotropic row's cumulative difference, HALVED, is the thing to
        # compare against the cumulative gap: the stage rows are 2*dt
        # increments and the slaved offset takes half back (see the UNITS note
        # above).  Printed as a row so the SHAPE is visible; no correlation
        # coefficient and no verdict -- the reader compares the two lines.
        _bcum = np.cumsum((acc_met[:, w, I_BARO] - nem_stage[:, w, I_BARO]
                           )[:, band_rows].sum(axis=1)) / 2.0
        print(f"  {'BARO cumulative / 2':22s}"
              + "".join(f"{x:9.4f}" for x in _bcum))

    if args.out_npz:
        np.savez_compressed(
            args.out_npz, stages=np.array(STAGES), metw=np.array(METW),
            lego_acc_met=acc_met, lego_acc_moffs=acc_moffs, lego_Mnow=Mnow,
            nemo_stage=nem_stage, nemo_real=nem_real, nemo_met=nemo_met,
            nemo_cancel=nem_cancel, full_rows=np.array(FULL_ROWS),
            chan_rows=np.array(CHAN_ROWS), git_sha=sha,
            lego_npz=args.lego_npz, nemo_run=args.nemo_run)
        print(f"\n[artifact] -> {args.out_npz}")


if __name__ == "__main__":
    main()
