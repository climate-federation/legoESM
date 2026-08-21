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

NEMO's ``jpdyn_spg`` bucket is the WHOLE change across ``dyn_spg_ts``, which
both removes the pre-spg RHS's column mean and adds the barotropic solve's
increment back (dynspg_ts.F90:337/351/1126, dynspg.F90:185-187).  So
(pre-spg trends + utrd_spg) is exactly "explicit deviation + barotropic
increment", the same object lego's combine forms -- which is what makes the
column-mean split the SAME split on both sides rather than an analogy.

WHAT THE STAGE ROWS ARE **NOT** UNDER THIS REDUCER.  Under the e3u_0 reducer
the three identities I1/I2/I3 collapse NEMO's step to a single row, and on the
shipped legoESM card the second reconciliation makes "POST fixer" cancel
"ZDF bt" exactly, so lego's realized row equals its barotropic row too.  Those
identities are e3u_0-SPECIFIC: they say the e3u_0-weighted column mean of the
discarded pieces is zero, not that the pieces are zero.  The gate metric weighs
with e3t_1d (or e3t_0), so the discarded pieces DO reach the ACC number and the
five rows are all live.  Both facts are measured below, not assumed.

CONTROLS, each of which CAN fail:
  G0  the imported metric weights must reproduce the lego npz's OWN day-0
      metric rows (two independent implementations, device and host).
  G1  day-0 lego minus NEMO must be zero to roundoff -- the bridge is
      bit-identical, so a non-zero day-0 gap means the two sides are being read
      on different arrays or different levels.
  G2  NEMO's accumulator run (RUN_ACC90) must sit on the SAME trajectory as the
      RUN_90D_TWIN the lego side is scored against: their day-90 ``un`` are
      compared cell by cell.  C0/C0b in ``nemo_accum_torque`` test the binary
      and the day-10 restart; this tests the endpoint the gap is quoted at.
  G3  NEMO's five stage increments must sum to (uacc_fin - uacc_bb) per window,
      which is the C1 closure carried through this reducer.
  G4  the stage x band table's grand total must reproduce the DIRECTLY measured
      day-90 gap, both sides reduced identically.
  G5  a PLANTED violation: with ``--plant-stage N --plant-sv X`` a constant Sv
      is injected into stage N's lego row.  G3/G4 must go red by exactly X.
      Run it once; a closure that cannot fail is not a closure.

CANCELLATION (the SG-B caveat, carried forward).  NEMO's barotropic row is a
difference of two large operands (``utrd_spg`` against the pre-spg trends).
The operands are fp64 model-side accumulators so the arithmetic is exact, but
the RATIO is printed per band and any band whose ratio exceeds 100:1 is
labelled PLAUSIBLE rather than CONFIRMED.

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
    spg_bt, _ = split_bt(_yxz(d["utrdacc_spg"]) / r1)
    return {
        "stages": np.stack([e_bt, e_bc, z_bt, z_bc,
                            np.where(B.umask, post, 0.0)]),
        "realized": np.where(B.umask, realized, 0.0),
        "cancel_operand": spg_bt,       # for the cancellation ratio
        "n_steps": int(d["nacc_steps"]),
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
    return f"{v:12.4f}"


def _band_table(title, per_stage, extra_rows, bands):
    """per_stage: (n_stage, NY).  extra_rows: list of (label, (NY,) array)."""
    print("\n" + "-" * 100)
    print(f"  {title}")
    print("-" * 100)
    print(f"  {'row':22s}" + "".join(f"{b:>12s}" for b, _ in bands))
    for t, name in enumerate(STAGES):
        print(f"  {name:22s}" + "".join(_fmt(per_stage[t, r].sum())
                                        for _, r in bands))
    for label, arr in extra_rows:
        print(f"  {label:22s}" + "".join(_fmt(arr[r].sum()) for _, r in bands))


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
    acc_met = np.asarray(z["acc_met"], np.float64)          # (n_int,2,5,NY) Sv
    acc_moffs = np.asarray(z["acc_moffs"], np.float64)      # (n_int,2,NY)  Sv
    Mnow = np.asarray(z["Mnow_series"], np.float64)         # (n_int+1,2,NY)
    med = np.asarray(z["acc_med_series"], np.float64)
    n_int = acc_met.shape[0]
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
        print(f"  [G0 reducer gate {name}] host vs device day-0 metric rows, "
              f"max|diff| = {e:.3e} Sv/row")
        if e > 1.0e-10:
            raise SystemExit("FATAL G0: the two metric reducers disagree")

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
    nem_cancel = np.zeros((n_int, 2, B.NY))
    nemo_met = np.zeros((n_int + 1, 2, B.NY))
    for w in range(2):
        nemo_met[0, w] = met_rows(u0, _WEIGHTS[w])
    for i in range(n_int):
        kt_lo = KT0 + i * 10 * STEPS_PER_DAY
        kt_hi = kt_lo + 10 * STEPS_PER_DAY
        r = nemo_stage_increments(args.nemo_run, None if i == 0 else kt_lo, kt_hi)
        # ---- G3: the five stage increments must sum to the realized increment.
        res = float(np.max(np.abs(r["stages"].sum(axis=0) - r["realized"])))
        sc = float(np.max(np.abs(r["realized"])))
        if sc > 0 and res / sc > 1.0e-9:
            raise SystemExit(f"FATAL G3: NEMO's stage increments do not close on "
                             f"the realized increment in window {i} "
                             f"(residual {res:.3e} on {sc:.3e})")
        un_i = nemo_un(args.nemo_run, kt_hi)
        for w in range(2):
            nem_stage[i, w] = np.stack([met_rows(x, _WEIGHTS[w])
                                        for x in r["stages"]])
            nem_real[i, w] = met_rows(r["realized"], _WEIGHTS[w])
            nem_cancel[i, w] = met_rows(r["cancel_operand"], _WEIGHTS[w])
            nemo_met[i + 1, w] = met_rows(un_i, _WEIGHTS[w])
        print(f"  window {i} (days {10*i}-{10*(i+1)}): NEMO steps="
              f"{r['n_steps']}, G3 residual {res:.2e} m/s on {sc:.2e}")

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
        print(f"CUMULATIVE day-90 ACC contribution [Sv], {wname} weighting")
        print("=" * 100)
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
        print(f"\n  [G4 total gate {wname}] table total {tot_b:+.6f} Sv vs "
              f"directly measured day-90 gap {tot_d:+.6f} Sv "
              f"(per-row max|diff| {e:.3e})")
        if abs(tot_d - tot_b) > 1.0e-9:
            raise SystemExit("FATAL G4: the stage x band table does not sum to "
                             "the gap it claims to decompose")

        # ---- the campaign's quoted number, for orientation.
        band_rows = FULL_ROWS if w == 0 else CHAN_ROWS
        print(f"  measured day-90 gap on THIS card, {wname}: "
              f"{float(direct[band_rows].sum()):+.4f} Sv "
              f"(MEAN over longitudes).  Campaign quoted {QUOTED[wname]:+.4f} "
              "on the card as it stood before the faithful pair shipped.")
        if w == 0:
            nm = A.acc_full(nemo_un(twin_run, kt90), A.umask)
            print(f"  MEDIAN-reduced (acc_full's own reduction): lego "
                  f"{med[-1]:.4f} - NEMO {nm:.4f} = {med[-1] - nm:+.4f} Sv")

        # ---- the spectrum: each stage's share of the gap.
        gap = float((lego_real - nemo_real_states)[band_rows].sum())
        print(f"\n  SPECTRUM over {'the FULL section' if w == 0 else 'the channel'}"
              f" (share of the {gap:+.4f} Sv gap):")
        shares = []
        for t, name in enumerate(STAGES):
            v = float((lego_cum[t] - nemo_cum[t])[band_rows].sum())
            shares.append((name, v))
        shares.append(("offset difference",
                       float(-(lego_off - nemo_off)[band_rows].sum())))
        for name, v in shares:
            print(f"    {name:22s}{v:+10.4f} Sv   "
                  f"{100.0 * v / gap if gap else float('nan'):7.1f} %")

        # ---- cancellation, per band (the SG-B caveat carried forward).
        print("\n  NEMO barotropic-row CANCELLATION (|spg operand| / |BARO row|):")
        for bname, rws in bands:
            op = abs(float(nem_cancel[:, w].sum(axis=0)[rws].sum()))
            row = abs(float(nemo_cum[0][rws].sum()))
            rat = op / max(row, 1e-30)
            print(f"    {bname:16s}{rat:10.1f}:1   "
                  f"{'PLAUSIBLE (>100:1)' if rat > 100 else 'readable'}")

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
