#!/usr/bin/env python
"""#1455 PHASE 1 -- THE TIME AXIS of the barotropic per-step ACC deposit.

WHAT THIS ANSWERS.  The full-section stage budget (c987db464) put 100% of the
remaining ACC gap in ONE stage, the split-explicit barotropic solve: -0.8069 Sv
of cumulative barotropic stage-sum difference over 90 days (2880 steps), which
after the slaved leap-frog halving is the -0.4014 Sv realized gap.  The substep
walk (f30e0169c -> f09758204 -> 3809dd006) then measured, at ONE state (day
180), a per-step deposit difference of +5.56e-03 Sv/step that splits 64% into
the frozen slow forcing and 36% into the substep loop.  Both survivors are
budget-scale.  BOTH ARE THE WRONG SIGN at that state, and n=1 in time cannot
tell an oscillation from an accumulation.  This probe turns n=1 into n=10 by
repeating the SAME measurement at every 10-day bridged state the 90-day twin
carries (day 180, 190, ... 270).

HOW.  It does NOT re-implement the deposit.  It runs the committed instrument
``substep_traj_compare.py`` once per state as a subprocess, with only the state
selectors changed (DINO_1226_IC_STEP, DINO_NEMO_RUN_SEQDUMP,
DINO_NEMO_RUN_TWIN_STEP1), and reads the three numbers that instrument already
prints and already validates with its own asserts:

    velocity-average deposit diff (lego - NEMO)   -> TOTAL   deposit(t)
    deposit with NEMO's frozen forcing            -> IN-LOOP share(t)
    TOTAL - IN-LOOP                               -> FORCING share(t)

A child that exits non-zero, or whose output is missing any of those lines, is
a FATAL for the whole walk -- never a dropped row, because a dropped row would
silently change the window the integral covers.

THE ONE INSTRUMENT CHANGE, and why it had to happen before this walk could run:
this card sets ``forcing_annual_cycle=True``, and the walk passed a bare
``t_seconds=DT`` (day 0.03) while replaying states at day 180-270.  At the
historical IC that was harmless by coincidence (230400 x 2700 s is exactly 20
360-day years, so the relative and absolute clocks coincide).  Across THIS
window it is not: the season mismatch would vary with the state and could fake
the very time dependence being measured.  ``substep_traj_compare.py`` now uses
the absolute clock ``(IC_STEP+1)*DT``.  ``--clock-ab`` re-runs one state under
both clocks so the size of that correction is measured, not assumed, and the
planted-violation controls inside the instrument (the two NEMO-side map asserts,
the legoESM-side boxcar assert, the substep-shift PLANT) re-run on EVERY child.

=====================================================================
PRE-REGISTERED CRITERIA -- written before the first state was run.
=====================================================================
Let d_i be a survivor's deposit at state i (i = 0..9 for days 180..270), and let
B_i be the barotropic stage row's own increment over window i, read off
c987db464's committed per-window table (BARO cum/2, doubled to undo the halving,
divided by the 320 steps in a window).  The walk's prediction for the window is
the trapezoid of the two endpoint deposits times 320 steps; the walk's
prediction for the whole campaign is the sum over the nine windows.

  ACCUMULATOR  -- a survivor is named the accumulator iff BOTH hold:
    (A1) its 90-day integral I = sum_i 320*(d_i + d_{i+1})/2 lands within a
         factor 2 of the budget's -0.8069 Sv, i.e. I in [-1.614, -0.403] Sv;
    (A2) it agrees in SIGN with the window increment B_i in at least 7 of the
         9 windows (the sign test, so a single large window cannot carry A1).

  EXONERATED FOR THE ACCUMULATION -- a survivor is exonerated iff BOTH hold:
    (E1) its sign coherence |mean(d)| / rms(d) < 0.5, i.e. it oscillates rather
         than accumulates; AND
    (E2) |I| < 0.403 Sv, i.e. even integrated it cannot pay for half the row.
    A survivor may be exonerated FOR THE ACCUMULATION while remaining large
    instantaneously; that is the whole point of the time axis.

  BOTH ACCUMULATE -- if both survivors satisfy A1+A2, or if both integrals are
    coherent with opposite signs that partially cancel, the pair is reported
    with both integrals and NOTHING is crowned.

  NEITHER -- if no survivor satisfies A1, the walk reports that the per-step
    deposit at bridged states does not reproduce the accumulated row, and the
    next lever is the bridge/feedback term, not a survivor.

  UNDERPOWERED (a fourth outcome, pre-registered because it is likely): if the
    standard error of the mean deposit, sqrt(var/n), exceeds |target mean|
    = 2.80e-04 Sv/step, then the SIGN test (A2) and the SHAPE (correlation with
    B) carry the verdict and the integral I is reported with its error bar
    rather than used as a pass/fail.  This is stated up front so a wide error
    bar cannot be reinterpreted after the fact.

This probe PRINTS NUMBERS AND THE PRE-REGISTERED THRESHOLDS SIDE BY SIDE.  It
does not print a verdict; the crowning belongs in the analysis after the dual
adversarial review, not baked into the tool.

USAGE
  CUDA_VISIBLE_DEVICES=0 LEGOESM_NEMO_E3T=both JAX_ENABLE_X64=1 \
    .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/baro_deposit_time_walk.py
  (add --clock-ab to also run the day-180 legacy-clock arm; --only 180,270 to
   restrict the states; --out to move the npz.)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_INSTRUMENT = os.path.join(_THIS_DIR, "substep_traj_compare.py")
_DINO = os.environ.get(
    "DINO_ORACLE_ROOT",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO")

DT = 2700.0
STEPS_PER_WINDOW = 320          # 10 days at rn_Dt = 2700 s
N_STEPS_90D = 2880

#: The budget row this walk has to pay for: c987db464's full-section
#: barotropic stage-sum difference, cumulative over 90 days [Sv].
BARO_ROW_90D = -0.8069
TARGET_PER_STEP = BARO_ROW_90D / N_STEPS_90D          # -2.80e-04 Sv/step

#: c987db464's committed per-window cumulative table, HALVED rows ("BARO cum/2")
#: at days 10..90 of the window.  Doubled below to recover the stage row itself,
#: which is the quantity the per-step deposit is measured in.
_BARO_CUM_HALVED = np.array(
    [-0.0094, -0.0042, -0.0364, -0.0772, -0.1210,
     -0.2055, -0.2389, -0.3405, -0.4035])
#: the realized GAP's own cumulative curve from the same table, for the report.
_GAP_CUM = np.array(
    [-0.0137, +0.0006, -0.0404, -0.0841, -0.1188,
     -0.2051, -0.2475, -0.3438, -0.4014])

#: (day, NEMO kt, seq-dump lane, restart-tile dir) for every bridged state.
#: Day 180's tiles live in RUN_D180_STEP1 (a 1-rank "_0000" tile); days 190+
#: are the 90-day twin's own 16-rank tiles.  The seq-dump lanes for 190+ are
#: built by scripts/data/gen_dino_nemo_seqdump_states.sh.
STATES = [(180, 5760, "RUN_SEQDUMP_D180_1R", "RUN_D180_STEP1")] + [
    (180 + 10 * k, 5760 + 320 * k, f"RUN_SEQDUMP_D{180 + 10 * k}_1R",
     "RUN_90D_TWIN")
    for k in range(1, 10)]

_RE_TOTAL = re.compile(
    r"velocity-average deposit diff \(lego - NEMO\)\s*=\s*([+-][0-9.]+e[+-][0-9]+)")
_RE_INLOOP = re.compile(
    r"deposit with NEMO's frozen forcing\s*=\s*([+-][0-9.]+e[+-][0-9]+)")
_RE_CLOCK = re.compile(r"seasonal clock: t_seconds=([0-9.]+) s")
_RE_TAU = re.compile(r"tau_x\[Pa\] range=\[([-0-9.]+),([-0-9.]+)\]")
_RE_PLANT = re.compile(r"shift-sensitivity ratio ([0-9.]+)")


def score(d: np.ndarray, baro_win_per_step: np.ndarray) -> dict:
    """The PRE-REGISTERED arithmetic for one survivor's deposit series.

    ``d`` is the per-step deposit at the ten bridged states [Sv/step];
    ``baro_win_per_step`` the nine committed per-window barotropic-row rates.
    Returns the numbers AND the boolean value of each pre-registered criterion.
    No verdict: A1/A2/E1/E2 are thresholds, the crowning is the analysis.
    """
    d = np.asarray(d, dtype=float)
    b = np.asarray(baro_win_per_step, dtype=float)
    if d.shape != (b.size + 1,):
        raise ValueError(
            f"score() needs one deposit per window BOUNDARY: got {d.shape} "
            f"deposits for {b.size} windows")
    mean = float(d.mean())
    sem = float(d.std(ddof=1) / np.sqrt(d.size))
    rms = float(np.sqrt((d ** 2).mean()))
    coh = abs(mean) / rms if rms else float("nan")
    win_mid = 0.5 * (d[:-1] + d[1:])                 # trapezoid rate per window
    integral = float((STEPS_PER_WINDOW * win_mid).sum())
    agree = int(np.sum(np.sign(win_mid) == np.sign(b)))
    corr = float(np.corrcoef(win_mid, b)[0, 1]) if win_mid.std() and b.std() \
        else float("nan")
    lo, hi = 2.0 * BARO_ROW_90D, 0.5 * BARO_ROW_90D  # both negative; lo < hi
    return {"mean": mean, "sem": sem, "rms": rms, "coherence": coh,
            "integral_90d": integral, "sign_agree": agree,
            "corr_with_baro_window": corr,
            "A1_in_band": bool(lo <= integral <= hi),
            "A2_sign_7of9": bool(agree >= 7),
            "E1_incoherent": bool(coh < 0.5),
            "E2_small_integral": bool(abs(integral) < abs(hi))}


def _assert_same_entry_state(day: int, kt: int, lane: str, tiles: str) -> dict:
    """CONTROL: NEMO and legoESM must enter the step from the SAME bytes.

    The two sides read the state through different files: legoESM stitches the
    per-rank restart tiles in memory (``rebuild_nemo_restart.rebuild``, what
    ``multistep_replay.build_replay_ic`` calls), while NEMO's 1-rank lane reads
    the single file produced by the shipped REBUILD_NEMO tool.  If those two
    disagree anywhere, the "deposit" is a difference of initial conditions and
    not of dynamics -- which is exactly the class of defect that produced this
    campaign's earlier retractions.  So compare EVERY shared variable, bit for
    bit, and refuse the state on any mismatch.

    Day 180 is included: there legoESM reads RUN_D180_STEP1's 1-rank tile and
    NEMO reads RUN_TRAJ's own restart, two independently written files whose
    agreement was never actually checked.
    """
    import glob

    import netCDF4 as nc

    lane_dir = os.path.join(_DINO, lane)
    # which file did NEMO read?  from the lane's own namelist, not assumed.
    ocerst = None
    with open(os.path.join(lane_dir, "namelist_cfg")) as fh:
        for line in fh:
            m = re.search(r'cn_ocerst_in\s*=\s*"([^"]+)"', line)
            if m:
                ocerst = m.group(1)
                break
    if ocerst is None:
        raise SystemExit(f"FATAL day {day}: no cn_ocerst_in in {lane_dir}")
    nemo_file = os.path.join(lane_dir, ocerst + ".nc")
    tile_glob = os.path.join(_DINO, tiles, f"DINO_{kt:08d}_restart_*.nc")
    tile_files = sorted(glob.glob(tile_glob))
    if not tile_files:
        raise SystemExit(f"FATAL day {day}: no restart tiles at {tile_glob}")

    dn = nc.Dataset(nemo_file)
    names = [v for v, var in dn.variables.items()
             if var.dtype.kind == "f" and var.ndim >= 3]
    sys.path.insert(0, os.path.dirname(_THIS_DIR))
    from rebuild_nemo_restart import rebuild
    stitched = rebuild(tile_glob, names)
    worst, worst_name = 0.0, ""
    for name in names:
        if name not in stitched:
            raise SystemExit(
                f"FATAL day {day}: the in-memory stitcher did not cover "
                f"{name!r}; the two entry states are not comparable.")
        a = np.asarray(dn.variables[name][0], dtype=np.float64)
        b = np.asarray(stitched[name], dtype=np.float64)
        if a.shape != b.shape:
            raise SystemExit(
                f"FATAL day {day}: {name} shape {a.shape} (NEMO file) vs "
                f"{b.shape} (stitched tiles)")
        d = float(np.abs(a - b).max())
        if d > worst:
            worst, worst_name = d, name
    dn.close()
    if worst != 0.0:
        raise SystemExit(
            f"FATAL day {day}: the two entry states differ (worst {worst:.3e} "
            f"on {worst_name!r}). NEMO read {nemo_file}; legoESM stitches "
            f"{len(tile_files)} tiles from {tile_glob}. A deposit measured "
            "from different states measures the difference of the states.")
    return {"entry_state_maxdiff": worst, "entry_nvars": len(names),
            "entry_ntiles": len(tile_files), "nemo_restart": nemo_file}


def _run_one(day: int, kt: int, lane: str, tiles: str, *,
             t_seconds: float | None, log_dir: str) -> dict:
    """One child run of the committed instrument at one bridged state."""
    seq = os.path.join(_DINO, lane)
    tdir = os.path.join(_DINO, tiles)
    for p in (seq, tdir, os.path.join(seq, "substep_dump.bin"),
              os.path.join(seq, "mesh_mask.nc")):
        if not os.path.exists(p):
            raise SystemExit(
                f"FATAL day {day}: missing {p}. Build the lane first with "
                "scripts/data/gen_dino_nemo_seqdump_states.sh -- a missing "
                "state must never become a dropped row.")
    entry = _assert_same_entry_state(day, kt, lane, tiles)
    env = dict(os.environ)
    env.update({
        "DINO_1226_IC_STEP": str(kt),
        "DINO_NEMO_RUN_SEQDUMP": seq,
        "DINO_NEMO_RUN_TWIN_STEP1": tdir,
        # the ladder and the precision are part of the comparison, not
        # defaults to inherit: pin them on every child.
        "LEGOESM_NEMO_E3T": os.environ.get("LEGOESM_NEMO_E3T", "both"),
        "JAX_ENABLE_X64": "1",
    })
    if t_seconds is not None:
        env["DINO_1226_T_SECONDS"] = repr(float(t_seconds))
    else:
        env.pop("DINO_1226_T_SECONDS", None)

    t0 = time.time()
    proc = subprocess.run([sys.executable, _INSTRUMENT], env=env,
                          capture_output=True, text=True, timeout=7200)
    out = proc.stdout + "\n" + proc.stderr
    tag = f"d{day}" + ("_legacyclock" if t_seconds is not None else "")
    with open(os.path.join(log_dir, f"substep_{tag}.log"), "w") as fh:
        fh.write(out)
    if proc.returncode != 0:
        raise SystemExit(
            f"FATAL day {day}: the instrument exited {proc.returncode} -- its "
            f"own controls did not pass, so this state has NO measurement and "
            f"the walk cannot proceed with a hole in it. Log tail:\n"
            + "\n".join(out.splitlines()[-40:]))

    tot = _RE_TOTAL.findall(out)
    inl = _RE_INLOOP.findall(out)
    if len(tot) != 1 or len(inl) != 1:
        raise SystemExit(
            f"FATAL day {day}: expected exactly one TOTAL and one IN-LOOP line, "
            f"found {len(tot)} and {len(inl)}. The instrument's output format "
            "changed; refusing to guess which number is which.")
    clock = _RE_CLOCK.findall(out)
    tau = _RE_TAU.findall(out)
    plant = _RE_PLANT.findall(out)
    rec = {
        "day": day, "kt": kt, "lane": lane, "tiles": tiles,
        "total": float(tot[0]), "in_loop": float(inl[0]),
        "clock_s": float(clock[0]) if clock else float("nan"),
        "tau_lo": float(tau[0][0]) if tau else float("nan"),
        "tau_hi": float(tau[0][1]) if tau else float("nan"),
        "plant_ratio": float(plant[0]) if plant else float("nan"),
        "wall_s": time.time() - t0,
    }
    rec.update(entry)
    rec["forcing"] = rec["total"] - rec["in_loop"]
    # the wind must be ON in every child (the whole #1455 wind-off retraction),
    # and the clock must be the one this state's kt implies unless the legacy
    # arm asked otherwise.
    if not (rec["tau_hi"] > 0.0 > rec["tau_lo"]):
        raise SystemExit(
            f"FATAL day {day}: wind stress range [{rec['tau_lo']},"
            f"{rec['tau_hi']}] is not the wind-on signature; a wind-off child "
            "would reproduce the retracted numbers.")
    if t_seconds is None and abs(rec["clock_s"] - (kt + 1) * DT) > 1.0:
        raise SystemExit(
            f"FATAL day {day}: child ran the clock at {rec['clock_s']} s, not "
            f"the absolute {(kt + 1) * DT} s this state implies.")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma-separated days to run (default: all ten)")
    ap.add_argument("--clock-ab", action="store_true",
                    help="also run day 180 under the LEGACY bare-dt clock, to "
                         "size the correction the three prior commits carry")
    ap.add_argument("--out", default=os.path.join(
        _THIS_DIR, "..", "..", "..", "..", "results", "dino_1455",
        "baro_deposit_time_walk.npz"))
    args = ap.parse_args(argv)

    sys.path.insert(0, _THIS_DIR)
    import multistep_replay as mr
    prov = mr.provenance("baro_deposit_time_walk")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_dir = os.path.join(os.path.dirname(out_path), "logs")
    os.makedirs(log_dir, exist_ok=True)

    want = ([int(x) for x in args.only.split(",") if x.strip()]
            if args.only else [s[0] for s in STATES])
    states = [s for s in STATES if s[0] in want]
    if len(states) != len(want):
        raise SystemExit(f"unknown day(s) in --only={args.only!r}")

    print(f"\n=== #1455 PHASE 1: the deposit's TIME AXIS "
          f"({len(states)} bridged states) ===")
    print("PRE-REGISTERED (see module docstring): ACCUMULATOR iff the 90-day "
          "trapezoid integral")
    print(f"  I in [{2 * BARO_ROW_90D:+.4f}, {0.5 * BARO_ROW_90D:+.4f}] Sv "
          f"(factor 2 either side of the {BARO_ROW_90D:+.4f} Sv budget row) "
          "AND sign-agrees with")
    print("  the per-window row in >=7 of 9 windows.  EXONERATED iff "
          "|mean|/rms < 0.5 AND |I| < "
          f"{abs(0.5 * BARO_ROW_90D):.4f} Sv.")

    recs = [_run_one(day, kt, lane, tiles, t_seconds=None, log_dir=log_dir)
            for (day, kt, lane, tiles) in states]

    ab = None
    if args.clock_ab:
        d180 = [s for s in STATES if s[0] == 180][0]
        ab = _run_one(d180[0], d180[1], d180[2], d180[3], t_seconds=DT,
                      log_dir=log_dir)

    days = np.array([r["day"] for r in recs], dtype=float)
    series = {k: np.array([r[k] for r in recs], dtype=float)
              for k in ("total", "in_loop", "forcing")}

    print("\n  deposit(t) and its split, Sv/step (lego - NEMO, gate weights):")
    print("    day |     TOTAL     |    FORCING    |    IN-LOOP    | "
          "forcing% | plant | wall s")
    for r in recs:
        fr = 100.0 * r["forcing"] / r["total"] if r["total"] else float("nan")
        print(f"    {r['day']:3.0f} | {r['total']:+.6e} | {r['forcing']:+.6e} "
              f"| {r['in_loop']:+.6e} | {fr:7.1f}% | {r['plant_ratio']:5.2f} "
              f"| {r['wall_s']:6.1f}")

    if ab is not None:
        print("\n  CLOCK A/B at day 180 (absolute vs the legacy bare-dt clock "
              "the three prior commits ran):")
        print(f"    absolute t={recs[0]['clock_s']:.0f}s  TOTAL="
              f"{recs[0]['total']:+.6e}  IN-LOOP={recs[0]['in_loop']:+.6e}")
        print(f"    legacy   t={ab['clock_s']:.0f}s  TOTAL="
              f"{ab['total']:+.6e}  IN-LOOP={ab['in_loop']:+.6e}")
        for k in ("total", "in_loop", "forcing"):
            d = recs[0][k] - ab[k]
            print(f"    delta {k:8s} = {d:+.4e} Sv/step "
                  f"({100 * d / ab[k] if ab[k] else float('nan'):+.2f}% of the "
                  "legacy value)")

    # ---- the pre-registered arithmetic ------------------------------------
    full = len(recs) == len(STATES)
    baro_cum = 2.0 * _BARO_CUM_HALVED                      # undo the halving
    baro_win = np.diff(np.concatenate([[0.0], baro_cum]))  # per-window [Sv]
    baro_win_per_step = baro_win / STEPS_PER_WINDOW

    print("\n  the curve this has to reproduce (c987db464's committed table):")
    print("    window (days) | BARO row [Sv] | per-step [Sv/step] | GAP cum [Sv]")
    for i in range(9):
        print(f"    {180 + 10 * i:3d}->{190 + 10 * i:3d}     | "
              f"{baro_win[i]:+.4f}       | {baro_win_per_step[i]:+.3e}         "
              f"| {_GAP_CUM[i]:+.4f}")

    summary = {"provenance": prov, "days": days.tolist(),
               "baro_win_per_step": baro_win_per_step.tolist(),
               "target_per_step": TARGET_PER_STEP,
               "baro_row_90d": BARO_ROW_90D, "records": recs}
    if ab is not None:
        summary["clock_ab_legacy"] = ab

    if full:
        print("\n  PRE-REGISTERED SCORE per survivor "
              "(numbers + thresholds; no verdict is printed):")
        print("    survivor |  mean [Sv/step] |  sem  | |mean|/rms | "
              "I_90d [Sv] | sign-agree | corr(d,B)")
        for name in ("total", "forcing", "in_loop"):
            sc = score(series[name], baro_win_per_step)
            print(f"    {name:8s} | {sc['mean']:+.6e} | {sc['sem']:.1e} | "
                  f"{sc['coherence']:9.3f} | {sc['integral_90d']:+10.4f} | "
                  f"{sc['sign_agree']:d}/9        | "
                  f"{sc['corr_with_baro_window']:+.3f}")
            print(f"             |   A1(band)={sc['A1_in_band']!s:5s} "
                  f"A2(sign>=7/9)={sc['A2_sign_7of9']!s:5s} "
                  f"E1(|mean|/rms<0.5)={sc['E1_incoherent']!s:5s} "
                  f"E2(|I|<{abs(0.5 * BARO_ROW_90D):.3f})="
                  f"{sc['E2_small_integral']!s:5s}")
            summary[name] = sc
        sem_tot = float(series["total"].std(ddof=1) / np.sqrt(len(recs)))
        print(f"\n    POWER (pre-registered): sem(total) = {sem_tot:.3e} vs "
              f"|target mean| = {abs(TARGET_PER_STEP):.3e} Sv/step -> "
              f"{'sem EXCEEDS target (underpowered arm: sign + shape carry)' if sem_tot > abs(TARGET_PER_STEP) else 'sem below target'}")
    else:
        print("\n  (partial state list: the pre-registered integral is NOT "
              "scored -- it is defined on all ten states)")

    np.savez(out_path, days=days, **{f"dep_{k}": v for k, v in series.items()},
             baro_win_per_step=baro_win_per_step,
             summary_json=json.dumps(summary, default=float))
    print(f"\n  artifact: {out_path}")
    print(f"  logs:     {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
