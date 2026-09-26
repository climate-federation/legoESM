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

THE ONE INSTRUMENT CHANGE.  This card sets ``forcing_annual_cycle=True`` and
the walk passed a bare ``t_seconds=DT`` (day 0.03) while replaying states at
day 180-270, so the seasonal restoring target and solar flux were evaluated in
the wrong season.  ``substep_traj_compare.py`` now uses the absolute clock
``(IC_STEP+1)*DT``.

THE RATIONALE THIS DOCSTRING USED TO GIVE FOR THAT CHANGE IS WITHDRAWN
(4c14b4d91).  It said the season mismatch "could fake the very time dependence
being measured".  It could not: ``t_seconds`` reaches only the tracer restoring
and the solar flux, the wind is built with no clock argument at all, and within
a single step the tracer update cannot reach the barotropic forcing.  So the
clock fix is correct hygiene for any probe that reads the TRACER channels, and
is provably inert for THIS metric.  ``--clock-ab`` therefore does NOT size a
correction -- it perturbs a channel with no path to the measured quantity, and
its zero is guaranteed by construction rather than being a bound on anything.
It is kept only to demonstrate that the override takes effect, and the probe
prints that warning above the numbers.

What every child DOES re-run is the instrument's own planted-violation controls
(the two NEMO-side map asserts, the legoESM-side boxcar assert, the v-face/ssh
alignment asserts and the substep-shift PLANT).

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
#: CONSECUTIVE-STEP states -- four NEMO steps in a row at day 180.  The 10-day
#: grid below samples every 320th step, which is an EVEN offset, so it aliases
#: any oscillation with a period that divides 320 -- in particular a leap-frog
#: computational mode that alternates sign every step would appear on that grid
#: as a perfect constant with sign coherence 1.000.  That is exactly what the
#: in-loop share looks like, so the aliasing must be excluded before the
#: constant can be called state-independent.  Both sides' restarts exist at
#: kt = 5760..5763 (RUN_D180_STEP1 tiles for legoESM, RUN_D180_STEP1_1R single
#: files for NEMO), so this costs four 4-step NEMO runs and five parent steps.
#: FIVE points, not four -- but that did NOT fix the provenance confound and an
#: earlier version of this comment wrongly implied it had.  kt=5760's restart is
#: the 16-rank 90-day twin's own checkpoint; kt=5761..5764 come from a 1-rank
#: continuation that STARTED from it.  Adding a point at the far end cannot
#: change the near end.  The confound stays OPEN, which is why every statistic
#: is reported with its leave-one-out twin and why the run PRINTS the reason.
CONSEC_STATES = [(5760, 5760, "RUN_SEQDUMP_D180_1R", "RUN_D180_STEP1")] + [
    (kt, kt, f"RUN_SEQDUMP_KT{kt}_1R", "RUN_D180_STEP1")
    for kt in (5761, 5762, 5763, 5764)]

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


def power_check(d: np.ndarray) -> dict:
    """The PRE-REGISTERED power test, as a function so it can be tested.

    It was previously an inline ternary in ``main()``; an adversarial review
    pointed out that the test named after it only re-derived the arithmetic
    and would still pass with the whole block deleted.
    """
    d = np.asarray(d, dtype=float)
    sem = float(d.std(ddof=1) / np.sqrt(d.size))
    return {"sem": sem, "target": abs(TARGET_PER_STEP),
            "underpowered": bool(sem > abs(TARGET_PER_STEP))}


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
    """CONTROL: the two sides must enter the step from the SAME bytes.

    WHAT THIS COVERS, stated exactly, because an earlier revision of this
    docstring over-claimed it and an adversarial review caught it.  legoESM
    stitches the per-rank restart tiles in memory
    (``rebuild_nemo_restart.rebuild``, what ``multistep_replay.build_replay_ic``
    calls); NEMO's 1-rank lane reads whichever single file its own namelist
    names.  This compares EVERY shared float variable of rank >= 3 (which on a
    NEMO restart INCLUDES the 2-D fields, stored as (time,y,x) -- ssh, uu_b and
    the surface-forcing carries are covered) bit for bit and refuses the state
    on any mismatch.

    So it is a check on the TWO STITCHERS, not on two independently produced
    ocean states:
      * days 190..270 -- NEMO reads REBUILD_NEMO's Fortran stitch of the twin's
        16 tiles and legoESM reads the Python stitch of the SAME 16 tiles.
        Different inodes, one byte source: it genuinely tests halo handling,
        axis order and variable coverage in the Python stitcher, and nothing
        about state provenance.
      * day 180 and the consecutive-step lanes -- both paths resolve to the
        SAME INODE (measured and printed below).  There the check is an
        identity and proves nothing at all; it is kept only so that no state is
        silently exempt from the gate.
    The inode pair is printed for every state so the reader can see which of
    the two situations a given row is in rather than inferring it.
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
    same_inode = False
    try:
        st_n = os.stat(os.path.realpath(nemo_file))
        st_t = os.stat(os.path.realpath(tile_files[0]))
        same_inode = (st_n.st_ino == st_t.st_ino and st_n.st_dev == st_t.st_dev)
    except OSError:
        pass
    if worst != 0.0:
        raise SystemExit(
            f"FATAL day {day}: the two entry states differ (worst {worst:.3e} "
            f"on {worst_name!r}). NEMO read {nemo_file}; legoESM stitches "
            f"{len(tile_files)} tiles from {tile_glob}. A deposit measured "
            "from different states measures the difference of the states.")
    return {"entry_state_maxdiff": worst, "entry_nvars": len(names),
            "entry_ntiles": len(tile_files), "nemo_restart": nemo_file,
            "entry_same_inode": bool(same_inode)}


def _run_one(day: int, kt: int, lane: str, tiles: str, *,
             t_seconds: float | None, log_dir: str,
             map_dir: str | None = None) -> dict:
    """One child run of the committed instrument at one bridged state.

    ``map_dir`` asks the child for its PER-CELL deposit map as well as the
    scalars this walk parses.  The map is what a per-cell two-step projection
    needs: this wrapper's regexes recover only band-reduced scalars, and a
    band sum annihilates an alternating signal whose sign varies along the
    wall (measured at 44x-80000x in the zonal-wall budget).  Off by default so
    the existing scalar arms are byte-unchanged.
    """
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
    map_path = None
    if map_dir is not None:
        os.makedirs(map_dir, exist_ok=True)
        map_path = os.path.join(map_dir, f"deposit_map_kt{kt}.npz")
        env["DINO_1455_DEPOSIT_MAP"] = map_path
    else:
        env.pop("DINO_1455_DEPOSIT_MAP", None)

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
    if map_path is not None:
        if not os.path.exists(map_path):
            raise SystemExit(
                f"FATAL day {day}: the child was asked for a deposit map at "
                f"{map_path} and did not write one. A missing map must never "
                "become a dropped state in a time projection.")
        rec["deposit_map"] = map_path
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


#: The 10-day grid's measured in-loop spread, (max-min)/mean, from the
#: committed run in 89dadfeb4.  The consecutive-step arm is scored against it.
GRID320_IN_LOOP_SPREAD = 3.1623e-03


def _assert_threshold_matches_artifact(out_path: str) -> None:
    """The pre-registered threshold is a literal; if the 10-day walk is ever
    re-run the literal goes stale silently.  Recompute it from that walk's own
    artifact when it is present and refuse on a mismatch."""
    grid = os.path.join(os.path.dirname(out_path), "baro_deposit_time_walk.npz")
    if not os.path.isfile(grid):
        print("    (10-day artifact absent: the pre-registered threshold "
              f"{GRID320_IN_LOOP_SPREAD:.4e} is NOT re-derived, only trusted)")
        return
    d = np.asarray(np.load(grid, allow_pickle=True)["dep_in_loop"], dtype=float)
    got = float((d.max() - d.min()) / abs(d.mean()))
    if abs(got - GRID320_IN_LOOP_SPREAD) > 1e-3 * GRID320_IN_LOOP_SPREAD:
        raise SystemExit(
            f"the pre-registered threshold {GRID320_IN_LOOP_SPREAD:.6e} no "
            f"longer matches the 10-day artifact's {got:.6e}; refusing to "
            "score against a stale criterion")
    print(f"    threshold re-derived from the 10-day artifact: {got:.6e} "
          f"(literal {GRID320_IN_LOOP_SPREAD:.6e})")


def consecutive_stats(d: np.ndarray) -> dict:
    """Spread, parity split and sign pattern of one consecutive-step series.

    Extracted from ``_report_consecutive`` so a test can plant a period-2
    signal and a null and have them reach THIS code -- the previous test
    re-implemented the arithmetic inline and passed with the whole block
    deleted, which is the same defect it was written to prevent.

    PARITY SPLIT is the direct statistic for the mode under test: a period-2
    component sits entirely in the even-minus-odd difference of the means.
    The range statistic only bounds it indirectly and is sample-size dependent
    (a 5-point range underestimates a 10-point range for the same scatter), so
    the range RATIO must never be quoted as a "times flatter".

    LEAVE-ONE-OUT on the FIRST point is reported because in this campaign's
    consecutive lanes the first state is the only one whose restart comes from
    a different run than the rest, and both adversarial reviews found a claim
    resting on it alone.
    """
    d = np.asarray(d, dtype=float)
    if d.size < 3:
        raise ValueError("need at least 3 consecutive samples")

    def _stats(x):
        ev, od = x[0::2], x[1::2]
        return (float((x.max() - x.min()) / abs(x.mean())),
                float(abs(ev.mean() - od.mean()) / abs(x.mean())))

    spread, parity = _stats(d)
    spread1, parity1 = _stats(d[1:])
    dif = np.diff(d)
    return {"spread": spread, "parity_split": parity,
            "spread_drop_first": spread1, "parity_split_drop_first": parity1,
            # an exact 0.0 is neither: map it to '0' rather than silently
            # calling it negative.
            "diff_signs": "".join("+" if x > 0 else "-" if x < 0 else "0"
                                  for x in dif),
            # a strictly alternating pattern (either phase) arises by chance on
            # n-1 differences with probability 2/2^(n-1)
            "p_sign_pattern_by_chance": 2.0 / (2.0 ** len(dif)),
            "values": d.tolist()}


def _raw_state_parity(kts, tiles: str) -> dict:
    """max|du| between consecutive NEMO restarts, its parity split, and the
    CELL each maximum sits in.

    SECONDARY control only.  It shows the state was not static.  It is NOT the
    control that makes the null meaningful -- that is the TOTAL deposit's own
    spread, which is the same functional from the same runs -- because a
    max-norm of a 3-D velocity increment and a signed section integral are
    different functionals, and because the argmax metadata shows the maximum
    moving between cells rather than one mode being tracked.
    """
    import netCDF4 as nc

    src = os.path.join(_DINO, tiles)
    prev, out, argmax = None, [], []
    for kt in kts:
        cand = [os.path.join(src, f"DINO_{kt:08d}_restart_0000.nc"),
                os.path.join(src, f"DINO_{kt:08d}_restart.nc")]
        f = next((c for c in cand if os.path.exists(c)), None)
        if f is None:
            raise SystemExit(f"positive control: no restart for kt={kt} in {src}")
        d = nc.Dataset(f)
        u = np.asarray(d.variables["un"][0], dtype=np.float64)
        d.close()
        if prev is not None:
            du = np.abs(u - prev)
            out.append(float(du.max()))
            argmax.append([int(v) for v in np.unravel_index(
                int(du.argmax()), du.shape)])
        prev = u
    a = np.array(out)
    if a.size < 2:
        raise SystemExit("positive control needs at least 3 restarts")
    ev, od = a[0::2], a[1::2]
    return {"max_du": a.tolist(), "argmax_cells": argmax,
            "parity_split": float(abs(ev.mean() - od.mean()) / abs(a.mean())),
            "parity_split_drop_first":
                float(abs(a[1::2].mean() - a[2::2].mean()) / abs(a[1:].mean()))
                if a.size > 3 else float("nan")}


def _report_consecutive(recs, out_path, log_dir, prov) -> int:
    """The ALIASING test, with its criterion pre-registered here in code.

    PRE-REGISTERED, before the three consecutive lanes were ever measured:
    the in-loop share on the 320-step grid has relative spread
    (max-min)/mean = 3.16e-03.  Let r_c be the same statistic over four
    CONSECUTIVE steps.

      ALIASED       iff r_c > 5 * 3.16e-03, i.e. the series moves far more
                    from step to step than the 320-step grid could see.  A
                    step-alternating (period-2) pattern is the specific mode
                    that the even 320-step offset hides completely, so the
                    sign pattern of the successive differences is printed
                    next to r_c.
      NOT ALIASED   iff r_c <= 2 * 3.16e-03: the share really is constant at
                    step resolution and the 10-day grid was not hiding
                    structure.
      INCONCLUSIVE  in between (2x..5x) -- named in advance so an awkward
                    number cannot be argued either way afterwards.

    Prints numbers and the thresholds.  No verdict.
    """
    _assert_threshold_matches_artifact(out_path)
    kts = np.array([r["kt"] for r in recs], dtype=int)
    if len(recs) < 3:
        raise SystemExit("the aliasing test needs at least 3 consecutive steps")
    print("\n  CONSECUTIVE-STEP arm (the 320-step grid cannot see this).")
    print("  OPEN CONFOUND, printed because a reader of this log cannot infer "
          "it: the FIRST state's restart is the 16-rank 90-day twin's own")
    print("  checkpoint, while the rest come from a 1-rank continuation that "
          "started from it.  A restart written as a run's terminal state need")
    print("  not carry the same leap-frog time levels as one written "
          "mid-chain, so the first row's excess has an unexcluded non-physical")
    print("  explanation.  Every statistic below is therefore reported with "
          "its leave-one-out twin.")
    print("     kt  |     TOTAL     |    FORCING    |    IN-LOOP")
    for r in recs:
        print(f"    {r['kt']:5d} | {r['total']:+.6e} | {r['forcing']:+.6e} "
              f"| {r['in_loop']:+.6e}")
    out = {}
    for name in ("total", "forcing", "in_loop"):
        d = np.array([r[name] for r in recs], dtype=float)
        st = consecutive_stats(d)
        out[name] = st
        print(f"    {name:8s} spread (max-min)/|mean| = {st['spread']:.4e}   "
              f"PARITY |even-odd|/|mean| = {st['parity_split']:.4e}   "
              f"diff signs '{st['diff_signs']}' (any strict alternation arises "
              f"by chance with p={st['p_sign_pattern_by_chance']:.3f})")
        print(f"             LEAVE-ONE-OUT dropping the first point: "
              f"spread {st['spread_drop_first']:.4e}  PARITY "
              f"{st['parity_split_drop_first']:.4e}")
    r_c = out["in_loop"]["spread"]
    r_c1 = out["in_loop"]["spread_drop_first"]
    print(f"\n    PRE-REGISTERED (on the SPREAD; the parity split has no "
          f"threshold and is reported, not scored):")
    print(f"      320-step-grid in-loop spread = {GRID320_IN_LOOP_SPREAD:.4e}")
    print(f"      consecutive/grid ratio = {r_c / GRID320_IN_LOOP_SPREAD:.3f}"
          f"   (ALIASED if > 5, NOT ALIASED if <= 2, INCONCLUSIVE between)")
    print(f"      same, dropping the cross-provenance first point = "
          f"{r_c1 / GRID320_IN_LOOP_SPREAD:.3f}")
    # RESOLUTION FLOOR.  The child instrument prints its deposits at five
    # significant figures, so on a 2.02e-03 value the print quantum is 1e-07,
    # i.e. ~5e-05 RELATIVE.  Any statistic below that is an upper BOUND, not a
    # measurement -- and this branch has already had to retract one claim
    # (a sign pattern) for exactly that reason.  Flag it mechanically instead
    # of leaving it to be noticed.
    print("\n    PRINT-RESOLUTION FLOOR: the child prints 5 significant "
          "figures, so each channel has its OWN relative quantum "
          "(1e-07 / |mean|).")
    quantum = 1e-7 / abs(np.mean([r["in_loop"] for r in recs]))
    for name in ("total", "forcing", "in_loop"):
        q = 1e-7 / abs(np.mean([r[name] for r in recs]))
        print(f"      {name:8s} quantum = {q:.1e}")
        for key in ("spread", "spread_drop_first", "parity_split",
                    "parity_split_drop_first"):
            v = out[name][key]
            if v < q:
                print(f"        {name}.{key} = {v:.3e} is BELOW one quantum "
                      f"-> an upper bound of order {q:.0e}, NOT a measured "
                      f"value")
    print(f"      the ALIASED arm needs an in-loop spread above "
          f"{5 * GRID320_IN_LOOP_SPREAD:.3e} = "
          f"{5 * GRID320_IN_LOOP_SPREAD / quantum:.0f} of that channel's "
          f"quanta, so the verdict itself is not resolution-limited")
    # POSITIVE CONTROL.  A flat in-loop share means nothing if the instrument
    # could not have seen a step-to-step change at all, so the null needs a
    # channel that DID move.  The primary control is the one in the table
    # above, and it is the strongest available because it shares everything
    # with the null: the TOTAL deposit is the SAME functional, from the SAME
    # child run at the SAME five states, and it moves ~100x more.  Printed
    # below.
    #
    # The raw-state series that follows is SECONDARY and is not load-bearing:
    # it is a max-norm of a 3-D velocity increment, a different functional
    # from the signed section integral the null is measured in, and its own
    # argmax metadata shows the maximum MOVING between cells rather than one
    # mode being tracked.  It shows only that the state was not static.
    _ctl = out["total"]["spread"] / out["in_loop"]["spread"]
    print(f"\n    PRIMARY POSITIVE CONTROL (same functional, same child runs, "
          f"same five states):")
    _ctl1 = (out["total"]["spread_drop_first"]
             / out["in_loop"]["spread_drop_first"])
    print(f"      the TOTAL deposit's spread is {out['total']['spread']:.3e} "
          f"against the in-loop channel's {out['in_loop']['spread']:.3e} "
          f"-> {_ctl:.0f}x   (leave-one-out twin, as promised above: "
          f"{_ctl1:.0f}x)")
    print("      So the instrument DOES respond to the step-to-step state "
          "change, in the null's own units, while the in-loop channel does "
          "not.")
    pc = _raw_state_parity([r["kt"] for r in recs], recs[0]["tiles"])
    out["raw_state_positive_control"] = pc
    print(f"\n    POSITIVE CONTROL (raw state, max|du| between consecutive "
          f"restarts, m/s):")
    print("      " + "  ".join(f"{v:.4e}" for v in pc["max_du"]))
    print(f"      argmax cells (k,j,i): {pc['argmax_cells']}")
    print(f"      parity split of that series = {pc['parity_split']:.4f}  "
          f"(dropping the first difference: "
          f"{pc['parity_split_drop_first']:.4f})")
    print("      -> SECONDARY, and weaker than it looks: the state changes "
          "between consecutive steps by amounts differing roughly two-fold,")
    print("         but the argmax cells above show that is PARTLY the "
          "maximum MOVING (3 of 4 on one row, 1 in the interior), so this is")
    print("         not one physical mode sampled four times and must not be "
          "read as a clean period-2 signature.  It is also a DIFFERENT")
    print("         FUNCTIONAL from the null (max-norm of a 3-D increment vs "
          "a signed section integral), so no ratio between them is quoted.")
    print("         Its only job is to show the state was not static.  The "
          "load-bearing control is the TOTAL-vs-in-loop contrast above.")
    np.savez(out_path, kts=kts,
             **{f"dep_{k}": np.array([r[k] for r in recs], dtype=float)
                for k in ("total", "forcing", "in_loop")},
             summary_json=json.dumps(
                 {"provenance": prov, "mode": "consecutive",
                  "grid320_in_loop_spread": GRID320_IN_LOOP_SPREAD,
                  "records": recs, "stats": out}, default=float))
    print(f"\n  artifact: {out_path}\n  logs:     {log_dir}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma-separated days to run (default: all ten)")
    ap.add_argument("--consecutive", action="store_true",
                    help="run the four CONSECUTIVE NEMO steps at day 180 "
                         "instead of the 10-day grid, to test whether the "
                         "in-loop share's constancy is aliasing")
    ap.add_argument("--deposit-map-dir", default="",
                    help="also write each child's PER-CELL deposit map into "
                         "this directory (deposit_map_kt<KT>.npz). Needed for "
                         "a per-cell two-step projection; the scalars this "
                         "walk parses are band sums and cannot carry one.")
    ap.add_argument("--clock-ab", action="store_true",
                    help="also run day 180 under the LEGACY bare-dt clock. "
                         "This does NOT size anything: the clock has no path "
                         "to this metric, so the zero is guaranteed. Kept only "
                         "to show the override takes effect.")
    ap.add_argument("--out", default=os.path.join(
        _THIS_DIR, "..", "..", "..", "..", "results", "dino_1455",
        "baro_deposit_time_walk.npz"))
    args = ap.parse_args(argv)

    sys.path.insert(0, _THIS_DIR)
    import multistep_replay as mr
    prov = mr.provenance("baro_deposit_time_walk")

    if args.consecutive and args.out.endswith("baro_deposit_time_walk.npz"):
        args.out = args.out.replace("baro_deposit_time_walk.npz",
                                    "baro_deposit_consecutive.npz")
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_dir = os.path.join(os.path.dirname(out_path), "logs")
    os.makedirs(log_dir, exist_ok=True)

    catalog = CONSEC_STATES if args.consecutive else STATES
    want = ([int(x) for x in args.only.split(",") if x.strip()]
            if args.only else [s[0] for s in catalog])
    states = [s for s in catalog if s[0] in want]
    if len(states) != len(want):
        raise SystemExit(f"unknown state key(s) in --only={args.only!r}; "
                         f"this mode offers {[s[0] for s in catalog]}")

    print(f"\n=== #1455 PHASE 1: the deposit's TIME AXIS "
          f"({len(states)} bridged states"
          f"{', CONSECUTIVE steps' if args.consecutive else ''}) ===")
    if args.consecutive:
        # the 10-day banner below does not apply to this arm and printing it
        # would attach the wrong pre-registration to the numbers.
        print("PRE-REGISTERED (see _report_consecutive): ALIASED iff the "
              "consecutive in-loop spread exceeds 5x the 320-step grid's "
              f"{GRID320_IN_LOOP_SPREAD:.4e},")
        print("  NOT ALIASED iff <= 2x, INCONCLUSIVE between.  The 90-day "
              "integral criteria are NOT defined on this arm and are not "
              "scored.")
    else:
        print("PRE-REGISTERED (see module docstring): ACCUMULATOR iff the "
              "90-day trapezoid integral")
        print(f"  I in [{2 * BARO_ROW_90D:+.4f}, {0.5 * BARO_ROW_90D:+.4f}]"
              f" Sv (factor 2 either side of the {BARO_ROW_90D:+.4f} Sv budget"
              " row) AND sign-agrees with")
        print("  the per-window row in >=7 of 9 windows.  EXONERATED iff "
              "|mean|/rms < 0.5 AND |I| < "
              f"{abs(0.5 * BARO_ROW_90D):.4f} Sv.")
        print("  NOTE (post-review): both criteria presuppose a retention "
              "factor of 1 between the matched-state injection they score and "
              "the free-running\n  accumulation they score it against.  That "
              "factor is UNMEASURED, so A1/A2 cannot be read as an "
              "attribution.  (R_deposit is NOT that\n  factor -- it is a "
              "within-step forcing-to-transport response ratio against a ramp "
              "model; that substitution is retracted in 4c14b4d91.)")

    recs = [_run_one(day, kt, lane, tiles, t_seconds=None, log_dir=log_dir,
                     map_dir=args.deposit_map_dir or None)
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
          "forcing% | plant | entry maxdiff (2 stitchers? )")
    for r in recs:
        fr = 100.0 * r["forcing"] / r["total"] if r["total"] else float("nan")
        _one = ("SAME INODE -> identity, proves nothing"
                if r.get("entry_same_inode") else
                f"2 stitchers, {r['entry_nvars']} vars")
        print(f"    {r['day']:3.0f} | {r['total']:+.6e} | {r['forcing']:+.6e} "
              f"| {r['in_loop']:+.6e} | {fr:7.1f}% | {r['plant_ratio']:5.2f} "
              f"| {r['entry_state_maxdiff']:.1e} ({_one})")

    if ab is not None:
        print("\n  CLOCK A/B at day 180 (absolute vs the legacy bare-dt clock "
              "the three prior commits ran).")
        print("  READ THIS BEFORE THE NUMBERS: this A/B PERTURBS A CHANNEL "
              "WITH NO PATH to the measured quantity, so a zero here is NOT a "
              "bound on\n  anything.  t_seconds enters only the tracer "
              "restoring and the solar flux; the wind is built without a clock "
              "argument at all and both arms\n  print the SAME tau_x range.  "
              "A control that perturbs a zero is not a control (this campaign's "
              "own rule).  The zero below is expected\n  by construction and is "
              "recorded only to show the override took effect.")
        print(f"    absolute t={recs[0]['clock_s']:.0f}s  TOTAL="
              f"{recs[0]['total']:+.6e}  IN-LOOP={recs[0]['in_loop']:+.6e}")
        print(f"    legacy   t={ab['clock_s']:.0f}s  TOTAL="
              f"{ab['total']:+.6e}  IN-LOOP={ab['in_loop']:+.6e}")
        for k in ("total", "in_loop", "forcing"):
            d = recs[0][k] - ab[k]
            print(f"    delta {k:8s} = {d:+.4e} Sv/step "
                  f"({100 * d / ab[k] if ab[k] else float('nan'):+.2f}% of the "
                  "legacy value)")

    if args.consecutive:
        return _report_consecutive(recs, out_path, log_dir, prov)

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
        pw = power_check(series["total"])
        summary["power"] = pw
        print(f"\n    POWER (pre-registered): sem(total) = {pw['sem']:.3e} vs "
              f"|target mean| = {pw['target']:.3e} Sv/step -> "
              f"underpowered={pw['underpowered']}"
              f"{'  (the pre-registration then forbids using the integral as a pass/fail)' if pw['underpowered'] else ''}")
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
