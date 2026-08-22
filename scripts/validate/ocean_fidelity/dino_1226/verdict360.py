#!/usr/bin/env python
"""#1455 THE VERDICT RUN: is the shipped DINO card statistically indistinguishable
from NEMO one YEAR after a shared restart?

Pre-registration: ``PREREG_verdict360.md``, committed BEFORE any member ran.
Read it before citing any number this prints.

WHAT THIS IS.  ``acceptance_gate_90d.py`` scores a 90-day twin against floor
constants; ``floor90_ensemble.py`` / ``floor90_nemo_ensemble.py`` measured what
those floors REALLY are at 90 days, on each model separately.  At 90 days the
1e-14 kick has not saturated, so those floors are ~1e-5 Sv and every gap is
thousands of floors wide -- which says nothing about climate fidelity, only
that 90 days is inside the deterministic-predictability window.  This probe
moves the whole apparatus to a CLIMATE-RELEVANT horizon: 360 days, where the
kick has saturated and the ensemble spread is the honest irreducible floor a
two-model comparison can be held to (the FESOM bar).

THE INSTRUMENT.  Eight runs, four per side, all from the SAME NEMO day-180
restart (``DINO_00005760_restart.nc``):

  legoESM  /tmp/dino_verdict360/m{0..3}   shipped card, no option flags,
                                          member 0 unperturbed, members 1-3
                                          with a 1e-14 relative now-level T
                                          kick at seeds 1/2/3
  NEMO     RUN_VERDICT360_M{0..3}         certified binary, RUN_90D_TWIN's
                                          namelist with nn_itend extended to
                                          360 days and NOTHING else changed,
                                          same seeds via perturb_nemo_tn_90d

  member 0 (each side)  ->  the TWIN pair.  Their difference at a matched day
                            is THE GAP.
  members 1-3 (each side) -> that side's own ensemble spread at that day, i.e.
                            how much a run of that model moves under a
                            perturbation far below any physical signal.  The
                            two spreads combine (RSS) into the two-sided floor
                            the gap is judged against.

METRICS -- all IMPORTED, none re-derived here:
  * the five acceptance-gate metrics                (acceptance_gate_90d.metrics)
  * the channel band, both recorded reductions      (floor90_ensemble.band_transport,
                                                     .band_transport_campaign)
  * south-of-band / band / north-of-band transport  (acc_driver_decomp.group_transport
                                                     with its own LAT_GROUPS + _avg)

WHY THE FULL SECTION IS REPORTED BUT NOT THE VERDICT.  ``acc_full`` integrates
every latitude row into one number, and the campaign's own decomposition shows
the three latitude groups carry OPPOSITE-SIGNED gaps (south-of-band negative,
channel positive).  A small full-section number can therefore be two large
cancelling errors, and a large one can be one small error in a group with
little compensation.  The gate metric is still printed -- it is the recorded
headline and dropping it would hide a regression -- but every verdict in the
table is per-GROUP.

VERDICT ARITHMETIC, stated once so it cannot drift:
  gap(day)   = lego member-0 metric - NEMO member-0 metric, SAME day, SAME
               reduction on both sides.
  floor(day) = sqrt( spread_lego(day)^2 + spread_NEMO(day)^2 ), the RSS of the
               two measured single-run spreads.  When the two sides happen to
               wobble equally this reduces to sqrt(2) x one side's spread,
               which is the difference-of-two-runs factor the 90-day lane
               established; the RSS is that rule without the equal-wobble
               assumption.
  INDISTINGUISHABLE  <=>  |gap| <= 2 x floor(day).
The primary spread statistic is the sample std (n=4 -> ~41% relative standard
error, so every floor is a factor-of-two estimate); the max-pairwise range is
printed beside it and, at n=4, runs ~2x the std BY CONSTRUCTION.

WINDOW STATISTIC.  An endpoint is one draw of a chaotic trajectory.  The
final-90-day window mean (days 280..360, 10-day samples, IDENTICAL sample days
on both sides) is the more climate-like statistic, and its floor is the spread
of the members' OWN window means -- not the endpoint floor reused.  Both are
reported; neither replaces the other.

WHAT IT IS NOT.  It does not edit any FLOORS constant.  It prints a verdict
because a verdict is the deliverable here -- but the verdict is a mechanical
comparison of two printed numbers, not an interpretation, and the numbers are
printed next to it.

Usage
-----
  verdict360.py --self-check                 # arithmetic, no runs needed
  verdict360.py --setup-nemo                 # build the 4 NEMO run dirs
  verdict360.py --run-nemo                   # launch all 4 NEMO members
  verdict360.py --run-lego --gpus 0,1        # launch the 4 legoESM members
  verdict360.py                              # score everything on disk
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))
import acc_driver_decomp as D          # noqa: E402  group_transport / LAT_GROUPS
import acc_thermal_wind as A           # noqa: E402  mesh, masks, J0/J1
import acceptance_gate_90d as G        # noqa: E402  the five gate metrics
import floor90_ensemble as F           # noqa: E402  spread(), band reductions
import perturb_nemo_tn_90d as P        # noqa: E402  the committed NEMO kick

# D._avg is the reduction the committed southern-basin numbers were produced
# with (southern_circulation_budget.py:823-833 imports it the same way).
# Re-spelling `mean over longitudes 2..-2` here would be a second chance for a
# reduction drift, which is the defect class this campaign has spent the most
# time on -- so the private name is imported deliberately, not worked around.
_avg = D._avg

SEEDS = (None, 1, 2, 3)
N_MEM = len(SEEDS)
N_DAYS = 360
SNAP_GRID = tuple(range(0, N_DAYS + 1, 10))     # legoESM 3-D snapshot days
CURVE_DAYS = (10, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360)
WINDOW_DAYS = tuple(range(280, N_DAYS + 1, 10))  # the final 90 days, 9 samples
HORIZONS = (90, 180, 270, 360)                   # the scored horizons
SCORE_DAYS = tuple(sorted(set(CURVE_DAYS) | set(WINDOW_DAYS) | set(HORIZONS)))

LEGO_DIR_DEFAULT = "/tmp/dino_verdict360"
NEMO_SRC = f"{A.DINO}/RUN_90D_TWIN"              # namelist + restart donor
NEMO_CERT = f"{A.DINO}/BLD/bin/nemo.exe.certified_d3cf9242"
MPIRUN = "/home/dbalwada/miniconda3/envs/nemo-build/bin/mpirun"
N_RANK = 16
KT_END = G.KT_RESTART + N_DAYS * G.STEPS_PER_DAY  # 17280

KEYS = G.KEYS + ("band", "band_c", "g_south", "g_band", "g_north")
LABELS = dict(G.LABELS,
              band="channel band [Sv] e3t_1d,median",
              band_c="channel band [Sv] e3t_0,mean",
              g_south="SOUTH of band transport [Sv]",
              g_band="channel band transport [Sv] (group)",
              g_north="NORTH of band transport [Sv]")
# The full section mixes the signs of the three groups, so it is reported and
# never given a per-group verdict of its own.
SIGN_MIXING = ("acc",)


def member_name(i):
    return "m0_control" if SEEDS[i] is None else f"m{i}_seed{SEEDS[i]}"


def nemo_dir(i):
    return f"{A.DINO}/RUN_VERDICT360_M{i}"


def kt_of(day):
    return G.KT_RESTART + day * G.STEPS_PER_DAY


# ------------------------------------------------------------------ metrics ---
def all_metrics(st, wet):
    """The five gate metrics + the two channel-band reductions + the three
    latitude-group transports, every one of them an imported reduction."""
    m = dict(G.metrics(st, wet))
    u = st["u"]
    m["band"] = F.band_transport(u, A.umask)
    m["band_c"] = F.band_transport_campaign(u, A.umask)
    for key, (_name, rows) in zip(("g_south", "g_band", "g_north"), D.LAT_GROUPS):
        m[key] = _avg(D.group_transport(u, A.umask, rows))
    return m


# ------------------------------------------------------------------ loaders ---
def lego_state(npz, day):
    return G.load_candidate(npz, day)


def nemo_state(run_dir, day):
    return G.load_nemo_day90(run_dir=run_dir, kt=kt_of(day))


# ------------------------------------------------------------- legoESM runs ---
def lego_npz(out_dir, i):
    return f"{out_dir}/{member_name(i)}.npz"


def lego_log(out_dir, i):
    return f"{out_dir}/{member_name(i)}.log"


def lego_cmd(out_dir, i):
    """The SHIPPED card, no option flags beyond the ones floor90_ensemble's own
    member launcher uses -- same card, same bridge, only the length and the
    snapshot grid differ."""
    cmd = [sys.executable, f"{_DIR}/run_fp64.py", f"{_DIR}/kamm_twin_90d.py",
           "nemo_dino_kamm_mlf", lego_npz(out_dir, i),
           "--days", str(N_DAYS), "--bridge-before", "--save-3d",
           "--snap-days", ",".join(str(d) for d in SNAP_GRID)]
    if SEEDS[i] is not None:
        cmd += ["--perturb-seed", str(SEEDS[i])]
    return cmd


def run_lego(out_dir, gpus):
    """Launch every missing member, one per GPU, at most len(gpus) at a time.

    Members are pinned to a GPU SLOT and a slot is refilled only after the run
    occupying it has exited.  The obvious spelling -- pick the GPU from the
    current queue length, then drain -- hands the next member a device that is
    still busy, because the drain happens after the choice.
    """
    os.makedirs(out_dir, exist_ok=True)
    todo = [i for i in range(N_MEM) if not os.path.exists(lego_npz(out_dir, i))]
    for i in range(N_MEM):
        if i not in todo:
            print(f"[skip] {lego_npz(out_dir, i)} exists", flush=True)
    slots = {g: None for g in gpus}          # gpu -> (member, Popen, filehandle)

    def _reap(gpu):
        j, pr, fh = slots[gpu]
        rc = pr.wait()
        fh.close()
        slots[gpu] = None
        print(f"[done] {member_name(j)} on GPU {gpu} rc={rc}", flush=True)

    for n, i in enumerate(todo):
        gpu = gpus[n % len(gpus)]
        if slots[gpu] is not None:
            _reap(gpu)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, JAX_ENABLE_X64="1",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false")
        cmd = lego_cmd(out_dir, i)
        print("RUN:", " ".join(cmd), f"[CUDA_VISIBLE_DEVICES={gpu}]", flush=True)
        fh = open(lego_log(out_dir, i), "w")
        slots[gpu] = (i, subprocess.Popen(cmd, env=env, stdout=fh,
                                          stderr=subprocess.STDOUT), fh)
    for gpu in gpus:
        if slots[gpu] is not None:
            _reap(gpu)


def lego_completed(log_path):
    """The harness's OWN success line -- an exit code is not evidence."""
    if not os.path.exists(log_path):
        return False, "no log"
    with open(log_path, errors="replace") as fh:
        txt = fh.read()
    if f"DONE nsteps={G.STEPS_PER_DAY * N_DAYS} STABLE=True" in txt:
        return True, "OK"
    for ln in txt.splitlines():
        if ln.startswith("BLOWUP") or ln.startswith("DONE "):
            return False, ln.strip()
    return False, "no DONE line (still running or killed)"


# ---------------------------------------------------------------- NEMO runs ---
def setup_nemo():
    """Build the four NEMO run dirs.  One variable against RUN_90D_TWIN: the
    run length.  Everything else -- namelist_ref, the certified binary, the
    source restart, nn_it000, nn_stock -- is copied, not retyped."""
    src_nml = f"{NEMO_SRC}/namelist_cfg"
    with open(src_nml) as fh:
        lines = fh.readlines()
    out, n_hit = [], 0
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("nn_itend") and "=" in stripped:
            out.append(f"   nn_itend    =   {KT_END}   !  360-day verdict run "
                       f"(5761->{KT_END}, {N_DAYS} d x {G.STEPS_PER_DAY} steps)\n")
            n_hit += 1
        else:
            out.append(ln)
    if n_hit != 1:
        raise SystemExit(f"expected exactly 1 active nn_itend in {src_nml}, "
                         f"found {n_hit} -- refusing to guess")
    for i in range(N_MEM):
        d = nemo_dir(i)
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/namelist_cfg", "w") as fh:
            fh.writelines(out)
        shutil.copy2(f"{NEMO_SRC}/namelist_ref", f"{d}/namelist_ref")
        link = f"{d}/nemo"
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(NEMO_CERT, link)
        if SEEDS[i] is None:
            shutil.copy2(P.SRC, f"{d}/{P.SRC.name}")
            print(f"[{member_name(i)}] {d}: unperturbed restart copied")
        else:
            rep = P.perturb(SEEDS[i], __import__("pathlib").Path(d))
            print(f"[{member_name(i)}] {d}: perturbed, "
                  f"max|dtn|={rep['max_abs_tn_diff']:.3e}")
    print(f"nn_itend set to {KT_END} in all {N_MEM} dirs; "
          f"every other namelist line byte-identical to {src_nml}")


def run_nemo():
    """Launch all four members CONCURRENTLY (4 x 16 = 64 of 80 cores)."""
    procs = []
    for i in range(N_MEM):
        d = nemo_dir(i)
        if not os.path.exists(f"{d}/nemo"):
            raise SystemExit(f"{d} not set up -- run --setup-nemo first")
        log = f"{d}/run_verdict360_m{i}.log"
        fh = open(log, "w")
        cmd = [MPIRUN, "-np", str(N_RANK), "./nemo"]
        print("RUN:", " ".join(cmd), f"[cwd={d}]", flush=True)
        procs.append((i, subprocess.Popen(cmd, cwd=d, stdout=fh,
                                          stderr=subprocess.STDOUT), fh))
    for i, pr, fh in procs:
        rc = pr.wait()
        fh.close()
        print(f"[done] NEMO {member_name(i)} rc={rc}", flush=True)


def nemo_completed(run_dir, i):
    ts = f"{run_dir}/time.step"
    if not os.path.exists(ts):
        return False, "no time.step"
    with open(ts) as fh:
        got = int(fh.read().strip())
    if got != KT_END:
        return False, f"time.step={got}, expected {KT_END}"
    oo = f"{run_dir}/ocean.output"
    if os.path.exists(oo):
        with open(oo, errors="replace") as fh:
            if "E R R O R" in fh.read():
                return False, "E R R O R in ocean.output"
    log = f"{run_dir}/run_verdict360_m{i}.log"
    if not os.path.exists(log):
        return False, "no run log"
    with open(log, errors="replace") as fh:
        tail = fh.read()[-200:]
    if "STOP 0" not in tail:
        return False, f"no 'STOP 0' in log tail: {tail!r}"
    return True, "OK"


# ------------------------------------------------------------------- scoring --
def collect(out_dir):
    """rows[side][member][day] -> metric dict.  Loads each state exactly once."""
    wet = A.tmask
    rows = {"lego": {}, "nemo": {}}
    for i in range(N_MEM):
        rows["lego"][i] = {}
        rows["nemo"][i] = {}
        npz = lego_npz(out_dir, i)
        for day in SCORE_DAYS:
            ml = all_metrics(lego_state(npz, day), wet)
            mn = all_metrics(nemo_state(nemo_dir(i), day), wet)
            for tag, m in (("lego", ml), ("nemo", mn)):
                bad = [k for k in KEYS if not np.isfinite(m[k])]
                if bad:
                    raise SystemExit(f"non-finite {bad} for {tag} "
                                     f"{member_name(i)} day {day} -- a "
                                     f"non-finite metric is a FINDING")
            rows["lego"][i][day] = ml
            rows["nemo"][i][day] = mn
        print(f"  scored {member_name(i)} ({len(SCORE_DAYS)} days, both sides)",
              flush=True)
    return rows


def spread_at(rows, side, key, day):
    return F.spread([rows[side][i][day][key] for i in range(N_MEM)])


def two_sided_floor(rows, key, day, stat=1):
    """RSS of the two sides' measured spreads.  stat 0 = max-pairwise, 1 = std."""
    ls = spread_at(rows, "lego", key, day)[stat]
    ns = spread_at(rows, "nemo", key, day)[stat]
    return float(np.sqrt(ls ** 2 + ns ** 2)), ls, ns


def verdict(gap, floor):
    """INDISTINGUISHABLE iff |gap| is within 2x the two-sided floor."""
    if floor == 0.0:
        return "NO-FLOOR"
    return "YES" if abs(gap) <= 2.0 * floor else "no"


def window_mean(rows, side, i, key):
    return float(np.mean([rows[side][i][d][key] for d in WINDOW_DAYS]))


# ---------------------------------------------------------------- self-check --
def _self_check():
    """Runnable, non-vacuous check of the two pieces of arithmetic this file
    owns: the RSS two-sided floor and the 2x verdict rule.  It must be able to
    FAIL, so a synthetic gap outside the band is asserted to be rejected."""
    F._self_check()
    fake = {"lego": {i: {90: {"x": v}} for i, v in enumerate([1.0, 1.1, 0.9, 1.0])},
            "nemo": {i: {90: {"x": v}} for i, v in enumerate([2.0, 2.2, 1.8, 2.0])}}
    fl, ls, ns = two_sided_floor(fake, "x", 90)
    ls_ref = float(np.std([1.0, 1.1, 0.9, 1.0], ddof=1))
    ns_ref = float(np.std([2.0, 2.2, 1.8, 2.0], ddof=1))
    assert abs(ls - ls_ref) < 1e-12 and abs(ns - ns_ref) < 1e-12, (ls, ns)
    assert abs(fl - np.sqrt(ls_ref ** 2 + ns_ref ** 2)) < 1e-12, fl
    # equal-wobble sides must reduce the RSS to exactly sqrt(2) x one side
    eq = {"lego": {i: {90: {"x": v}} for i, v in enumerate([0.0, 1.0, 2.0, 3.0])},
          "nemo": {i: {90: {"x": v}} for i, v in enumerate([5.0, 6.0, 7.0, 8.0])}}
    fe, le, ne = two_sided_floor(eq, "x", 90)
    assert abs(le - ne) < 1e-12 and abs(fe - np.sqrt(2.0) * le) < 1e-12, (fe, le)
    # the 2x rule ACCEPTS inside the band and REJECTS outside it -- both arms
    assert verdict(1.9 * fl, fl) == "YES"
    assert verdict(-1.9 * fl, fl) == "YES"
    assert verdict(2.1 * fl, fl) == "no", "the verdict rule cannot fail -- vacuous"
    assert verdict(-2.1 * fl, fl) == "no"
    assert verdict(1.0, 0.0) == "NO-FLOOR"
    # the window mean is the mean over the pre-registered sample days, and its
    # sample days are the SAME set on both sides
    w = {"lego": {0: {d: {"x": float(d)} for d in WINDOW_DAYS}}}
    assert abs(window_mean(w, "lego", 0, "x") - float(np.mean(WINDOW_DAYS))) < 1e-12
    assert len(WINDOW_DAYS) == 9 and WINDOW_DAYS[0] == 280 and WINDOW_DAYS[-1] == 360
    assert set(HORIZONS) <= set(SCORE_DAYS) and set(WINDOW_DAYS) <= set(SCORE_DAYS)
    assert set(SCORE_DAYS) <= set(SNAP_GRID), "a scored day has no legoESM snapshot"
    assert KT_END == 17280, KT_END
    print("SELF-CHECK OK: RSS two-sided floor (and its sqrt(2) equal-wobble "
          "reduction), the 2x verdict rule in BOTH arms, window-mean sampling, "
          "and the scored-day/snapshot-day coverage.")
    return 0


# -------------------------------------------------------------------- report --
def controls(out_dir):
    print("=" * 118)
    print(f"VERDICT RUN -- {N_DAYS} DAYS, {N_MEM} members per side, from the "
          f"SAME NEMO day-180 restart")
    print("=" * 118)
    shas = {}
    for i in range(N_MEM):
        ok, why = lego_completed(lego_log(out_dir, i))
        print(f"[lego {member_name(i)}] {lego_npz(out_dir, i)}: "
              f"{'COMPLETE' if ok else 'INCOMPLETE'} ({why})")
        if not ok:
            raise SystemExit(f"legoESM {member_name(i)} did not complete: {why}. "
                             "A member that did not finish is a FINDING, not a "
                             "member to drop -- report it and stop.")
        shas[member_name(i)] = F.member_sha(lego_log(out_dir, i))
    if len(set(shas.values())) != 1 or None in shas.values():
        raise SystemExit(f"legoESM members sit at different source SHAs: {shas}")
    print(f"[control] all legoESM members at one HEAD: {set(shas.values()).pop()}")
    for i in range(N_MEM):
        d = nemo_dir(i)
        ok, why = nemo_completed(d, i)
        print(f"[NEMO {member_name(i)}] {d}: "
              f"{'COMPLETE' if ok else 'INCOMPLETE'} ({why})")
        if not ok:
            raise SystemExit(f"NEMO {member_name(i)} did not complete: {why}.")
        real = os.path.realpath(f"{d}/nemo")
        if real != os.path.realpath(NEMO_CERT):
            raise SystemExit(f"{member_name(i)} ran {real}, not {NEMO_CERT}")
    print(f"[control] all NEMO members ran the certified binary {NEMO_CERT}")
    G.instrument_self_checks(A.tmask)


def separation_control(rows):
    """The kick must reach every metric on BOTH sides.  An ensemble that is
    identical across members has a floor of exactly zero BY CONSTRUCTION, and
    every verdict built on it would be a division by the perturbation never
    having propagated."""
    for side in ("lego", "nemo"):
        for k in KEYS:
            for day in HORIZONS:
                vals = [rows[side][i][day][k] for i in range(N_MEM)]
                if len(set(vals)) < 2:
                    raise SystemExit(
                        f"{side} metric {k!r} at day {day} is IDENTICAL across "
                        f"ALL members ({vals}) -- the perturbation did not "
                        f"reach it, so its floor is zero by construction")
    print("[control] every metric separates at least two members, both sides, "
          "at every horizon: OK")


def spread_curves(rows):
    print("\n--- SPREAD(t): single-run ensemble spread (sample std, n=4) by day ---")
    print("    (this is the SATURATION curve: the 1e-14 kick is invisible early "
          "and saturates late;")
    print("     a floor still falling at day 360 means the year is not yet the "
          "climate floor)")
    for side in ("lego", "nemo"):
        print(f"\n  {side}")
        print(f"    {'metric':<38}" + "".join(f"{'d' + str(d):>12}" for d in CURVE_DAYS))
        for k in KEYS:
            cells = "".join(f"{spread_at(rows, side, k, d)[1]:>12.4e}"
                            for d in CURVE_DAYS)
            print(f"    {LABELS[k]:<38}{cells}")


def verdict_table(rows):
    print("\n" + "=" * 118)
    print("THE VERDICT TABLE -- gap = legoESM(member 0) - NEMO(member 0), "
          "matched day, matched reduction")
    print("floor = sqrt(lego_std^2 + NEMO_std^2) at that day; "
          "INDISTINGUISHABLE <=> |gap| <= 2 x floor")
    print("=" * 118)
    for day in HORIZONS:
        print(f"\n--- day {day} ---")
        print(f"{'metric':<38}{'lego m0':>14}{'NEMO m0':>14}{'gap':>13}"
              f"{'2-sided floor':>15}{'|gap|/floor':>13}{'INDIST':>9}")
        for k in KEYS:
            lv = rows["lego"][0][day][k]
            nv = rows["nemo"][0][day][k]
            gap = lv - nv
            fl, _, _ = two_sided_floor(rows, k, day)
            ratio = "inf" if fl == 0 else f"{abs(gap) / fl:.2f}"
            tag = verdict(gap, fl)
            if k in SIGN_MIXING:
                tag += "*"
            print(f"{LABELS[k]:<38}{lv:>14.6f}{nv:>14.6f}{gap:>+13.4e}"
                  f"{fl:>15.4e}{ratio:>13}{tag:>9}")
        print("  * sign-mixing reduction (three latitude groups summed into one "
              "number) -- reported, not a per-group verdict")


def window_table(rows):
    print("\n" + "=" * 118)
    print(f"FINAL-90-DAY WINDOW MEANS -- days {WINDOW_DAYS[0]}..{WINDOW_DAYS[-1]} "
          f"every 10 days, {len(WINDOW_DAYS)} samples, IDENTICAL days both sides")
    print("floor = RSS of the two sides' spread of their OWN window means "
          "(not the endpoint floor reused)")
    print("=" * 118)
    print(f"{'metric':<38}{'lego m0':>14}{'NEMO m0':>14}{'gap':>13}"
          f"{'2-sided floor':>15}{'|gap|/floor':>13}{'INDIST':>9}")
    for k in KEYS:
        lv = window_mean(rows, "lego", 0, k)
        nv = window_mean(rows, "nemo", 0, k)
        gap = lv - nv
        ls = F.spread([window_mean(rows, "lego", i, k) for i in range(N_MEM)])[1]
        ns = F.spread([window_mean(rows, "nemo", i, k) for i in range(N_MEM)])[1]
        fl = float(np.sqrt(ls ** 2 + ns ** 2))
        ratio = "inf" if fl == 0 else f"{abs(gap) / fl:.2f}"
        tag = verdict(gap, fl) + ("*" if k in SIGN_MIXING else "")
        print(f"{LABELS[k]:<38}{lv:>14.6f}{nv:>14.6f}{gap:>+13.4e}"
              f"{fl:>15.4e}{ratio:>13}{tag:>9}")


def growth_control(rows, out_dir):
    """Did the kick reach the integrated state?  Read as max|dT| between each
    perturbed member and its own side's control, on the SAME side."""
    print("\n--- GROWTH CONTROL: max|dT| of each perturbed member vs its own "
          "side's control [K] ---")
    print("    (legoESM snapshots are float32: quantum ~2e-06 K on a ~20 K "
          "field, so an early 0.0 is storage, not physics)")
    days = (10, 90, 180, 270, 360)
    from rebuild_nemo_restart import rebuild
    for side in ("lego", "nemo"):
        print(f"  {side}")
        print(f"    {'member':<14}" + "".join(f"{'day ' + str(d):>16}" for d in days))
        for i in range(1, N_MEM):
            cells = []
            for day in days:
                if side == "lego":
                    a = np.asarray(np.load(lego_npz(out_dir, i))[f"T3d_day{day}"],
                                   dtype=np.float64)
                    b = np.asarray(np.load(lego_npz(out_dir, 0))[f"T3d_day{day}"],
                                   dtype=np.float64)
                else:
                    a = np.asarray(rebuild(f"{nemo_dir(i)}/DINO_{kt_of(day):08d}"
                                           f"_restart_*.nc", ["tn"])["tn"],
                                   dtype=np.float64)
                    b = np.asarray(rebuild(f"{nemo_dir(0)}/DINO_{kt_of(day):08d}"
                                           f"_restart_*.nc", ["tn"])["tn"],
                                   dtype=np.float64)
                v = np.abs(a - b)
                if not np.isfinite(v).all():
                    raise SystemExit(f"non-finite T at day {day}, {side} "
                                     f"{member_name(i)} -- a blown member is a "
                                     f"finding")
                cells.append(f"{float(v.max()):>16.4e}")
            print(f"    {member_name(i):<14}" + "".join(cells))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=LEGO_DIR_DEFAULT)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--setup-nemo", action="store_true")
    p.add_argument("--run-nemo", action="store_true")
    p.add_argument("--run-lego", action="store_true")
    p.add_argument("--gpus", default="0,1")
    args = p.parse_args(argv)

    if args.self_check:
        return _self_check()
    if args.setup_nemo:
        setup_nemo()
        return 0
    if args.run_nemo:
        run_nemo()
        return 0
    if args.run_lego:
        run_lego(args.dir, args.gpus.split(","))
        return 0

    _self_check()
    print()
    controls(args.dir)
    print("\nscoring...", flush=True)
    rows = collect(args.dir)
    separation_control(rows)
    growth_control(rows, args.dir)
    spread_curves(rows)
    verdict_table(rows)
    window_table(rows)
    print("\nEvery number above is a measurement; the pre-registration "
          "(PREREG_verdict360.md) says which ones were predicted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
