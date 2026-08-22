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
import itertools
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

KEYS = G.KEYS + ("acc_mean", "band", "band_c", "g_south", "g_band", "g_north")
LABELS = dict(G.LABELS,
              acc_mean="full section [Sv] e3t_1d,MEAN",
              band="channel band [Sv] e3t_1d,median",
              band_c="channel band [Sv] e3t_0,mean",
              g_south="SOUTH of band transport [Sv]",
              g_band="channel band transport [Sv] (group)",
              g_north="NORTH of band transport [Sv]")
# The full section mixes the signs of the three groups, so it is reported and
# never given a per-group verdict of its own.
SIGN_MIXING = ("acc",)

# The PRE-REGISTERED rule is |gap| <= 2 x floor and is not adjustable.  But the
# floor is a variance ESTIMATE with 3 degrees of freedom per side, so the
# two-sided ~95% constant for an ESTIMATED floor is a Welch t (nu ~ 6) of about
# 2.45, not 2.0.  The consequence runs one way: ratios between 2.0 and 2.45 are
# printed as `no` -- DISTINGUISHABLE -- when they are not statistically
# resolved, which is the direction that manufactures a false fidelity failure.
# The registered rule still decides; this constant only marks the band where it
# is over-confident, and it is printed as its own column (both reviews).
K_PREREG = 2.0
K_WELCH = 2.45
# A metric whose measured spread is not comfortably above the fp32 storage
# quantum of the legoESM snapshots is measuring the npz dtype, not the physics.
QUANTUM_MARGIN = 10.0
# Registered here, BEFORE any day-360 gap exists (the NEMO members had not
# finished when this was committed): the ensemble is SATURATED at day 360 if
# spread(360)/spread(270) < 1.3, i.e. the spread has stopped growing over the
# last quarter of the year.  An UNSATURATED verdict is asymmetric and the table
# says so: a YES still stands (the runs did not separate even though they had
# room to), a `no` does not (the denominator is still growing).
SATURATION_RATIO_MAX = 1.3


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
    # The gate's ACC is a MEDIAN over longitudes; the three latitude groups are
    # MEANS.  A median is not additive, so the groups cannot be checked against
    # `acc` -- they can be checked against the MEAN-reduced full section, and
    # they are, right here.  Without this row the decomposition footnote in the
    # table is an assertion the output cannot support (physics review B6).
    m["acc_mean"] = _avg(D.section_total(u, A.umask))
    m["band"] = F.band_transport(u, A.umask)
    m["band_c"] = F.band_transport_campaign(u, A.umask)
    for key, (_name, rows) in zip(("g_south", "g_band", "g_north"), D.LAT_GROUPS):
        m[key] = _avg(D.group_transport(u, A.umask, rows))
    resid = abs(m["g_south"] + m["g_band"] + m["g_north"] - m["acc_mean"])
    if resid > 1e-9:
        raise SystemExit(f"the three latitude groups do not partition the "
                         f"full section: residual {resid:.3e} Sv")
    return m


# ------------------------------------------------------------------ loaders ---
def lego_state(npz, day):
    """The legoESM state, with the mask shortcut ASSERTED rather than assumed.

    The recorded gate intersects NEMO's reference ocean mask with the model's
    own land mask; this probe scores on the reference mask alone.  For these
    artifacts the two are bit-identical, so it is a no-op -- but a card with
    different land would then be silently scored on NEMO's ocean, which is the
    one asymmetry that would invalidate the whole comparison.  Checked, not
    assumed (both reviews).
    """
    st = G.load_candidate(npz, day)
    own = np.asarray(st["land_mask"], dtype=np.float64) > 0.5
    ref = np.asarray(A.tmask[:, :, 0]) if A.tmask.ndim == 3 else np.asarray(A.tmask)
    if own.shape == ref.shape and not np.array_equal(own, ref):
        raise SystemExit(
            f"{npz} day {day}: the candidate's own land mask disagrees with "
            f"NEMO's reference mask on {int((own != ref).sum())} cells -- "
            "scoring it on the reference mask would compare two different "
            "ocean domains")
    return st


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
    todo = []
    for i in range(N_MEM):
        if not os.path.exists(lego_npz(out_dir, i)):
            todo.append(i)
            continue
        # A reused artifact gets the same provenance gate the subprocess would
        # have applied.  Skipping on a bare file-exists test re-opens, from the
        # other side, the hole floor90_ensemble.run_member closed deliberately:
        # four members left over from an older revision agree with EACH OTHER
        # and sail through every downstream control (code review finding 3).
        want, got = F.head_sha(), F.member_sha(lego_log(out_dir, i))
        if got != want:
            raise SystemExit(
                f"{lego_npz(out_dir, i)} exists but its log records "
                f"HEAD={got}, not the current HEAD={want}. An ensemble whose "
                f"members sit at different source revisions measures the "
                f"revision as well as the noise. Delete it to regenerate.")
        print(f"[skip] {lego_npz(out_dir, i)} exists at the current HEAD",
              flush=True)
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
        # The pre-registration says "nothing existing is overwritten".  That is
        # true of OTHER runs' directories and was NOT true inside these four:
        # setup rewrote the namelist, the binary link and the input restart of
        # whatever was already there, and left any stale model output in place
        # for the scorer to read as current (code review finding 6).  RETRACTED
        # and now enforced: a directory carrying output refuses.
        stale = (glob.glob(f"{d}/DINO_*_restart_*.nc") + glob.glob(f"{d}/time.step")
                 + glob.glob(f"{d}/ocean.output"))
        if stale:
            raise SystemExit(
                f"{d} already carries {len(stale)} output file(s) from an "
                f"earlier run (e.g. {os.path.basename(stale[0])}). Scoring a "
                f"mix of old and new states is a silent confound -- clear the "
                f"directory deliberately, or point at a fresh one.")
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


def run_nemo(concurrency=1):
    """Launch the four members, `concurrency` at a time.

    SERIAL BY DEFAULT, and that is a measurement, not caution.  Launched
    4-wide (4 x 16 = 64 of this machine's 80 cores) the members ran at 14
    steps/min each -- 55 steps/min aggregate -- against 137 steps/min measured
    for a single 16-rank run of the same binary on the same machine.  Going
    4-wide therefore made the ENSEMBLE 2.5x slower, not 4x faster: the
    instrumented oracle binary writes per-step diagnostic dumps and 64 ranks
    contend on the filesystem rather than on the cores.  Serial reproduces the
    exact conditions the recorded 90-day twin was measured under, which is also
    the configuration nemo_continuation_control() checks against.
    """
    todo = list(range(N_MEM))
    while todo:
        batch, todo = todo[:concurrency], todo[concurrency:]
        procs = []
        for i in batch:
            d = nemo_dir(i)
            if not os.path.exists(f"{d}/nemo"):
                raise SystemExit(f"{d} not set up -- run --setup-nemo first")
            fh = open(f"{d}/run_verdict360_m{i}.log", "w")
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


def fp32_quantum(st, wet):
    """The metric change from round-tripping a state through float32.

    legoESM snapshots are stored float32 and NEMO restarts are read float64, so
    the two sides' spreads are measured at DIFFERENT precision.  Three of the
    five recorded 90-day legoESM floors sit at or below this quantum -- they are
    storage, not physics.  Measured on a real state rather than argued from
    machine epsilon, because the metrics are integrals whose conditioning is
    not the scalar eps (physics review B3, code review 8).
    """
    lo = {k: (np.asarray(v, dtype=np.float32).astype(np.float64)
              if k in ("T", "S", "u") else v) for k, v in st.items()}
    a, b = all_metrics(st, wet), all_metrics(lo, wet)
    return {k: abs(a[k] - b[k]) for k in KEYS}


def permutation_p(rows, key, day):
    """EXACT two-sided permutation test on the 4-vs-4 members, using ALL EIGHT
    runs instead of the two control members.

    The registered rule compares one legoESM run to one NEMO run and spends the
    other six only on the denominator.  This uses every run, assumes no
    distribution, and needs no floor constant: pool the eight values, enumerate
    all C(8,4)=70 splits, and count how often |mean(A)-mean(B)| reaches the
    observed separation.  The smallest attainable p is 2/70 = 0.029, so this
    can never be more significant than that -- stated because a floor of 0.029
    printed as `p = 0.029` reads like a result and is partly a sample-size
    artifact.  Reported ALONGSIDE the registered rule, never instead of it
    (physics review B1).
    """
    a = [rows["lego"][i][day][key] for i in range(N_MEM)]
    b = [rows["nemo"][i][day][key] for i in range(N_MEM)]
    pool = np.asarray(a + b, dtype=np.float64)
    obs = abs(np.mean(a) - np.mean(b))
    idx = range(len(pool))
    hits = tot = 0
    for c in itertools.combinations(idx, N_MEM):
        m = np.zeros(len(pool), dtype=bool)
        m[list(c)] = True
        stat = abs(pool[m].mean() - pool[~m].mean())
        tot += 1
        if stat >= obs - 1e-15:
            hits += 1
    return hits / tot, float(obs)


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
    # --- the permutation test, on cases whose exact p is known by counting ---
    # Perfectly separated 4-vs-4: only the observed split and its mirror reach
    # the observed statistic, so p is exactly 2/70 -- the design's FLOOR.
    sep = {"lego": {i: {90: {"x": float(v)}} for i, v in enumerate([1, 2, 3, 4])},
           "nemo": {i: {90: {"x": float(v)}} for i, v in enumerate([11, 12, 13, 14])}}
    pv, obs = permutation_p(sep, "x", 90)
    assert abs(pv - 2.0 / 70.0) < 1e-12, pv
    assert abs(obs - 10.0) < 1e-12, obs
    # Two IDENTICAL ensembles: the observed separation is 0, every split
    # reaches it, so p is exactly 1 -- the test cannot manufacture significance.
    ident = {s_: {i: {90: {"x": float(v)}} for i, v in enumerate([1, 2, 3, 4])}
             for s_ in ("lego", "nemo")}
    assert abs(permutation_p(ident, "x", 90)[0] - 1.0) < 1e-12
    # It must be able to sit STRICTLY between the two, or it is a two-valued
    # indicator dressed as a p-value.
    mid = {"lego": {i: {90: {"x": float(v)}} for i, v in enumerate([1, 2, 3, 9])},
           "nemo": {i: {90: {"x": float(v)}} for i, v in enumerate([2, 4, 5, 6])}}
    pm = permutation_p(mid, "x", 90)[0]
    assert 2.0 / 70.0 < pm < 1.0, pm
    # --- the honest-band constants, and their ORDER ---
    assert K_PREREG == 2.0 and K_WELCH > K_PREREG, (K_PREREG, K_WELCH)
    assert SATURATION_RATIO_MAX > 1.0
    assert QUANTUM_MARGIN >= 1.0
    print("SELF-CHECK OK: RSS two-sided floor (and its sqrt(2) equal-wobble "
          "reduction), the 2x verdict rule in BOTH arms, window-mean sampling, "
          "the scored-day/snapshot-day coverage, and the exact "
          "permutation test at its floor (2/70), at 1.0, and strictly "
          "between.")
    return 0


# -------------------------------------------------------------------- report --
def controls(out_dir):
    print("=" * 130)
    print(f"VERDICT RUN -- {N_DAYS} DAYS, {N_MEM} members per side, from the "
          f"SAME NEMO day-180 restart")
    print("=" * 130)
    shas, stamps = {}, {}
    for i in range(N_MEM):
        ok, why = lego_completed(lego_log(out_dir, i))
        print(f"[lego {member_name(i)}] {lego_npz(out_dir, i)}: "
              f"{'COMPLETE' if ok else 'INCOMPLETE'} ({why})")
        if not ok:
            raise SystemExit(f"legoESM {member_name(i)} did not complete: {why}. "
                             "A member that did not finish is a FINDING, not a "
                             "member to drop -- report it and stop.")
        shas[member_name(i)] = F.member_sha(lego_log(out_dir, i))
        d = np.load(lego_npz(out_dir, i))
        stamps[member_name(i)] = (str(d["nemo_ladder_mode"]),
                                  float(d["seasonal_t0_seconds"]),
                                  str(d["control_dtype"]))
    if len(set(shas.values())) != 1 or None in shas.values():
        raise SystemExit(f"legoESM members sit at different source SHAs: {shas}")
    print(f"[control] all legoESM members at one HEAD: {set(shas.values()).pop()}")
    # The launcher sets only CUDA_VISIBLE_DEVICES / JAX_ENABLE_X64 / the XLA
    # allocator; LEGOESM_NEMO_E3T and the seasonal clock are resolved INSIDE
    # each subprocess and were previously only PRINTED, never compared.  A
    # member that silently resolved a different vertical ladder would inflate
    # the ensemble spread -- i.e. the denominator -- and inflation buys a PASS.
    # This is the one failure mode the instrument could not see, and it failed
    # toward INDISTINGUISHABLE (physics review B4).
    if len(set(stamps.values())) != 1:
        raise SystemExit(
            "legoESM members do not share (vertical ladder, seasonal clock, "
            f"precision): {stamps}. A configuration difference between members "
            "inflates the floor the verdict divides by.")
    lad, t0, dt = stamps["m0_control"]
    print(f"[control] all legoESM members share ladder={lad!r} "
          f"seasonal_t0={t0:.0f}s dtype={dt}")
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
    nemo_continuation_control()


def nemo_continuation_control():
    """NEMO's unperturbed member must reproduce the EXISTING 90-day twin's
    day-90 state exactly.

    Same binary, same restart, same namelist bar `nn_itend` -- so the first 90
    days of this year-long run ARE the recorded 90-day twin, and if they are
    not, something in the setup moved that nobody asked to move.  This
    certifies the whole NEMO side directly instead of inferring it through the
    two-model gap, and it costs one comparison (physics review, missing
    control 8).  The DATA is compared, not the file bytes: netCDF metadata
    carries timestamps that differ between runs by construction.
    """
    print("\n[control] NEMO m0 vs the recorded 90-day twin, day 90 "
          "(same binary, same restart, only nn_itend differs)")
    a = nemo_state(nemo_dir(0), 90)
    b = nemo_state(G.RUN_90D_TWIN, 90)
    worst = 0.0
    for k in ("T", "S", "u"):
        v = np.abs(np.asarray(a[k]) - np.asarray(b[k]))
        v = v[np.isfinite(v)]
        worst = max(worst, float(v.max()) if v.size else 0.0)
        print(f"    max|d{k}| = {float(v.max()) if v.size else 0.0:.3e}")
    if worst != 0.0:
        raise SystemExit(
            f"NEMO m0's day-90 state is NOT bit-identical to the recorded "
            f"90-day twin (worst {worst:.3e}). The two runs differ only in "
            f"nn_itend, so a nonzero difference means something else moved -- "
            f"chase it before reading any day-360 number.")
    print("    BIT-IDENTICAL: this year-long run IS the recorded 90-day twin, "
          "continued.")


def p5_provenance_gate(rows):
    """P5, MECHANIZED.  The pre-registration says a day-90 miss is a provenance
    failure to be chased BEFORE any day-360 number is read; that ordering was
    left to a human reading a table row.  It is a gate now (code review 4).

    The two recorded numbers are named explicitly because two different day-90
    gaps circulate in this campaign: 0.4107 Sv is the PRE-drag-fix arm (the
    floor90 lane's control) and 0.3823 Sv is THIS branch's HEAD after the three
    bottom-drag fixes.  This gates the latter.
    """
    lego_d90 = 64.986934      # commit 59e6070a4, arm D = this branch's numerics
    nemo_d90 = 65.3692043752875   # acc_metric_reconciliation.NEMO_D90_ACC_SV
    tol = 1e-3
    lv, nv = rows["lego"][0][90]["acc"], rows["nemo"][0][90]["acc"]
    print(f"\n[P5 GATE] day-90 full-section ACC  lego {lv:.6f} (recorded "
          f"{lego_d90:.6f})  NEMO {nv:.6f} (recorded {nemo_d90:.6f})")
    print(f"[P5 GATE] gap {lv - nv:+.6f} Sv (recorded {lego_d90 - nemo_d90:+.6f})")
    bad = []
    if abs(lv - lego_d90) > tol:
        bad.append(f"legoESM off by {abs(lv - lego_d90):.3e}")
    if abs(nv - nemo_d90) > tol:
        bad.append(f"NEMO off by {abs(nv - nemo_d90):.3e}")
    if bad:
        raise SystemExit(
            f"P5 REFUTED ({'; '.join(bad)}, tolerance {tol}). This is a "
            f"provenance failure, not a physics result -- chase it before any "
            f"day-360 number is read.")
    print("[P5 GATE] CONFIRMED: member 0 reproduces the recorded day-90 state "
          "on both sides.")


def separation_control(rows, quantum):
    """The kick must reach EVERY metric on BOTH sides, on ALL FOUR members.

    The earlier spelling accepted any two of four members differing: three
    identical members and one outlier passed while making the std meaningless
    (both reviews).  It now requires four distinct values, and reports how many
    sit within the storage quantum of each other, because a spread built out of
    float32 ties is a dtype measurement.
    """
    for side in ("lego", "nemo"):
        for k in KEYS:
            for day in HORIZONS:
                vals = [rows[side][i][day][k] for i in range(N_MEM)]
                if len(set(vals)) < N_MEM:
                    raise SystemExit(
                        f"{side} metric {k!r} at day {day} does not separate "
                        f"all {N_MEM} members ({vals}) -- a spread built from "
                        f"tied members is not a floor")
    print(f"[control] every metric separates ALL {N_MEM} members, both sides, "
          f"at every horizon: OK")
    thin = [k for k in KEYS
            if spread_at(rows, "lego", k, N_DAYS)[1] < QUANTUM_MARGIN * quantum[k]]
    if thin:
        print(f"[control] WARNING: at day {N_DAYS} these metrics' legoESM "
              f"spread is under {QUANTUM_MARGIN:.0f}x the float32 storage "
              f"quantum, so their floor is partly dtype: {thin}")
    else:
        print(f"[control] every metric's day-{N_DAYS} legoESM spread clears "
              f"{QUANTUM_MARGIN:.0f}x its float32 storage quantum: OK")
    return set(thin)


def empirical_rule_controls(rows, quantum):
    """The registered rule, applied to two cases whose answer is KNOWN.

    Arithmetic self-checks prove the formula; they do not prove the rule is
    CALIBRATED on this data.  Two controls do, and both are free from members
    already on disk (physics review B2):

      POSITIVE -- legoESM m1 against legoESM m2 at day 360.  Same model, same
        card, same revision; the only difference is the perturbation seed.  The
        rule MUST return YES on every metric.  A `no` here means the floor is
        too small and every `no` in the verdict table is suspect.
      NEGATIVE -- legoESM m0 at day 360 against NEMO m0 at day 350.  A
        deliberate 10-day mismatch of two states that are otherwise the closest
        pair in the campaign.  The rule MUST return `no` on the transports.  A
        YES here means the floor is so wide the test cannot fail.
    """
    print("\n--- EMPIRICAL CONTROLS ON THE REGISTERED RULE (known answers) ---")
    fails = []
    print(f"  POSITIVE (lego m1 vs lego m2 at day {N_DAYS}; must be YES)")
    print(f"    {'metric':<38}{'gap':>13}{'floor':>14}{'ratio':>9}{'verdict':>9}")
    for k in KEYS:
        gap = rows["lego"][1][N_DAYS][k] - rows["lego"][2][N_DAYS][k]
        fl, _, _ = two_sided_floor(rows, k, N_DAYS)
        v = verdict(gap, fl)
        if v != "YES":
            fails.append(f"POSITIVE {k} -> {v}")
        print(f"    {LABELS[k]:<38}{gap:>+13.4e}{fl:>14.4e}"
              f"{(abs(gap) / fl if fl else float('inf')):>9.2f}{v:>9}")
    print(f"  NEGATIVE (lego m0 day {N_DAYS} vs NEMO m0 day {N_DAYS - 10}; "
          f"transports must be `no`)")
    print(f"    {'metric':<38}{'gap':>13}{'floor':>14}{'ratio':>9}{'verdict':>9}")
    for k in ("acc", "acc_mean", "band", "band_c", "g_south", "g_band", "g_north"):
        gap = rows["lego"][0][N_DAYS][k] - rows["nemo"][0][N_DAYS - 10][k]
        fl, _, _ = two_sided_floor(rows, k, N_DAYS)
        v = verdict(gap, fl)
        if v == "YES":
            fails.append(f"NEGATIVE {k} -> YES")
        print(f"    {LABELS[k]:<38}{gap:>+13.4e}{fl:>14.4e}"
              f"{(abs(gap) / fl if fl else float('inf')):>9.2f}{v:>9}")
    if fails:
        print(f"  RULE NOT CALIBRATED on this data: {fails}")
    else:
        print("  BOTH CONTROLS PASS: the rule says YES where the answer is "
              "known-same and `no` where it is known-different.")
    return fails


def saturation_table(rows):
    """Has the 1e-14 kick saturated by day 360, or is the floor still growing?

    Registered before any day-360 gap existed: SATURATED if
    spread(360)/spread(270) < 1.3.  The verdict is ASYMMETRIC under an
    unsaturated floor -- a YES still stands (the two runs did not separate even
    though the ensemble had room to grow), a `no` does not (the denominator is
    still growing, so the ratio is an upper bound that will fall).
    """
    print(f"\n--- SATURATION: spread(360)/spread(270), saturated if < "
          f"{SATURATION_RATIO_MAX} ---")
    print(f"{'metric':<38}{'lego ratio':>13}{'NEMO ratio':>13}{'saturated?':>12}")
    unsat = []
    for k in KEYS:
        r = []
        for side in ("lego", "nemo"):
            a = spread_at(rows, side, k, 270)[1]
            b = spread_at(rows, side, k, 360)[1]
            r.append(b / a if a > 0 else float("inf"))
        ok = max(r) < SATURATION_RATIO_MAX
        if not ok:
            unsat.append(k)
        print(f"{LABELS[k]:<38}{r[0]:>13.3f}{r[1]:>13.3f}"
              f"{('yes' if ok else 'NO'):>12}")
    return set(unsat)


def spread_curves(rows, quantum):
    print("\n--- SPREAD(t): single-run ensemble spread by day ---")
    print("    std (primary) on the first line of each metric, max-pairwise "
          "range on the second.")
    print("    At n=4 the range is ~2x the std BY CONSTRUCTION; they are never "
          "compared across.")
    print("    legoESM states are stored float32, NEMO states are read "
          "float64 -- the quantum column is")
    print("    the metric's own float32 round-trip, and a spread below ~10x it "
          "is dtype, not physics.")
    for side in ("lego", "nemo"):
        print(f"\n  {side}")
        print(f"    {'metric':<38}{'fp32 quantum':>14}"
              + "".join(f"{'d' + str(d):>12}" for d in CURVE_DAYS))
        for k in KEYS:
            for stat, tag in ((1, "std"), (0, "rng")):
                cells = "".join(f"{spread_at(rows, side, k, d)[stat]:>12.4e}"
                                for d in CURVE_DAYS)
                head = f"{LABELS[k]:<38}{quantum[k]:>14.2e}" if stat == 1 \
                    else f"{'  (range)':<38}{'':>14}"
                print(f"    {head}{cells}")


def _row(k, lv, nv, gap, rows, day, quantum, thin, unsat, use_window=False,
         wfl=None):
    fl_sd, ls_sd, ns_sd = (wfl if use_window
                           else two_sided_floor(rows, k, day, stat=1))
    fl_rg, ls_rg, ns_rg = (wfl if use_window
                           else two_sided_floor(rows, k, day, stat=0))
    ratio = float("inf") if fl_sd == 0 else abs(gap) / fl_sd
    tag = verdict(gap, fl_sd)
    if tag == "no" and ratio <= K_WELCH:
        tag = "unres"          # inside the honest Welch band: not resolved
    flags = ""
    if k in SIGN_MIXING:
        flags += "*"
    if k in thin:
        flags += "q"
    if k in unsat:
        flags += "u"
    p, _ = permutation_p(rows, k, day) if not use_window else (float("nan"), 0)
    return (f"{LABELS[k]:<38}{lv:>14.6f}{nv:>14.6f}{gap:>+13.4e}"
            f"{ls_sd:>12.3e}{ns_sd:>12.3e}{fl_sd:>13.3e}{fl_rg:>13.3e}"
            f"{ratio:>9.2f}{(f'{p:.3f}' if p == p else '   n/a'):>8}"
            f"{tag:>8}{flags:>4}")


def _header():
    return (f"{'metric':<38}{'lego m0':>14}{'NEMO m0':>14}{'gap':>13}"
            f"{'lego std':>12}{'NEMO std':>12}{'floor(std)':>13}"
            f"{'floor(rng)':>13}{'ratio':>9}{'perm p':>8}{'verd':>8}{'fl':>4}")


def _legend():
    print(f"  verdict: YES = |gap| <= {K_PREREG} x floor(std), the "
          f"PRE-REGISTERED rule.  `unres` = the registered rule says `no` but")
    print(f"           the ratio is under {K_WELCH}, the honest two-sided ~95% "
          f"constant for a floor ESTIMATED from 3 dof")
    print( "           per side -- not resolved, NOT a fidelity failure.  "
           "`no` = outside both.")
    print(f"  n={N_MEM}: ~41% relative standard error on every std, so every "
          f"floor is a factor-of-two estimate.")
    print( "  perm p: exact 4-vs-4 permutation test on all eight runs; floor "
           "2/70 = 0.029, so 0.029 is `as small as")
    print( "           this design can report`, not `p < 0.03`.  ~50 cells are "
           "printed and there is no multiplicity")
    print( "           correction; the metrics are strongly correlated, so the "
           "inflation is well below 50 independent tests.")
    print( "  flags: * sign-mixing reduction (median over longitudes; not "
           "additive over the latitude groups)")
    print( "         q measured spread under 10x the float32 storage quantum "
           "-- partly dtype")
    print(f"         u ensemble not saturated at day {N_DAYS} (spread still "
           f"growing): a YES stands, a `no` does not")


def verdict_table(rows, quantum, thin, unsat):
    print("\n" + "=" * 130)
    print("THE VERDICT TABLE -- gap = legoESM(member 0) - NEMO(member 0), "
          "matched day, matched reduction")
    print("floor = sqrt(lego_std^2 + NEMO_std^2) at that day")
    print("=" * 130)
    for day in HORIZONS:
        print(f"\n--- day {day} ---")
        print(_header())
        for k in KEYS:
            lv, nv = rows["lego"][0][day][k], rows["nemo"][0][day][k]
            print(_row(k, lv, nv, lv - nv, rows, day, quantum, thin, unsat))
    print()
    _legend()


def window_table(rows, quantum, thin, unsat):
    print("\n" + "=" * 130)
    print(f"FINAL-90-DAY WINDOW MEANS -- days {WINDOW_DAYS[0]}..{WINDOW_DAYS[-1]} "
          f"every 10 days, {len(WINDOW_DAYS)} samples, IDENTICAL days both sides")
    print("floor = RSS of the two sides' spread of their OWN window means "
          "(not the endpoint floor reused)")
    print("=" * 130)
    print(_header())
    for k in KEYS:
        lv = window_mean(rows, "lego", 0, k)
        nv = window_mean(rows, "nemo", 0, k)
        lsd, lrg = F.spread([window_mean(rows, "lego", i, k)
                             for i in range(N_MEM)])[1], \
            F.spread([window_mean(rows, "lego", i, k) for i in range(N_MEM)])[0]
        nsd, nrg = F.spread([window_mean(rows, "nemo", i, k)
                             for i in range(N_MEM)])[1], \
            F.spread([window_mean(rows, "nemo", i, k) for i in range(N_MEM)])[0]
        wfl_sd = (float(np.hypot(lsd, nsd)), lsd, nsd)
        gap = lv - nv
        ratio = float("inf") if wfl_sd[0] == 0 else abs(gap) / wfl_sd[0]
        tag = verdict(gap, wfl_sd[0])
        if tag == "no" and ratio <= K_WELCH:
            tag = "unres"
        flags = ("*" if k in SIGN_MIXING else "") + ("q" if k in thin else "") \
            + ("u" if k in unsat else "")
        print(f"{LABELS[k]:<38}{lv:>14.6f}{nv:>14.6f}{gap:>+13.4e}"
              f"{lsd:>12.3e}{nsd:>12.3e}{wfl_sd[0]:>13.3e}"
              f"{float(np.hypot(lrg, nrg)):>13.3e}{ratio:>9.2f}{'   n/a':>8}"
              f"{tag:>8}{flags:>4}")


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
    p.add_argument("--nemo-concurrency", type=int, default=1,
                   help="NEMO members to run at once (default 1: measured "
                        "fastest -- see run_nemo's docstring)")
    args = p.parse_args(argv)

    if args.self_check:
        return _self_check()
    if args.setup_nemo:
        setup_nemo()
        return 0
    if args.run_nemo:
        run_nemo(args.nemo_concurrency)
        return 0
    if args.run_lego:
        run_lego(args.dir, args.gpus.split(","))
        return 0

    _self_check()
    print()
    controls(args.dir)
    print("\nscoring...", flush=True)
    rows = collect(args.dir)
    p5_provenance_gate(rows)
    quantum = fp32_quantum(nemo_state(nemo_dir(0), N_DAYS), A.tmask)
    thin = separation_control(rows, quantum)
    unsat = saturation_table(rows)
    growth_control(rows, args.dir)
    spread_curves(rows, quantum)
    empirical_rule_controls(rows, quantum)
    verdict_table(rows, quantum, thin, unsat)
    window_table(rows, quantum, thin, unsat)
    print("\nEvery number above is a measurement; the pre-registration "
          "(PREREG_verdict360.md) says which ones were predicted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
