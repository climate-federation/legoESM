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
every latitude row into one number and reduces with a MEDIAN over longitudes,
so it is not additive over the three latitude groups (``acc_mean`` is, and the
partition is checked at score time).  The three groups do carry opposite-signed
gaps -- but MEASURED at day 90 they are south -0.4405, channel +0.0632, north
-0.0241 Sv against a full-section gap of -0.4107, i.e. the full-section number
is DOMINATED BY ONE GROUP, lightly offset.  An earlier revision of this
docstring, and the pre-registration, said "two large cancelling errors"; that
is RETRACTED as overstated on this data.  The reason to score per-group stands
either way: which group owns the gap is the actionable fact, and a single
summed number cannot say.  The gate metric is still printed -- it is the
recorded headline and dropping it would hide a regression.

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
import json
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
import acc_metric_reconciliation as R   # noqa: E402  NEMO_D90_ACC_SV, imported not pasted
import floor90_ensemble as F           # noqa: E402  spread(), band reductions
import kamm_twin_90d as _twin         # noqa: E402  snapshot_storage_dtypes
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
# The full section is a MEDIAN over longitudes and therefore NOT additive over
# the three latitude groups, so it is reported and never given a per-group
# verdict of its own.  (It also mixes their signs -- but measured, it is
# dominated by the southern basin rather than built from cancelling terms; see
# the module docstring's retraction.)
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
# N3: ONE quarter under the ratio is not saturation.  Measured false positive on
# the existing 90-day NEMO ensemble: NEMO's ACC spread FELL from day 30 to 60
# (ratio 0.52, "saturated") and then grew ~300x -- the Asselin filter damping the
# leapfrog computational mode before physical error growth takes over.  Two
# CONSECUTIVE quarters are required, plus a dispersion check (below).
SATURATION_QUARTERS = ((180, 270), (270, 360))
# The two sides' spreads must be within ~1 order of magnitude for the RSS to be
# a genuinely TWO-sided floor.  Measured at day 90 on the existing ensembles,
# lego/NEMO runs 9.3x to 16596x, so the RSS is in practice LEGO'S OWN DISPERSION
# and the "sqrt(2) x one side" reading of it is empirically false.  Flagged per
# metric rather than argued.
ONE_SIDED_DECADES = 1.0


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
    if own.shape != ref.shape:
        raise SystemExit(
            f"{npz} day {day}: the candidate's land mask is {own.shape} and "
            f"NEMO's reference mask is {ref.shape}. A shape mismatch used to "
            f"SKIP this check, which is the one case where it matters most.")
    if not np.array_equal(own, ref):
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


def launch_sha(out_dir):
    """The revision the ENSEMBLE was launched at, recorded in the output dir.

    Gating a partially-complete ensemble against the LIVE HEAD makes it
    unresumable the moment anything else is committed -- and this campaign
    committed twice while its members were integrating.  The ensemble's own
    revision is the invariant that matters: every member must match EACH OTHER
    and the recorded launch, not whatever HEAD happens to be now (round-2
    review).
    """
    path = f"{out_dir}/.launch_sha"
    if os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    sha = F.head_sha()
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(sha + "\n")
    return sha


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
        want, got = launch_sha(out_dir), F.member_sha(lego_log(out_dir, i))
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

    SERIAL BY DEFAULT, and that is a measurement, not caution.  Both rates
    measured on this machine, on this binary, on this case:

        4-wide (4 x 16 = 64 of 80 cores)      14 steps/min PER MEMBER
                                              (55 aggregate), ETA 13.7 h
        serial (16 ranks, nothing else)     1343 steps/min, 8.6 min per
                                              member, ~35 min for all four

    Going 4-wide made the ENSEMBLE ~24x SLOWER in aggregate, not 4x faster: the
    instrumented oracle writes per-step diagnostic dumps and 64 ranks contend on
    the filesystem rather than on the cores.  (An earlier revision of this
    docstring quoted 137 steps/min as the single-run baseline, taken from file
    mtimes of an older ensemble that had other work on the machine; it is ~10x
    low and is corrected here.)  Serial also reproduces the exact conditions the
    recorded 90-day twin was measured under -- the configuration
    nemo_continuation_control() checks against.
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


def fp32_quantum(st, wet, fp32_fields=("T", "S", "u")):
    """The metric change from round-tripping a state through float32.

    legoESM snapshots were stored float32 and NEMO restarts are read float64,
    so the two sides' spreads were measured at DIFFERENT precision.  Three of
    the five recorded 90-day legoESM floors sit at or below this quantum --
    they are storage, not physics.  Measured on a real state rather than
    argued from machine epsilon, because the metrics are integrals whose
    conditioning is not the scalar eps (physics review B3, code review 8).

    ``fp32_fields`` is the set of 3-D fields the CANDIDATE artifacts actually
    stored in float32, read off their per-field ``storage_dtypes`` stamp (see
    :func:`storage_fp32_fields`).  It defaults to all three, so an artifact
    written before ``kamm_twin_90d --fp64-3d`` existed is scored exactly as
    before; an fp64-stored field contributes no quantum, because there is none
    to contribute, and passing an empty set gives an all-zero quantum -- no
    metric is then downgraded for a storage limit the artifact does not have.
    """
    fp32_fields = tuple(fp32_fields)
    if not fp32_fields:
        return {k: 0.0 for k in KEYS}
    lo = {k: (np.asarray(v, dtype=np.float32).astype(np.float64)
              if k in fp32_fields else v) for k, v in st.items()}
    a, b = all_metrics(st, wet), all_metrics(lo, wet)
    return {k: abs(a[k] - b[k]) for k in KEYS}


def storage_fp32_fields(npzs):
    """Which of T/S/u the legoESM members actually stored in float32.

    EXTEND-ONLY.  An artifact with no ``storage_dtypes`` stamp predates the
    per-field stamp and is treated as all-float32, which is what it was, so no
    recorded artifact's score moves.  The members must AGREE: a mixed-precision
    ensemble would have two different storage quanta under one floor, and the
    floor is the denominator every verdict divides by.

    ONLY the three scored 3-D fields are compared, never the whole stamp map.
    Comparing the map falsely aborted a legitimate ensemble two ways (code
    review): a recorded member resolves to a nine-entry legacy map while a new
    single-precision member writes a ten-entry one, and two brand-new members
    disagree if one member's reduction succeeded and the other's did not --
    neither of which is a difference in storage precision.
    """
    seen = set()
    for path in npzs:
        d = np.load(path, allow_pickle=False)
        st = _twin.snapshot_storage_dtypes(d)
        # the gate reduces T3d/S3d/u3d; eta/v never enter a scored metric
        # SAFE DIRECTION: a field counts as float32 unless the stamp says
        # exactly "float64".  An unstamped artifact, and a stamp that says
        # "absent" (a run that wrote no 3-D block at all), both keep today's
        # quantum rather than being granted a waiver -- a wrongly-granted
        # waiver removes a downgrade that exists, which is the failure that
        # manufactures a false measurement.
        seen.add(tuple(f for f, k in (("T", "T3d"), ("S", "S3d"), ("u", "u3d"))
                       if st.get(k, "float32") != "float64"))
    if len(seen) != 1:
        raise SystemExit(
            f"legoESM members stored their 3-D snapshots at different "
            f"precisions {sorted(seen)} -- one floor cannot span two storage "
            f"quanta")
    return seen.pop()


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
        if stat >= obs * (1.0 - 1e-12) - 1e-300:
            hits += 1
    return hits / tot, float(obs)


def verdict(gap, floor):
    """INDISTINGUISHABLE iff |gap| is within 2x the two-sided floor."""
    if floor == 0.0:
        return "NO-FLOOR"
    return "YES" if abs(gap) <= K_PREREG * floor else "no"


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
        missing = [q for q in ("nemo_ladder_mode", "seasonal_t0_seconds",
                               "control_dtype") if q not in d.files]
        if missing:
            raise SystemExit(
                f"{lego_npz(out_dir, i)} carries no {missing} stamp -- it "
                f"predates the provenance stamps and does not record the "
                f"configuration it ran; re-run the member.")
        stamps[member_name(i)] = (str(d["nemo_ladder_mode"]),
                                  float(d["seasonal_t0_seconds"]),
                                  str(d["control_dtype"]),
                                  json.dumps(_twin.snapshot_storage_dtypes(d),
                                             sort_keys=True))
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
    lad, t0, dt, sto = stamps["m0_control"]
    print(f"[control] all legoESM members share ladder={lad!r} "
          f"seasonal_t0={t0:.0f}s build-dtype={dt} storage={sto}")
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
    """P5, MECHANIZED, on the quantity the pre-registration actually registered.

    Round-2 review: the first spelling of this gate tested the two PER-SIDE
    ACC values within 1e-3.  The registered criterion is that the GAP
    reproduces -0.382 Sv within 1e-3 -- a different test in both directions
    (two absolutes can each drift 9e-4 the same way and leave the gap perfect,
    or drift opposite ways and blow a gap the per-side test passes).  Silently
    re-specifying a registered gate after launch is exactly what the
    pre-registration exists to prevent, so the gap is the gate and the two
    absolutes are printed as diagnostics beside it.

    Two day-90 gaps circulate in this campaign and they are named rather than
    assumed: 0.4107 Sv is the PRE-drag-fix arm (the floor90 lane's control) and
    0.3823 Sv is THIS branch's HEAD after the three bottom-drag fixes.  The
    NEMO side is IMPORTED from the recorded reconciliation harness, not pasted.
    """
    lego_d90 = 64.986934              # commit 59e6070a4, arm D = this branch
    nemo_d90 = R.NEMO_D90_ACC_SV      # imported, not pasted
    registered_gap = lego_d90 - nemo_d90
    tol = 1e-3
    lv, nv = rows["lego"][0][90]["acc"], rows["nemo"][0][90]["acc"]
    gap = lv - nv
    print(f"\n[P5 GATE] day-90 full-section ACC   lego {lv:.6f} (recorded "
          f"{lego_d90:.6f})   NEMO {nv:.6f} (recorded {nemo_d90:.6f})")
    print(f"[P5 GATE] REGISTERED QUANTITY = the GAP: {gap:+.6f} Sv against "
          f"{registered_gap:+.6f} Sv, tolerance {tol}")
    if abs(gap - registered_gap) > tol:
        raise SystemExit(
            f"P5 REFUTED: day-90 gap {gap:+.6f} Sv vs the recorded "
            f"{registered_gap:+.6f} Sv, off by {abs(gap - registered_gap):.3e} "
            f"> {tol}. This is a PROVENANCE failure, not a physics result -- "
            f"chase it before any day-360 number is read.")
    print(f"[P5 GATE] CONFIRMED: the day-90 gap reproduces the recorded value "
          f"to {abs(gap - registered_gap):.1e} Sv.")


def tie_report(rows, side, key, day):
    """(n distinct members, n members) for one metric/side/day."""
    vals = [rows[side][i][day][key] for i in range(N_MEM)]
    return len(set(vals)), N_MEM


def separation_control(rows, quantum_by_day):
    """The kick must reach every metric -- but a TIE IS A FLAG, NOT AN ABORT
    before the final horizon.

    Round-1 review asked for all four members to separate; I tightened to that
    everywhere and it would have aborted the entire report after eight
    integrations.  Measured on the four finished legoESM members: at day 90 the
    southern-band sigma MAX has 2 of 4 distinct values and the deep contrast 3
    of 4.  That is not a broken ensemble -- sigma MAX is a single-cell maximum
    of a float32-stored field, so early members tie at the STORAGE quantum
    while the trajectories differ.  The floor lane had already documented the
    relaxed form; re-tightening it was my error in both directions.

    So: at the final horizon a metric must separate all four members (there the
    spread is the published floor and a tie means a zero denominator); at
    earlier horizons two of four suffice and the tie count is PRINTED, per
    horizon, so a reader sees which numbers are dtype-limited.
    """
    fp64_stored = not any(quantum_by_day[d][k]
                          for d in HORIZONS for k in KEYS)
    print("\n--- SEPARATION / TIES (distinct members out of "
          f"{N_MEM}; " + ("ties are NOT a storage quantum here -- the "
                          "snapshots are float64"
                          if fp64_stored else
                          "ties early are the float32 storage quantum, not a "
                          "defect") + ") ---")
    if fp64_stored:
        print("[control] the legoESM members stored their 3-D snapshots at "
              "float64, so there is no storage quantum to downgrade a metric "
              "for; every `q` flag below is off by measurement, not by "
              "assumption, and an early tie must separate like any other")
    print(f"{'metric':<38}" + "".join(f"{'d' + str(d) + ' L/N':>12}" for d in HORIZONS))
    for k in KEYS:
        cells = []
        for day in HORIZONS:
            dl = tie_report(rows, "lego", k, day)[0]
            dn = tie_report(rows, "nemo", k, day)[0]
            cells.append(f"{str(dl) + '/' + str(dn):>12}")
            # The relaxed early-horizon rule exists ONLY because a float32
            # max-type metric can tie at the storage quantum while the
            # trajectories differ.  Remove the quantum and that excuse is
            # gone, so every horizon requires full separation (physics
            # review).
            need = N_MEM if (day == N_DAYS or fp64_stored) else 2
            for side, dd in (("lego", dl), ("nemo", dn)):
                if dd < need:
                    raise SystemExit(
                        f"{side} metric {k!r} at day {day} has only {dd} "
                        f"distinct member values (need {need} at this "
                        f"horizon) -- a spread built from tied members is not "
                        f"a floor")
        print(f"{LABELS[k]:<38}" + "".join(cells))
    print(f"[control] every metric separates >= 2 members early and all "
          f"{N_MEM} at day {N_DAYS}, both sides: OK")
    thin = {}
    for day in HORIZONS:
        thin[day] = {k for k in KEYS
                     if spread_at(rows, "lego", k, day)[1]
                     < QUANTUM_MARGIN * quantum_by_day[day][k]}
        if thin[day]:
            print(f"[control] day {day}: legoESM spread under "
                  f"{QUANTUM_MARGIN:.0f}x the float32 storage quantum (partly "
                  f"dtype) for {sorted(thin[day])}")
    return thin


def one_sided_flags(rows, day):
    """Which metrics' RSS floor is really only ONE side's dispersion.

    The pre-registration justified the RSS as "sqrt(2) x one side when the two
    sides wobble equally".  They do not.  Measured at day 90 on the existing
    ensembles the legoESM/NEMO spread ratio runs 9.3x to 16596x across the ten
    metrics, so the RSS is numerically legoESM's own dispersion and the sqrt(2)
    reading is empirically false.  That is not a defect in the arithmetic -- the
    RSS is still the right combination -- but a reader must not be allowed to
    believe the floor is a joint property of the two models when it is one
    model's.  Flagged `1` per metric, per horizon.
    """
    out = set()
    for k in KEYS:
        ls = spread_at(rows, "lego", k, day)[1]
        ns = spread_at(rows, "nemo", k, day)[1]
        if ns <= 0 or ls <= 0:
            out.add(k)
            continue
        if abs(np.log10(ls / ns)) > ONE_SIDED_DECADES:
            out.add(k)
    return out


def saturation_table(rows):
    """Has the kick saturated by day 360?  TWO consecutive quarters, not one.

    Registered as spread(360)/spread(270) < 1.3 before any day-360 gap existed.
    Round-2 review then produced a measured FALSE POSITIVE for that single-
    quarter form on the existing 90-day NEMO ensemble: NEMO's ACC spread FELL
    from day 30 to day 60 (ratio 0.52, which the single test calls "saturated")
    and then grew by ~300x -- the Asselin filter damping the leapfrog
    computational mode before physical error growth takes over.  A transient
    plateau is not saturation.  The criterion is therefore TIGHTENED (never
    loosened) to both 180->270 AND 270->360 under 1.3.  Tightening a registered
    criterion after launch can only make a claim harder to make, which is the
    safe direction; it is recorded here rather than quietly applied.
    """
    print(f"\n--- SATURATION: consecutive quarter ratios, saturated only if "
          f"BOTH < {SATURATION_RATIO_MAX} ---")
    hdr = "".join(f"{'L ' + str(a) + '->' + str(b):>13}{'N ' + str(a) + '->' + str(b):>13}"
                  for a, b in SATURATION_QUARTERS)
    print(f"{'metric':<38}{hdr}{'saturated?':>12}")
    unsat = set()
    for k in KEYS:
        cells, ok = [], True
        for a, b in SATURATION_QUARTERS:
            for side in ("lego", "nemo"):
                sa = spread_at(rows, side, k, a)[1]
                sb = spread_at(rows, side, k, b)[1]
                r = sb / sa if sa > 0 else float("inf")
                cells.append(r)
                if not (r < SATURATION_RATIO_MAX):
                    ok = False
        if not ok:
            unsat.add(k)
        print(f"{LABELS[k]:<38}"
              + "".join(f"{c:>13.3f}" for c in (cells[0], cells[1], cells[2], cells[3]))
              + f"{('yes' if ok else 'NO'):>12}")
    return unsat


# How many more quarters of growth we are willing to credit an unsaturated
# floor with, when asking whether "not saturated" could ever overturn a `no`.
# Four quarters = one more year at the OBSERVED last-quarter rate.
UNSAT_CREDIT_QUARTERS = 4


def unsaturated_materiality(rows, unsat, day):
    """Which `u` flags could ACTUALLY overturn a `no`, and which are decoration.

    "Not saturated" was voiding every `no` in the table, because 10 of the 11
    metrics fail the two-quarter test on the legoESM side.  That reading voids
    the run's own headline for free, and it is not honest: a gap sitting 15x
    its floor needs the floor to grow ~8x before it becomes a YES, and a floor
    whose LAST-QUARTER growth is ~1.0x is not going to do that.  So the flag is
    attached only where the arithmetic permits it to matter:

        x2YES   = ratio / K_PREREG        (the growth the floor needs)
        g_lastQ = spread(360)/spread(270) (the growth actually observed)

    `u` is MATERIAL iff the metric is unsaturated AND g_lastQ > 1 AND
    x2YES <= g_lastQ ** UNSAT_CREDIT_QUARTERS -- i.e. another year at the rate
    the last quarter actually showed could close it.  Everything else keeps the
    saturation FACT in the printed table below but does not get to void its
    verdict.
    """
    material, table = set(), []
    for k in KEYS:
        lv, nv = rows["lego"][0][day][k], rows["nemo"][0][day][k]
        fl, ls, ns = two_sided_floor(rows, k, day)
        ratio = float("inf") if fl == 0 else abs(lv - nv) / fl
        x2yes = ratio / K_PREREG
        g = []
        for side in ("lego", "nemo"):
            a = spread_at(rows, side, k, 270)[1]
            b = spread_at(rows, side, k, day)[1]
            g.append(b / a if a > 0 else float("inf"))
        gl = max(g)
        can = (k in unsat) and gl > 1.0 and x2yes <= gl ** UNSAT_CREDIT_QUARTERS
        if can:
            material.add(k)
        share = (ls ** 2 / fl ** 2) if fl > 0 else float("nan")
        table.append((k, ratio, x2yes, gl, can, share))
    return material, table


def unsaturated_table(table, day):
    print(f"\n--- DOES `u` MATTER? day {day}: the growth an unsaturated floor "
          f"would need, against the growth it shows ---")
    print(f"{'metric':<38}{'ratio':>10}{'x2YES':>10}{'g_lastQ':>10}"
          f"{'u material?':>13}{'lego share of floor':>21}")
    for k, ratio, x2yes, gl, can, share in table:
        print(f"{LABELS[k]:<38}{ratio:>10.2f}{x2yes:>10.2f}{gl:>10.2f}"
              f"{('YES' if can else 'no'):>13}{share:>20.1%}")
    print(f"  x2YES = ratio / {K_PREREG} (growth the floor needs for a `no` to "
          f"become YES); g_lastQ = spread({day})/spread(270).")
    print(f"  `u` is attached to a verdict only when another "
          f"{UNSAT_CREDIT_QUARTERS} quarters at g_lastQ could close x2YES.")
    print( "  lego share of floor = lego_std^2 / floor^2: how much of the "
           "two-sided floor is legoESM's own dispersion.")


def spread_curves(rows, quantum, fp32_fields=("T", "S", "u")):
    print("\n--- SPREAD(t): single-run ensemble spread by day ---")
    print("    std (primary) on the first line of each metric, max-pairwise "
          "range on the second.")
    print("    At n=4 the range is ~2x the std BY CONSTRUCTION; they are never "
          "compared across.")
    print(f"    legoESM 3-D fields stored float32: "
          f"{list(fp32_fields) or 'none -- float64 snapshots'}; NEMO states "
          f"are read float64 -- the quantum column is")
    print(f"    the metric's own float32 round-trip MEASURED AT DAY {N_DAYS} "
          f"-- ONE number labelling all")
    print("    columns, shown for scale only; the per-horizon quantum used by "
          "the `q` flag is measured at")
    print("    each scored horizon separately.")
    for side in ("lego", "nemo"):
        print(f"\n  {side}")
        print(f"    {'metric':<38}{'quantum@' + str(N_DAYS):>14}"
              + "".join(f"{'d' + str(d):>12}" for d in CURVE_DAYS))
        for k in KEYS:
            for stat in (1, 0):
                cells = "".join(f"{spread_at(rows, side, k, d)[stat]:>12.4e}"
                                for d in CURVE_DAYS)
                head = (f"{LABELS[k]:<38}{quantum[k]:>14.2e}" if stat == 1
                        else f"{'  (range)':<38}{'':>14}")
                print(f"    {head}{cells}")


def _label(gap, floor):
    """The registered rule, plus the one band where it is over-confident.

    Spelled ONCE, here, and unit-tested -- it was inline in two printers and
    tested in neither (round-2 review).
    """
    v = verdict(gap, floor)
    if v == "no" and floor > 0 and abs(gap) <= K_WELCH * floor:
        return "unres"
    return v


def positive_control_null(k):
    """P(|gap|/floor <= k) for the leave-two-out positive control, EXACTLY.

    Derived rather than assumed, because the leave-two-out fix CHANGED the null
    and the control was briefly printing a fraction against no expectation at
    all.  Within one side, gap = x_i - x_j with x ~ N(0, sigma^2), so
    gap = sqrt(2) sigma Z.  The denominator is the std of the two members NOT
    in the pair, which at n=2 is |x_k - x_l| / sqrt(2) = sigma |Z'|.  When the
    other side's spread is negligible -- which is this system, measured 9x to
    16600x tighter -- the RSS adds nothing, so

        ratio = sqrt(2) |Z / Z'|,  a folded Cauchy scaled by sqrt(2)
        P(ratio <= k) = (2/pi) arctan(k / sqrt(2))
        median ratio  = sqrt(2) = 1.41421...

    So ~61% inside the registered 2.0x band is the HEALTHY value here, not a
    failure; the MEDIAN is the stable statistic and the fraction is noisy at
    n=12 pairs.  If the two sides' spreads were comparable instead, the
    denominator gains a second independent term and the median falls to ~1.16.
    """
    return float(2.0 / np.pi * np.arctan(k / np.sqrt(2.0)))


def empirical_rule_controls(rows):
    """The registered rule applied to cases whose answer is KNOWN.

    POSITIVE, PER SIDE -- and per side is not cosmetic.  The floor is the
    two-sided RSS, so a pair drawn from the legoESM side is divided by a floor
    legoESM dominates (the other side is 9x-16600x tighter and adds nothing),
    while a pair drawn from the NEMO side is divided by a floor that is almost
    entirely LEGOESM'S spread.  Those are two different nulls, and pooling them
    reports a median that belongs to neither: the NEMO-side ratios are tiny BY
    CONSTRUCTION and drag the pooled statistic down.  Only the legoESM side is
    compared against the analytic sqrt(2)|Z/Z'| null; the NEMO side is printed
    for completeness with its ratios explicitly labelled uninformative.

    NEGATIVE, AT TWO LAGS.  A single lag can have no power for a metric that
    happens to return near its own value over exactly that interval, which is
    not the same thing as a floor that is too wide.  Measured here: the
    southern-basin transport differs by only -0.012 Sv between day 330 and day
    360 because it is oscillating on that timescale, so the 30-day lag cannot
    reject it however good the rule is.  A 90-day lag is run alongside, and a
    metric is only counted as a control FAILURE if it passes at BOTH lags.
    """
    print("\n--- EMPIRICAL CONTROLS ON THE REGISTERED RULE (known answers) ---")
    print(f"  POSITIVE: within-side member pairs at day {N_DAYS}, floor from "
          f"the two members NOT in the pair")
    print( "            (a pair scored against a floor it is part of is capped "
           "at sqrt(6)=2.449 and cannot fail)")
    stats = {}
    for side in ("lego", "nemo"):
        other = "nemo" if side == "lego" else "lego"
        ratios, npass, ntot = [], 0, 0
        for a, b in itertools.combinations(range(N_MEM), 2):
            rest = [m for m in range(N_MEM) if m not in (a, b)]
            for k in KEYS:
                gap = rows[side][a][N_DAYS][k] - rows[side][b][N_DAYS][k]
                s_rest = F.spread([rows[side][m][N_DAYS][k] for m in rest])[1]
                s_oth = F.spread([rows[other][m][N_DAYS][k]
                                  for m in range(N_MEM)])[1]
                fl = float(np.hypot(s_rest, s_oth))
                ntot += 1
                if fl > 0:
                    ratios.append(abs(gap) / fl)
                if _label(gap, fl) in ("YES", "unres"):
                    npass += 1
        stats[side] = (float(np.median(ratios)) if ratios else float("nan"),
                       npass / ntot if ntot else float("nan"), ntot)
    med_l, frac_l, n_l = stats["lego"]
    med_n, frac_n, n_n = stats["nemo"]
    print(f"            legoESM side (the informative one -- its floor is its "
          f"own spread):")
    print(f"              HEADLINE median ratio {med_l:.3f}   PREDICTED "
          f"{np.sqrt(2.0):.3f} (analytic sqrt(2)|Z/Z'| null)")
    print(f"              inside the band {100 * frac_l:.1f}% of {n_l}   "
          f"PREDICTED {100 * positive_control_null(K_WELCH):.1f}% "
          f"(inside {K_PREREG}x alone: {100 * positive_control_null(K_PREREG):.1f}%)")
    print(f"            NEMO side: median ratio {med_n:.3f}, "
          f"{100 * frac_n:.1f}% of {n_n} inside -- UNINFORMATIVE BY "
          f"CONSTRUCTION:")
    print( "              its numerator is NEMO's tiny member spread and its "
           "denominator is dominated by legoESM's,")
    print( "              so these ratios are near zero however good or bad "
           "the rule is.  Not compared to the null.")
    print(f"  NEGATIVE: legoESM day {N_DAYS} against its OWN earlier state, at "
          f"TWO lags (must be `no`)")
    print(f"    {'metric':<38}{'lag30 gap':>13}{'r30':>8}{'v30':>7}"
          f"{'lag90 gap':>13}{'r90':>8}{'v90':>7}{'power?':>9}")
    neg_fail = []
    for k in ("acc", "acc_mean", "band", "band_c", "g_south", "g_band", "g_north"):
        fl, _, _ = two_sided_floor(rows, k, N_DAYS)
        cells, labs = [], []
        for lag in (30, 90):
            gap = rows["lego"][0][N_DAYS][k] - rows["lego"][0][N_DAYS - lag][k]
            r = abs(gap) / fl if fl else float("inf")
            lab = _label(gap, fl)
            cells.append((gap, r, lab))
            labs.append(lab)
        if all(l != "no" for l in labs):
            neg_fail.append(k)
        power = "BOTH" if all(l == "no" for l in labs) else (
            "one lag" if any(l == "no" for l in labs) else "NEITHER")
        print(f"    {LABELS[k]:<38}{cells[0][0]:>+13.4e}{cells[0][1]:>8.2f}"
              f"{cells[0][2]:>7}{cells[1][0]:>+13.4e}{cells[1][1]:>8.2f}"
              f"{cells[1][2]:>7}{power:>9}")
    if neg_fail:
        print(f"    CONTROL FAILS for {neg_fail}: the rule accepts a "
              f"known-different state at BOTH lags for these metrics")
    else:
        print( "    every transport is rejected at at least one lag; a metric "
               "passing at one lag only is that lag")
        print( "    having no power for it (the quantity returns near its own "
               "value over that interval), not a wide floor")
    print(f"  CALIBRATION: legoESM-side positive median {med_l:.3f} vs "
          f"predicted {np.sqrt(2.0):.3f}; negative control "
          f"{'FAILS for ' + str(neg_fail) if neg_fail else 'rejects every transport'}")
    return frac_l, (not neg_fail), med_l


def _header():
    return (f"{'metric':<38}{'lego m0':>14}{'NEMO m0':>14}{'gap':>13}"
            f"{'lego std':>12}{'NEMO std':>12}{'floor(std)':>13}"
            f"{'floor(rng)':>13}{'ratio':>9}{'perm p':>8}{'verd':>8}{'fl':>5}")


def _flags(k, thin_day, unsat, onesided):
    return (("*" if k in SIGN_MIXING else "")
            + ("q" if k in thin_day else "")
            + ("u" if k in unsat else "")
            + ("1" if k in onesided else ""))


def _legend():
    print(f"  verdict: YES = |gap| <= {K_PREREG} x floor(std), the "
          f"PRE-REGISTERED rule.  `unres` = the registered rule says `no` but")
    print(f"           the ratio is under {K_WELCH}, the honest two-sided ~95% "
          f"constant for a floor ESTIMATED from 3 dof")
    print( "           per side -- not resolved, NOT a fidelity failure.  "
           "`no` = outside both.")
    print(f"  n={N_MEM}: ~41% relative standard error on every std, so every "
          f"floor is a factor-of-two estimate.")
    print( "  perm p: exact 4-vs-4 permutation test on all eight runs.  Its "
           "FLOOR is 2/70 = 0.029, so 0.029 means")
    print( "           `as small as this design can report`, not `p < 0.03`.  "
           "It is a supporting column, never the")
    print( "           headline.  ~50 cells print with no multiplicity "
           "correction; the metrics are strongly correlated.")
    print( "  flags: * full-section reduction: a MEDIAN over longitudes, so it "
           "is NOT additive over the three latitude")
    print( "           groups (acc_mean is, and is checked).  At day 90 this "
           "gap is DOMINATED by the southern basin")
    print( "           rather than built from cancelling terms -- the "
           "pre-registration's `two large cancelling errors`")
    print( "           is RETRACTED.")
    print( "         q measured spread under 10x the float32 storage quantum "
           "AT THIS HORIZON -- partly dtype")
    print(f"         u ensemble not saturated (needs BOTH 180->270 and "
           f"270->360 under {SATURATION_RATIO_MAX}) AND the observed growth")
    print( "           could still close the gap -- see the `u` materiality "
           "table.  Where it IS attached, a YES")
    print( "           stands and a `no` is an UPPER BOUND.  Most metrics here "
           "are unsaturated but NOT material:")
    print( "           the floor would have to grow far faster than its last "
           "quarter shows to overturn the `no`.")
    print(f"         1 the two sides' spreads differ by more than "
           f"{ONE_SIDED_DECADES:.0f} order(s) of magnitude, so the RSS floor is")
    print( "           effectively ONE model's dispersion -- read it as "
           "legoESM's own, not as a joint property")


def verdict_table(rows, thin, unsat):
    print("\n" + "=" * 135)
    print("THE VERDICT TABLE -- gap = legoESM(member 0) - NEMO(member 0), "
          "matched day, matched reduction")
    print("floor = sqrt(lego_std^2 + NEMO_std^2) at that day")
    print("=" * 135)
    for day in HORIZONS:
        onesided = one_sided_flags(rows, day)
        material, _ = unsaturated_materiality(rows, unsat, day)
        print(f"\n--- day {day} ---")
        print(_header())
        for k in KEYS:
            lv, nv = rows["lego"][0][day][k], rows["nemo"][0][day][k]
            gap = lv - nv
            fl_sd, ls, ns = two_sided_floor(rows, k, day, stat=1)
            fl_rg, _, _ = two_sided_floor(rows, k, day, stat=0)
            ratio = float("inf") if fl_sd == 0 else abs(gap) / fl_sd
            pv, _ = permutation_p(rows, k, day)
            print(f"{LABELS[k]:<38}{lv:>14.6f}{nv:>14.6f}{gap:>+13.4e}"
                  f"{ls:>12.3e}{ns:>12.3e}{fl_sd:>13.3e}{fl_rg:>13.3e}"
                  f"{ratio:>9.2f}{pv:>8.3f}{_label(gap, fl_sd):>8}"
                  f"{_flags(k, thin.get(day, set()), material, onesided):>5}")
    _, tbl = unsaturated_materiality(rows, unsat, N_DAYS)
    unsaturated_table(tbl, N_DAYS)
    print()
    _legend()


def window_table(rows, unsat):
    """Flags here are computed from the WINDOW's own spreads, not inherited
    from the endpoint -- an endpoint-derived `q`/`1` on a window row labels the
    wrong statistic (round-2 review)."""
    print("\n" + "=" * 135)
    print(f"FINAL-90-DAY WINDOW MEANS -- days {WINDOW_DAYS[0]}..{WINDOW_DAYS[-1]} "
          f"every 10 days, {len(WINDOW_DAYS)} samples, IDENTICAL days both sides")
    print("floor = RSS of the two sides' spread of their OWN window means "
          "(not the endpoint floor reused)")
    print("=" * 135)
    print(_header())
    for k in KEYS:
        lw = [window_mean(rows, "lego", i, k) for i in range(N_MEM)]
        nw = [window_mean(rows, "nemo", i, k) for i in range(N_MEM)]
        lrg, lsd = F.spread(lw)
        nrg, nsd = F.spread(nw)
        fl_sd = float(np.hypot(lsd, nsd))
        gap = lw[0] - nw[0]
        ratio = float("inf") if fl_sd == 0 else abs(gap) / fl_sd
        one = set()
        if lsd > 0 and nsd > 0 and abs(np.log10(lsd / nsd)) > ONE_SIDED_DECADES:
            one.add(k)
        elif lsd <= 0 or nsd <= 0:
            one.add(k)
        print(f"{LABELS[k]:<38}{lw[0]:>14.6f}{nw[0]:>14.6f}{gap:>+13.4e}"
              f"{lsd:>12.3e}{nsd:>12.3e}{fl_sd:>13.3e}"
              f"{float(np.hypot(lrg, nrg)):>13.3e}{ratio:>9.2f}{'   n/a':>8}"
              f"{_label(gap, fl_sd):>8}{_flags(k, set(), unsat, one):>5}")


def growth_control(rows, out_dir, fp32_fields=("T", "S", "u")):
    """Did the kick reach the integrated state?  Read as max|dT| between each
    perturbed member and its own side's control, on the SAME side."""
    print("\n--- GROWTH CONTROL: max|dT| of each perturbed member vs its own "
          "side's control [K] ---")
    print("    (legoESM snapshots are float32: quantum ~2e-06 K on a ~20 K "
          "field, so an early 0.0 is storage, not physics)"
          if "T" in fp32_fields else
          "    (legoESM snapshots are float64: an early 0.0 here is NOT the "
          "storage quantum and needs a physical explanation)")
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
    # The storage quantum is a property of the STATE, so it is measured at each
    # scored horizon rather than measured once at day 360 and stamped on all of
    # them: at day 90 three of the five gate metrics are dtype-dominated and a
    # day-360 quantum would not have flagged them (round-2 review).
    # WHICH storage quantum applies is a property of the CANDIDATE artifacts,
    # not of this script's history: an ensemble run with --fp64-3d has none,
    # and downgrading its metrics for one would manufacture a false
    # UNMEASURABLE.  Read off the members' own per-field stamp; unstamped
    # (every recorded artifact) resolves to all-float32, unchanged.
    _fp32 = storage_fp32_fields([lego_npz(args.dir, i) for i in range(N_MEM)])
    print(f"[control] legoESM 3-D fields stored in float32: "
          f"{list(_fp32) or 'none (float64 snapshots)'}")
    quantum_by_day = {d: fp32_quantum(nemo_state(nemo_dir(0), d), A.tmask,
                                      _fp32)
                      for d in HORIZONS}
    thin = separation_control(rows, quantum_by_day)
    unsat = saturation_table(rows)
    growth_control(rows, args.dir, _fp32)
    spread_curves(rows, quantum_by_day[N_DAYS], _fp32)
    frac, neg_ok, med = empirical_rule_controls(rows)
    verdict_table(rows, thin, unsat)
    window_table(rows, unsat)
    # Re-printed AFTER the tables: a calibration number quoted 200 lines above
    # the verdicts it calibrates does not travel with them (round-3 review).
    print(f"\nCALIBRATION OF THE RULE THAT PRODUCED THE TABLES ABOVE: "
          f"positive-control median ratio {med:.3f} against the analytic "
          f"leave-two-out null {np.sqrt(2.0):.3f},")
    print(f"  and {100 * frac:.1f}% of the legoESM-side within-side pairs x "
          f"{len(KEYS)} metrics inside the band against a predicted "
          f"{100 * positive_control_null(K_WELCH):.1f}%")
    print(f"  (the median is the stable statistic; the fraction is noisy at "
          f"this sample size and the comparisons are correlated),")
    neg_msg = ('rejects every transport as required' if neg_ok else
               'FAILS -- the rule accepts a known-different state at BOTH '
               'lags somewhere')
    print(f"  and the two-lag self-mismatch negative control {neg_msg}.")
    print("\nEvery number above is a measurement; the pre-registration "
          "(PREREG_verdict360.md) says which ones were predicted, and the "
          "result commit carries its corrections.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
