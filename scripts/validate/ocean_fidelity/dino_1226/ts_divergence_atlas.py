#!/usr/bin/env python
"""#1455 follow-on: the 3-D T/S DIVERGENCE ATLAS of the verdict year.

Pre-registration: ``PREREG_ts_divergence_atlas.md``, committed BEFORE any
statistic here was computed (commit 7e088914e). Read it before citing a
number; the bars for questions 2 and 3 are registered there, not chosen here.

WHAT THIS IS.  ``verdict360.py`` scored TRANSPORT metrics over a year from a
shared NEMO day-180 restart and found the circumpolar channel matching NEMO to
17 ppm while the southern basin does not.  That says which latitudes carry the
gap in the MOMENTUM field.  This probe asks the tracer question underneath it:
where in three dimensions do the two models' T and S fields depart, how fast,
and does the departure pattern stay put or move.

It is DESCRIPTIVE.  It computes no new transport reduction (the per-row
deficit is ``acc_driver_decomp.group_transport``, imported), edits no floor
constant, and prints no verdict string -- each statistic is printed next to
the bar registered for it, and the classification is made in the findings
document by a human reading both.

WHAT IT READS
  legoESM  /tmp/dino_verdict360/m{0..3}_*.npz   T3d_day{d}/S3d_day{d}/u3d_day{d}
  NEMO     .../DINO/RUN_VERDICT360_M{0..3}/DINO_<kt>_restart_*.nc   tn/sn/un
Scored horizons are the INTERSECTION of the two cadences -- legoESM saves every
10 days, NEMO writes restarts at 20 of those -- and a horizon present on only
one side is dropped, never interpolated.

CONTROLS, all fatal, all printed before any number is used:
  C1 PROVENANCE.  Every legoESM member's log must carry the same
     ``PROVENANCE: HEAD=`` as the directory's ``.launch_sha``; every npz must
     carry ``control_dtype``, ``nemo_ladder_mode`` and the seasonal-clock
     stamps (enforced by ``acceptance_gate_90d.load_candidate``, which refuses
     an unstamped or wrong-clock artifact).  Every NEMO member dir must carry
     all scored restart stamps.
  C2 KICK.  Each NEMO member i>0 must differ from member 0 AT DAY 0 in tn only,
     at the registered 1e-14 relative level -- i.e. the ensemble really is the
     perturbation ensemble and not four copies of one run.
  C3 DAY-0 IDENTITY.  lego and NEMO must agree at day 0 in T and S to the fp32
     snapshot quantum, or the two sides are not the same twin.
  C4 NaN.  A single non-finite value on the wet mask is fatal, and the
     difference field is zeroed off the mask so a dry-cell NaN cannot reach a
     sum.  No NaN-tolerant reduction appears in ANY REPORTED STATISTIC; the two
     ``nanpercentile`` calls in this file set colourbar limits on arrays that
     are deliberately NaN-masked for display, downstream of that fatal check.
  C5 MASK BITE (planted violation).  A dry cell is given a difference of 1e6
     and every reported reduction must be unchanged to 1e-12 relative.  A
     statistic that moves means the mask does not bite.
  C6 DTYPE.  Every comparison array and every weight is float64.
  C7 KNOWN ANSWER.  The per-row transport deficits must SUM to
     ``acc_driver_decomp``'s own south-of-band group gap at every horizon -- a
     probe that cannot reproduce a number the campaign already recorded is not
     usable on one it has not.  At day 360 that sum is -0.9519 Sv against the
     -0.95191 Sv in the verdict-run result commit (a1387f1f7).

WEIGHTS.  Thickness-weighted throughout: volume = e1t * e2t * e3t_0 (partial
cell), zero on land.  The campaign's layer-averaging retraction forbids an
unweighted mean over levels or over cells of unequal volume, so there is no
plain ``.mean()`` on any field axis in this file.

Usage
-----
  JAX_ENABLE_X64=1 ts_divergence_atlas.py --self-test    # controls only
  JAX_ENABLE_X64=1 ts_divergence_atlas.py --out-dir DIR  # atlas + figures
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
import acc_driver_decomp as D  # noqa: E402  group_transport, LAT_GROUPS, _avg
import acc_thermal_wind as A  # noqa: E402  mesh, masks, J0/J1
import acceptance_gate_90d as G  # noqa: E402  load_candidate (stamp refusals)
import kamm_twin_90d as T  # noqa: E402  the twin's OWN clock guard
from rebuild_nemo_restart import rebuild  # noqa: E402

LEGO_DIR = "/tmp/dino_verdict360"
SEEDS = (None, 1, 2, 3)
N_MEM = len(SEEDS)

# fp32 storage quantum of the legoESM snapshots on a ~20 K / ~35 g/kg field.
DAY0_BAR = 1e-5
# the registered kick: 1e-14 relative on the now-level T of NEMO members 1..3
KICK_REL_LO, KICK_REL_HI = 1e-16, 1e-12
# The verdict run's OWN recorded day-360 south-of-band transport gap
# (a1387f1f7, "SOUTH of band -0.95191"). C7 reproduces THIS -- a control that
# only checks the row split against its own group sum is an algebraic identity
# of two linear reductions and cannot fail, which is what the first version was.
D360_SOUTH_GAP_RECORDED_SV = -0.95191
D360_SOUTH_GAP_TOL_SV = 1e-4
# member 3's NEMO run was relocated off the certified tree when the home quota
# filled mid-campaign; a1387f1f7 records that its restart was re-verified as the
# seed-3 perturbation and reproduced independently before it was re-run. Named
# here so the refusal below is a NAMED exception with a reason, not a print line
# a reader is expected to eyeball.
NEMO_DIR_ALLOWED_OUTSIDE_TREE = {"/tmp/dino_v360_m3"}
# the vertical ladder the registration scores on; refused on VALUE, not printed
LADDER_REQUIRED = "both"

# ---- registered partitions (PREREG sec.2) -----------------------------------
UPPER_M, DEEP_M = 200.0, A.DEEP_M           # 200 m, 1400 m
SOUTH_ROWS = list(range(1, A.J0))           # basin rows 1..13 (row 0 is dry)
# southern_circulation_budget.py's own split: wall rows 1-5, main rows 6-13.
# Expressed ONCE, as absolute T-rows, and every consumer derives its positional
# slice into SOUTH_ROWS from it -- a second hand-written `[:5]` is how the two
# definitions drift apart.  The task brief said "the 4 wall rows"; the committed
# split is FIVE and it was not changed to match prose.
WALL_ROWS = list(range(1, 6))
MAIN_ROWS = [j for j in SOUTH_ROWS if j not in WALL_ROWS]
N_WALL = len(WALL_ROWS)
assert WALL_ROWS + MAIN_ROWS == SOUTH_ROWS, "wall/main split is not a partition"
# registered bars, question 2
Q2_R_BAR, Q2_RET_BAR = 0.50, 0.25
Q2_SPREAD_RET, Q2_SPREAD_ROWS, Q2_SPREAD_M = 0.15, 10.0, 500.0
Q2_NULL = 0.05
# registered bar, question 3
Q3_RHO_BAR = 0.56                           # two-sided p<0.05 at n=13


# ---------------------------------------------------------------- geometry ---
def _f64(a):
    return np.asarray(a, dtype=np.float64)


e1t = _f64(A.mm["e1t"][0]).squeeze()                      # (y,x) [m]
e2t = _f64(A.mm["e2t"][0]).squeeze()                      # (y,x) [m]
CELL_VOL = e1t[:, :, None] * e2t[:, :, None] * _f64(A.e3t0)   # (y,x,z) [m3]

DEPTH_CLASSES = (
    ("upper   <200m", _f64(A.gdept1d) < UPPER_M),
    ("interior 200-1400m", (_f64(A.gdept1d) >= UPPER_M) & (_f64(A.gdept1d) < DEEP_M)),
    ("abyss   >1400m", _f64(A.gdept1d) >= DEEP_M),
)
REGIONS = (("south basin", slice(0, A.J0)),
           ("channel band", slice(A.J0, A.J1 + 1)),
           ("north basin", slice(A.J1 + 1, A.NY)))


def build_weights(land_mask):
    """(wet 3-D mask, volume weight zeroed off it).  SAME mask for both models:
    mesh_mask tmask AND the legoESM land_mask, exactly as acc_thermal_wind and
    acc_driver_decomp apply it.  A per-model mask would be a protocol
    difference, i.e. a confound."""
    wet = A.tmask & (_f64(land_mask)[:, :, None] > 0.5)
    w = np.where(wet, CELL_VOL, 0.0)
    return wet, w


# --------------------------------------------------------------- reductions ---
def wrms(d, w, sel=None):
    """Volume-weighted rms of the difference field over `sel` (a boolean 3-D
    mask; None = the whole weight support).  sqrt(sum w d^2 / sum w).  The
    weight is already zero on land, so a dry cell contributes nothing whatever
    it holds -- that is what C5 plants against."""
    ww = w if sel is None else np.where(sel, w, 0.0)
    tot = float(ww.sum())
    # An empty selection ABORTS.  Returning NaN would print a plausible-looking
    # blank into a table and let a mis-specified region or depth band survive
    # to the findings document.
    if not tot > 0.0:
        raise SystemExit("FATAL: wrms over a selection with zero wet volume -- "
                         "a region or depth band that selects nothing is a "
                         "defect in the partition, not a NaN to print")
    return float(np.sqrt(float((ww * d * d).sum()) / tot))


def wcorr(a, b, w):
    """Volume-weighted Pearson correlation of two fields over the weight
    support."""
    tot = float(w.sum())
    if not tot > 0.0:
        raise SystemExit("FATAL: wcorr over a zero-volume support")
    ma = float((w * a).sum()) / tot
    mb = float((w * b).sum()) / tot
    da, db = a - ma, b - mb
    cov = float((w * da * db).sum()) / tot
    va = float((w * da * da).sum()) / tot
    vb = float((w * db * db).sum()) / tot
    if not (va > 0.0 and vb > 0.0):
        raise SystemExit("FATAL: wcorr on a field with zero weighted variance "
                         "-- the correlation is undefined, not NaN to print")
    return float(cov / np.sqrt(va * vb))


def centroid(d, w):
    """Volume-weighted centroid of d^2 in (T-row index, depth [m]).

    DEVIATION FROM THE REGISTRATION: PREREG sec.3 S2 says ``gdept_1d``; this
    uses ``gdept_0``, the 3-D partial-cell T-point depth, so a bottom partial
    cell is placed at its own depth rather than at the reference ladder's.
    That is the physically correct depth of the water the difference sits in,
    and it differs from the registered choice only in bottom partial cells.
    Recorded rather than silently taken.  (The depth CLASSES use ``gdept_1d``
    as registered, because a class boundary must be one number for all
    columns, not a per-column one.)
    """
    m = w * d * d
    tot = float(m.sum())
    if not tot > 0.0:
        raise SystemExit("FATAL: centroid of a zero squared-difference field "
                         "-- undefined, not NaN to print")
    rows = np.arange(A.NY, dtype=np.float64)[:, None, None]
    return (float((m * rows).sum()) / float(tot),
            float((m * _f64(A.gdept0)).sum()) / float(tot))


def hotspot_set(d0, w, frac=Q2_NULL):
    """The cells holding the top `frac` of the WET VOLUME when ranked by |d0|.

    DEVIATION FROM THE REGISTRATION, recorded here so it is not read as "as
    registered": PREREG sec.3 S3 says "top 5% of V*d^2".  Ranking by ``V*d^2``
    would rank a large cell above a more strongly diverged small one, which
    makes the set a statement about cell size and destroys the exact 0.05 null.
    Ranking is therefore by INTENSITY (|d0|) and the CUT is by cumulative
    volume, which is what gives the set a null share of exactly `frac` of any
    later sum of w*d^2.  The self-test measures that null on unequal weights.
    """
    flat_i = np.argsort(np.abs(d0), axis=None)[::-1]
    wflat = w.reshape(-1)[flat_i]
    cum = np.cumsum(wflat)
    n = int(np.searchsorted(cum, frac * float(w.sum()))) + 1
    sel = np.zeros(w.size, dtype=bool)
    sel[flat_i[:n]] = True
    return sel.reshape(w.shape)


def retention(d, w, hot):
    """Share of the total volume-weighted d^2 sitting inside `hot`."""
    m = w * d * d
    tot = float(m.sum())
    if not tot > 0.0:
        raise SystemExit("FATAL: retention of a zero squared-difference field "
                         "-- undefined, not NaN to print")
    return float(m[hot].sum()) / tot


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(_f64(v))).astype(np.float64)  # noqa: E731
    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if not den > 0:
        raise SystemExit("FATAL: spearman on a constant profile -- the rank "
                         "correlation is undefined, not NaN to print")
    return float((rx * ry).sum() / den)


def pearson(x, y):
    x, y = _f64(x) - np.mean(x), _f64(y) - np.mean(y)
    den = np.sqrt((x * x).sum() * (y * y).sum())
    if not den > 0:
        raise SystemExit("FATAL: pearson on a constant profile -- the "
                         "correlation is undefined, not NaN to print")
    return float((x * y).sum() / den)


def lag1(x):
    """Lag-1 autocorrelation of a 1-D profile (about its own mean)."""
    x = _f64(x) - np.mean(x)
    den = float((x * x).sum())
    return float((x[:-1] * x[1:]).sum() / den) if den > 0 else 0.0


def n_eff(x, y):
    """Dawdy-Matalas effective sample size for a correlation between two
    AUTOCORRELATED profiles: n_eff = n * (1 - r1x*r1y) / (1 + r1x*r1y).

    The registered bar |rho| >= 0.56 is the two-sided p<0.05 critical value for
    THIRTEEN INDEPENDENT samples.  These 13 samples are adjacent latitude rows
    in one basin and are strongly autocorrelated, so that bar is too generous by
    construction.  This is a first-order correction quoted from standard
    practice, not derived here; the shift/wrong-time nulls below are exact and
    are what the conclusion should rest on.
    """
    p = lag1(x) * lag1(y)
    return float(len(x) * (1.0 - p) / (1.0 + p)) if p > -1.0 else float(len(x))


def shift_null(a, b, stat=None):
    """Exact cyclic-shift null for a profile correlation.

    Rolling `b` by 1..n-1 rows preserves BOTH profiles' own shapes and their
    autocorrelation exactly, and destroys only the row-by-row alignment -- which
    is the thing being tested.  Returns (observed, p) where p is the fraction of
    shifts whose |statistic| is at least the observed one, including the
    observed alignment itself.  The resolution floor is 1/n, so p can never go
    below 1/13 = 0.077 here: that IS the test's power, and quoting it is the
    honest alternative to quoting a normal-theory p-value the samples do not
    support.
    """
    stat = spearman if stat is None else stat
    obs = stat(a, b)
    alt = [abs(stat(a, np.roll(b, k))) for k in range(1, len(b))]
    return obs, float((1 + sum(v >= abs(obs) for v in alt)) / (1 + len(alt)))


def row_volume(w):
    """Wet volume of each southern-basin row [m3], in SOUTH_ROWS order."""
    v = np.array([float(w[j].sum()) for j in SOUTH_ROWS], dtype=np.float64)
    if not (v > 0.0).all():
        raise SystemExit(f"FATAL: southern-basin rows with zero wet volume: "
                         f"{[SOUTH_ROWS[i] for i in np.nonzero(v <= 0)[0]]} -- "
                         f"a dry row inside the basin would silently drop out "
                         f"of every share and correlation below")
    return v


def _wall_share(rms_rows, row_vol):
    """Share of the southern-basin volume-weighted d^2 sitting in the wall rows.

    ``rms_rows`` is the per-row volume-weighted rms, so ``rms^2`` is that row's
    volume-weighted MEAN square and must be multiplied by the row's own volume
    to recover the row's contribution to the total.  Weighting is not optional
    here: the 13 rows span 4.6 degrees of latitude and differ in wet volume.
    """
    contrib = np.asarray(row_vol, dtype=np.float64) * np.asarray(rms_rows, dtype=np.float64) ** 2
    tot = float(contrib.sum())
    if not tot > 0.0:
        raise SystemExit("FATAL: southern-basin squared difference is zero -- "
                         "the wall share is undefined, not 0.0")
    return float(contrib[:N_WALL].sum() / tot)


def row_deficit(u_lego, u_nemo):
    """Per-row zonal transport deficit [Sv], lego - NEMO, on the campaign's own
    reduction: acc_driver_decomp.group_transport restricted to one T-row, with
    A.umask on BOTH sides and D._avg (mean over lons 2..-2) as the reducer."""
    out = np.zeros(A.NY, dtype=np.float64)
    for j in range(A.NY):
        sl = slice(j, j + 1)
        out[j] = (D._avg(D.group_transport(u_lego, A.umask, sl))
                  - D._avg(D.group_transport(u_nemo, A.umask, sl)))
    return out


# ------------------------------------------------------------------ loaders ---
def nemo_dir(i):
    return f"{A.DINO}/RUN_VERDICT360_M{i}"


def lego_npz(i):
    return f"{LEGO_DIR}/{'m0_control' if SEEDS[i] is None else f'm{i}_seed{SEEDS[i]}'}.npz"


def lego_log(i):
    return lego_npz(i).replace(".npz", ".log")


def kt_of(day):
    return G.KT_RESTART + day * G.STEPS_PER_DAY


def load_nemo(i, day, fields=("tn", "sn")):
    pat = f"{nemo_dir(i)}/DINO_{kt_of(day):08d}_restart*.nc"
    if not glob.glob(pat):
        raise SystemExit(f"FATAL: no NEMO restart for member {i} day {day}: {pat}")
    raw = rebuild(pat, list(fields))
    yxz = lambda a: np.moveaxis(_f64(a), 0, -1)                     # noqa: E731
    out = {}
    for src, dst in (("tn", "T"), ("sn", "S"), ("un", "u")):
        if src in raw:
            out[dst] = yxz(raw[src])
    missing = [s for s in fields if s not in raw]
    if missing:
        raise SystemExit(f"FATAL: NEMO member {i} day {day} restart lacks {missing}")
    return out


def scored_days(nemo_i=0):
    """The intersection of the two cadences, as a sorted list of days."""
    lego_days = {int(k[len("T3d_day"):]) for k in np.load(lego_npz(0)).files
                 if k.startswith("T3d_day")}
    nemo_days = set()
    for p in glob.glob(f"{nemo_dir(nemo_i)}/DINO_*_restart*.nc"):
        kt = int(os.path.basename(p).split("_")[1])
        d, rem = divmod(kt - G.KT_RESTART, G.STEPS_PER_DAY)
        if rem == 0 and d >= 0:
            nemo_days.add(d)
    return sorted(lego_days & nemo_days)


# ------------------------------------------------------------------ controls ---
def control_provenance():
    print("[C1 PROVENANCE]")
    sha_path = f"{LEGO_DIR}/.launch_sha"
    if not os.path.exists(sha_path):
        raise SystemExit(f"FATAL: {LEGO_DIR} has no .launch_sha -- an artifact "
                         f"that does not stamp the revision it was produced at "
                         f"is refused, not guessed at")
    with open(sha_path) as fh:
        sha = fh.read().strip()
    print(f"  launch sha              : {sha}")
    for i in range(N_MEM):
        log = lego_log(i)
        if not os.path.exists(log):
            raise SystemExit(f"FATAL: {log} missing -- member has no run log")
        head = None
        dtype = None
        with open(log) as fh:
            for line in fh:
                if line.startswith("PROVENANCE: HEAD="):
                    head = line.split("HEAD=")[1].split()[0]
                if "materialized state dtype" in line:
                    dtype = line.split(":")[-1].strip()
                if head and dtype:
                    break
        if head != sha:
            raise SystemExit(f"FATAL: {log} PROVENANCE HEAD={head} != launch "
                             f"sha {sha} -- members are not one revision")
        d = np.load(lego_npz(i))
        for key in ("control_dtype", "nemo_ladder_mode", "seasonal_t0_seconds"):
            if key not in d.files:
                raise SystemExit(f"FATAL: {lego_npz(i)} lacks the {key} stamp "
                                 f"-- a legacy artifact is refused")
        if str(d["control_dtype"]) != "float64":
            raise SystemExit(f"FATAL: {lego_npz(i)} control_dtype "
                             f"{d['control_dtype']} -- fp64 required")
        # the ladder stamp is REFUSED on value, not merely printed: a member run
        # on a different vertical ladder would otherwise pass this control and
        # inflate the chaotic floor every table divides by
        if str(d["nemo_ladder_mode"]) != LADDER_REQUIRED:
            raise SystemExit(f"FATAL: {lego_npz(i)} ran vertical ladder "
                             f"{d['nemo_ladder_mode']!r}, not the registered "
                             f"{LADDER_REQUIRED!r} -- members on different "
                             f"ladders are not one ensemble")
        print(f"  m{i} head={head[:9]} dtype={dtype} ladder={d['nemo_ladder_mode']} "
              f"clock_t0={float(d['seasonal_t0_seconds']):.0f}s")
    control_clock()
    days = scored_days()
    for i in range(N_MEM):
        have = set(scored_days(i))
        if not set(days) <= have:
            raise SystemExit(f"FATAL: NEMO member {i} is missing scored horizons "
                             f"{sorted(set(days) - have)}")
    print(f"  scored horizons ({len(days)}) : {days}")
    # The registration refuses a member outside the certified NEMO tree.  It is
    # a REFUSAL, not a print: member 3 really does resolve outside it, and it
    # feeds the chaotic floor in every row of both growth tables, so "a reader
    # will notice the path" is not a control.
    cert_root = os.path.realpath(A.DINO)
    for i in range(N_MEM):
        real = os.path.realpath(nemo_dir(i))
        inside = os.path.commonpath([real, cert_root]) == cert_root
        allowed = real in NEMO_DIR_ALLOWED_OUTSIDE_TREE
        if not inside and not allowed:
            raise SystemExit(
                f"FATAL: NEMO member {i} resolves to {real}, outside the "
                f"certified tree {cert_root}, and is not in the named "
                f"allow-list. A member on unaudited storage is refused, not "
                f"printed for a reader to eyeball.")
        note = "" if inside else "  [NAMED EXCEPTION: relocated, a1387f1f7]"
        print(f"  NEMO m{i} dir           : {real}{note}")
    try:
        sha_now = subprocess.run(["git", "-C", _DIR, "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        print(f"  this probe at           : {sha_now}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  this probe at           : UNKNOWN ({exc})")
    return days


def control_clock():
    """C1b SEASONAL CLOCK, run from the ORACLE side because the stamp the gate
    now demands did not exist when these members ran.

    ``kamm_twin_90d.assert_nemo_seasonal_clock`` refuses an artifact that does
    not carry BOTH ``seasonal_t0_seconds`` (the phase the twin was forced on)
    and ``seasonal_t0_reference_seconds`` (NEMO's own phase, for the
    comparison).  The second stamp was introduced by a98019fb6, which is NOT an
    ancestor of a7b940f75 -- the revision every verdict member ran at -- so
    these artifacts CANNOT carry it and no re-scoring can make them.

    Refusing them outright would mean refusing the campaign's only year-long
    twin.  Instead the missing HALF of the guard is supplied from the oracle:
    NEMO's own phase is read out of the restart the members started from
    (``adatrj``, cross-checked against ``kt * rn_Dt`` by the twin's own reader),
    and the SAME comparison the guard makes is then made here, on the SAME
    function.  This substitutes for the stamp; it does not weaken the check.
    Only after it passes is the gate's stamp refusal bypassed for the load.
    """
    print("[C1b SEASONAL CLOCK] the reference stamp postdates these members "
          "(a98019fb6 is not an ancestor of a7b940f75);\n  supplying NEMO's own "
          "phase from the restart and running the twin's OWN guard on it")
    restart = f"{nemo_dir(0)}/DINO_{G.KT_RESTART:08d}_restart.nc"
    if not os.path.exists(restart):
        raise SystemExit(f"FATAL: {restart} missing -- NEMO's own seasonal "
                         f"phase cannot be read and the clock cannot be checked")
    t0_nemo = T.restart_elapsed_seconds(restart)
    for i in range(N_MEM):
        d = np.load(lego_npz(i))
        if "seasonal_t0_reference_seconds" in d.files:
            raise SystemExit(
                f"FATAL: {lego_npz(i)} DOES carry the reference stamp, so this "
                f"substitute path is not the one to take -- score it through "
                f"the gate's own guard instead.")
        t0, ref = T.assert_nemo_seasonal_clock(
            {"seasonal_t0_seconds": float(d["seasonal_t0_seconds"]),
             "seasonal_t0_reference_seconds": t0_nemo}, lego_npz(i))
        print(f"  m{i} forced at {t0 / 86400.0:.2f} d of the 360-day year; "
              f"NEMO restart at {ref / 86400.0:.2f} d")
    os.environ["DINO_GATE_ALLOW_LEGACY_CLOCK"] = "1"
    print("  clock verified against the oracle -- the gate's STAMP refusal is "
          "bypassed for the loads below, its PHASE check is not")


def control_kick(wet):
    print("[C2 KICK] NEMO members 1-3 vs member 0 at day 0, relative on tn")
    t0 = load_nemo(0, 0)["T"]
    for i in range(1, N_MEM):
        ti = load_nemo(i, 0)["T"]
        # on the WET mask, like every other statistic here.  The previous
        # version used isfinite(), which is the one place a control in this
        # file touched dry cells.
        rel = (np.abs(ti[wet] - t0[wet])
               / np.maximum(np.abs(t0[wet]), 1e-12))
        mx = float(rel.max())
        print(f"  m{i} max relative |dT| = {mx:.3e}")
        if not (KICK_REL_LO <= mx <= KICK_REL_HI):
            raise SystemExit(f"FATAL: NEMO member {i} day-0 kick {mx:.3e} is "
                             f"outside the registered [{KICK_REL_LO:.0e},"
                             f"{KICK_REL_HI:.0e}] band -- the ensemble is not "
                             f"the perturbation ensemble it is being read as")


def control_day0(wet):
    print("[C3 DAY-0 IDENTITY] lego m0 - NEMO m0 at day 0 on the wet mask")
    lg = G.load_candidate(lego_npz(0), 0)
    nm = load_nemo(0, 0)
    for key, unit in (("T", "K"), ("S", "g/kg")):
        mx = float(np.abs(lg[key][wet] - nm[key][wet]).max())
        print(f"  max|d{key}| = {mx:.3e} {unit}   (bar {DAY0_BAR:.0e})")
        if not (mx <= DAY0_BAR):
            raise SystemExit(f"FATAL: day-0 {key} differs by {mx:.3e} > "
                             f"{DAY0_BAR:.0e} -- the two sides are not the same twin")


def control_finite(tag, field, wet):
    bad = int(np.sum(~np.isfinite(field[wet])))
    if bad:
        raise SystemExit(f"FATAL: {tag} has {bad} non-finite values on the wet "
                         f"mask -- a blown field is a finding, not a nanmean")


def control_mask_bite(d, w, wet):
    """C5: a DRY cell given a 1e6 difference must move nothing."""
    print("[C5 MASK BITE] planted 1e6 on a dry cell")
    dry = ~wet
    if not dry.any():
        raise SystemExit("FATAL: no dry cell exists -- the plant cannot be made")
    idx = tuple(int(v[0]) for v in np.nonzero(dry))
    poisoned = d.copy()
    poisoned[idx] = 1e6
    ref = np.asarray(A.gdept0, dtype=np.float64)      # a fixed second field for wcorr
    rows_of = lambda x: np.array(                                       # noqa: E731
        [wrms(x, w, _sel(slice(j, j + 1), np.ones(A.NZ, bool))) for j in SOUTH_ROWS])
    hot0 = hotspot_set(d, w)
    # EVERY reported reduction, as registered -- not the four that were easiest
    # to reach.  A reduction absent from this list is one whose mask behaviour
    # was argued rather than measured.
    checks = [
        ("global wrms", lambda x: wrms(x, w)),
        ("south-upper wrms", lambda x: wrms(x, w, _sel(REGIONS[0][1], DEPTH_CLASSES[0][1]))),
        ("share south-upper", lambda x: float(
            (w * x * x)[_sel(REGIONS[0][1], DEPTH_CLASSES[0][1])].sum()
            / max(float((w * x * x).sum()), 1e-300))),
        ("centroid row", lambda x: centroid(x, w)[0]),
        ("centroid depth", lambda x: centroid(x, w)[1]),
        ("retention", lambda x: retention(x, w, hot0)),
        ("hot overlap", lambda x: float(
            w[hotspot_set(x, w) & hot0].sum() / w[hot0].sum())),
        ("wcorr vs depth", lambda x: wcorr(np.abs(x), ref, w)),
        ("row rms (basin sum)", lambda x: float(rows_of(x).sum())),
        ("wall share", lambda x: _wall_share(rows_of(x), row_volume(w))),
    ]
    for name, fn in checks:
        a, b = fn(d), fn(poisoned)
        rel = abs(b - a) / max(abs(a), 1e-30)
        print(f"  {name:<20} clean {a:.12e}  planted {b:.12e}  rel {rel:.2e}")
        # NEGATED form deliberately: `rel > tol` is FALSE for NaN, so the
        # original comparison PASSED on a NaN -- in the one control whose whole
        # job is dry-cell behaviour, and NaN is exactly what an unwritten
        # restart region carries.
        if not (rel <= 1e-12):
            raise SystemExit(f"FATAL: the mask does not bite -- {name} moved "
                             f"{rel:.2e} when a DRY cell was poisoned "
                             f"(a non-finite value here is a failure, not a pass)")
    # and prove the check is not vacuous: the same plant on a WET cell MUST move it
    wet_idx = tuple(int(v[0]) for v in np.nonzero(wet))
    live = d.copy()
    live[wet_idx] = 1e6
    a, b = wrms(d, w), wrms(live, w)
    print(f"  non-vacuity: same plant on a WET cell moves global wrms "
          f"{a:.6e} -> {b:.6e}")
    if not (b > a * 1.000001):
        raise SystemExit("FATAL: the planted violation is VACUOUS -- poisoning a "
                         "WET cell did not move the statistic either")


def control_dtype(*arrays):
    for name, a in arrays:
        if np.asarray(a).dtype != np.float64:
            raise SystemExit(f"FATAL: {name} is {np.asarray(a).dtype}, not float64")


def _sel(rows, levs):
    """3-D boolean selection from a row slice and a level boolean."""
    m = np.zeros((A.NY, A.NX, A.NZ), dtype=bool)
    m[rows, :, :] = True
    return m & levs[None, None, :]


# ---------------------------------------------------------------- self-test ---
def self_test():
    """Analytic checks of every reduction defined here, on synthetic fields
    with known answers.  Each one FAILS if the reduction is removed."""
    print("=" * 78)
    print("SELF-TEST -- every reduction against a case whose answer is known")
    print("=" * 78)
    rng = np.random.default_rng(0)
    w = np.zeros((4, 3, 2))
    w[:, :, 0] = 1.0
    w[:, :, 1] = 3.0                       # deliberately unequal layer thickness
    d = np.zeros_like(w)
    d[:, :, 0] = 2.0
    d[:, :, 1] = 4.0
    # thickness-weighted rms of {2 with weight 1, 4 with weight 3}
    want = float(np.sqrt((1 * 4 + 3 * 16) / 4))
    got = wrms(d, w)
    assert abs(got - want) < 1e-12, (got, want)
    # the UNWEIGHTED rms would be sqrt((4+16)/2)=3.162 -- prove they differ, i.e.
    # that the weighting is load-bearing and not decoration
    assert abs(got - float(np.sqrt(10.0))) > 0.1
    print(f"  wrms thickness-weighted   {got:.6f} (want {want:.6f}; the "
          f"unweighted answer {np.sqrt(10.0):.6f} is excluded)")

    a = rng.normal(size=(6, 5, 4))
    ww = rng.uniform(0.5, 2.0, size=a.shape)
    assert abs(wcorr(a, a, ww) - 1.0) < 1e-12
    assert abs(wcorr(a, -a, ww) + 1.0) < 1e-12
    print("  wcorr self/anti           +1 / -1 exactly")

    # centroid of a single spike sits on the spike
    dd = np.zeros((A.NY, A.NX, A.NZ))
    dd[100, 10, 20] = 1.0
    ww3 = np.ones_like(dd)
    r, z = centroid(dd, ww3)
    assert abs(r - 100.0) < 1e-9 and abs(z - float(A.gdept0[100, 10, 20])) < 1e-9
    print(f"  centroid of a spike       row {r:.1f}, {z:.1f} m (spike at row 100)")

    # hotspot retention: if d is UNCHANGED the top-5%-by-volume set holds far
    # more than 5% of d^2 when d is concentrated, and exactly 5% when d is flat
    flat = np.ones_like(dd)
    # On UNEQUAL weights, so the "selection is by cumulative VOLUME" property is
    # actually exercised.  The earlier version used all-ones weights and asserted
    # retention == the selected volume share, which is true BY CONSTRUCTION for a
    # flat field and ANY selector whatsoever -- it passed for a selector taking
    # half the cells.  What must be asserted is that the selected VOLUME is the
    # registered 0.05, because that is what makes the null 0.05.
    w2 = rng.uniform(0.1, 10.0, size=dd.shape)
    h = hotspot_set(flat, w2)
    assert abs(float(w2[h].sum() / w2.sum()) - Q2_NULL) < 1e-3
    assert abs(retention(flat, w2, h) - Q2_NULL) < 1e-3
    # ...and a case where cutting by VOLUME and cutting by COUNT give visibly
    # different sets, so the assertion above cannot pass a count-based selector.
    # Rank the field so the HEAVIEST cells come first: the top 5% of volume is
    # then far fewer than 5% of cells.
    heavy = w2.copy()                       # |d0| ordering == weight ordering
    h2 = hotspot_set(heavy, w2)
    vol_share = float(w2[h2].sum() / w2.sum())
    cell_share = float(h2.sum() / h2.size)
    assert abs(vol_share - Q2_NULL) < 1e-3, vol_share
    # a COUNT-based cut would give exactly Q2_NULL here; a volume cut gives far
    # fewer cells because the heaviest ones rank first
    assert cell_share < 0.7 * Q2_NULL, (cell_share, "cut is by COUNT, not volume")
    print(f"  hotspot cuts by VOLUME    {vol_share:.4f} of volume but only "
          f"{cell_share:.4f} of cells when the heaviest cells rank first "
          f"(a count-based cut would give {Q2_NULL})")
    spike = np.zeros_like(dd)
    spike[:2, :, :] = 10.0
    hot2 = hotspot_set(spike, ww3)
    assert retention(spike, ww3, hot2) > 0.9
    print(f"  retention of a CONCENTRATED field {retention(spike, ww3, hot2):.4f}")

    # the wall share must weight each row's mean-square by that ROW's volume.
    # Case: wall rows carry rms 1 on tiny volume, main rows rms 1 on large
    # volume -- the correct share is the VOLUME share, and the unweighted
    # answer (N_WALL/13) is excluded.
    vol = np.array([1.0] * N_WALL + [9.0] * (len(SOUTH_ROWS) - N_WALL))
    got = _wall_share(np.ones(len(SOUTH_ROWS)), vol)
    want = float(vol[:N_WALL].sum() / vol.sum())
    assert abs(got - want) < 1e-12, (got, want)
    assert abs(got - N_WALL / len(SOUTH_ROWS)) > 0.1, "unweighted answer not excluded"
    print(f"  wall share volume-weighted {got:.6f} (want {want:.6f}; the "
          f"unweighted answer {N_WALL / len(SOUTH_ROWS):.6f} is excluded)")
    # and it must respond to intensity, not only to volume
    hot = np.ones(len(SOUTH_ROWS))
    hot[:N_WALL] = 3.0
    assert _wall_share(hot, vol) > got

    x = np.arange(13, dtype=np.float64)
    assert abs(spearman(x, x ** 3) - 1.0) < 1e-12       # monotone, non-linear
    assert abs(spearman(x, -x) + 1.0) < 1e-12
    assert abs(pearson(x, 2 * x + 1) - 1.0) < 1e-12
    print("  spearman monotone/anti    +1 / -1 exactly; pearson affine +1")
    print("SELF-TEST PASSED\n")
    return 0


# -------------------------------------------------------------------- atlas ---
def run(out_dir):
    days = control_provenance()

    lego0 = np.load(lego_npz(0))
    wet, w = build_weights(lego0["land_mask"])
    control_kick(wet)
    control_dtype(("volume weight", w), ("e3t_0", A.e3t0), ("gdept_0", A.gdept0),
                  ("gdept_1d", A.gdept1d), ("cell volume", CELL_VOL),
                  ("gphit", _f64(A.gphit)))
    print(f"[mask] {int(wet.sum())} wet T-cells of {wet.size}; "
          f"total volume {w.sum():.6e} m3")
    print("[regions] " + "; ".join(
        f"{n} rows {s.start}..{s.stop - 1}" for n, s in REGIONS))
    print("[depths]  " + "; ".join(
        f"{n} = {int(m.sum())} levels" for n, m in DEPTH_CLASSES))
    control_day0(wet)

    sels = {(rn, dn): _sel(rs, dm) for rn, rs in REGIONS for dn, dm in DEPTH_CLASSES}
    wreg = {rn: np.where(_sel(rs, np.ones(A.NZ, bool)), w, 0.0) for rn, rs in REGIONS}
    hot_reg = {}

    rows = []                     # per-horizon record
    zm = {"T": [], "S": []}       # zonal-mean difference (y,z) per horizon
    keep = {}                     # full 3-D difference at the figure horizons
    hot = {}
    d10 = {}
    d90 = {}
    d_prev = {}
    fig_days = [d for d in (10, 90, 180, 270, 360) if d in days]

    for day in days:
        lg = G.load_candidate(lego_npz(0), day)
        nm = load_nemo(0, day, fields=("tn", "sn", "un"))
        rec = {"day": day}
        for fld in ("T", "S"):
            control_finite(f"lego m0 {fld} day{day}", lg[fld], wet)
            control_finite(f"NEMO m0 {fld} day{day}", nm[fld], wet)
            # ZERO the difference off the wet mask.  rebuild() fills unwritten
            # restart regions with NaN and control_finite only polices the WET
            # mask, so a dry-cell NaN would reach `0.0 * nan = nan` and poison
            # every global sum.  The weight is already zero there, so this
            # changes no reported number -- it only removes the one route by
            # which a dry cell can affect a wet-cell statistic.
            d = np.where(wet, _f64(lg[fld]) - _f64(nm[fld]), 0.0)
            control_dtype((f"lego {fld} day{day}", lg[fld]),
                          (f"NEMO {fld} day{day}", nm[fld]),
                          (f"difference {fld} day{day}", d))
            if day == days[0] and fld == "T":
                control_mask_bite(d, w, wet)
            if day == 10:
                d10[fld] = d.copy()
                hot[fld] = hotspot_set(d, w)
                hot_reg[fld] = {rn: hotspot_set(d, wreg[rn]) for rn, _ in REGIONS}
            rec[f"{fld}_all"] = wrms(d, w)
            # INCREMENT NORM.  A flat rms is a statement about AMPLITUDE only:
            # a difference field that is completely re-drawn between two
            # horizons at constant amplitude has a flat rms too.  The norm of
            # the CHANGE in the difference field is what separates them, and
            # without it "the divergence saturates" cannot be said at all.
            rec[f"{fld}_incr_prev"] = (wrms(d - d_prev[fld], w)
                                       if fld in d_prev else float("nan"))
            rec[f"{fld}_incr_d90"] = (wrms(d - d90[fld], w)
                                      if fld in d90 else float("nan"))
            d_prev[fld] = d.copy()
            if day == 90:
                d90[fld] = d.copy()
            tot_sq = float((w * d * d).sum())
            for key, sel in sels.items():
                rec[f"{fld}_{key[0]}|{key[1]}"] = wrms(d, w, sel)
                # SHARE of the total volume-weighted d^2, so "where the
                # divergence is" is separable from "how intense it is there"
                if not tot_sq > 0.0:
                    raise SystemExit(f"FATAL: day {day} {fld} difference is "
                                     f"identically zero on the wet mask -- two "
                                     f"models cannot be bit-identical after "
                                     f"{day} days, so this is an artifact read "
                                     f"error, not a result")
                rec[f"{fld}_share_{key[0]}|{key[1]}"] = (
                    float((w * d * d)[sel].sum()) / tot_sq)
            for rn, rs in REGIONS:
                rec[f"{fld}_{rn}"] = wrms(d, w, _sel(rs, np.ones(A.NZ, bool)))
            for dn, dm in DEPTH_CLASSES:
                rec[f"{fld}_{dn}"] = wrms(d, w, _sel(slice(0, A.NY), dm))
            # Q2 statistics against the earliest scored horizon that both sides
            # carry.  Day 0 is the IDENTITY (its difference is the fp32 storage
            # quantum, not a divergence pattern), so it carries no Q2 row.
            if fld in d10:
                rec[f"{fld}_r_abs"] = wcorr(np.abs(d), np.abs(d10[fld]), w)
                rec[f"{fld}_r_signed"] = wcorr(d, d10[fld], w)
                rec[f"{fld}_retention"] = retention(d, w, hot[fld])
                hot_t = hotspot_set(d, w)
                # does the HOTSPOT SET itself move?  Volume overlap of this
                # horizon's own top-5%-by-volume set with day 10's.  Null = 0.05.
                rec[f"{fld}_hot_overlap"] = float(
                    w[hot_t & hot[fld]].sum() / w[hot[fld]].sum())
                # REVERSE retention: how much of the DAY-10 squared difference
                # sits inside THIS horizon's hotspot.  Forward retention alone
                # cannot distinguish "the divergence moved somewhere new" from
                # "it moved somewhere that was already enriched", and those are
                # different findings.  Same 0.05 null.
                rec[f"{fld}_retention_rev"] = retention(d10[fld], w, hot_t)
                rec[f"{fld}_selfret"] = retention(d, w, hot_t)
                for rn, _rs in REGIONS:
                    hr = hotspot_set(d, wreg[rn])
                    rec[f"{fld}_{rn}_retention_rev"] = retention(d10[fld], wreg[rn], hr)
                    rec[f"{fld}_{rn}_selfret"] = retention(d, wreg[rn], hr)
                # SUPPLEMENTARY, POST-HOC (not registered): the same three Q2
                # statistics computed INSIDE each region.  Needed because the
                # share table shows one box holds ~98% of the global d^2, so
                # the registered whole-domain statistics answer the question
                # for that box and for nothing else.  Labelled post-hoc.
                for rn, _rs in REGIONS:
                    wr = wreg[rn]
                    rec[f"{fld}_{rn}_r_abs"] = wcorr(np.abs(d), np.abs(d10[fld]), wr)
                    rec[f"{fld}_{rn}_retention"] = retention(d, wr, hot_reg[fld][rn])
                    rec[f"{fld}_{rn}_cent_row"] = centroid(d, wr)[0]
                    rec[f"{fld}_{rn}_cent_z"] = centroid(d, wr)[1]
            else:
                rec[f"{fld}_r_abs"] = float("nan")
                rec[f"{fld}_r_signed"] = float("nan")
                rec[f"{fld}_retention"] = float("nan")
                rec[f"{fld}_hot_overlap"] = float("nan")
            cr, cz = centroid(d, w)
            rec[f"{fld}_cent_row"], rec[f"{fld}_cent_z"] = cr, cz
            # per-row rms over the southern basin (Q3)
            rec[f"{fld}_rows"] = np.array(
                [wrms(d, w, _sel(slice(j, j + 1), np.ones(A.NZ, bool)))
                 for j in SOUTH_ROWS])
            # zonal mean of the difference, volume-weighted over longitude
            num = (w * d).sum(axis=1)
            den = w.sum(axis=1)
            zm[fld].append(np.where(den > 0, num / np.maximum(den, 1e-30), np.nan))
            if day in fig_days:
                keep[(fld, day)] = d
        # per-row transport deficit on the campaign's committed reduction
        control_finite(f"lego m0 u day{day}", lg["u"], A.umask)
        control_finite(f"NEMO m0 u day{day}", nm["u"], A.umask)
        full_deficit = row_deficit(lg["u"], nm["u"])
        rec["deficit_rows"] = full_deficit[SOUTH_ROWS]
        # C7a LINEARITY, and it is ONLY that.  group_transport is an einsum over
        # the row axis and _avg is a mean over longitudes, so the per-row split
        # summing back to the group total is an ALGEBRAIC IDENTITY of two linear
        # reductions -- it holds for garbage `u` just as well as for this one and
        # CANNOT FAIL on physics.  It is kept because it would catch a future
        # refactor that made the row split non-linear, and for NO other reason.
        # It is not the known-answer control; C7b below is.
        want = (D._avg(D.group_transport(lg["u"], A.umask, D.LAT_GROUPS[0][1]))
                - D._avg(D.group_transport(nm["u"], A.umask, D.LAT_GROUPS[0][1])))
        got = float(full_deficit[D.LAT_GROUPS[0][1]].sum())
        if not (abs(got - want) <= 1e-9 * max(abs(want), 1e-9) + 1e-12):
            raise SystemExit(
                f"FATAL: day {day} per-row deficits sum to {got:.9f} Sv but "
                f"acc_driver_decomp's own south-of-band group gap is "
                f"{want:.9f} Sv -- the row split is not the recorded reduction")
        rec["deficit_south_group"] = want
        # C7b KNOWN ANSWER, the real one: at day 360 the number this instrument
        # computes must equal the value the campaign ALREADY RECORDED in
        # a1387f1f7, compared IN CODE rather than read off a table by a human.
        # A probe that cannot reproduce an answer we already know is not usable
        # on one we do not.
        if day == 360:
            if not (abs(want - D360_SOUTH_GAP_RECORDED_SV) <= D360_SOUTH_GAP_TOL_SV):
                raise SystemExit(
                    f"FATAL: day-360 south-of-band gap is {want:.5f} Sv but the "
                    f"verdict run recorded {D360_SOUTH_GAP_RECORDED_SV:.5f} Sv "
                    f"(tolerance {D360_SOUTH_GAP_TOL_SV:.0e}). Either this probe "
                    f"is reading different artifacts or the recorded number is "
                    f"wrong -- both are findings, neither is a number to publish.")
            print(f"  [C7b known answer] day-360 south-of-band gap {want:.5f} Sv "
                  f"vs recorded {D360_SOUTH_GAP_RECORDED_SV:.5f} Sv "
                  f"(|d| = {abs(want - D360_SOUTH_GAP_RECORDED_SV):.2e})")
        # within-side chaotic floor: rms difference between each member and its
        # own side's control, RSS'd across the two sides (verdict360's convention)
        mem = {i: (G.load_candidate(lego_npz(i), day),
                   load_nemo(i, day, fields=("tn", "sn", "un")))
               for i in range(1, N_MEM)}
        for fld in ("T", "S"):
            lo, no = [], []
            for i in range(1, N_MEM):
                li, ni = mem[i][0][fld], mem[i][1][fld]
                control_finite(f"lego m{i} {fld} day{day}", li, wet)
                control_finite(f"NEMO m{i} {fld} day{day}", ni, wet)
                lo.append(wrms(_f64(li) - _f64(lg[fld]), w))
                no.append(wrms(_f64(ni) - _f64(nm[fld]), w))
            rec[f"{fld}_floor_lego"] = float(np.sqrt(np.mean(np.square(lo))))
            rec[f"{fld}_floor_nemo"] = float(np.sqrt(np.mean(np.square(no))))
            rec[f"{fld}_floor"] = float(np.hypot(rec[f"{fld}_floor_lego"],
                                                 rec[f"{fld}_floor_nemo"]))
            # THE FLOOR'S OWN GEOGRAPHY.  Reducing the ensemble spread to one
            # global number and throwing away where it lives makes the gap's
            # share table uninterpretable: if one model's difference from
            # ITSELF is concentrated in the same box, "98% of the divergence
            # lives there" is a statement about where any perturbation to this
            # configuration grows, not about the two models.  Member 1 of each
            # side, same mask, same weights, same reduction.
            df_l = np.where(wet, _f64(mem[1][0][fld]) - _f64(lg[fld]), 0.0)
            df_n = np.where(wet, _f64(mem[1][1][fld]) - _f64(nm[fld]), 0.0)
            for tag, df in (("floorL", df_l), ("floorN", df_n)):
                tsq = float((w * df * df).sum())
                for key, sel in sels.items():
                    rec[f"{fld}_{tag}_share_{key[0]}|{key[1]}"] = (
                        float((w * df * df)[sel].sum()) / tsq if tsq > 0 else 0.0)
                    rec[f"{fld}_{tag}_{key[0]}|{key[1]}"] = (
                        wrms(df, w, sel) if tsq > 0 else 0.0)
        # PER-ROW TRANSPORT FLOOR.  F1 rank-correlates against the per-row
        # deficit; without its floor there is no way to know how many of the 13
        # ranks are set by numbers below the chaotic noise, and an arbitrary
        # rank is a large perturbation on a 13-point rank correlation.
        rd_floor = []
        for i in range(1, N_MEM):
            rd_floor.append(np.abs(row_deficit(mem[i][0]["u"], lg["u"]))[SOUTH_ROWS])
            rd_floor.append(np.abs(row_deficit(mem[i][1]["u"], nm["u"]))[SOUTH_ROWS])
        rec["deficit_rows_floor"] = np.sqrt(np.mean(np.square(rd_floor), axis=0))
        rows.append(rec)
        print(f"  day {day:>3}  rms dT {rec['T_all']:.4e} K (floor "
              f"{rec['T_floor']:.2e})   rms dS {rec['S_all']:.4e} g/kg (floor "
              f"{rec['S_floor']:.2e})", flush=True)

    report(rows, days, w, out_dir)
    figures(rows, days, zm, keep, fig_days, wet, w, out_dir)
    return 0


# ------------------------------------------------------------------- report ---
def _tbl(title, header, lines):
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    print(header)
    for ln in lines:
        print(ln)


def _ratio(gap, floor):
    """gap/floor, printed as `n/a` when the floor is exactly zero.

    A zero floor is not an infinitely strong result: before the 1e-14 kick has
    grown, the three perturbed members can be BIT-IDENTICAL to their control in
    fp32 storage, so the denominator is a storage artifact.  verdict360 makes
    the same point for the transports; the ratio column is only readable once
    the kick has saturated.
    """
    return "n/a" if not floor > 0.0 else f"{gap / floor:.1f}"


def _amp_row(by, days, fld, d):
    """One row of the amplitude-vs-pattern table."""
    prev = by[days[days.index(d) - 1]][f"{fld}_all"]
    damp = abs(by[d][f"{fld}_all"] - prev)
    incr = by[d][f"{fld}_incr_prev"]
    return (f"{d:>5}{by[d][f'{fld}_all']:>13.4e}{damp:>17.4e}{incr:>16.4e}"
            f"{incr / max(damp, 1e-300):>9.1f}"
            f"{by[d][f'{fld}_incr_d90']:>19.4e}"
            f"{by[d][f'{fld}_incr_d90'] / by[d][f'{fld}_all']:>10.3f}")


def _snr_row(r, fld, day):
    """One row of the signal-to-noise share table: gap^2/floor^2 per box,
    renormalised.  The floor is the RSS of the two sides' member-1 rms in that
    box, matching the global floor's own convention."""
    vals = []
    for rn, _ in REGIONS:
        for dn, _ in DEPTH_CLASSES:
            gap = r[f"{fld}_{rn}|{dn}"]
            fl = float(np.hypot(r[f"{fld}_floorL_{rn}|{dn}"],
                                r[f"{fld}_floorN_{rn}|{dn}"]))
            vals.append((gap / fl) ** 2 if fl > 0 else 0.0)
    v = np.array(vals)
    tot = float(v.sum())
    if not tot > 0.0:
        raise SystemExit(f"FATAL: day {day} {fld} signal-to-noise is zero "
                         f"everywhere -- the floor cannot exceed the gap in "
                         f"every box at once")
    return f"{day:>5}" + "".join(f"{x:>13.4f}" for x in v / tot)


def report(rows, days, w, out_dir):
    by = {r["day"]: r for r in rows}

    for fld, unit in (("T", "K"), ("S", "g/kg")):
        _tbl(f"GROWTH CURVE -- volume-weighted rms of (lego - NEMO) {fld} [{unit}]. "
             f"Thickness-weighted: sqrt(sum V d^2 / sum V), V = e1t*e2t*e3t_0.",
             f"{'day':>5}{'ALL':>12}{'floor':>11}{'gap/floor':>11}"
             + "".join(f"{n[:12]:>13}" for n, _ in REGIONS)
             + "".join(f"{n.split()[0][:9]:>11}" for n, _ in DEPTH_CLASSES),
             [f"{d:>5}{by[d][f'{fld}_all']:>12.4e}{by[d][f'{fld}_floor']:>11.2e}"
              f"{_ratio(by[d][f'{fld}_all'], by[d][f'{fld}_floor']):>11}"
              + "".join(f"{by[d][f'{fld}_{n}']:>13.4e}" for n, _ in REGIONS)
              + "".join(f"{by[d][f'{fld}_{n}']:>11.3e}" for n, _ in DEPTH_CLASSES)
              for d in days])

        _tbl(f"GROWTH CURVE, region x depth class -- rms d{fld} [{unit}]",
             f"{'day':>5}" + "".join(
                 f"{(rn.split()[0][:5] + '/' + dn.split()[0][:4]):>13}"
                 for rn, _ in REGIONS for dn, _ in DEPTH_CLASSES),
             [f"{d:>5}" + "".join(f"{by[d][f'{fld}_{rn}|{dn}']:>13.4e}"
                                  for rn, _ in REGIONS for dn, _ in DEPTH_CLASSES)
              for d in days])

        _tbl("AMPLITUDE vs PATTERN -- is the difference FIELD settling, or only "
             "its SIZE?\n'change in rms' is how much the amplitude moved since "
             "the previous scored horizon; 'rms of the change'\nis the size of "
             "the difference between the two horizons' difference FIELDS.  A "
             "field that is completely\nre-drawn at constant amplitude has a "
             "flat rms too, so the first column alone can never support the\n"
             "word 'saturates'.  Ratio >> 1 means the pattern is turning over "
             "far faster than the amplitude is.",
             f"{'day':>5}{'rms':>13}{'|change in rms|':>17}"
             f"{'rms of change':>16}{'ratio':>9}{'rms of (d - d90)':>19}"
             f"{'.. / rms':>10}",
             [_amp_row(by, days, fld, d) for d in days[1:]])

        _tbl(f"WHERE IT IS -- share of the TOTAL volume-weighted d{fld}^2 held by "
             f"each region x depth class (rows sum to 1).\nThe rms table above says "
             f"how INTENSE the divergence is in a box; this says how much of it "
             f"the box HOLDS.",
             f"{'day':>5}" + "".join(
                 f"{(rn.split()[0][:5] + '/' + dn.split()[0][:4]):>13}"
                 for rn, _ in REGIONS for dn, _ in DEPTH_CLASSES),
             [f"{d:>5}" + "".join(f"{by[d][f'{fld}_share_{rn}|{dn}']:>13.4f}"
                                  for rn, _ in REGIONS for dn, _ in DEPTH_CLASSES)
              for d in days])

    for fld in ("T", "S"):
        _tbl(f"THE SAME SHARE TABLE FOR THE CHAOTIC FLOOR -- d{fld}^2 of ONE "
             f"MODEL against ITSELF (member 1 - member 0),\nsame mask, same "
             f"weights, same reduction.  If the floor is concentrated in the "
             f"same box as the gap, then\n'the divergence lives there' is a "
             f"statement about where ANY perturbation to this configuration "
             f"grows,\nnot about the two models.  L = legoESM side, N = NEMO "
             f"side.",
             f"{'day':>5}{'side':>6}" + "".join(
                 f"{(rn.split()[0][:5] + '/' + dn.split()[0][:4]):>13}"
                 for rn, _ in REGIONS for dn, _ in DEPTH_CLASSES),
             [f"{d:>5}{side:>6}" + "".join(
                 f"{by[d][f'{fld}_{tag}_share_{rn}|{dn}']:>13.4f}"
                 for rn, _ in REGIONS for dn, _ in DEPTH_CLASSES)
              for d in (days[-1],) for side, tag in (("L", "floorL"), ("N", "floorN"))])

        _tbl(f"SIGNAL-TO-NOISE SHARE, d{fld} -- each box's gap^2 divided by that "
             f"box's OWN floor^2, renormalised to\nsum to 1.  This asks WHERE "
             f"THE TWO MODELS DIFFER BY MORE THAN ONE OF THEM DIFFERS FROM "
             f"ITSELF, which is\na different question from where the raw "
             f"difference is largest, and the two need not have the same "
             f"answer.\nREAD ONLY THE SATURATED ROW. Before the 1e-14 kick has "
             f"grown, the denominator is a storage artifact, so\nthe day-90 row "
             f"is printed for contrast and is NOT usable as a measurement.",
             f"{'day':>5}" + "".join(
                 f"{(rn.split()[0][:5] + '/' + dn.split()[0][:4]):>13}"
                 for rn, _ in REGIONS for dn, _ in DEPTH_CLASSES),
             [_snr_row(by[d], fld, d) for d in (90, days[-1]) if d in by])

    # ---- QUESTION 2 -------------------------------------------------------
    print("\n" + "=" * 118)
    print("QUESTION 2 -- FIRST DEPARTURE.  Earliest scored horizon = day "
          f"{days[1] if days[0] == 0 else days[0]}.  Statistics and bars are "
          "the ones registered in\nPREREG_ts_divergence_atlas.md sec.3; the "
          "classification is NOT made here.")
    print("=" * 118)
    for fld in ("T", "S"):
        print(f"\n  {fld}: pattern correlation against the day-10 field, "
              f"d^2 centroid, and day-10 hotspot retention")
        print(f"    (registered bars: r >= {Q2_R_BAR}, retention >= {Q2_RET_BAR}; "
              f"null retention = {Q2_NULL}; spread if retention < {Q2_SPREAD_RET}\n"
              f"     AND the centroid moves > {Q2_SPREAD_ROWS} rows or > {Q2_SPREAD_M} m)")
        print("    'retention' is FORWARD (day-10 hotspot, this horizon's d^2). "
              "'rev' is REVERSE (this horizon's\n     hotspot, day-10's d^2), "
              "and 'self' is this horizon's own concentration. Forward alone "
              "cannot tell\n     'the divergence moved somewhere new' from 'it "
              "moved somewhere already enriched'; all three nulls are 0.05.")
        print(f"    {'day':>5}{'r(|d|,|d10|)':>14}{'r(d,d10)':>11}"
              f"{'ret fwd':>9}{'ret rev':>9}{'self':>8}{'overlap':>9}"
              f"{'cent row':>10}{'d row':>8}{'cent z':>9}{'d z':>8}")
        c0 = (by[10][f"{fld}_cent_row"], by[10][f"{fld}_cent_z"])
        for d in days:
            if d == 0:
                continue
            r = by[d]
            print(f"    {d:>5}{r[f'{fld}_r_abs']:>14.4f}{r[f'{fld}_r_signed']:>11.4f}"
                  f"{r[f'{fld}_retention']:>9.4f}{r[f'{fld}_retention_rev']:>9.4f}"
                  f"{r[f'{fld}_selfret']:>8.4f}{r[f'{fld}_hot_overlap']:>9.4f}"
                  f"{r[f'{fld}_cent_row']:>10.2f}"
                  f"{r[f'{fld}_cent_row'] - c0[0]:>+8.2f}{r[f'{fld}_cent_z']:>9.1f}"
                  f"{r[f'{fld}_cent_z'] - c0[1]:>+8.1f}")

    print("\n" + "=" * 118)
    print("QUESTION 2, SUPPLEMENTARY and POST-HOC -- the SAME statistics computed "
          "INSIDE each region.\nNOT pre-registered.  It exists because the share "
          "table shows one region x depth box holds ~98% of the\nglobal squared "
          "difference, so the registered whole-domain numbers above are that "
          "box's answer and\nsay nothing about the other two.  Same bars are "
          "printed for reference; they were registered for the\nwhole domain, so "
          "reading them here is a comparison, not a test.")
    print("=" * 118)
    for fld in ("T", "S"):
        print(f"\n  {fld}")
        print(f"    {'day':>5}" + "".join(
            f"{rn.split()[0][:5] + ' r':>10}{rn.split()[0][:5] + ' fwd':>12}"
            f"{rn.split()[0][:5] + ' rev':>12}{rn.split()[0][:5] + ' self':>13}"
            for rn, _ in REGIONS))
        for d in days:
            if d == 0:
                continue
            print(f"    {d:>5}" + "".join(
                f"{by[d][f'{fld}_{rn}_r_abs']:>10.3f}"
                f"{by[d][f'{fld}_{rn}_retention']:>12.3f}"
                f"{by[d][f'{fld}_{rn}_retention_rev']:>12.3f}"
                f"{by[d][f'{fld}_{rn}_selfret']:>13.3f}" for rn, _ in REGIONS))
        print("    day-360 centroid per region (row, depth m): " + "; ".join(
            f"{rn} ({by[360][f'{fld}_{rn}_cent_row']:.1f}, "
            f"{by[360][f'{fld}_{rn}_cent_z']:.0f})" for rn, _ in REGIONS))
        print("    day-10  centroid per region (row, depth m): " + "; ".join(
            f"{rn} ({by[10][f'{fld}_{rn}_cent_row']:.1f}, "
            f"{by[10][f'{fld}_{rn}_cent_z']:.0f})" for rn, _ in REGIONS))

    # ---- QUESTION 3 -------------------------------------------------------
    q3_days = [d for d in days if d != 0]
    print("\n" + "=" * 118)
    print("QUESTION 3 -- FEEDBACK FINGERPRINT.  Per-row tracer rms against the "
          "per-row zonal transport deficit\n(acc_driver_decomp.group_transport, "
          f"A.umask both sides, mean over lons 2..-2), southern basin rows "
          f"{SOUTH_ROWS[0]}..{SOUTH_ROWS[-1]}.")
    print("=" * 118)
    print(f"  F1 co-location: Spearman rank correlation across the "
          f"{len(SOUTH_ROWS)} rows (registered bar |rho| >= {Q3_RHO_BAR})")
    print("     Day 0 is EXCLUDED from F1/F2/F3: the transport deficit there is "
          "identically zero, so any\n     correlation or share against it is a "
          "statistic of rounding noise.")
    print("     The registered bar is applied IDENTICALLY to both fields and at "
          "EVERY horizon, two-sided.\n     A qualifier that admits one field's "
          "early negative values as a finding and excludes the other's is\n"
          "     not the registered criterion.")
    print("     The bar is the p<0.05 critical value for 13 INDEPENDENT samples. "
          "These 13 rows are adjacent\n     latitudes and are autocorrelated, so "
          "n_eff and the EXACT cyclic-shift p are printed beside it;\n"
          "     the shift null is what the conclusion should rest on (its "
          "resolution floor is 1/13 = 0.077).")
    print(f"    {'day':>5}{'rho(T,|D|)':>12}{'p_shift':>9}{'rho(S,|D|)':>12}"
          f"{'p_shift':>9}{'n_eff':>7}{'|rho|crit':>10}{'sum|D|':>10}"
          f"{'wall D':>10}{'main D':>10}{'rows<2fl':>9}")
    for d in q3_days:
        r = by[d]
        Dd = np.abs(r["deficit_rows"])
        wall = slice(0, N_WALL)                  # WALL_ROWS, positionally
        main = slice(N_WALL, len(SOUTH_ROWS))    # MAIN_ROWS, positionally
        rt, pt = shift_null(r["T_rows"], Dd)
        rs, ps = shift_null(r["S_rows"], Dd)
        ne = n_eff(r["T_rows"], Dd)
        crit = (float(2.0 / np.sqrt(max(ne - 2.0, 1e-9) + 4.0))
                if ne > 2.0 else float("inf"))
        below = int((Dd < 2.0 * r["deficit_rows_floor"]).sum())
        print(f"    {d:>5}{rt:>12.3f}{pt:>9.3f}{rs:>12.3f}{ps:>9.3f}"
              f"{ne:>7.1f}{crit:>10.3f}{Dd.sum():>10.4f}"
              f"{r['deficit_rows'][wall].sum():>+10.4f}"
              f"{r['deficit_rows'][main].sum():>+10.4f}{below:>9d}")
    print(f"    ('rows<2fl' = how many of the {len(SOUTH_ROWS)} rows carry a "
          f"deficit below twice their own ensemble floor;\n     those rows' "
          f"ranks are set by chaotic noise and each one perturbs the rank "
          f"correlation.)")

    # WRONG-TIME NULL: does the alignment happen to be TIME-SPECIFIC at all?  If
    # the day-t tracer profile matches the day-t deficit no better than it
    # matches some OTHER horizon's deficit, what is being measured is a standing
    # shape the two share, not a relationship in time.
    print("\n  F1b wrong-time null: rank of the MATCHED pairing among all "
          "horizon pairings.\n     A matched pairing that does not beat the "
          "mismatched ones is not evidence of a time relationship.")
    print(f"    {'day':>5}{'rho matched T':>16}{'rank/N':>9}{'mean mismatched T':>20}"
          f"{'rho matched S':>16}{'rank/N':>9}")
    for d in q3_days:
        out = [f"{d:>5}"]
        for fld in ("T", "S"):
            prof = by[d][f"{fld}_rows"]
            vals = [(dd, spearman(prof, np.abs(by[dd]["deficit_rows"])))
                    for dd in q3_days]
            matched = dict(vals)[d]
            order = sorted(vals, key=lambda kv: -kv[1])
            rank = 1 + [k for k, _ in order].index(d)
            mism = float(np.mean([v for dd, v in vals if dd != d]))
            out.append(f"{matched:>16.3f}{f'{rank}/{len(vals)}':>9}"
                       + (f"{mism:>20.3f}" if fld == "T" else ""))
        print("".join(out))
    print("    (last column: acc_driver_decomp's OWN south-of-band group gap, "
          "which the wall+main columns must sum to -- C7)")

    ks = (-2, -1, 0, 1, 2)
    # ONE COMMON SET OF HORIZONS for every k.  Averaging each k over a different
    # subset makes the argmax a protocol difference rather than a lag signal --
    # and the cadence is uneven (half the horizons sit in days 270-360), so the
    # subsets differ in WHERE they are concentrated, not only in count.  Day 0
    # is excluded on BOTH sides: its deficit is identically zero, so any pair
    # touching it correlates a profile against numerical noise.
    idx = [a for a in range(len(days))
           if days[a] != 0 and all(0 <= a + k < len(days) and days[a + k] != 0
                                   for k in ks)]
    if not idx:
        raise SystemExit("FATAL: no horizon supports the full lead/lag sweep")
    print(f"\n  F2 lead/lag: mean over ONE COMMON SET of {len(idx)} horizons of "
          f"the {len(SOUTH_ROWS)}-row Pearson r\n     between the tracer profile "
          f"at t and |D| at t+k, k in scored-horizon steps.")
    print("     SIGN, stated once because it is easy to invert: |D| is read at "
          "the LATER horizon for k>0,\n     so a peak at k<0 means the tracer "
          "matches an EARLIER deficit, i.e. MOMENTUM LEADS THE TRACER.\n"
          "     (PREREG sec.4 F2 states this the wrong way round; the formula "
          "is the registered one, its legend was not.)\n"
          "     The cadence is uneven, so the median day offset of each k is "
          "printed. 'sd' is the horizon-to-horizon\n     standard deviation of "
          "r, i.e. the scatter any difference between k rows has to beat.")
    print(f"    {'k':>4}{'median dt [d]':>15}{'mean r (T)':>13}{'sd (T)':>10}"
          f"{'mean r (S)':>13}{'sd (S)':>10}{'n':>8}{'r(k)-r(0) +- SE':>18}")
    r_at = {k: [pearson(by[days[a]]["T_rows"],
                        np.abs(by[days[a + k]]["deficit_rows"])) for a in idx]
            for k in ks}
    for k in ks:
        rT = r_at[k]
        rS = [pearson(by[days[a]]["S_rows"],
                      np.abs(by[days[a + k]]["deficit_rows"])) for a in idx]
        dt = [days[a + k] - days[a] for a in idx]
        # PAIRED difference against k=0 on the SAME horizons, with its standard
        # error.  "The profile declines from its peak" is not a finding until
        # the decline is compared with the scatter of the thing declining --
        # and the paired SE is one line, so "no error bar could be computed"
        # was never true.
        dif = np.array(rT) - np.array(r_at[0])
        se = float(np.std(dif, ddof=1) / np.sqrt(len(dif))) if k != 0 else 0.0
        pair = ("      --" if k == 0
                else f"{float(dif.mean()):>+7.3f}+-{se:.3f}")
        print(f"    {k:>+4}{np.median(dt):>15.0f}"
              f"{np.mean(rT):>13.3f}{np.std(rT, ddof=1):>10.3f}"
              f"{np.mean(rS):>13.3f}{np.std(rS, ddof=1):>10.3f}{len(rT):>8}"
              f"{pair:>18}")

    print("\n  F3 wall-vs-main share.  Tracer share = share of the southern-basin "
          "volume-weighted d^2\n     in wall rows 1-5; deficit share = the same "
          "rows' share of sum|D|.")
    print(f"    {'day':>5}{'T wall share':>15}{'S wall share':>15}{'|D| wall share':>17}")
    # The tracer share is a share of the total volume-weighted d^2, so each
    # row's mean-square must be re-weighted by that ROW's own wet volume before
    # summing.  Summing the per-row rms^2 unweighted would be an unweighted mean
    # over cells of unequal volume -- the exact reduction the campaign's
    # layer-averaging retraction forbids -- and these rows differ in volume by
    # ~20% through cos(lat) alone, before bathymetry.
    rv = row_volume(w)
    shares = []
    for d in q3_days:
        r = by[d]
        Dd = np.abs(r["deficit_rows"])
        sT = _wall_share(r["T_rows"], rv)
        sS = _wall_share(r["S_rows"], rv)
        if not float(Dd.sum()) > 0.0:
            raise SystemExit(f"FATAL: day {d} transport deficit is identically "
                             f"zero; day 0 should already be excluded here")
        sD = float(Dd[:N_WALL].sum() / Dd.sum())
        shares.append((sT, sS, sD))
        print(f"    {d:>5}{sT:>15.4f}{sS:>15.4f}{sD:>17.4f}")
    sh = np.array(shares)
    # TIME-WEIGHTED over the horizon spacing, and day 0 already excluded.  A
    # plain mean over these horizons is roughly a mean of the final quarter,
    # because half of them sit in days 270-360 -- calling that "the full-year
    # mean" would be a sampling artifact wearing the name of a yearly average.
    tw = np.gradient(np.asarray(q3_days, dtype=np.float64))
    tw = tw / tw.sum()
    print(f"    {'MEAN':>5}" + "".join(
        f"{float((sh[:, c] * tw).sum()):>15.4f}" if c < 2
        else f"{float((sh[:, c] * tw).sum()):>17.4f}" for c in range(3)))
    print(f"    (MEAN row is TIME-WEIGHTED over the uneven horizon spacing; the "
          f"unweighted mean of the same\n     column would be "
          f"{sh[:, 0].mean():.4f}/{sh[:, 1].mean():.4f}/{sh[:, 2].mean():.4f}, "
          f"i.e. mostly the final quarter, since half the\n     horizons sit in "
          f"days 270-360)")
    vol_null = float(rv[:N_WALL].sum() / rv.sum())
    print(f"    GEOMETRIC NULL: wall rows {WALL_ROWS[0]}-{WALL_ROWS[-1]} hold "
          f"{vol_null:.3f} of the basin's wet VOLUME.")
    print(f"    That -- not the {N_WALL / len(SOUTH_ROWS):.3f} share BY ROW "
          f"COUNT -- is the null a volume-weighted d^2 share is judged\n"
          f"    against; a share equal to {vol_null:.3f} means the divergence is "
          f"spread as the water is.")
    print("    The registered 'AVOIDS' criterion is a RELATIVE one (tracer "
          "share < half the deficit share) and is\n    scored as registered; "
          "it is a different statement from sitting below the geometric null, "
          "and a field\n    can clear one and not the other.")
    dwall = np.array([float(np.abs(by[d]["deficit_rows"])[:N_WALL].sum()
                            / np.abs(by[d]["deficit_rows"]).sum())
                      for d in q3_days])
    print(f"    The deficit's OWN wall share over the {len(q3_days)} scored "
          f"horizons ranges {dwall.min():.3f}..{dwall.max():.3f} "
          f"(mean {dwall.mean():.3f}) --\n    quote the range, not a band read "
          f"off the horizons that happen to be flat.")

    print("\n" + "=" * 118)
    print("SEASONAL CONFOUND -- read this before any statement about WHEN "
          "something happened.")
    print("=" * 118)
    print("  The restart is day 180 of a 360-day year, so elapsed time and "
          "season are DEGENERATE in this run:\n  the 'final quarter' (days "
          "270-360) is day-of-year 90-180, one particular season, and the early "
          "horizons\n  (days 10-60 = day-of-year 190-240) are another. Every "
          "sign change and every 'grows in the final\n  quarter' statement here "
          "is confounded with season and CANNOT be deconfounded from one year.\n"
          "  The only same-season pair in the run is day 0 and day 360, and day "
          "0 is the identity -- so no\n  seasonal control exists inside this "
          "run at all. The upstream verdict result carries the same\n  "
          "limitation and uses a full-cycle mean as its remedy; the same remedy "
          "applies to every series above.")
    print("  The horizon cadence is also uneven -- 30-day for the first three "
          "quarters, 10-day for the last --\n  so half the scored horizons sit "
          "in the final quarter. Any unweighted mean over horizons is\n  "
          "therefore mostly a mean of that quarter, which is why the F3 mean is "
          "time-weighted.")

    npz = os.path.join(out_dir, "ts_divergence_atlas.npz")
    np.savez_compressed(npz, days=np.array(days),
                        **{f"{k}": np.array([r[k] for r in rows])
                           for k in rows[0] if k != "day"})
    print(f"\n[out] {npz}")


# ------------------------------------------------------------------ figures ---
def figures(rows, days, zm, keep, fig_days, wet, w, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    by = {r["day"]: r for r in rows}
    lat = A.gphit[:, 25]
    depth = _f64(A.gdept1d)
    # characteristic levels: surface, and the level nearest 500 m and 2500 m
    k_surf = 0
    k_int = int(np.argmin(np.abs(depth - 500.0)))
    k_aby = int(np.argmin(np.abs(depth - 2500.0)))

    for fld, unit in (("T", "K"), ("S", "g/kg")):
        fig, ax = plt.subplots(3, 5, figsize=(26, 14))
        fig.suptitle(
            f"DINO verdict year: 3-D {fld} divergence atlas, legoESM - NEMO "
            f"[{unit}]\nsame day-180 restart, member 0 both sides; volume-"
            f"weighted (e1t*e2t*e3t_0); wet = tmask & land_mask; "
            f"probe ts_divergence_atlas.py, prereg PREREG_ts_divergence_atlas.md",
            fontsize=13)

        # --- row 1: zonal-mean depth-latitude difference ---------------------
        zall = np.array(zm[fld])
        vmax = float(np.nanpercentile(np.abs(zall), 99.5))
        for c, day in enumerate(fig_days):
            a = ax[0, c]
            z = zm[fld][days.index(day)]
            im = a.pcolormesh(lat, depth, z.T, cmap="RdBu_r",
                              norm=TwoSlopeNorm(0.0, -vmax, vmax), shading="auto")
            # log depth so the upper 200 m -- which holds most of the signal --
            # is legible; limits set AFTER the scale, because inverting first
            # and switching to log afterwards silently clips the top decade
            a.set_yscale("log")
            a.set_ylim(float(depth.max()), float(depth.min()))
            a.set_title(f"zonal-mean d{fld}, day {day}  (+-{vmax:.2g})")
            a.set_xlabel("latitude")
            if c == 0:
                a.set_ylabel("depth [m]")
            for j in (A.J0, A.J1):
                a.axvline(A.gphit[j, 25], color="k", lw=0.6, ls="--")
            fig.colorbar(im, ax=a, fraction=0.046)

        # --- row 2: horizontal maps ------------------------------------------
        panels = [(fig_days[0], k_surf), (90, k_surf), (360, k_surf),
                  (360, k_int), (360, k_aby)]
        for c, (day, k) in enumerate(panels):
            a = ax[1, c]
            if (fld, day) not in keep:
                a.axis("off")
                continue
            f2 = np.where(wet[:, :, k], keep[(fld, day)][:, :, k], np.nan)
            v = float(np.nanpercentile(np.abs(f2), 99.0))
            im = a.pcolormesh(np.arange(A.NX), lat, f2, cmap="RdBu_r",
                              norm=TwoSlopeNorm(0.0, -v, v), shading="auto")
            a.set_title(f"d{fld} day {day}, {depth[k]:.0f} m")
            a.set_xlabel("longitude index")
            if c == 0:
                a.set_ylabel("latitude")
            for j in (A.J0, A.J1):
                a.axhline(A.gphit[j, 25], color="k", lw=0.6, ls="--")
            fig.colorbar(im, ax=a, fraction=0.046)

        # --- row 3: growth curves + question statistics ----------------------
        a = ax[2, 0]
        for name, _ in REGIONS:
            a.plot(days, [by[d][f"{fld}_{name}"] for d in days], marker="o", label=name)
        a.plot(days, [by[d][f"{fld}_all"] for d in days], "k-", lw=2, label="all")
        a.plot(days, [by[d][f"{fld}_floor"] for d in days], "k:", lw=1.5,
               label="chaotic floor (RSS)")
        a.set_yscale("log")
        a.set_xlabel("day")
        a.set_ylabel(f"volume-weighted rms d{fld} [{unit}]")
        a.set_title("growth by region")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)

        a = ax[2, 1]
        for name, _ in DEPTH_CLASSES:
            a.plot(days, [by[d][f"{fld}_{name}"] for d in days], marker="s", label=name)
        a.plot(days, [by[d][f"{fld}_floor"] for d in days], "k:", lw=1.5, label="floor")
        a.set_yscale("log")
        a.set_xlabel("day")
        a.set_title("growth by depth class")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)

        a = ax[2, 2]
        dd = [d for d in days if d != 0]
        a.plot(dd, [by[d][f"{fld}_r_abs"] for d in dd], marker="o", label="r(|d|,|d10|)")
        a.plot(dd, [by[d][f"{fld}_retention"] for d in dd], marker="s",
               label="day-10 hotspot retention")
        a.axhline(Q2_R_BAR, color="C0", ls="--", lw=1)
        a.axhline(Q2_RET_BAR, color="C1", ls="--", lw=1)
        a.axhline(Q2_NULL, color="grey", ls=":", lw=1)
        a.set_ylim(-0.05, 1.05)
        a.set_xlabel("day")
        a.set_title("Q2: grow-in-place statistics\n(dashed = registered bars, "
                    "dotted = null)")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)

        a = ax[2, 3]
        jlat = A.gphit[SOUTH_ROWS, 25]
        for day, style in ((fig_days[0], ":"), (180, "--"), (360, "-")):
            if day in by:
                a.plot(jlat, by[day][f"{fld}_rows"], style, marker="o",
                       color="C0", label=f"rms d{fld} day {day}")
        a2 = a.twinx()
        for day, style in ((fig_days[0], ":"), (180, "--"), (360, "-")):
            if day in by:
                a2.plot(jlat, np.abs(by[day]["deficit_rows"]), style, marker="x",
                        color="C3", label=f"|transport deficit| day {day}")
        a.set_xlabel("latitude (southern basin rows 1-13)")
        a.set_ylabel(f"rms d{fld} [{unit}]", color="C0")
        a2.set_ylabel("|deficit| [Sv]", color="C3")
        a.set_title("Q3: tracer divergence vs transport deficit, per row")
        a.legend(fontsize=7, loc="upper left")
        a2.legend(fontsize=7, loc="upper right")
        a.grid(alpha=0.3)

        a = ax[2, 4]
        cr = [by[d][f"{fld}_cent_row"] for d in dd]
        cz = [by[d][f"{fld}_cent_z"] for d in dd]
        # the centroid row is fractional; interpolate the latitude rather than
        # rounding to a row, which would quantise a sub-degree drift away
        clat = np.interp(cr, np.arange(A.NY, dtype=np.float64), lat)
        sc = a.scatter(clat, cz, c=dd, cmap="viridis", s=45)
        a.plot(clat, cz, "-", color="grey", lw=0.8)
        crs = [by[d][f"{fld}_south basin_cent_row"] for d in dd]
        czs = [by[d][f"{fld}_south basin_cent_z"] for d in dd]
        clats = np.interp(crs, np.arange(A.NY, dtype=np.float64), lat)
        a.scatter(clats, czs, c=dd, cmap="viridis", s=45, marker="^")
        a.plot(clats, czs, "--", color="grey", lw=0.8)
        a.invert_yaxis()
        a.set_xlabel("centroid latitude")
        a.set_ylabel("centroid depth [m]")
        a.set_title("Q2: where the squared difference sits\n(circles = whole "
                    "domain, triangles = southern basin only)")
        fig.colorbar(sc, ax=a, fraction=0.046, label="day")
        a.grid(alpha=0.3)

        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = os.path.join(out_dir, f"ts_divergence_atlas_{fld}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[fig] {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out-dir", default=".")
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    self_test()
    os.makedirs(args.out_dir, exist_ok=True)
    return run(args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
