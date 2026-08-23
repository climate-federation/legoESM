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
  C4 NaN.  A single non-finite value on the wet mask is fatal.  No nanmean.
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
import acc_driver_decomp as D           # noqa: E402  group_transport, LAT_GROUPS, _avg
import acc_thermal_wind as A            # noqa: E402  mesh, masks, J0/J1
import acceptance_gate_90d as G         # noqa: E402  load_candidate (stamp refusals)
import kamm_twin_90d as T               # noqa: E402  the twin's OWN clock guard
from rebuild_nemo_restart import rebuild  # noqa: E402

LEGO_DIR = "/tmp/dino_verdict360"
SEEDS = (None, 1, 2, 3)
N_MEM = len(SEEDS)

# fp32 storage quantum of the legoESM snapshots on a ~20 K / ~35 g/kg field.
DAY0_BAR = 1e-5
# the registered kick: 1e-14 relative on the now-level T of NEMO members 1..3
KICK_REL_LO, KICK_REL_HI = 1e-16, 1e-12

# ---- registered partitions (PREREG sec.2) -----------------------------------
UPPER_M, DEEP_M = 200.0, A.DEEP_M           # 200 m, 1400 m
WALL_ROWS = slice(1, 6)                     # southern_circulation_budget's 1-5
MAIN_ROWS = slice(6, 14)                    # its main rows 6-13
SOUTH_ROWS = list(range(1, A.J0))           # basin rows 1..13 (row 0 is dry)
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
    tot = ww.sum()
    if tot <= 0.0:
        return float("nan")
    return float(np.sqrt(float((ww * d * d).sum()) / float(tot)))


def wcorr(a, b, w):
    """Volume-weighted Pearson correlation of two fields over the weight
    support."""
    tot = w.sum()
    ma = float((w * a).sum()) / tot
    mb = float((w * b).sum()) / tot
    da, db = a - ma, b - mb
    cov = float((w * da * db).sum()) / tot
    va = float((w * da * da).sum()) / tot
    vb = float((w * db * db).sum()) / tot
    if va <= 0.0 or vb <= 0.0:
        return float("nan")
    return float(cov / np.sqrt(va * vb))


def centroid(d, w):
    """Volume-weighted centroid of d^2 in (T-row index, depth [m])."""
    m = w * d * d
    tot = m.sum()
    if tot <= 0.0:
        return float("nan"), float("nan")
    rows = np.arange(A.NY, dtype=np.float64)[:, None, None]
    return (float((m * rows).sum()) / float(tot),
            float((m * _f64(A.gdept0)).sum()) / float(tot))


def hotspot_set(d0, w, frac=Q2_NULL):
    """The cells holding the top `frac` of the WET VOLUME when ranked by |d0|.
    Ranking is by intensity, selection is by cumulative volume, so the set's
    null share of any later sum-of-w*d^2 is exactly `frac`."""
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
    return float(m[hot].sum()) / tot if tot > 0.0 else float("nan")


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(_f64(v))).astype(np.float64)  # noqa: E731
    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def pearson(x, y):
    x, y = _f64(x) - np.mean(x), _f64(y) - np.mean(y)
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


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
    print(f"  NEMO member dirs        : {[os.path.realpath(nemo_dir(i)) for i in range(N_MEM)]}")
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


def control_kick():
    print("[C2 KICK] NEMO members 1-3 vs member 0 at day 0, relative on tn")
    t0 = load_nemo(0, 0)["T"]
    fin = np.isfinite(t0)
    for i in range(1, N_MEM):
        ti = load_nemo(i, 0)["T"]
        rel = np.abs(ti[fin] - t0[fin]) / np.maximum(np.abs(t0[fin]), 1e-12)
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
    checks = [("global wrms", lambda x: wrms(x, w)),
              ("south-upper wrms", lambda x: wrms(x, w, _sel(REGIONS[0][1], DEPTH_CLASSES[0][1]))),
              ("centroid row", lambda x: centroid(x, w)[0]),
              ("centroid depth", lambda x: centroid(x, w)[1])]
    for name, fn in checks:
        a, b = fn(d), fn(poisoned)
        rel = abs(b - a) / max(abs(a), 1e-30)
        print(f"  {name:<18} clean {a:.12e}  planted {b:.12e}  rel {rel:.2e}")
        if rel > 1e-12:
            raise SystemExit(f"FATAL: the mask does not bite -- {name} moved "
                             f"{rel:.2e} when a DRY cell was poisoned")
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
    hot = hotspot_set(flat, ww3)
    assert abs(retention(flat, ww3, hot) - float(ww3[hot].sum() / ww3.sum())) < 1e-12
    print(f"  retention of a FLAT field {retention(flat, ww3, hot):.4f} "
          f"(== its volume share, the {Q2_NULL} null)")
    spike = np.zeros_like(dd)
    spike[:2, :, :] = 10.0
    hot2 = hotspot_set(spike, ww3)
    assert retention(spike, ww3, hot2) > 0.9
    print(f"  retention of a CONCENTRATED field {retention(spike, ww3, hot2):.4f}")

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
    control_kick()

    lego0 = np.load(lego_npz(0))
    wet, w = build_weights(lego0["land_mask"])
    control_dtype(("volume weight", w), ("e3t_0", A.e3t0), ("gdept_0", A.gdept0),
                  ("gdept_1d", A.gdept1d))
    print(f"[mask] {int(wet.sum())} wet T-cells of {wet.size}; "
          f"total volume {w.sum():.6e} m3")
    print(f"[regions] " + "; ".join(
        f"{n} rows {s.start}..{s.stop - 1}" for n, s in REGIONS))
    print(f"[depths]  " + "; ".join(
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
    fig_days = [d for d in (10, 90, 180, 270, 360) if d in days]

    for day in days:
        lg = G.load_candidate(lego_npz(0), day)
        nm = load_nemo(0, day, fields=("tn", "sn", "un"))
        rec = {"day": day}
        for fld in ("T", "S"):
            control_finite(f"lego m0 {fld} day{day}", lg[fld], wet)
            control_finite(f"NEMO m0 {fld} day{day}", nm[fld], wet)
            d = _f64(lg[fld]) - _f64(nm[fld])
            if day == days[0] and fld == "T":
                control_mask_bite(d, w, wet)
            if day == 10:
                d10[fld] = d.copy()
                hot[fld] = hotspot_set(d, w)
                hot_reg[fld] = {rn: hotspot_set(d, wreg[rn]) for rn, _ in REGIONS}
            rec[f"{fld}_all"] = wrms(d, w)
            tot_sq = float((w * d * d).sum())
            for key, sel in sels.items():
                rec[f"{fld}_{key[0]}|{key[1]}"] = wrms(d, w, sel)
                # SHARE of the total volume-weighted d^2, so "where the
                # divergence is" is separable from "how intense it is there"
                rec[f"{fld}_share_{key[0]}|{key[1]}"] = (
                    float((w * d * d)[sel].sum()) / tot_sq if tot_sq > 0 else float("nan"))
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
                # does the HOTSPOT SET itself move?  Volume overlap of this
                # horizon's own top-5%-by-volume set with day 10's.  Null = 0.05.
                rec[f"{fld}_hot_overlap"] = float(
                    w[hotspot_set(d, w) & hot[fld]].sum() / w[hot[fld]].sum())
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
        # C7 INSTRUMENT VALIDATION: the per-row split must reproduce the number
        # the campaign already recorded.  The 13 basin rows plus the dry row 0
        # ARE acc_driver_decomp's "south of band" group, so their sum must equal
        # that group's own transport gap.  A probe that cannot reproduce a known
        # answer is not usable on an unknown one.
        want = (D._avg(D.group_transport(lg["u"], A.umask, D.LAT_GROUPS[0][1]))
                - D._avg(D.group_transport(nm["u"], A.umask, D.LAT_GROUPS[0][1])))
        got = float(full_deficit[D.LAT_GROUPS[0][1]].sum())
        if abs(got - want) > 1e-9 * max(abs(want), 1e-9) + 1e-12:
            raise SystemExit(
                f"FATAL: day {day} per-row deficits sum to {got:.9f} Sv but "
                f"acc_driver_decomp's own south-of-band group gap is "
                f"{want:.9f} Sv -- the row split is not the recorded reduction")
        rec["deficit_south_group"] = want
        # within-side chaotic floor: rms difference between each member and its
        # own side's control, RSS'd across the two sides (verdict360's convention)
        for fld in ("T", "S"):
            lo, no = [], []
            for i in range(1, N_MEM):
                li = G.load_candidate(lego_npz(i), day)[fld]
                ni = load_nemo(i, day, fields=("tn", "sn"))[fld]
                control_finite(f"lego m{i} {fld} day{day}", li, wet)
                control_finite(f"NEMO m{i} {fld} day{day}", ni, wet)
                lo.append(wrms(_f64(li) - _f64(lg[fld]), w))
                no.append(wrms(_f64(ni) - _f64(nm[fld]), w))
            rec[f"{fld}_floor_lego"] = float(np.sqrt(np.mean(np.square(lo))))
            rec[f"{fld}_floor_nemo"] = float(np.sqrt(np.mean(np.square(no))))
            rec[f"{fld}_floor"] = float(np.hypot(rec[f"{fld}_floor_lego"],
                                                 rec[f"{fld}_floor_nemo"]))
        rows.append(rec)
        print(f"  day {day:>3}  rms dT {rec['T_all']:.4e} K (floor "
              f"{rec['T_floor']:.2e})   rms dS {rec['S_all']:.4e} g/kg (floor "
              f"{rec['S_floor']:.2e})", flush=True)

    report(rows, days, out_dir)
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


def report(rows, days, out_dir):
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
        print(f"    {'day':>5}{'r(|d|,|d10|)':>15}{'r(d,d10)':>12}"
              f"{'retention':>12}{'hot overlap':>13}{'cent row':>11}{'d row':>9}"
              f"{'cent z [m]':>12}{'d z [m]':>10}")
        c0 = (by[10][f"{fld}_cent_row"], by[10][f"{fld}_cent_z"])
        for d in days:
            if d == 0:
                continue
            r = by[d]
            print(f"    {d:>5}{r[f'{fld}_r_abs']:>15.4f}{r[f'{fld}_r_signed']:>12.4f}"
                  f"{r[f'{fld}_retention']:>12.4f}{r[f'{fld}_hot_overlap']:>13.4f}"
                  f"{r[f'{fld}_cent_row']:>11.2f}"
                  f"{r[f'{fld}_cent_row'] - c0[0]:>+9.2f}{r[f'{fld}_cent_z']:>12.1f}"
                  f"{r[f'{fld}_cent_z'] - c0[1]:>+10.1f}")

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
            f"{rn.split()[0][:6] + ' r':>12}{rn.split()[0][:6] + ' ret':>13}"
            for rn, _ in REGIONS))
        for d in days:
            if d == 0:
                continue
            print(f"    {d:>5}" + "".join(
                f"{by[d][f'{fld}_{rn}_r_abs']:>12.3f}"
                f"{by[d][f'{fld}_{rn}_retention']:>13.3f}" for rn, _ in REGIONS))
        print(f"    day-360 centroid per region (row, depth m): " + "; ".join(
            f"{rn} ({by[360][f'{fld}_{rn}_cent_row']:.1f}, "
            f"{by[360][f'{fld}_{rn}_cent_z']:.0f})" for rn, _ in REGIONS))
        print(f"    day-10  centroid per region (row, depth m): " + "; ".join(
            f"{rn} ({by[10][f'{fld}_{rn}_cent_row']:.1f}, "
            f"{by[10][f'{fld}_{rn}_cent_z']:.0f})" for rn, _ in REGIONS))

    # ---- QUESTION 3 -------------------------------------------------------
    print("\n" + "=" * 118)
    print("QUESTION 3 -- FEEDBACK FINGERPRINT.  Per-row tracer rms against the "
          "per-row zonal transport deficit\n(acc_driver_decomp.group_transport, "
          f"A.umask both sides, mean over lons 2..-2), southern basin rows "
          f"{SOUTH_ROWS[0]}..{SOUTH_ROWS[-1]}.")
    print("=" * 118)
    print(f"  F1 co-location: Spearman rank correlation across the "
          f"{len(SOUTH_ROWS)} rows (registered bar |rho| >= {Q3_RHO_BAR})")
    print(f"    {'day':>5}{'rho(T,|D|)':>13}{'rho(S,|D|)':>13}"
          f"{'sum|D| [Sv]':>14}{'wall 1-5 D':>13}{'main 6-13 D':>13}"
          f"{'S-of-band gap':>16}")
    for d in days:
        r = by[d]
        Dd = np.abs(r["deficit_rows"])
        wall = slice(0, 5)                       # rows 1-5 within SOUTH_ROWS
        main = slice(5, len(SOUTH_ROWS))         # rows 6-13
        print(f"    {d:>5}{spearman(r['T_rows'], Dd):>13.3f}"
              f"{spearman(r['S_rows'], Dd):>13.3f}{Dd.sum():>14.4f}"
              f"{r['deficit_rows'][wall].sum():>+13.4f}"
              f"{r['deficit_rows'][main].sum():>+13.4f}"
              f"{r['deficit_south_group']:>+16.4f}")
    print("    (last column: acc_driver_decomp's OWN south-of-band group gap, "
          "which the wall+main columns must sum to -- C7)")

    print(f"\n  F2 lead/lag: mean over horizons of the {len(SOUTH_ROWS)}-row "
          f"Pearson r between the tracer profile at t and |D| at t+k,\n"
          f"     k in scored-horizon steps (the cadence is UNEVEN -- the median "
          f"day offset of each k is printed).")
    print(f"    {'k':>4}{'median dt [d]':>15}{'mean r (T)':>13}{'mean r (S)':>13}{'n pairs':>10}")
    for k in (-2, -1, 0, 1, 2):
        rT, rS, dt = [], [], []
        for a in range(len(days)):
            b = a + k
            if b < 0 or b >= len(days) or days[a] == 0:
                continue
            Dd = np.abs(by[days[b]]["deficit_rows"])
            rT.append(pearson(by[days[a]]["T_rows"], Dd))
            rS.append(pearson(by[days[a]]["S_rows"], Dd))
            dt.append(days[b] - days[a])
        print(f"    {k:>+4}{np.median(dt) if dt else float('nan'):>15.0f}"
              f"{np.mean(rT):>13.3f}{np.mean(rS):>13.3f}{len(rT):>10}")

    print(f"\n  F3 wall-vs-main share.  Tracer share = share of the southern-basin "
          f"volume-weighted d^2\n     in wall rows 1-5; deficit share = the same "
          f"rows' share of sum|D|.")
    print(f"    {'day':>5}{'T wall share':>15}{'S wall share':>15}{'|D| wall share':>17}")
    shares = []
    for d in days:
        r = by[d]
        Dd = np.abs(r["deficit_rows"])
        sT = float((r["T_rows"][:5] ** 2).sum() / (r["T_rows"] ** 2).sum())
        sS = float((r["S_rows"][:5] ** 2).sum() / (r["S_rows"] ** 2).sum())
        sD = float(Dd[:5].sum() / Dd.sum()) if Dd.sum() > 0 else float("nan")
        shares.append((sT, sS, sD))
        print(f"    {d:>5}{sT:>15.4f}{sS:>15.4f}{sD:>17.4f}")
    sh = np.array(shares)
    print(f"    {'MEAN':>5}{sh[:, 0].mean():>15.4f}{sh[:, 1].mean():>15.4f}"
          f"{sh[:, 2].mean():>17.4f}")
    print(f"    (the 5 wall rows are {5 / len(SOUTH_ROWS):.3f} of the rows by count; "
          f"'AVOIDS' was registered as tracer share < half the deficit share)")

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
