#!/usr/bin/env python
"""#1455 basin-budget lane, PHASE 1: the SOUTHERN-BASIN transport gap decomposed
by SEASON, by LATITUDE ROW and by DEPTH -- entirely offline, from the verdict
run's eight member states already on disk.

Pre-registration: ``PREREG_basin_seasonal_decomp.md``.  It was WRITTEN before
this file existed and is committed in the SAME commit as this file; it is not a
separately-timestamped artifact, and its day-90 constant carries a logged
retraction (see RECORDED_GAP below).  Read it before citing anything here.

WHAT THIS IS, AND WHAT IT IS NOT.  ``verdict360.py`` established that one year
from a shared NEMO day-180 restart leaves the circumpolar channel matched to 17
parts per million and the southern basin short by ~0.95 Sv, along a trajectory
that is NOT monotonic: the deficit almost vanishes at day 270 and then grows
steeply.  This probe asks WHERE that gap lives -- in the calendar, in latitude,
and in the vertical -- so that the accumulated stage budget (a separate,
compute-bearing lane) knows which rows and which part of the year to look at.

IT NAMES NO MECHANISM.  A correlation between the gap and a forcing phase is a
correlation; every such statement printed here is labelled PLAUSIBLE by
construction and the header says so.  This probe prints tables; the verdict
sentences are written by a human from the values.

EVERYTHING IS IMPORTED, NOTHING IS RE-DERIVED:
  * transports         acc_driver_decomp.group_transport / LAT_GROUPS / _avg
                       (the reducer the recorded southern-basin numbers used)
  * geometry + masks   acc_thermal_wind  (mesh_mask e3t_0/e3t_1d/e2u/gphit,
                       tmask & the legoESM land_mask)
  * state loaders      acceptance_gate_90d.load_candidate / load_nemo_day90
  * seasonal forcing   legoesm.ocean.experiments.dino -- the SAME analytic
                       functions the legoESM member actually ran with, so the
                       overlay is the model's own forcing rather than a second
                       transcription of usrdef_sbc.F90

FIVE FORCING COLUMNS, TWO INDEPENDENT CHANNELS.  R1 prints c1, c2, Qsr, T*
and tau_u.  T* is EXACTLY affine in c2 over the southern basin (usrdef_sbc
case 4: ``ztstar_s = rn_tstar_s - 0.5*c2``, with a fixed meridional profile), so
r(T*) == -r(c2) identically; Qsr is a near-duplicate of c1 the same way.  And
``dino_wind_stress`` takes no time argument at all, so its printed zero range
is a STATEMENT ABOUT DINO's forcing DESIGN, read off a function signature and
the oracle source -- not a measurement of this run, and it could not have come
out otherwise.  The table is two channels and one structural fact, not a
five-way scan.

THE ONE NEW PIECE OF NUMERICS is ``bc_bt_rows``: the exact bottom-referenced
barotropic/baroclinic split of a transport, generalised from the channel band
to an arbitrary row slice.  ``acc_thermal_wind.bc_bt_band`` hard-codes the band
rows, and the southern basin is a different slice.  Self-check S1 below feeds
this function the band rows and requires it to reproduce ``bc_bt_band`` to
1e-12 Sv -- a check that fails if the generalisation drifted in any way.

SAMPLING.  The NEMO side exists only at its 19 restart dumps
(days 10/30/60/90/120/150/180/210/240/270 and then every 10 days to 360); the
legoESM side has every 10th day.  Every number here is on the 19 days BOTH
sides have, and the day list is printed.  No interpolation anywhere.

Usage
-----
  basin_seasonal_decomp.py --self-check     # the five gates, no member data
  basin_seasonal_decomp.py                  # score (uses/writes a cache)
  basin_seasonal_decomp.py --no-cache       # ignore the cache and reload
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# The forcing overlay is evaluated with the model's own jnp functions; without
# this the fp64 they explicitly request is silently truncated to fp32 (review
# finding 16).  Set before any jax import.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))
import acc_driver_decomp as D          # noqa: E402
import acc_thermal_wind as A           # noqa: E402
import acceptance_gate_90d as G        # noqa: E402

_avg = D._avg                          # the recorded reducer (mean, lons 2..-2)

# ------------------------------------------------------------------ protocol ---
# The 19 days at which BOTH sides have a state.  NEMO writes a restart at each;
# legoESM stores a 3-D snapshot every 10 days.  Fixed here, not discovered, so
# a missing member file is a fatal error rather than a silently shorter curve.
DAYS = (10, 30, 60, 90, 120, 150, 180, 210, 240, 270,
        280, 290, 300, 310, 320, 330, 340, 350, 360)
HORIZONS = (90, 180, 270, 360)
MEMBERS = ("m0_control", "m1_seed1", "m2_seed2", "m3_seed3")
LEGO_DIR = os.environ.get("DINO_V360_LEGO_DIR", "/tmp/dino_verdict360")
NEMO_DIR = f"{A.DINO}/RUN_VERDICT360_M%d"
CACHE = os.environ.get("DINO_BSD_CACHE", "/tmp/dino_basin_seasonal_decomp.npz")

# Run day 0 IS day-of-year 180 on DINO's 360-day calendar: the verdict members
# stamp seasonal_t0_seconds = 15552000 s = 180 d, and NEMO's own clock is
# absolute kt (the restart carries adatrj = 180.0).  Checked, not assumed:
# self-check S4 reads the stamp out of every legoESM member file.
T0_SECONDS_EXPECTED = 15552000.0
SEC_PER_DAY = 86400.0

# Recorded values the curve must reproduce, read off the VERDICT RUN's own
# report (/tmp/verdict360_report.txt, "SOUTH of band transport [Sv]" rows for
# the day-90 and day-360 horizons; verdict360.py, commit a1387f1f7).
#
# RETRACTION, logged here because it was caught by this gate rather than by
# reading: the first revision of this file used -0.440537 for day 90, taken
# from ``acc_driver_decomp.py``'s DOCSTRING.  That number belongs to a
# DIFFERENT run -- the 90-day gate2 twin -- not to the verdict run, and using
# it here would have compared two protocols.  The gate failed, which is what it
# is for.  Prose is a pointer, never a citable fact.
RECORDED_GAP = {90: -0.42578, 360: -0.95191}
RECORDED_FLOOR = {90: 1.450e-04, 360: 6.174e-02}
RECORDED_TOL = 1e-4


# ------------------------------------------------------------------ numerics ---
def bc_bt_rows(u, wet_u, rows):
    """EXACT split of the `rows` transport into a bottom-referenced BAROCLINIC
    part and a BAROTROPIC (reference-level) part [Sv per longitude, each].

    Generalises ``acc_thermal_wind.bc_bt_band`` from the channel band to an
    arbitrary row slice; identical arithmetic, identical e3t_0 weighting.
    u(z) = u_bot + (u(z) - u_bot); bc + bt is the row-slice transport exactly.
    """
    us = np.asarray(u, dtype=np.float64)[rows]
    ws, e3s = wet_u[rows], A.e3t0[rows]
    col = ws.any(axis=2)
    kbot = np.where(col, ws.sum(axis=2) - 1, 0)
    ubot = np.where(col, np.take_along_axis(us, kbot[:, :, None], axis=2)[:, :, 0], 0.0)
    H = np.sum(np.where(ws, e3s, 0.0), axis=2)
    e2 = np.asarray(A.e2u_col, dtype=np.float64)[rows]
    bt = np.einsum("ji,ji,j->i", ubot, H, e2) / 1e6
    bc = np.einsum("jik,j->i", np.where(ws, (us - ubot[:, :, None]) * e3s, 0.0), e2) / 1e6
    return bc, bt


def transport_moments(u, wet_u, rows):
    """(M0, M1) [Sv, Sv*m] per longitude over `rows`, e3t_0 weighting.

    M0 = int u dz * e2  -- the transport itself.
    M1 = int z u dz * e2 -- its first moment about the surface (z = gdept_0,
         positive DOWN), so M1/M0 is the depth at which the transport is
         centred.

    WHY THIS EXISTS ALONGSIDE ``bc_bt_rows``.  The bottom-referenced split
    multiplies ONE cell per column (the deepest wet u value) by the full water
    depth, so a bottom-cell difference of 1e-3 m/s is amplified by ~3300 m and
    lands in BOTH halves with opposite sign -- which is why the two halves are
    ~95% anti-correlated and why neither is independently interpretable.  M0/M1
    are integrals: no single cell can move them.  M1 is NOT the algebraic
    complement of M0, so the pair carries information the bt/bc pair does not.
    """
    us = np.asarray(u, dtype=np.float64)[rows]
    ws, e3s = wet_u[rows], A.e3t0[rows]
    e2 = np.asarray(A.e2u_col, dtype=np.float64)[rows]
    z = A.gdept0[rows]
    m0 = np.einsum("jik,j->i", np.where(ws, us * e3s, 0.0), e2) / 1e6
    m1 = np.einsum("jik,j->i", np.where(ws, us * e3s * z, 0.0), e2) / 1e6
    # per-LEVEL transport [Sv per level], summed over rows and longitudes: the
    # vertical map whose first moment is M1/M0.  Sums to _avg-reduced M0 only
    # under a mean reducer, which is what _avg is.
    lev = np.einsum("jik,j->k", np.where(ws, us * e3s, 0.0)[:, 2:-2, :],
                    e2) / 1e6 / (us.shape[1] - 4)
    return m0, m1, lev


def row_transport(u, wet_u, rows):
    """Per-T-ROW transport [Sv], full-section e3t_1d weighting, already reduced
    over longitude with the recorded reducer.  Summing over rows reproduces
    ``group_transport`` reduced the same way (self-check S2)."""
    us = np.asarray(u, dtype=np.float64)[rows]
    per_row_lon = np.einsum("jik,k,j->ji", np.where(wet_u[rows], us, 0.0),
                            np.asarray(A.e3t1d, dtype=np.float64),
                            np.asarray(A.e2u_col, dtype=np.float64)[rows]) / 1e6
    return np.array([_avg(per_row_lon[j]) for j in range(per_row_lon.shape[0])])


# ------------------------------------------------------------------ forcing ---
def forcing_channels(days):
    """The DINO seasonal forcing at each run day, evaluated with the model's OWN
    functions (legoesm.ocean.experiments.dino), reduced over the southern-basin
    T-rows with area weights.

    Returns a dict of (n_days,) arrays.  ``tau_u`` is a STRUCTURAL fact, not a
    measurement: ``dino_wind_stress`` takes no time argument, so its zero range
    is true by signature for any run.  It is printed because the
    pre-registration needed the wind channel ruled out in the same table, and
    the ruling actually comes from the oracle source (usrdef_sbc.F90:162-163
    and 221, under nn_forcingtype=4), not from this column.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from legoesm.ocean.experiments.dino import (
        DINOConfig, dino_Q_sr_seasonal, dino_T_star_seasonal,
        dino_seasonal_cosines, dino_wind_stress)
    cfg = DINOConfig()
    south = slice(0, A.J0)
    lat_t = np.asarray(A.gphit, dtype=np.float64)[south]        # (ny_s, nx)
    wet_s = A.tmask[south, :, 0] & (LAND[south] > 0.5)          # surface wet
    # AREA weights, not a cell count: cos(lat) runs 0.344..0.424 across these
    # 14 rows, a 23% spread, so an unweighted mean is not a basin mean.
    e1t = np.asarray(A.mm["e1t"][0]).squeeze()[south]
    e2t = np.asarray(A.mm["e2t"][0]).squeeze()[south]
    w = np.where(wet_s, e1t * e2t, 0.0)
    wsum = float(w.sum())
    assert wsum > 0, "southern basin has no wet surface cells"

    out = {k: [] for k in ("c1", "c2", "qsr", "tstar", "tau_u", "doy")}
    for d in days:
        t = T0_SECONDS_EXPECTED + d * SEC_PER_DAY
        c1, c2 = dino_seasonal_cosines(t, cfg)
        qsr = np.asarray(dino_Q_sr_seasonal(lat_t, t, cfg), dtype=np.float64)
        ts = np.asarray(dino_T_star_seasonal(lat_t, t, cfg), dtype=np.float64)
        tau = np.asarray(dino_wind_stress(lat_t, cfg), dtype=np.float64)
        out["c1"].append(float(c1))
        out["c2"].append(float(c2))
        out["qsr"].append(float((qsr * w).sum() / wsum))
        out["tstar"].append(float((ts * w).sum() / wsum))
        out["tau_u"].append(float((tau * w).sum() / wsum))
        out["doy"].append((180 + d) % 360)
    return {k: np.asarray(v) for k, v in out.items()}


# ------------------------------------------------------------------ loaders ---
LAND = None      # set in main() from the member-0 npz; the SAME mask both sides


def lego_state(member, day):
    return G.load_candidate(f"{LEGO_DIR}/{member}.npz", day=day)


def nemo_state(i, day):
    kt = G.KT_RESTART + day * G.STEPS_PER_DAY
    return G.load_nemo_day90(run_dir=NEMO_DIR % i, kt=kt)


def transports(st, wet_u):
    """The three latitude groups (recorded reducer) plus the south group's own
    barotropic/baroclinic split and its per-row profile."""
    u = np.asarray(st["u"], dtype=np.float64)
    grp = [D.group_transport(u, wet_u, sl) for _, sl in D.LAT_GROUPS]
    tot = np.einsum("jik,k,j->i", np.where(wet_u, u, 0.0),
                    np.asarray(A.e3t1d, dtype=np.float64),
                    np.asarray(A.e2u_col, dtype=np.float64)) / 1e6
    # The partition residual is STORED, not just asserted, so the gate still
    # runs when the numbers come back from the cache -- the pre-registration
    # promises this check on every scored day of both sides, and an assert
    # inside a cache-miss branch does not deliver that (review finding 3).
    part = float(np.max(np.abs(sum(grp) - tot)))
    south = slice(0, A.J0)
    bc, bt = bc_bt_rows(u, wet_u, south)
    m0, m1, lev = transport_moments(u, wet_u, south)
    out = dict(g_south=_avg(grp[0]), g_band=_avg(grp[1]), g_north=_avg(grp[2]),
               s_bc=_avg(bc), s_bt=_avg(bt), part_resid=part,
               s_m0=_avg(m0), s_m1=_avg(m1), s_lev=lev,
               rows=row_transport(u, wet_u, south))
    # THERMAL WIND from this model's OWN density over the same rows.  This is
    # the density route: if the two models' southern-basin shear difference came
    # from different water masses (different convection, different mixing), the
    # bottom-referenced transport their densities PREDICT would differ by the
    # same amount as their velocities do.  Same functional both sides.
    wet_t = A.tmask & (LAND[:, :, None] > 0.5)
    rho = A.rho_of(st, wet_t)
    tw, _ = A.thermal_wind_rows(rho, wet_t, 0, A.J0 - 1)
    out["s_tw"] = _avg(tw)
    out["rho_mean"] = float(np.nanmean(rho[south]))
    return out


# -------------------------------------------------------------- self-checks ---
def self_checks(wet_u=None, verbose=True):
    """The pre-run gates.  Their strength is NOT uniform, and the header says so.

    S1 compares ``bc_bt_rows`` against ``bc_bt_band``, which shares its ``kbot``
    expression, and S1b (bc + bt == the transport) telescopes for ANY value of
    ``u_bot`` -- so NEITHER validates the deepest-wet-level index.  S1c exists
    for exactly that: it asserts the wet columns are contiguous from k = 0,
    which is the assumption ``kbot = ws.sum(axis=2) - 1`` rests on, and it fails
    if any column has a wet cell under a dry one.  S4 checks this repo's cosines
    against constants that live in the same module, so it is a REGRESSION gate,
    not an oracle comparison; the oracle values were read by hand from
    usrdef_sbc.F90:526-527 and are quoted in the pre-registration.
    S3 (the recorded-gap reproduction) is NOT here -- it needs the member states
    and runs in ``main``.
    """
    ok = []
    # S1 -- the rows-general split reproduces the recorded band split exactly.
    rng = np.random.default_rng(20260822)
    u = rng.normal(size=A.umask.shape) * 0.05
    band = slice(A.J0, A.J1 + 1)
    bc_r, bt_r = bc_bt_rows(u, A.umask, band)
    bc_b, bt_b = A.bc_bt_band(u, A.umask)
    e = max(float(np.abs(bc_r - bc_b).max()), float(np.abs(bt_r - bt_b).max()))
    ok.append(("S1 bc_bt_rows == bc_bt_band on the band rows",
               f"max|diff| = {e:.3e} Sv", e < 1e-12))
    # S1b -- and the split is exact: bc + bt is the row-slice transport itself.
    south = slice(0, A.J0)
    bc_s, bt_s = bc_bt_rows(u, A.umask, south)
    direct = np.einsum("jik,j->i", np.where(A.umask[south], u[south] * A.e3t0[south], 0.0),
                       np.asarray(A.e2u_col, dtype=np.float64)[south]) / 1e6
    e = float(np.abs(bc_s + bt_s - direct).max())
    ok.append(("S1b south bc + bt == the e3t_0 south transport",
               f"max|resid| = {e:.3e} Sv", e < 1e-10))
    # S2 -- the per-row profile sums to the group transport (same weights).
    grp = D.group_transport(u, A.umask, south)
    rows = row_transport(u, A.umask, south)
    e = abs(float(rows.sum()) - float(_avg(grp)))
    ok.append(("S2 per-row profile sums to group_transport",
               f"|diff| = {e:.3e} Sv", e < 1e-10))
    # S1c -- the assumption kbot rests on.  A wet cell beneath a dry one would
    # make ws.sum()-1 point at the wrong level, and neither S1 nor S1b would
    # see it (shared expression / telescopes over any u_bot).
    n_bad = 0
    for msk in (A.umask, A.tmask):
        cnt = msk.sum(axis=2)
        idx = np.arange(msk.shape[2])[None, None, :]
        n_bad += int((msk != (idx < cnt[:, :, None])).sum())
    ok.append(("S1c wet columns contiguous from k=0 (kbot's premise)",
               f"{n_bad} cells violate it", n_bad == 0))
    # S1d -- the per-LEVEL profile must reduce to the transport it decomposes.
    # Without this a divisor of NX instead of NX-4 would scale every per-level
    # number by 52/48 = 1.083 and every other gate would still pass.
    m0_, _m1_, lev_ = transport_moments(u, A.umask, south)
    e = abs(float(lev_.sum()) - float(_avg(m0_)))
    ok.append(("S1d per-level profile sums to _avg(M0)",
               f"|diff| = {e:.3e} Sv", e < 1e-12))
    # S4 -- the seasonal cosines carry the oracle's phase (c1 max at doy 171,
    # c2 at 201).  Scanned at 1-day resolution over a full year.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from legoesm.ocean.experiments.dino import DINOConfig, dino_seasonal_cosines
    cfg = DINOConfig()
    tt = np.arange(0, 360) * SEC_PER_DAY
    c1, c2 = dino_seasonal_cosines(tt, cfg)
    a1, a2 = int(np.argmax(np.asarray(c1))), int(np.argmax(np.asarray(c2)))
    ok.append(("S4 c1 peaks at day-of-year 171 (21 June)",
               f"argmax = {a1}", abs(a1 - 171) <= 1))
    ok.append(("S4 c2 peaks at day-of-year 201 (21 July)",
               f"argmax = {a2}", abs(a2 - 201) <= 1))
    # S5 -- every member state this probe will read exists on disk.
    missing = []
    for m in MEMBERS:
        p = f"{LEGO_DIR}/{m}.npz"
        if not os.path.exists(p):
            missing.append(p)
    for i in range(len(MEMBERS)):
        for d in DAYS:
            kt = G.KT_RESTART + d * G.STEPS_PER_DAY
            import glob as _g
            if not _g.glob(f"{NEMO_DIR % i}/DINO_{kt:08d}_restart*.nc"):
                missing.append(f"{NEMO_DIR % i}/DINO_{kt:08d}_restart*.nc")
    ok.append(("S5 all 4+4 members present at all 19 days",
               f"{len(missing)} missing" + (f" (first: {missing[0]})" if missing else ""),
               not missing))
    if verbose:
        print("=" * 100)
        print("PRE-RUN GATES -- see self_checks.__doc__ for what each one does and\n                 does NOT cover; they are not equally strong")
        print("=" * 100)
        for name, val, good in ok:
            print(f"  [{'PASS' if good else 'FAIL'}] {name:58s} {val}")
    return all(g for _, _, g in ok)


# ------------------------------------------------------------------ scoring ---
def spread(vals):
    """Sample std across members (the verdict run's primary spread statistic)."""
    return float(np.std(np.asarray(vals, dtype=np.float64), ddof=1))


def pearson(x, y):
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    dx, dy = np.linalg.norm(x), np.linalg.norm(y)
    if dx == 0 or dy == 0:
        return float("nan")
    return float(x @ y / (dx * dy))


def perm_p(x, y, n=20000, seed=20260822):
    """Two-sided permutation p-value for |r|.  19 samples of a TIME SERIES are
    autocorrelated, so this p-value is optimistic and is printed as an ordering
    aid, never as a significance claim."""
    r0 = abs(pearson(x, y))
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    c = sum(abs(pearson(x, rng.permutation(y))) >= r0 for _ in range(n))
    return (c + 1) / (n + 1)


def main():
    global LAND
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    sha = subprocess.run(["git", "-C", _DIR, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    _dirty = subprocess.run(["git", "-C", _DIR, "status", "--porcelain", "--",
                             os.path.abspath(__file__)],
                            capture_output=True, text=True).stdout.strip()
    print(f"[stamp] repo git HEAD {sha}"
          + (f"   *** THIS FILE IS {_dirty.split()[0]} AT THAT COMMIT -- the "
             "SHA does NOT identify the code that produced the numbers below ***"
             if _dirty else "   (this file is clean at that commit)"))
    print(f"[stamp] lego dir {LEGO_DIR}   nemo dirs {NEMO_DIR % 0} ..M3")
    print(f"[stamp] days {DAYS}")
    print(f"[stamp] run day 0 == day-of-year 180 (seasonal_t0 {T0_SECONDS_EXPECTED:.0f} s)")

    d0 = np.load(f"{LEGO_DIR}/{MEMBERS[0]}.npz")
    LAND = d0["land_mask"].astype(np.float64)
    for m in MEMBERS:
        t0 = float(np.load(f"{LEGO_DIR}/{m}.npz")["seasonal_t0_seconds"])
        if t0 != T0_SECONDS_EXPECTED:
            raise SystemExit(f"{m} seasonal_t0_seconds = {t0}, expected "
                             f"{T0_SECONDS_EXPECTED} -- the calendar mapping "
                             "in this probe would be wrong")
    print(f"[stamp] all {len(MEMBERS)} legoESM members stamp seasonal_t0 = "
          f"{T0_SECONDS_EXPECTED:.0f} s")

    if not self_checks():
        raise SystemExit("SELF-CHECKS FAILED -- no number below is trustworthy")
    if args.self_check:
        return

    wet_u = A.umask & (LAND[:, :, None] > 0.5)
    # verdict360 refuses a candidate whose land mask disagrees with NEMO's
    # surface tmask; this probe reads the same members, so it makes the same
    # check rather than printing "identical mask" on trust.
    mism = int(np.sum((LAND > 0.5) != A.tmask[:, :, 0]))
    if mism:
        raise SystemExit(f"legoESM land_mask disagrees with NEMO tmask surface "
                         f"on {mism} cells -- the two sides are not on one mask")
    print(f"[stamp] wet u-cells {int(wet_u.sum())}; legoESM land_mask == NEMO "
          f"surface tmask on all {LAND.size} cells")

    # ---- load / cache -------------------------------------------------------
    key = lambda side, i, d, f: f"{side}{i}_{d}_{f}"     # noqa: E731
    # The cache key must cover EVERY input that can change a stored number:
    # the two source directories (both env-overridable), the member list, the
    # day list, the wet mask, and this file's own source -- ``bc_bt_rows`` is
    # new numerics and an edit to it must invalidate the stored split.
    import hashlib
    _src = ""
    for _m in (sys.modules[__name__], A, D, G):
        _f = getattr(_m, "__file__", None)
        if _f:
            _src += hashlib.sha256(open(os.path.abspath(_f), "rb").read()).hexdigest()
    ck = hashlib.sha256(
        (repr(DAYS) + repr(MEMBERS) + LEGO_DIR + NEMO_DIR
         + hashlib.sha256(np.ascontiguousarray(wet_u).tobytes()).hexdigest()
         + _src).encode()).hexdigest()
    store = {}
    if os.path.exists(CACHE) and not args.no_cache:
        z = np.load(CACHE)
        if "cache_key" in z.files and str(z["cache_key"]) == ck:
            store = {k: z[k] for k in z.files if k != "cache_key"}
            print(f"[cache] reusing {CACHE} ({len(store)} entries, key {ck[:12]})")
        else:
            print(f"[cache] {CACHE} does not match this configuration -- reloading")
    need = [(s, i, d) for s in "LN" for i in range(len(MEMBERS)) for d in DAYS
            if key(s, i, d, "g_south") not in store]
    if need:
        t_start = time.time()
        print(f"[load] {len(need)} states to read ...", flush=True)
        for n, (s, i, d) in enumerate(need):
            st = lego_state(MEMBERS[i], d) if s == "L" else nemo_state(i, d)
            tr = transports(st, wet_u)
            for f, v in tr.items():
                store[key(s, i, d, f)] = np.asarray(v)
            if (n + 1) % 10 == 0:
                print(f"  {n+1}/{len(need)}  ({time.time()-t_start:.0f}s)", flush=True)
        np.savez_compressed(CACHE, cache_key=np.asarray(ck), **store)
        print(f"[load] done in {time.time()-t_start:.0f}s, cached to {CACHE}")

    def get(s, i, d, f):
        return store[key(s, i, d, f)]

    # The pre-registered partition check, run over every stored state whether
    # it came from disk or from the cache.
    worst = max(float(get(sd, i, d, "part_resid"))
                for sd in "LN" for i in range(len(MEMBERS)) for d in DAYS)
    n_states = 2 * len(MEMBERS) * len(DAYS)
    ok_part = worst < 1e-9
    print(f"  [{'PASS' if ok_part else 'FAIL'}] S2b the three latitude groups sum "
          f"to the full section on all {n_states} states: worst |resid| = "
          f"{worst:.3e} Sv")
    if not ok_part:
        raise SystemExit("S2b FAILED -- the latitude partition is not additive")

    # ---- S3: the recorded gaps must come back out ---------------------------
    for d, rec in RECORDED_GAP.items():
        g = float(get("L", 0, d, "g_south")) - float(get("N", 0, d, "g_south"))
        sl = spread([float(get("L", i, d, "g_south")) for i in range(4)])
        sn = spread([float(get("N", i, d, "g_south")) for i in range(4)])
        fl = float(np.hypot(sl, sn))
        good = (abs(g - rec) < RECORDED_TOL
                and abs(fl - RECORDED_FLOOR[d]) / RECORDED_FLOOR[d] < 1e-3)
        print(f"  [{'PASS' if good else 'FAIL'}] S3 day-{d} south gap {g:+.6f} Sv "
              f"vs recorded {rec:+.6f}   floor {fl:.4e} vs recorded "
              f"{RECORDED_FLOOR[d]:.4e}")
        if not good:
            raise SystemExit("S3 FAILED -- this probe does not reproduce the "
                             "verdict run's own southern-basin gap")

    # ---- R1: the gap curve vs the calendar ----------------------------------
    F = forcing_channels(DAYS)
    gap, floor = [], []
    for d in DAYS:
        gap.append(float(get("L", 0, d, "g_south")) - float(get("N", 0, d, "g_south")))
        sl = spread([float(get("L", i, d, "g_south")) for i in range(4)])
        sn = spread([float(get("N", i, d, "g_south")) for i in range(4)])
        floor.append(float(np.hypot(sl, sn)))
    gap, floor = np.asarray(gap), np.asarray(floor)

    print()
    print("=" * 100)
    print("R1  THE SOUTHERN-BASIN GAP THROUGH THE YEAR, AGAINST DINO'S OWN FORCING")
    print("=" * 100)
    print("gap = legoESM m0 - NEMO m0, south-of-band transport, mean reducer over lons 2..-2.")
    print("floor = RSS of the two 4-member sample stds at the SAME day.  season = austral.")
    print()
    print(f"{'day':>5}{'doy':>6}{'season':>10}{'gap[Sv]':>11}{'floor':>11}"
          f"{'g/f':>13}{'c1':>8}{'c2':>8}{'Qsr[W/m2]':>11}{'T*[C]':>8}"
          f"{'tau[Pa]':>9}")
    seas = lambda doy: ("summer" if doy < 60 or doy >= 330 else      # noqa: E731
                        "autumn" if doy < 150 else
                        "winter" if doy < 240 else "spring")
    for k, d in enumerate(DAYS):
        print(f"{d:>5}{F['doy'][k]:>6.0f}{seas(F['doy'][k]):>10}{gap[k]:>11.4f}"
              f"{floor[k]:>11.2e}{gap[k]/floor[k]:>13.1f}"
              f"{F['c1'][k]:>8.3f}{F['c2'][k]:>8.3f}{F['qsr'][k]:>11.2f}"
              f"{F['tstar'][k]:>8.3f}{F['tau_u'][k]:>9.4f}")
    print()
    print(f"  wind stress over the basin: min {F['tau_u'].min():.6f} max "
          f"{F['tau_u'].max():.6f} Pa  (range {np.ptp(F['tau_u']):.3e})")

    # envelope removal: the gap is amplitude x shape, so regress out a linear
    # trend in run day and correlate the RESIDUAL with each forcing channel.
    t = np.asarray(DAYS, float)
    P = np.polyfit(t, gap, 1)
    resid = gap - np.polyval(P, t)
    print()
    print(f"  linear envelope   gap ~ {P[0]:+.5f} Sv/day * day {P[1]:+.4f}"
          f"   (r = {pearson(t, gap):+.3f})")
    print()
    print(f"  {'channel':<28}{'r(gap)':>10}{'p':>10}{'r(resid)':>12}{'p':>10}")
    for ch, lab in (("c1", "solar phase c1 (21 Jun)"),
                    ("c2", "T* phase c2 (21 Jul)"),
                    ("qsr", "basin-mean Qsr [W/m2]"),
                    ("tstar", "basin-mean T* [degC]"),
                    ("tau_u", "basin-mean tau_u [Pa]")):
        r1, r2 = pearson(F[ch], gap), pearson(F[ch], resid)
        if np.isnan(r1):
            print(f"  {lab:<28}{'constant -- no seasonal cycle exists in this channel':>10}")
            continue
        print(f"  {lab:<28}{r1:>10.3f}{perm_p(F[ch], gap):>10.4f}"
              f"{r2:>12.3f}{perm_p(F[ch], resid):>10.4f}")
    print()
    print("  These are CORRELATIONS over 19 autocorrelated samples.  They order")
    print("  candidates; they name no cause.  Any sentence built on them is PLAUSIBLE.")
    rbest = max(abs(pearson(F[ch], resid)) for ch in ("c1", "c2", "qsr", "tstar"))
    # The registered CONFIRM has TWO clauses: |r| >= 0.7 AND the extremum in
    # austral winter.  Testing |r| alone would let a summer extremum print
    # CONFIRMED, so the phase clause is evaluated too.
    _worst = int(np.argmin(gap))
    _winter = F["doy"][_worst] >= 150 and F["doy"][_worst] < 240
    q1 = ("CONFIRMED seasonal" if (rbest >= 0.7 and _winter) else
          "NO VERDICT (|r| clears 0.7 but the extremum is not in austral winter)"
          if rbest >= 0.7 else
          "REFUTED seasonal" if rbest < 0.3 else "NO VERDICT (registered dead zone)")
    print(f"  most negative gap at day-of-year {F['doy'][_worst]:.0f} "
          f"(austral winter: {_winter})")
    print()
    print(f"  [Q1, per the pre-registered bins: CONFIRM >= 0.7, REFUTE < 0.3]")
    print(f"  best |r| against any forcing channel, envelope removed = {rbest:.3f}")
    print(f"  => {q1}")
    print("  Only TWO of the four channels are independent: T* is exactly affine in")
    print("  c2 over this basin and Qsr is near-affine in c1, so r(T*) == -r(c2).")
    print("  And ONE seasonal cycle cannot separate 'seasonal' from 'a curve that is")
    print("  not linear in time': after a LINEAR detrend, any smooth residual")
    print("  correlates ~0.5 with a cosine of that same single period.  The effective")
    print("  degrees of freedom here are ~3-4, not 19.")

    # ---- R2: latitude rows ---------------------------------------------------
    print()
    print("=" * 100)
    print("R2  WHERE IN LATITUDE -- per-T-row transport gap inside the southern basin")
    print("=" * 100)
    lat_row = np.asarray(A.gphit, float)[:, 25]
    prof = {d: get("L", 0, d, "rows") - get("N", 0, d, "rows") for d in HORIZONS}
    tot = {d: float(prof[d].sum()) for d in HORIZONS}
    print(f"  rows 0..{A.J0-1}  (latitudes {lat_row[0]:.1f} .. {lat_row[A.J0-1]:.1f} degN)")
    print(f"  row-sum gap: " + "  ".join(f"d{d} {tot[d]:+.4f}" for d in HORIZONS) + " Sv")
    print()
    # per-row member floor at day 360, so a row-level gap is judged the same way
    # every other number here is (review finding 7: the row table had none).
    rfl = np.hypot(
        np.std(np.stack([get("L", i, 360, "rows") for i in range(4)]), axis=0, ddof=1),
        np.std(np.stack([get("N", i, 360, "rows") for i in range(4)]), axis=0, ddof=1))
    print(f"  {'row':>5}{'lat':>8}" + "".join(f"{'d'+str(d):>11}" for d in HORIZONS)
          + f"{'floor360':>11}{'d360/fl':>9}   (ALL {A.J0} rows)")
    for j in range(A.J0):
        print(f"  {j:>5}{lat_row[j]:>8.2f}"
              + "".join(f"{prof[d][j]:>11.4f}" for d in HORIZONS)
              + f"{rfl[j]:>11.4f}{prof[360][j]/rfl[j]:>9.1f}")
    nsig = int(np.sum(np.abs(prof[360]) > 2 * rfl))
    print(f"  rows whose day-360 gap clears 2x its own floor: {nsig} of {A.J0}")
    neg = prof[360] < 0
    print(f"  the {int(neg.sum())} negative rows sum to "
          f"{float(prof[360][neg].sum()):+.4f} Sv, the "
          f"{int((~neg).sum())} positive ones to "
          f"{float(prof[360][~neg].sum()):+.4f} Sv, net {tot[360]:+.4f}")
    print()
    # Normalising by a signed sum FLIPS the sign of r when the two sums have
    # opposite signs, which would invert the verdict.  All four horizons are
    # negative here (printed above); the correlation is taken on the RAW
    # profiles so no normalisation can flip it, and the normalised value is
    # printed beside it only as a scale check.
    nrm = lambda a: a / abs(a.sum())      # noqa: E731
    for d in HORIZONS[:-1]:
        print(f"  row-profile shape, day {d:>3} vs day 360:  "
              f"r(raw) = {pearson(prof[d], prof[360]):+.3f}   "
              f"r(|sum|-normalised) = {pearson(nrm(prof[d]), nrm(prof[360])):+.3f}")

    # ---- R3: depth ----------------------------------------------------------
    print()
    print("=" * 100)
    print("R3  BOTTOM-REFERENCED split of the southern basin's transport")
    print("=" * 100)
    print("  bt = the DEEPEST WET CELL's velocity times the full water depth (NOT a")
    print("  depth mean); bc = everything above it.  bc is bt's ALGEBRAIC COMPLEMENT")
    print("  (bc == total - bt exactly), so any statement about bc is the same")
    print("  statement about bt and the total -- they are not two measurements.")
    print("  Because bt multiplies ONE cell by ~3300 m, both halves are inflated and")
    print("  ~95% anti-correlated; R3b below is the robust companion.")
    print("  Both on e3t_0 geometry, so bt + bc is the basin's e3t_0 transport, not")
    print("  the e3t_1d group transport in R1 -- the two weightings differ and the")
    print("  totals are printed side by side so the difference is visible.")
    print()
    print(f"  {'day':>5}{'lego bt':>11}{'NEMO bt':>11}{'D bt':>10}"
          f"{'lego bc':>11}{'NEMO bc':>11}{'D bc':>10}{'D bt+bc':>10}{'D e3t1d':>10}")
    for d in DAYS:
        lbt, nbt = float(get("L", 0, d, "s_bt")), float(get("N", 0, d, "s_bt"))
        lbc, nbc = float(get("L", 0, d, "s_bc")), float(get("N", 0, d, "s_bc"))
        g1d = float(get("L", 0, d, "g_south")) - float(get("N", 0, d, "g_south"))
        print(f"  {d:>5}{lbt:>11.3f}{nbt:>11.3f}{lbt-nbt:>10.4f}"
              f"{lbc:>11.3f}{nbc:>11.3f}{lbc-nbc:>10.4f}"
              f"{(lbt-nbt)+(lbc-nbc):>10.4f}{g1d:>10.4f}")
    # member floors for the two PARTS, so "the parts disagree" is a measurement
    # against the same 2x-floor rule the verdict used, not an eyeball.
    fbt, fbc = [], []
    for d in DAYS:
        fbt.append(float(np.hypot(spread([float(get("L", i, d, "s_bt")) for i in range(4)]),
                                  spread([float(get("N", i, d, "s_bt")) for i in range(4)]))))
        fbc.append(float(np.hypot(spread([float(get("L", i, d, "s_bc")) for i in range(4)]),
                                  spread([float(get("N", i, d, "s_bc")) for i in range(4)]))))
    fbt, fbc = np.asarray(fbt), np.asarray(fbc)
    dbt = np.array([float(get("L", 0, d, "s_bt")) - float(get("N", 0, d, "s_bt")) for d in DAYS])
    dbc = np.array([float(get("L", 0, d, "s_bc")) - float(get("N", 0, d, "s_bc")) for d in DAYS])
    print()
    print(f"  day 360:  D bt {dbt[-1]:+.4f}   D bc {dbc[-1]:+.4f}   "
          f"ratio |bt|/|bc| = {abs(dbt[-1])/abs(dbc[-1]):.2f}")
    print(f"  full year (mean of the 19 days):  D bt {dbt.mean():+.4f}  "
          f"D bc {dbc.mean():+.4f}")
    # Correlating a DETRENDED gap against UNDETRENDED, strongly trending parts
    # is not a seasonality statement -- it mostly measures the trend.  Both
    # sides are detrended here, and the per-part seasonal phases are reported
    # in their own block below against the forcing directly.  (An earlier
    # revision printed the undetrended version and it supported the OPPOSITE
    # sentence to the detrended block; caught in review, retracted.)
    rbt = dbt - np.polyval(np.polyfit(t, dbt, 1), t)
    rbc = dbc - np.polyval(np.polyfit(t, dbc, 1), t)
    print(f"  which part carries the seasonality (BOTH sides detrended; note the")
    print(f"  gap is e3t_1d-weighted and the parts are e3t_0-weighted):")
    print(f"    r(resid_gap, resid D bt) = {pearson(resid, rbt):+.3f}"
          f"    r(resid_gap, resid D bc) = {pearson(resid, rbc):+.3f}")

    print()
    print("  ARE THE TWO PARTS THEMSELVES DISTINGUISHABLE?  same 2x-floor rule as the")
    print("  verdict, applied to each part separately.")
    print(f"  {'day':>5}{'D bt':>10}{'floor bt':>11}{'bt/floor':>13}"
          f"{'D bc':>10}{'floor bc':>11}{'bc/floor':>13}{'D bt+bc':>10}"
          f"{'|bt|+|bc|':>11}")
    for k, d in enumerate(DAYS):
        print(f"  {d:>5}{dbt[k]:>10.4f}{fbt[k]:>11.2e}{dbt[k]/fbt[k]:>13.1f}"
              f"{dbc[k]:>10.4f}{fbc[k]:>11.2e}{dbc[k]/fbc[k]:>13.1f}"
              f"{dbt[k]+dbc[k]:>10.4f}{abs(dbt[k])+abs(dbc[k]):>11.4f}")
    print("  NOTE: at days 10-60 the member floors are ~1e-9 Sv (the four members")
    print("  have barely separated from a shared restart), so the ratios there are")
    print("  ratios against roundoff and carry no meaning.  Read them from day 90.")
    print()
    # Both halves on the SAME (e3t_0) weighting: the denominator is the parts'
    # own sum, not the e3t_1d group transport.  Mixing the two inflated the
    # day-270 figure by 49% in an earlier revision (caught in review).
    i270 = DAYS.index(270)
    net_e30 = dbt + dbc
    print(f"  cancellation ratio (|D bt| + |D bc|) / |D bt + D bc|, one weighting:")
    print(f"    day 270 {(abs(dbt[i270])+abs(dbc[i270]))/abs(net_e30[i270]):.1f}x"
          f"    day 360 {(abs(dbt[-1])+abs(dbc[-1]))/abs(net_e30[-1]):.1f}x")

    print()
    print("  WHO OWNS THE FINAL-QUARTER COMPOUNDING?  the day-270 -> day-360 change,")
    print("  split into the two parts (they are additive by construction).")
    d_net = (dbt[-1] + dbc[-1]) - (dbt[i270] + dbc[i270])
    d_bt_ch, d_bc_ch = dbt[-1] - dbt[i270], dbc[-1] - dbc[i270]
    print(f"    net (bt+bc)   {d_net:+.4f} Sv")
    print(f"    of which bt   {d_bt_ch:+.4f} Sv  ({100*d_bt_ch/d_net:5.1f}%)")
    print(f"    of which bc   {d_bc_ch:+.4f} Sv  ({100*d_bc_ch/d_net:5.1f}%)")
    print("  and the FIRST quarter (day 10 -> day 90), for contrast:")
    i10, i90 = DAYS.index(10), DAYS.index(90)
    d_net0 = (dbt[i90] + dbc[i90]) - (dbt[i10] + dbc[i10])
    print(f"    net {d_net0:+.4f}   bt {dbt[i90]-dbt[i10]:+.4f} "
          f"({100*(dbt[i90]-dbt[i10])/d_net0:5.1f}%)   bc {dbc[i90]-dbc[i10]:+.4f} "
          f"({100*(dbc[i90]-dbc[i10])/d_net0:5.1f}%)")
    print("    (percentages divide by a near-cancelling net, so they exceed 100%"
          " and are shares of a difference, not of a magnitude)")

    print()
    print("  RELATIVE strength of each part, legoESM / NEMO:")
    print(f"  {'day':>5}{'bt ratio':>11}{'bc ratio':>11}")
    for k, d in enumerate(DAYS):
        lbt, nbt = float(get("L", 0, d, "s_bt")), float(get("N", 0, d, "s_bt"))
        lbc, nbc = float(get("L", 0, d, "s_bc")), float(get("N", 0, d, "s_bc"))
        print(f"  {d:>5}{lbt/nbt:>11.4f}{lbc/nbc:>11.4f}")

    print()
    print("  SEASONAL PHASE of each part (linear envelope removed from each):")
    for lab, arr in (("D bt", dbt), ("D bc", dbc)):
        r_ = arr - np.polyval(np.polyfit(t, arr, 1), t)
        print(f"    {lab}: r(resid, c2) = {pearson(F['c2'], r_):+.3f}"
              f"   r(resid, c1) = {pearson(F['c1'], r_):+.3f}"
              f"   r(resid, Qsr) = {pearson(F['qsr'], r_):+.3f}")

    # ================================================================ R3b ===
    print()
    print("=" * 100)
    print("R3b  THE ROBUST COMPANION -- transport and its FIRST MOMENT (no single")
    print("     cell can move either), plus the DENSITY route")
    print("=" * 100)
    print("  M0 = the southern-basin transport itself [Sv], e3t_0 weighting.")
    print("  M1/M0 = the DEPTH at which that transport is centred [m, +down].")
    print("  tw = the bottom-referenced transport this model's OWN density field")
    print("       PREDICTS over the same rows (thermal wind).  If the two models'")
    print("       shear differed because their water masses differ, D tw would be")
    print("       the same size as D bc.  Same functional on both sides.")
    print()
    print(f"  {'day':>5}{'M0 lego':>10}{'M0 NEMO':>10}{'D M0':>9}"
          f"{'z_c lego':>10}{'z_c NEMO':>10}{'D z_c':>8}"
          f"{'tw lego':>10}{'tw NEMO':>10}{'D tw':>9}{'D bc':>9}{'tw/bc':>8}")
    dtw = []
    for d in DAYS:
        lm0, nm0 = float(get("L", 0, d, "s_m0")), float(get("N", 0, d, "s_m0"))
        lm1, nm1 = float(get("L", 0, d, "s_m1")), float(get("N", 0, d, "s_m1"))
        ltw, ntw = float(get("L", 0, d, "s_tw")), float(get("N", 0, d, "s_tw"))
        k = DAYS.index(d)
        dtw.append(ltw - ntw)
        frac = (ltw - ntw) / dbc[k] if dbc[k] != 0 else float("nan")
        print(f"  {d:>5}{lm0:>10.3f}{nm0:>10.3f}{lm0-nm0:>9.4f}"
              f"{lm1/lm0:>10.1f}{nm1/nm0:>10.1f}{lm1/lm0-nm1/nm0:>8.1f}"
              f"{ltw:>10.4f}{ntw:>10.4f}{ltw-ntw:>9.4f}{dbc[k]:>9.4f}{frac:>8.3f}")
    dtw = np.asarray(dtw)
    print()
    print(f"  the density route explains "
          f"{100*abs(dtw[-1])/abs(dbc[-1]):.1f}% of the day-360 shear difference "
          f"and {100*np.mean(np.abs(dtw))/np.mean(np.abs(dbc)):.1f}% on the "
          f"year mean")
    print()
    print("  PER-LEVEL southern-basin transport [Sv per model level], the vertical")
    print("  map whose centroid is z_c.  This is what discriminates 'too little")
    print("  vertical momentum mixing' (excess shallow, deficit deep) from 'bottom")
    print("  drag / bottom boundary layer' (the difference confined to the deepest")
    print("  wet levels).  Levels are printed to the deepest one carrying >0.5% of")
    print("  the transport on either side.")
    lv90L, lv90N = get("L", 0, 90, "s_lev"), get("N", 0, 90, "s_lev")
    lvL, lvN = get("L", 0, 360, "s_lev"), get("N", 0, 360, "s_lev")
    zc = np.asarray(A.gdept1d, float)
    keep = np.where((np.abs(lvL) + np.abs(lvN)) >
                    0.005 * (np.abs(lvL).sum() + np.abs(lvN).sum()))[0]
    print()
    # DIVIDE BY THE LAYER THICKNESS.  A per-level transport difference that
    # "grows with depth" grows because e3t grows from 10 m to 450 m; the
    # physically meaningful quantity is the difference PER METRE, i.e. the
    # velocity-times-width error.  Reading the raw column instead inverts the
    # conclusion, and an earlier revision of the write-up did exactly that.
    # NORMALISE BY THE LEVEL'S OWN WET CROSS-SECTION, so the column is a mean
    # VELOCITY difference [m/s] rather than a transport difference that grows
    # simply because the layers get thicker (10 m at the surface, 450 m at the
    # bottom) -- reading the raw column instead inverts the conclusion, and an
    # earlier revision of the write-up did exactly that.
    south = slice(0, A.J0)
    e2 = np.asarray(A.e2u_col, float)[south]
    area = np.einsum("jik,j->k",
                     np.where(wet_u[south][:, 2:-2, :],
                              A.e3t0[south][:, 2:-2, :], 0.0),
                     e2) / (A.NX - 4)                       # m2 per level
    dv90 = 1e6 * (lv90L - lv90N) / np.maximum(area, 1e-30)  # m/s
    dv360 = 1e6 * (lvL - lvN) / np.maximum(area, 1e-30)
    print(f"  {'k':>4}{'depth[m]':>10}{'area[m2]':>12}{'D d90':>9}{'D d360':>9}"
          f"{'dU d90':>12}{'dU d360':>12}   (dU in 1e-4 m/s)")
    for k in keep:
        print(f"  {k:>4}{zc[k]:>10.1f}{area[k]:>12.3e}{lv90L[k]-lv90N[k]:>9.4f}"
              f"{lvL[k]-lvN[k]:>9.4f}{1e4*dv90[k]:>12.2f}{1e4*dv360[k]:>12.2f}")
    dlev = lvL - lvN
    up = zc < 500.0
    print(f"  day-360 difference above 500 m: {float(dlev[up].sum()):+.4f} Sv;"
          f" below: {float(dlev[~up].sum()):+.4f} Sv;"
          f" deepest 5 levels: {float(dlev[-5:].sum()):+.4f} Sv")
    print(f"  ... and 500 m is {100*500.0/float(zc[keep[-1]]):.1f}% of the depth"
          " these levels span, so a DEPTH-UNIFORM error already puts most of"
          " itself below 500 m.")
    # TWO COMPETING NULLS, fitted the same way and scored the same way.
    #   (a) DEPTH-UNIFORM VELOCITY: dU is the same at every level -- what a
    #       barotropic momentum-budget difference (drag, the barotropic solve,
    #       a lateral flux at the wall) produces.
    #   (b) UNIFORM SCALING of NEMO's own profile: legoESM = (1+a) x NEMO at
    #       every level, i.e. the same flow, weaker -- which does NOT move the
    #       centroid and is a different hypothesis from (a).
    # They are distinguishable here because NEMO's own profile is strongly
    # surface-intensified in VELOCITY while the level cross-sections grow with
    # depth.  Reporting the raw per-level transports instead of dU makes both
    # look like "the deficit grows with depth", which is the level thickness.
    lm0 = float(get("L", 0, 360, "s_m0")); nm0 = float(get("N", 0, 360, "s_m0"))
    lm1 = float(get("L", 0, 360, "s_m1")); nm1 = float(get("N", 0, 360, "s_m1"))
    zk = np.asarray(A.gdept1d, float)
    A_sv = area / 1e6                                   # Sv per (m/s)
    c = float(np.sum(dlev * A_sv) / np.sum(A_sv * A_sv))     # best constant dU
    ra = dlev - c * A_sv
    a = float(np.sum(dlev * lvN) / np.sum(lvN * lvN))        # best scaling
    rb = dlev - a * lvN
    print()
    print("  WHICH NULL FITS?  both fitted by least squares on the per-level")
    print("  transport difference, both scored on the same residual.")
    for lab, pred, res in (("(a) depth-uniform velocity "
                            f"dU = {c*1e4:+.2f}e-4 m/s", c * A_sv, ra),
                           ("(b) uniform scaling      "
                            f"lego = {1+a:.4f} x NEMO", a * lvN, rb)):
        print(f"    {lab}:  explains "
              f"{100*(1-np.sum(res**2)/np.sum(dlev**2)):5.1f}% by variance, "
              f"{100*(1-np.abs(res).sum()/np.abs(dlev).sum()):5.1f}% by size, "
              f"residual total {float(res.sum()):+.4f} of {float(dlev.sum()):+.4f} Sv")
    best = ra if np.sum(ra**2) <= np.sum(rb**2) else rb
    top2 = np.argsort(-np.abs(best))[:2]
    print(f"    the better fit's two largest residual levels: k={int(top2[0])} "
          f"({zk[top2[0]]:.0f} m, {float(best[top2[0]]):+.4f} Sv) and "
          f"k={int(top2[1])} ({zk[top2[1]]:.0f} m, {float(best[top2[1]]):+.4f} Sv)")
    # what each null predicts for the CENTROID -- the scaling null predicts no
    # shift at all, so a measured shift is evidence for (a) over (b).
    zc_a = (nm1 + float(np.sum(c * A_sv * zk))) / (nm0 + float(np.sum(c * A_sv)))
    print(f"    centroid: NEMO {nm1/nm0:.1f} m; measured legoESM {lm1/lm0:.1f} m "
          f"({lm1/lm0 - nm1/nm0:+.1f} m)")
    print(f"              null (a) predicts {zc_a:.1f} m "
          f"({zc_a - nm1/nm0:+.1f} m); null (b) predicts no shift at all.")

    lrho = np.array([float(get("L", 0, d, "rho_mean")) for d in DAYS])
    nrho = np.array([float(get("N", 0, d, "rho_mean")) for d in DAYS])
    print(f"  southern-basin mean density, lego - NEMO:  day 90 "
          f"{lrho[DAYS.index(90)]-nrho[DAYS.index(90)]:+.3e}   day 360 "
          f"{lrho[-1]-nrho[-1]:+.3e} kg/m3   (basin mean rho "
          f"{nrho[-1]:.3f})")

    # ================================================================= R4 ===
    print()
    print("=" * 100)
    print("R4  MEMBER-0 IS ONE DRAW -- the same differences on the 4-MEMBER MEAN")
    print("=" * 100)
    print("  Everything above is member 0 minus member 0.  The other three members")
    print("  exist and cost nothing to average, and where the net gap is small the")
    print("  member-0 draw is not representative.")
    print()
    emean = lambda sd, d, f: float(np.mean(          # noqa: E731
        [float(get(sd, i, d, f)) for i in range(len(MEMBERS))]))
    print(f"  {'day':>5}{'gap m0':>10}{'gap ens':>10}{'floor':>11}"
          f"{'D bt m0':>10}{'D bt ens':>10}{'D bc m0':>10}{'D bc ens':>10}")
    for k, d in enumerate(DAYS):
        print(f"  {d:>5}{gap[k]:>10.4f}"
              f"{emean('L', d, 'g_south')-emean('N', d, 'g_south'):>10.4f}"
              f"{floor[k]:>11.2e}"
              f"{dbt[k]:>10.4f}{emean('L', d, 's_bt')-emean('N', d, 's_bt'):>10.4f}"
              f"{dbc[k]:>10.4f}{emean('L', d, 's_bc')-emean('N', d, 's_bc'):>10.4f}")

    json_out = dict(days=list(DAYS), doy=[float(x) for x in F["doy"]],
                    gap=[float(x) for x in gap], floor=[float(x) for x in floor],
                    d_bt=[float(x) for x in dbt], d_bc=[float(x) for x in dbc],
                    **{k: [float(x) for x in v] for k, v in F.items() if k != "doy"})
    out = "/tmp/dino_basin_seasonal_decomp.json"
    with open(out, "w") as f:
        json.dump(dict(git_head=sha, **json_out), f, indent=1)
    print(f"\n[stamp] series written to {out}")


if __name__ == "__main__":
    main()
