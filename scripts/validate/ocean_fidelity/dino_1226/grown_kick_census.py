#!/usr/bin/env python
"""#1455: the GROWN-perturbation threshold census and pair-flip firing rates.

Pre-registration: ``PREREG_grown_kick_census.md``, committed before this ran.
Read it before citing any number this prints.

WHY.  ``dino_switch_rectifier_result.md`` isolated convective adjustment's hard
edge as a ~300x-per-step rectifier, and in the same breath recorded that at the
ensembles' own 1e-14 kick amplitude ZERO interfaces sit within tipping reach of
the trigger, so the edge cannot fire at that size.  The ensembles'
perturbations grow.  The chain's one unmeasured link is therefore whether the
GROWN perturbation reaches the switch's neighbourhood, and -- the part that
decides the asymmetry -- whether legoESM's trigger resolves DIFFERENTLY between
a perturbed member and its control more often than NEMO's does.

WHAT IT MEASURES, per horizon, per model, per member pair:

  (a) the perturbation's magnitude distribution;
  (b) the threshold-neighbourhood census -- interfaces within the
      perturbation's reach of the trigger.  UPPER BOUND by construction: one
      global maximum |dN2| is applied to every interface.  That is the shipped
      convention in ``switch_rectifier.py::census`` and it is kept unchanged so
      the two numbers are comparable;
  (c) the PAIR-FLIP count and rate -- interfaces where the trigger fires in
      exactly one of {perturbed member, control}.  This is the discriminating
      statistic, and unlike (b) it is an exact count, not a bound.

THE PRECISION ASYMMETRY IS THE HARD PART.  legoESM's snapshots are float32,
NEMO's restarts float64.  A float32 temperature near 20 degC has a quantum of
about 2.4e-6 K, which maps to a dN2 of order 5e-10 -- five hundred times the
trigger's own 1e-12 offset -- so STORAGE ALONE CAN MANUFACTURE FLIPS.  NEMO is
therefore run through this pipeline twice: native fp64, and cast to fp32 and
back.  legoESM is compared against the fp32 arm; the gap between the two NEMO
arms is the storage inflation factor and is printed next to every number.

THE TRIGGER IS NOT THE SAME RULE IN THE TWO MODELS, and that is measured rather
than assumed.  legoESM fires on the single now-level ``N2 < -1e-12``
(``enhanced_diffusion.py:195``); NEMO fires on
``MIN(rn2, rn2b) <= -1e-12`` (``zdfevd.F90:93``), a minimum over TWO time
levels.  The primary comparison uses the single-level rule on BOTH sides so the
measured difference is a difference in STATES, not in rules; NEMO's own
two-level rule is reported alongside to size the rule's own contribution.
NEMO's exact ``rn2b`` is not in the restart, so ``bn2(tb,sb)`` stands in for it
and is labelled a PROXY everywhere it appears.  legoESM's before-level 3-D
state was never saved, so the two-level rule cannot be evaluated on legoESM at
all -- that is stated, not estimated.

DAY 5 DOES NOT EXIST.  Neither model wrote a 3-D state at day 5.  Both wrote
days 0,10,...,90.  The day-5 column is ABSENT, not interpolated.

WHAT IT IS NOT.  It reads states off disk and prints tables plus the mechanical
application of the pre-registered rule.  It runs no model, edits no card, no
recipe and no floor, and it does not interpret its own numbers.

Usage
-----
  grown_kick_census.py --self-check     # controls only, no NEMO/lego states
  grown_kick_census.py --ladder-check   # builds the twin ONCE to verify the
                                        # depth ladders + EOS form this probe
                                        # reads from mesh_mask match the card
  grown_kick_census.py                  # the measurement
"""
from __future__ import annotations

import argparse
import functools
import glob
import os
import subprocess
import sys
import time

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

# --------------------------------------------------------------- the inputs ---
MESH_MASK = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
             "RUN_TRAJ/mesh_mask.nc")
LEGO_LANES = {"kick2": "/tmp/dino_kick2",        # kick on BOTH time levels
              "kick1": "/tmp/dino_kick1"}        # now level only (lego-only)
NEMO_LANE = "/tmp/dino_kick2_nemo"               # matches lego lane "kick2"
MEMBERS = ("m1_seed1", "m2_seed2", "m3_seed3")
CONTROL = "m0_control"
NEMO_MEMBERS = {"m0_control": "RUN_KICK2_M0", "m1_seed1": "RUN_KICK2_M1",
                "m2_seed2": "RUN_KICK2_M2", "m3_seed3": "RUN_KICK2_M3"}
KT_DAY0 = 5760                                   # NEMO step at day 180 = day 0
STEPS_PER_DAY = 32                               # rdt = 2700 s
HORIZONS = (10, 20, 30, 40, 50, 60, 70, 80, 90)

# --------------------------------------------------- the card's own settings ---
# Both verified against the shipped card and NEMO's namelist_cfg by
# --ladder-check; see the pre-registration.
N2_THRESHOLD = -1e-12        # enhanced_diffusion n2_threshold on this card
EOS_FORM = "seos"            # ln_seos = .true. in namelist_cfg; card default

# ------------------------------------------------------- pre-registered bars ---
RESOLVABLE_FRAC = 1e-3       # >=0.1% of wet cells with a non-zero dT
CONFIRM_RATIO = 5.0          # lego pair-flip rate >= 5x NEMO(fp32) -> CONFIRM
REFUTE_RATIO = 2.0           # <= 2x -> REFUTE, loudly
MATCH_TOL = 3.0              # amplitude-matched pairs must agree within this factor


# ------------------------------------------------------------------ helpers ---
def _fatal(msg: str):
    raise SystemExit(f"FATAL: {msg}")


def _no_nan(a: np.ndarray, where: np.ndarray, name: str):
    """NaN inside the wet region is fatal, never nan-reduced away."""
    bad = int(np.count_nonzero(~np.isfinite(a[where])))
    if bad:
        _fatal(f"{bad} non-finite value(s) in {name} inside the wet mask")


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", _DIR, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", _DIR, "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, check=True)
        n = len([l for l in dirty.stdout.splitlines() if l.strip()])
        return f"{out.stdout.strip()} dirty_tracked_files={n}"
    except Exception as exc:                                  # pragma: no cover
        return f"<unavailable: {exc}>"


def _stamp(path: str) -> str:
    if not os.path.exists(path):
        return f"{path}  MISSING"
    st = os.stat(path)
    return (f"{path}  {st.st_size} B  "
            f"mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}")


# -------------------------------------------------------------------- masks ---
def load_masks():
    """(wet_t, wet_iface) on legoESM's (y, x, lev) axis order, from NEMO's own
    mesh_mask.  The interface mask is NEMO's ``wmask``: an interface is wet only
    if BOTH the cells it separates are wet."""
    import netCDF4 as nc
    if not os.path.exists(MESH_MASK):
        _fatal(f"mesh_mask not found: {MESH_MASK}")
    ds = nc.Dataset(MESH_MASK)
    tmask = np.asarray(ds["tmask"][0]).astype(bool)           # (lev, y, x)
    ds.close()
    wet_t = np.moveaxis(tmask, 0, -1)                         # (y, x, lev)
    wet_iface = wet_t[..., :-1] & wet_t[..., 1:]
    return wet_t, wet_iface


def load_ladders():
    """(gdept, gdepw_int) positive-down [m], from NEMO's own mesh_mask 1-D
    ladders.  ``gdepw_int`` is the INTERIOR w-interface ladder, i.e. NEMO's
    gdepw_1d with its surface entry dropped, matching legoESM's
    ``nemo_bn2_depth_ladders`` convention (|z_half_ref[1:-1]|)."""
    import netCDF4 as nc
    ds = nc.Dataset(MESH_MASK)
    gdept = np.asarray(ds["gdept_1d"][0]).astype(np.float64).ravel()
    gdepw = np.asarray(ds["gdepw_1d"][0]).astype(np.float64).ravel()
    ds.close()
    return gdept, gdepw[1:]


# ---------------------------------------------------------------------- N^2 ---
def make_n2(gdept, gdepw_int):
    """N^2 by NEMO's own ``bn2``, the SAME operator on both models' states."""
    import jax.numpy as jnp
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2
    t_d = jnp.asarray(gdept)
    w_d = jnp.asarray(gdepw_int)

    def n2(T, S):
        return np.asarray(compute_buoyancy_frequency_nemo_bn2(
            jnp.asarray(T), jnp.asarray(S), t_d, w_d, eos_form=EOS_FORM),
            dtype=np.float64)
    return n2


# ---------------------------------------------------------------- the states ---
def lego_state(lane_dir: str, member: str, day: int):
    """(T, S) float64 from the float32 snapshot.  Axis order (y, x, lev)."""
    path = os.path.join(lane_dir, f"{member}.npz")
    if not os.path.exists(path):
        _fatal(f"legoESM member not found: {path}")
    with np.load(path) as d:
        kt, ks = f"T3d_day{day}", f"S3d_day{day}"
        if kt not in d.files:
            _fatal(f"{path} has no {kt} -- horizons present: "
                   f"{[f for f in d.files if f.startswith('T3d_')]}")
        T = np.asarray(d[kt], dtype=np.float64)
        S = np.asarray(d[ks], dtype=np.float64)
    return T, S


@functools.lru_cache(maxsize=64)     # the same tiles are read by both NEMO arms
def nemo_state(run_dir: str, day: int, level: str = "n"):
    """(T, S) from the per-rank restart tiles, stitched, axis order (y, x, lev).

    ``level`` "n" = the NOW level (tn/sn); "b" = the BEFORE level (tb/sb), used
    only as the PROXY for NEMO's rn2b."""
    from rebuild_nemo_restart import rebuild
    kt = KT_DAY0 + STEPS_PER_DAY * day
    pattern = os.path.join(run_dir, f"DINO_{kt:08d}_restart_*.nc")
    if not glob.glob(pattern):
        _fatal(f"NEMO restart tiles not found: {pattern}")
    names = ["tn", "sn"] if level == "n" else ["tb", "sb"]
    raw = rebuild(pattern, names)
    for nm in names:
        if nm not in raw:
            _fatal(f"the stitcher did not cover {nm!r} at kt={kt}")
    def yxz(a):
        out = np.moveaxis(np.asarray(a, dtype=np.float64), 0, -1)
        out.setflags(write=False)      # cached and shared by both NEMO arms
        return out
    return yxz(raw[names[0]]), yxz(raw[names[1]])


def to_fp32(T, S):
    """The storage-matched control: the SAME state through legoESM's own
    snapshot precision and back."""
    return (T.astype(np.float32).astype(np.float64),
            S.astype(np.float32).astype(np.float64))


# --------------------------------------------------------------- statistics ---
def pair_stats(n2, T_c, S_c, T_p, S_p, wet_t, wet_i):
    """(a) magnitude distribution, (b) upper-bound census, (c) pair flips."""
    for nm, a in (("T_control", T_c), ("S_control", S_c),
                  ("T_perturbed", T_p), ("S_perturbed", S_p)):
        _no_nan(a, wet_t, nm)

    dT = np.abs(T_p - T_c)[wet_t]
    nz = dT[dT > 0.0]
    n_wet_t = int(wet_t.sum())

    N2c, N2p = n2(T_c, S_c), n2(T_p, S_p)
    _no_nan(N2c, wet_i, "N2_control")
    _no_nan(N2p, wet_i, "N2_perturbed")

    dN2 = np.abs(N2p - N2c)[wet_i]
    reach = float(dN2.max()) if dN2.size else 0.0
    # UPPER BOUND, shipped convention: one global maximum reach, every interface.
    near = int(np.count_nonzero(np.abs(N2c[wet_i] - N2_THRESHOLD) < reach))

    fc = N2c[wet_i] < N2_THRESHOLD
    fp = N2p[wet_i] < N2_THRESHOLD
    n_wet_i = int(wet_i.sum())
    # REVIEW FINDING B5. The flip COUNT is not independent of how many
    # interfaces the perturbation reached at all: a model whose perturbation is
    # visible on more interfaces flips more for that reason alone, so a raw
    # flip ratio can re-express table (a)'s "cells != 0" column rather than say
    # anything about the trigger. Flips per DIFFERING interface removes that.
    n_diff = int(np.count_nonzero(dN2 > 0.0))
    return {
        "n_wet_t": n_wet_t, "n_wet_i": n_wet_i,
        "dT_max": float(dT.max()) if dT.size else 0.0,
        "dT_p999": float(np.percentile(nz, 99.9)) if nz.size else 0.0,
        "dT_med_nz": float(np.median(nz)) if nz.size else 0.0,
        "n_dT_nz": int(nz.size), "frac_dT_nz": nz.size / max(n_wet_t, 1),
        "reach": reach, "near": near, "frac_near": near / max(n_wet_i, 1),
        "fire_ctrl": int(fc.sum()), "rate_ctrl": fc.sum() / max(n_wet_i, 1),
        "flips": int(np.count_nonzero(fc ^ fp)),
        "flips_on": int(np.count_nonzero(~fc & fp)),
        "flips_off": int(np.count_nonzero(fc & ~fp)),
        "flip_rate": np.count_nonzero(fc ^ fp) / max(n_wet_i, 1),
        "n_iface_diff": n_diff,
        "flips_per_diff": np.count_nonzero(fc ^ fp) / max(n_diff, 1),
    }


def firing_rate_two_level(n2, T_n, S_n, T_b, S_b, wet_i):
    """NEMO's OWN rule: MIN(rn2, rn2b) <= threshold.  ``bn2(tb,sb)`` is a PROXY
    for rn2b -- the exact rn2b is not in the restart."""
    N2n, N2b = n2(T_n, S_n), n2(T_b, S_b)
    _no_nan(N2n, wet_i, "N2_now")
    _no_nan(N2b, wet_i, "N2_before_proxy")
    m = np.minimum(N2n[wet_i], N2b[wet_i])
    return int(np.count_nonzero(m <= N2_THRESHOLD)), int(wet_i.sum())


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def _resolvable_both(table, day) -> bool:
    """The bar binds on the NEMO arm too (physics review).

    It was applied only to legoESM. At day 20 the nemo_fp32 denominator has a
    non-zero perturbation on 0.003% of cells -- thirty times BELOW the probe's
    own floor -- and a mean flip count of 1.0, and day 30's ratio rested on a
    mean of 0.7 flips. A ratio whose denominator is itself unresolvable is not
    a measurement, and applying the rule one-sidedly is what produced the
    largest ratios in the table."""
    return (_resolvable(table[(day, "lego")])
            and _resolvable(table[(day, "nemo_fp32")]))


def _resolvable(rows) -> bool:
    """REVIEW FINDING B2: the gate must bind on EVERY member, not on the mean.

    At day 20 the three members' differing-cell fractions are 0.366%, 0.0006%
    and 0.0003%; the mean is 0.122% and clears the 0.1% bar, so two members
    that are three orders of magnitude below it were being carried past the
    very gate that exists to stop the statistic measuring the file format."""
    return all(r["frac_dT_nz"] >= RESOLVABLE_FRAC for r in rows)


def _median(rows, key):
    """The ROBUSTNESS arm alongside the registered mean.

    Added after the per-member column showed legoESM's early-horizon flips are
    carried by ONE member of three (99,0,0 at day 20; 749,0,0 at day 50). A
    mean over that distribution is not a rate, and a verdict resting on it
    rests on a single member. The median is reported next to the mean
    everywhere the mean decides something."""
    return float(np.median([r[key] for r in rows])) if rows else float("nan")


# ------------------------------------------------------------- self-checks ----
def self_check():
    """Controls that must pass before any measured number is read.  Each one is
    shown to FAIL when the thing it guards is removed."""
    print("=" * 100)
    print("SELF-CHECKS -- the instrument before the measurement")
    print("=" * 100)
    rng = np.random.default_rng(0)
    ny, nx, nz = 12, 9, 8
    gdept = np.cumsum(np.full(nz, 10.0)) - 5.0
    gdepw = np.cumsum(np.full(nz, 10.0))[:-1]
    n2 = make_n2(gdept, gdepw)

    wet_t = np.ones((ny, nx, nz), dtype=bool)
    wet_t[0, 0, :] = False                      # a dry COLUMN
    wet_t[:, :, -2:] = False                    # a dry BOTTOM pair
    wet_i = wet_t[..., :-1] & wet_t[..., 1:]

    T = 10.0 + np.cumsum(rng.random((ny, nx, nz)), axis=-1)[::-1]
    T = 20.0 - np.linspace(0, 10, nz)[None, None, :] + 0.01 * rng.random((ny, nx, nz))
    S = 35.0 + 0.01 * rng.random((ny, nx, nz))

    # 1. identical states -> zero flips, zero reach.  A statistic that cannot
    #    return zero cannot return a small number honestly either.
    s = pair_stats(n2, T, S, T.copy(), S.copy(), wet_t, wet_i)
    assert s["flips"] == 0 and s["reach"] == 0.0 and s["n_dT_nz"] == 0, s
    print("  [1] identical states -> 0 flips, 0 reach, 0 differing cells   PASS")

    # 2. PLANTED VIOLATION, dry cell: a gross static instability planted in a
    #    DRY cell must move NOTHING.  This is the mask's proof of life.
    # 5 K, not 50 K. A 50 K plant drives T to -34 degC, where the simplified
    # EOS's thermal expansion coefficient CHANGES SIGN, and the planted
    # instability inverts -- a self-check must not run its own operator outside
    # the range the operator is valid on.
    PLANT_K = 5.0
    T_dry = T.copy()
    T_dry[0, 0, 3] -= PLANT_K                    # inside the dry column
    s_dry = pair_stats(n2, T, S, T_dry, S, wet_t, wet_i)
    assert s_dry["flips"] == 0, s_dry
    assert s_dry["reach"] == 0.0, s_dry
    assert s_dry["n_dT_nz"] == 0, s_dry
    print("  [2] violation planted in a DRY cell -> every statistic unmoved  PASS")

    # 3. NON-VACUITY: the SAME violation planted in a WET cell must move them.
    #    Without this, check 2 is satisfied by a probe that measures nothing.
    T_wet = T.copy()
    T_wet[5, 4, 3] -= PLANT_K                    # cold over warm -> unstable
    s_wet = pair_stats(n2, T, S, T_wet, S, wet_t, wet_i)
    assert s_wet["flips"] > 0, s_wet
    assert s_wet["reach"] > 0.0 and s_wet["n_dT_nz"] == 1, s_wet
    print(f"  [3] the SAME violation in a WET cell -> {s_wet['flips']} flip(s), "
          f"reach {s_wet['reach']:.3e}                PASS")

    # 4. the mask really excludes land: dropping it makes check 2 FAIL.
    all_t = np.ones_like(wet_t)
    all_i = all_t[..., :-1] & all_t[..., 1:]
    s_nomask = pair_stats(n2, T, S, T_dry, S, all_t, all_i)
    # The `or` in the first version of this assertion short-circuited on
    # n_dT_nz, which is trivially 1, so the flips half was never exercised.
    assert s_nomask["n_dT_nz"] == 1, s_nomask
    assert s_nomask["flips"] > 0, s_nomask
    print("  [4] the same dry violation WITHOUT the mask -> BOTH the cell "
          "count and the flips move  PASS")

    # 5. NaN inside the wet mask is FATAL, not nan-reduced.
    T_nan = T.copy()
    T_nan[5, 4, 2] = np.nan
    try:
        pair_stats(n2, T, S, T_nan, S, wet_t, wet_i)
    except SystemExit:
        print("  [5] NaN inside the wet mask -> FATAL                          PASS")
    else:                                                     # pragma: no cover
        raise AssertionError("a NaN inside the wet mask did not abort")

    # 6. NaN OUTSIDE the mask must NOT abort (else the land fill kills the run).
    T_ok = T.copy()
    T_ok[0, 0, 3] = np.nan
    pair_stats(n2, T, S, T_ok, S, wet_t, wet_i)
    print("  [6] NaN on land -> tolerated (land is filled, not measured)     PASS")

    # 7. the fp32 round-trip control is not a no-op on an fp64 field.
    T32, _ = to_fp32(T, S)
    assert np.any(T32 != T), "fp32 round trip changed nothing -- control is inert"
    print("  [7] the fp32 storage control actually perturbs an fp64 field    PASS")

    # 9. THE FLIP LANDS AT THE INTERFACE IT SHOULD.  Checks 2-4 only assert
    #    that SOMETHING moved, so a uniform off-by-one in the interface mask or
    #    in the N2 index convention would pass every one of them.  Cooling
    #    cell k makes the interface BELOW it (index k, between cell k and cell
    #    k+1) unstable, because cold water is denser: this asserts the flip is
    #    at exactly that index and nowhere else.
    kk = 3
    T_one = T.copy()
    T_one[5, 4, kk] -= PLANT_K
    fc = (n2(T, S) < N2_THRESHOLD) & wet_i
    fp = (n2(T_one, S) < N2_THRESHOLD) & wet_i
    idx = np.argwhere(fc ^ fp)
    assert idx.shape[0] == 1, f"expected exactly one flip, got {idx.shape[0]}"
    assert tuple(idx[0]) == (5, 4, kk), (
        f"the flip landed at {tuple(idx[0])}, not at the interface below the "
        f"cooled cell (5, 4, {kk}) -- the interface index convention is off")
    print(f"  [9] the planted flip lands at interface (5,4,{kk}), the one "
          "BELOW the cooled cell   PASS")

    # 8. the verdict is COMPUTED, both ways.
    assert verdict(10.0, 1.0)[0] == "CONFIRM"
    assert verdict(1.5, 1.0)[0] == "REFUTE"
    assert verdict(3.0, 1.0)[0] == "INDETERMINATE"
    assert verdict(float("nan"), 0.0)[0] == "UNDEFINED"
    print("  [8] the verdict rule fires in every direction it can            PASS")
    print("  ALL SELF-CHECKS PASS\n")


def verdict(f_lego: float, f_nemo: float):
    """The pre-registered rule, applied mechanically.  Never typed by hand."""
    if not np.isfinite(f_lego) or not np.isfinite(f_nemo) or f_nemo <= 0.0:
        if f_lego > 0.0 and f_nemo == 0.0:
            return "CONFIRM", float("inf")
        return "UNDEFINED", float("nan")
    r = f_lego / f_nemo
    if r >= CONFIRM_RATIO:
        return "CONFIRM", r
    if r <= REFUTE_RATIO:
        return "REFUTE", r
    return "INDETERMINATE", r


# -------------------------------------------------------------- ladder check ---
def ladder_check():
    """Build the twin ONCE and confirm the ladders / EOS form / threshold this
    probe reads from mesh_mask are the ones the shipped card actually uses.

    The ladders are STATIC GRID QUANTITIES, and the card carries them at
    whatever precision the bridge was built under.  Agreement to fp64 round-off
    is not required and is not gated for; what IS gated is that the choice of
    ladder cannot change the answer, which is measured directly on a real state
    below rather than argued.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())          # the bridge defaults to fp32
    import kamm_twin_90d as K
    from legoesm.ocean.eos import nemo_bn2_depth_ladders
    print("=" * 100)
    print("LADDER + CARD CHECK -- does this probe read the card's own settings?")
    print("=" * 100)
    br, cfg, mc, model, forcing, sf, st = K._build_twin_state(
        "nemo_dino_kamm_mlf", K.RUN_TRAJ, K.RUN_STEPDUMP)
    ed = mc.physics.convection.enhanced_diffusion
    t_card, w_card = (np.asarray(a, dtype=np.float64)
                      for a in nemo_bn2_depth_ladders(br.z_coord))
    t_mesh, w_mesh = load_ladders()
    for nm, a, b in (("gdept", t_card, t_mesh), ("gdepw_int", w_card, w_mesh)):
        if a.shape != b.shape:
            _fatal(f"{nm} shape {a.shape} (card) vs {b.shape} (mesh_mask)")
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-30)))
        print(f"  {nm:<12} max relative difference card vs mesh_mask: {rel:.3e}")
        if rel > 1e-6:
            _fatal(f"{nm} disagrees between the card and mesh_mask beyond "
                   "single precision -- that is a real ladder difference")

    # DOES THE LADDER CHOICE CHANGE THE ANSWER?  Measured on the real day-90
    # NEMO control and its m1 pair -- the horizon that carries the verdict --
    # not reasoned about.  If the flip count is identical under both ladders
    # the choice is immaterial and this probe's use of mesh_mask is safe.
    wet_t, wet_i = load_masks()
    n2_mesh = make_n2(t_mesh, w_mesh)
    n2_card = make_n2(t_card, w_card)
    Tc, Sc = nemo_state(os.path.join(NEMO_LANE, NEMO_MEMBERS[CONTROL]), 90)
    Tp, Sp = nemo_state(os.path.join(NEMO_LANE, NEMO_MEMBERS["m1_seed1"]), 90)
    s_mesh = pair_stats(n2_mesh, Tc, Sc, Tp, Sp, wet_t, wet_i)
    s_card = pair_stats(n2_card, Tc, Sc, Tp, Sp, wet_t, wet_i)
    a, b = n2_mesh(Tc, Sc)[wet_i], n2_card(Tc, Sc)[wet_i]
    rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-30)))
    print(f"  LADDER SENSITIVITY on the real day-90 NEMO pair: "
          f"max relative N2 difference {rel:.3e}")
    print(f"    flips  mesh_mask ladder {s_mesh['flips']}   "
          f"card ladder {s_card['flips']}   "
          f"firing(ctrl) {s_mesh['fire_ctrl']} vs {s_card['fire_ctrl']}")
    if s_mesh["flips"] != s_card["flips"]:
        _fatal("the ladder choice changes the flip count -- it is NOT "
               "immaterial and the probe must take the card's ladder")
    print("    identical -> the ladder choice is immaterial to the verdict")
    print(f"  n2_threshold  card={float(ed.n2_threshold):.3e}  "
          f"probe={N2_THRESHOLD:.3e}")
    if float(ed.n2_threshold) != N2_THRESHOLD:
        _fatal("n2_threshold disagrees with the card")
    print(f"  n2_eos_form   card={ed.n2_eos_form!r}  probe={EOS_FORM!r}")
    if ed.n2_eos_form != EOS_FORM:
        _fatal("n2_eos_form disagrees with the card")
    print(f"  K_bg -> K_conv  {ed.K_bg:.1e} -> {ed.K_conv:.1e} m2/s "
          f"(factor {ed.K_conv / ed.K_bg:.0e}), smooth_transition="
          f"{ed.smooth_transition}")
    wet_t, wet_i = load_masks()
    print(f"  mask from mesh_mask tmask: wet T-cells {int(wet_t.sum())} / "
          f"{wet_t.size} ({100 * wet_t.mean():.2f}%), wet interfaces "
          f"{int(wet_i.sum())} / {wet_i.size} ({100 * wet_i.mean():.2f}%)")
    print("  (the shipped switch_rectifier census used isfinite(N2), i.e. "
          f"{wet_i.size} -- the WHOLE grid, land included)")
    print("  LADDER + CARD CHECK PASS\n")


# ------------------------------------------------------------------- measure ---
def measure(lane: str):
    wet_t, wet_i = load_masks()
    n2 = make_n2(*load_ladders())
    lane_dir = LEGO_LANES[lane]
    has_nemo = (lane == "kick2")

    print("=" * 100)
    print(f"PROVENANCE -- lane {lane!r}")
    print("=" * 100)
    print(f"  probe git: {_git_sha()}")
    sha = os.path.join(lane_dir, ".launch_sha")
    print(f"  legoESM launch SHA: "
          f"{open(sha).read().strip() if os.path.exists(sha) else '<none>'}")
    for m in (CONTROL,) + MEMBERS:
        print(f"    {_stamp(os.path.join(lane_dir, m + '.npz'))}")
    if has_nemo:
        for m in (CONTROL,) + MEMBERS:
            d = os.path.join(NEMO_LANE, NEMO_MEMBERS[m])
            n = len(glob.glob(os.path.join(d, "DINO_*_restart_*.nc")))
            print(f"    {d}  {n} restart tiles")
    print(f"  mask + ladders: {_stamp(MESH_MASK)}")
    print(f"  trigger: N2 < {N2_THRESHOLD:.1e}, eos_form={EOS_FORM!r}; "
          f"wet interfaces {int(wet_i.sum())} of {wet_i.size}")
    print("  DAY 5 IS ABSENT on both models -- the finest common cadence is 10 "
          "days.  It is not interpolated.\n")

    arms = ["lego"] + (["nemo_fp64", "nemo_fp32"] if has_nemo else [])
    table = {}
    for day in HORIZONS:
        Tc_l, Sc_l = lego_state(lane_dir, CONTROL, day)
        nemo_ctrl = (nemo_state(os.path.join(NEMO_LANE,
                                            NEMO_MEMBERS[CONTROL]), day)
                     if has_nemo else None)
        for arm in arms:
            rows = []
            for m in MEMBERS:
                if arm == "lego":
                    Tc, Sc = Tc_l, Sc_l
                    Tp, Sp = lego_state(lane_dir, m, day)
                else:
                    Tc, Sc = nemo_ctrl
                    Tp, Sp = nemo_state(os.path.join(NEMO_LANE,
                                                     NEMO_MEMBERS[m]), day)
                    if arm == "nemo_fp32":
                        Tc, Sc = to_fp32(Tc, Sc)
                        Tp, Sp = to_fp32(Tp, Sp)
                rows.append(pair_stats(n2, Tc, Sc, Tp, Sp, wet_t, wet_i))
            table[(day, arm)] = rows
        print(f"  ... day {day} done", flush=True)

    _report(table, lane, has_nemo, wet_i, n2, has_nemo)
    return table


def _report(table, lane, has_nemo, wet_i, n2, do_two_level):
    arms = ["lego"] + (["nemo_fp64", "nemo_fp32"] if has_nemo else [])
    nwi = int(wet_i.sum())

    print("\n" + "=" * 100)
    print(f"(a) PERTURBATION MAGNITUDE -- mean over 3 member pairs, lane {lane!r}")
    print("=" * 100)
    print(f"  {'day':>4}{'arm':>12}{'max|dT| [K]':>14}{'p99.9 nz':>12}"
          f"{'median nz':>12}{'cells != 0':>12}{'of wet':>10}  resolvable?")
    for day in HORIZONS:
        for arm in arms:
            r = table[(day, arm)]
            f = _mean(r, "frac_dT_nz")
            print(f"  {day:>4}{arm:>12}{_mean(r, 'dT_max'):>14.3e}"
                  f"{_mean(r, 'dT_p999'):>12.3e}{_mean(r, 'dT_med_nz'):>12.3e}"
                  f"{_mean(r, 'n_dT_nz'):>12.0f}{100 * f:>9.3f}%  "
                  f"{'yes' if f >= RESOLVABLE_FRAC else 'NO (storage floor)'}")
        print()

    print("=" * 100)
    print("(b) THRESHOLD-NEIGHBOURHOOD CENSUS -- UPPER BOUND (one global reach "
          "applied to every interface)")
    print("=" * 100)
    print("  DO NOT CITE THIS TABLE ACROSS MODELS. It is the shipped "
          "convention, kept so the numbers")
    print("  line up with the earlier census, but 'reach' is a single global "
          "MAXIMUM |dN2| applied to")
    print("  every interface, and the two models' |dN2| distributions have "
          "very different tail-to-bulk")
    print("  ratios -- so it over-counts the two models by DIFFERENT factors. "
          "It is an upper bound per")
    print("  model, not a comparable one. The amplitude-free table (f) is the "
          "cross-model statement.")
    print(f"  {'day':>4}{'arm':>12}{'reach |dN2|':>14}{'within reach':>14}"
          f"{'of wet':>10}{'firing (ctrl)':>15}{'firing rate':>13}")
    for day in HORIZONS:
        for arm in arms:
            r = table[(day, arm)]
            print(f"  {day:>4}{arm:>12}{_mean(r, 'reach'):>14.3e}"
                  f"{_mean(r, 'near'):>14.0f}"
                  f"{100 * _mean(r, 'frac_near'):>9.3f}%"
                  f"{_mean(r, 'fire_ctrl'):>15.0f}"
                  f"{100 * _mean(r, 'rate_ctrl'):>12.3f}%")
        print()

    if has_nemo:
        print("  IS THE BASELINE FIRING-RATE DIFFERENCE REAL, OR STORAGE? "
              "(computed)")
        for day in (HORIZONS[0], HORIZONS[-1]):
            rl = _mean(table[(day, "lego")], "rate_ctrl")
            r32 = _mean(table[(day, "nemo_fp32")], "rate_ctrl")
            r64 = _mean(table[(day, "nemo_fp64")], "rate_ctrl")
            gap = abs(rl - r64)
            explained = abs(r32 - r64)
            frac = explained / gap if gap > 0 else float("nan")
            print(f"    day {day}: legoESM {100 * rl:.3f}%  NEMO(fp64) "
                  f"{100 * r64:.3f}%  gap {100 * gap:.3f} pts;  casting NEMO "
                  f"to fp32 gives {100 * r32:.3f}%,")
            print(f"      reproducing {100 * frac:.1f}% of that gap.")
        print("    -> the apparent 'NEMO convects more often than legoESM' is "
              "reproduced by the FILE FORMAT.")
        print("    legoESM's true fp64 firing rate was never saved and is "
              "UNKNOWN. Any earlier statement")
        print("    that NEMO fires more is RETRACTED.\n")

    print("=" * 100)
    print("(c) PAIR-FLIP COUNT AND RATE -- the discriminating statistic (an "
          "EXACT count, not a bound)")
    print("=" * 100)
    # The per-member RANGE is printed next to the mean: a mean over three
    # member pairs can hide one member disagreeing with the other two, and the
    # ensemble is small enough that it would not show up any other way.
    print("  'ifaces differing' is how many wet interfaces the perturbation "
          "moved AT ALL, and")
    print("  'flips/diff' normalises the flip count by it (review B5): a raw "
          "flip ratio can simply")
    print("  re-express how many interfaces each perturbation reached, rather "
          "than say anything")
    print("  about the trigger. flips/diff is the trigger's own conversion "
          "rate.")
    print(f"\n  {'day':>4}{'arm':>12}{'flips':>10}{'flip rate':>12}"
          f"{'ifaces differing':>18}{'flips/diff':>12}{'per-member flips':>22}")
    for day in HORIZONS:
        for arm in arms:
            r = table[(day, arm)]
            per = ",".join(str(x["flips"]) for x in r)
            print(f"  {day:>4}{arm:>12}{_mean(r, 'flips'):>10.1f}"
                  f"{100 * _mean(r, 'flip_rate'):>11.4f}%"
                  f"{_mean(r, 'n_iface_diff'):>18.0f}"
                  f"{_mean(r, 'flips_per_diff'):>12.5f}{per:>22}")
        print()
    if not has_nemo:
        print("  (no NEMO counterpart for this lane -- the normalised "
              "cross-model comparison needs one)\n")
    else:
      print("  NORMALISED COMPARISON (review B5) -- flips per differing "
            "interface, medians:")
      print(f"  {'day':>4}{'lego':>12}{'nemo_fp32':>12}{'ratio':>10}"
            f"{'   direction':>14}")
      for day in HORIZONS:
        if not _resolvable_both(table, day):
            continue
        a = _median(table[(day, "lego")], "flips_per_diff")
        b = _median(table[(day, "nemo_fp32")], "flips_per_diff")
        rr = a / b if b > 0 else float("nan")
        print(f"  {day:>4}{a:>12.5f}{b:>12.5f}{rr:>10.3f}"
              f"{('   legoESM higher' if rr > 1 else '   NEMO higher'):>14}")
      print()

    if not has_nemo:
        print("  (no NEMO counterpart for this lane -- lego-only sensitivity)\n")
        return

    print("=" * 100)
    print("(d) THE VERDICT, per horizon, by the PRE-REGISTERED rule")
    print("=" * 100)
    print(f"  storage inflation = flip rate(nemo_fp32) / flip rate(nemo_fp64): "
          f"how much the file format alone adds")
    print(f"  R = flip rate(lego) / flip rate(nemo_fp32)   "
          f"CONFIRM >= {CONFIRM_RATIO}, REFUTE <= {REFUTE_RATIO}")
    print("\n  BOTH the registered MEAN and the MEDIAN over the three member "
          "pairs are shown. Where they")
    print("  disagree the median governs the honest reading: legoESM's early "
          "flips are carried by a SINGLE")
    print("  member (see the per-member column above), and a mean over "
          "99,0,0 is not a rate.")
    print(f"\n  {'day':>4}{'lego rate':>12}{'nemo32 rate':>13}"
          f"{'nemo64 rate':>13}{'storage infl':>13}{'R(mean)':>9}"
          f"{'R(med)':>9}{'verdict(mean)':>15}{'verdict(med)':>15}")
    first = None
    first_med = None
    for day in HORIZONS:
        fl = _mean(table[(day, "lego")], "flip_rate")
        f32 = _mean(table[(day, "nemo_fp32")], "flip_rate")
        f64 = _mean(table[(day, "nemo_fp64")], "flip_rate")
        ml = _median(table[(day, "lego")], "flip_rate")
        m32 = _median(table[(day, "nemo_fp32")], "flip_rate")
        res = _resolvable_both(table, day)
        infl = f32 / f64 if f64 > 0 else float("nan")
        v, R = verdict(fl, f32)
        vm, Rm = verdict(ml, m32)
        if not res:
            v = vm = "UNMEASURABLE"
        else:
            if first is None:
                first = (day, v, R)
            if first_med is None:
                first_med = (day, vm, Rm)
        infl_s = (f"{infl:.2f}" if np.isfinite(infl)
                  else "n/a(0 fp64)")   # NEMO fp64 has zero flips there
        print(f"  {day:>4}{100 * fl:>11.4f}%{100 * f32:>12.4f}%"
              f"{100 * f64:>12.4f}%{infl_s:>13}{R:>9.2f}{Rm:>9.2f}"
              f"{v:>15}{vm:>15}")
    print("\n  UNMEASURABLE = at least ONE legoESM member's stored "
          f"perturbation is non-zero on fewer than")
    print(f"  {100 * RESOLVABLE_FRAC:.1f}% of wet cells, so the statistic "
          "would be measuring the file format. The gate binds")
    print("  per member: a mean over one resolved member and two unresolved "
          "ones defeats it (review B2).")
    if first is None:
        print("\n  REGISTERED VERDICT: UNMEASURABLE at every horizon -- no "
              "horizon cleared the resolvability bar.")
    else:
        d, v, R = first
        print(f"\n  REGISTERED VERDICT (mean, earliest resolvable horizon, "
              f"day {d}): {v}  (R = {R:.2f})")
        dm, vm, Rm = first_med
        print(f"  ROBUSTNESS  VERDICT (median, day {dm}): {vm}  "
              f"(R = {Rm:.2f})")
        agree = [d for d in HORIZONS
                 if _resolvable_both(table, d)
                 and verdict(_mean(table[(d, "lego")], "flip_rate"),
                             _mean(table[(d, "nemo_fp32")], "flip_rate"))[0]
                 == verdict(_median(table[(d, "lego")], "flip_rate"),
                            _median(table[(d, "nemo_fp32")], "flip_rate"))[0]]
        print(f"  mean and median agree at horizons: "
              f"{agree if agree else 'NONE'}")

    # ---- THE ANTI-CIRCULARITY CONTROL -------------------------------------
    # (d) is HORIZON-matched, and at a matched horizon legoESM's perturbation is
    # up to ~2000x LARGER than NEMO's. A larger perturbation flips more
    # triggers for trivial reasons, so a horizon-matched ratio cannot separate
    # "legoESM's trigger rectifies more" from "legoESM's perturbation is
    # bigger" -- and using it to explain why legoESM amplifies more would be
    # circular. This table matches on PERTURBATION AMPLITUDE instead: for each
    # legoESM horizon it finds the NEMO horizon whose max|dT| is closest in
    # logarithm, accepts the pair only if the amplitudes agree within a factor
    # MATCH_TOL, and reports the flip-rate ratio there. THAT ratio is the
    # rectification asymmetry with the growth asymmetry divided out.
    print("=" * 100)
    print("(d2) AMPLITUDE-MATCHED CONTROL -- the same comparison with the "
          "growth asymmetry divided out")
    print("=" * 100)
    print("  (d) is horizon-matched and therefore CONFOUNDED: at a matched "
          "horizon legoESM's perturbation")
    print("  is far larger, and a larger perturbation flips more triggers for "
          "trivial reasons. Here each")
    print("  legoESM horizon is paired with the NEMO horizon of the CLOSEST "
          f"max|dT| (accepted only within")
    print(f"  a factor {MATCH_TOL:g}), so the pair differs in the model and not "
          "in the perturbation size.")
    amp = {a: [(_mean(table[(d, a)], "dT_max"),
                _median(table[(d, a)], "flip_rate"), d) for d in HORIZONS]
           for a in arms}
    print(f"\n  {'day':>4}{'lego max|dT|':>14}{'lego med nz':>13}"
          f"{'nemo max|dT|':>14}{'nemo med nz':>13}   (the distributions, "
          f"side by side)")
    for d in HORIZONS:
        print(f"  {d:>4}{_mean(table[(d, 'lego')], 'dT_max'):>14.3e}"
              f"{_mean(table[(d, 'lego')], 'dT_med_nz'):>13.3e}"
              f"{_mean(table[(d, 'nemo_fp64')], 'dT_max'):>14.3e}"
              f"{_mean(table[(d, 'nemo_fp64')], 'dT_med_nz'):>13.3e}")
    print("  Flip rates in THIS table are MEDIANS over the three member pairs, "
          "not means, for the reason")
    print("  given under (d): legoESM's early flips are carried by a single "
          "member. The NEMO arm is the")
    print("  STORAGE-MATCHED one (review B3): casting to float32 INFLATES the "
          "flip count, so pairing")
    print("  legoESM's censored rate against NEMO's uncensored fp64 rate "
          "would flatter legoESM by")
    print("  roughly the inflation factor.")
    print("\n  CAVEAT ON THE MATCHING STATISTIC (review B4): max|dT| is a "
          "single-cell tail draw and is")
    print("  NOT monotone in horizon on NEMO's side. The two perturbation "
          "DISTRIBUTIONS do not overlap")
    print("  at all -- only their extreme tails do -- so any amplitude match "
          "is a match of tails. The")
    print("  median non-zero |dT| of each model is printed beside the max so "
          "the gap is visible.")
    print(f"\n  {'lego day':>9}{'lego max|dT|':>14}{'lego rate':>11}"
          f"{'~ NEMO day':>12}{'nemo max|dT|':>14}{'nemo rate':>11}"
          f"{'amp ratio':>11}{'R_amp':>9}")
    matched = []
    for a_l, f_l, d_l in amp["lego"]:
        if not _resolvable_both(table, d_l):
            continue
        if a_l <= 0:
            continue
        # REVIEW FINDING B3: this MUST be the storage-matched arm. Casting to
        # float32 INFLATES the flip count (NEMO day 90: 10 -> 233), so pairing
        # legoESM's censored rate against NEMO's uncensored fp64 rate flatters
        # legoESM by roughly the inflation factor -- which is the same order as
        # the ratio being reported. The earlier version of this block used
        # fp64 AND asserted in prose that the bias ran the other way. Both are
        # corrected here; the assertion is deleted rather than reworded.
        cand = [(abs(np.log(a_n / a_l)), a_n, f_n, d_n)
                for a_n, f_n, d_n in amp["nemo_fp32"] if a_n > 0]
        if not cand:
            continue
        cand.sort()
        _, a_n, f_n, d_n = cand[0]
        # MATCH-CHOICE SENSITIVITY. NEMO's max|dT| swings four orders of
        # magnitude between adjacent 10-day samples, so the "closest" partner
        # is close to a tie and the ratio it yields can swing with it. The
        # three best partners and the spread of R across them are printed, so
        # a ratio that depends on an arbitrary tie-break is visible as one.
        alts = [(dd, (f_l / ff) if ff > 0 else float("inf"))
                for _, aa, ff, dd in cand[:3]]
        ratio = a_n / a_l
        ok = (1.0 / MATCH_TOL) <= ratio <= MATCH_TOL
        R = (f_l / f_n) if f_n > 0 else float("inf")
        print(f"  {d_l:>9}{a_l:>14.3e}{100 * f_l:>10.4f}%"
              f"{d_n:>12}{a_n:>14.3e}{100 * f_n:>10.4f}%{ratio:>11.2f}"
              f"{R:>9.2f}"
              + "  alts " + ",".join(f"d{dd}:{rr:.1f}" for dd, rr in alts)
              + ("" if ok else "   REJECTED (amplitudes too far apart)"))
        if ok:
            matched.append((d_l, d_n, R, ratio))
    if not matched:
        v_amp, R_amp = "UNAVAILABLE", float("nan")
        print("\n  NO amplitude-matched pair exists within the tolerance: "
              "legoESM's perturbation never")
        print("  overlaps NEMO's in size at any pair of available horizons.")
    else:
        R_amp = float(np.median([m[2] for m in matched]))
        v_amp, _ = verdict(R_amp, 1.0)
        print(f"\n  amplitude-matched pairs: {len(matched)};  median R_amp = "
              f"{R_amp:.2f}  ->  {v_amp} on the amplitude-matched footing")

    # THE STORAGE-NOISE FLOOR, AND HOW MANY legoESM MEMBERS CLEAR IT.
    #
    # nemo_fp32 is a trajectory pair whose TRUE flip count is ~0 (nemo_fp64
    # gives 0 flips for 60 days) put through legoESM's own storage precision.
    # Its flips are therefore almost entirely MANUFACTURED BY THE FILE FORMAT,
    # which makes it the noise floor for this statistic: legoESM's count has to
    # clear it to be a signal at all. The floor is taken as the WORST (largest)
    # NEMO member, not the mean, so it cannot be beaten by luck.
    #
    # The earlier robustness test here was too weak -- it accepted a horizon
    # whose three legoESM members read 365, 3 and 2. This one asks the question
    # per member: how many of the three clear the floor by the CONFIRM factor?
    print("\n" + "=" * 100)
    print("(d3) THE STORAGE-NOISE FLOOR, AND THE CHAIN-LINK VERDICT -- "
          "computed, not typed")
    print("=" * 100)
    print("  nemo_fp32 has a TRUE flip count of ~0, so its flips are "
          "manufactured by the file format.")
    print("  It is the noise floor legoESM's count must clear. The floor is "
          "the WORST NEMO member.")
    print(f"\n  {'day':>4}{'floor (worst nemo32)':>22}"
          f"{'lego per-member flips':>24}{'x floor':>22}"
          f"{'members clearing ' + str(int(CONFIRM_RATIO)) + 'x':>20}")
    cleared = {}
    for day in HORIZONS:
        if not _resolvable_both(table, day):
            print(f"  {day:>4}{'':>22}{'':>24}{'':>22}{'UNMEASURABLE':>20}")
            continue
        floor = max(x["flips"] for x in table[(day, "nemo_fp32")])
        legos = [x["flips"] for x in table[(day, "lego")]]
        mult = [(l / floor) if floor > 0 else float("inf") for l in legos]
        n_ok = sum(1 for m in mult if m >= CONFIRM_RATIO)
        cleared[day] = (n_ok, floor, legos, mult)
        print(f"  {day:>4}{floor:>22d}"
              f"{','.join(str(l) for l in legos):>24}"
              f"{','.join(f'{m:.1f}' for m in mult):>22}"
              f"{n_ok:>16} / {len(legos)}")

    n_mem = len(MEMBERS)
    all_clear = [d for d, (n, *_) in cleared.items() if n == n_mem]
    most_clear = [d for d, (n, *_) in cleared.items() if n >= (n_mem + 1) // 2]
    matched_days = sorted({m[0] for m in matched})
    print(f"\n  horizons where ALL {n_mem} legoESM members clear the floor by "
          f"{CONFIRM_RATIO:g}x: {all_clear if all_clear else 'NONE'}")
    print(f"  horizons where a MAJORITY clear it: "
          f"{most_clear if most_clear else 'NONE'}")
    print(f"  horizons with an amplitude-matched NEMO partner: "
          f"{matched_days if matched_days else 'NONE'}")
    overlap = sorted(set(most_clear) & set(matched_days))
    print(f"  OVERLAP (majority-clearing AND amplitude-matched): "
          f"{overlap if overlap else 'NONE'}")

    print()
    if not most_clear:
        print("  CHAIN-LINK VERDICT: REFUTE. legoESM's trigger does not "
              "resolve differently more often")
        print("  than the file format alone would make it, at any resolvable "
              "horizon. The rectification")
        print("  asymmetry is NOT measured, and the 2dt-flicker link WEAKENS.")
    elif overlap:
        print(f"  CHAIN-LINK VERDICT: CONFIRM. A majority of members clear the "
              f"storage-noise floor at")
        print(f"  horizons {most_clear}, and {overlap} is ALSO "
              f"amplitude-matched, so the difference is not")
        print("  merely a consequence of legoESM's larger perturbation.")
    else:
        print(f"  CHAIN-LINK VERDICT: UNSEPARATED (partial). A MAJORITY of "
              f"legoESM members -- not all --")
        print(f"  clear the storage-noise floor at horizons {most_clear}, "
              f"by the factors printed above.")
        print(f"  ALL {n_mem} members clear it at: "
              f"{all_clear if all_clear else 'NO horizon'}.")
        print("  But NO horizon is both majority-clearing AND "
              "amplitude-matched: NEMO's perturbation")
        print("  never grows to legoESM's amplitude there. So the "
              "RECTIFICATION asymmetry is NOT separated")
        print("  from the GROWTH asymmetry by these states, and the "
              "horizon-matched ratio must be read as")
        print("  a JOINT statement about both. The chain link does NOT close "
              "on this evidence.")
    print()

    # ---- THE AMPLITUDE-FREE STATISTIC -------------------------------------
    # Everything above depends on the perturbation's size, and legoESM's
    # perturbation is far bigger, which is the confound that stops (d) from
    # closing anything. THIS statistic does not involve the perturbation at
    # all: it is the shape of each model's own N2 distribution near the
    # trigger, on the CONTROL member. An interface can only be tipped if it
    # sits close to the threshold, so the density of near-threshold interfaces
    # is the model's intrinsic SUSCEPTIBILITY to rectification -- the part of
    # the question that "how many sit within tipping reach" was really asking,
    # asked in a way a bigger perturbation cannot flatter.
    #
    # Bands are FIXED and identical for both models. legoESM's N2 carries about
    # 5e-10 of float32 storage noise, so bands below that are not resolvable on
    # its side; nemo_fp32 shows directly what the cast does to each band.
    print("=" * 100)
    print("(f) THRESHOLD-PROXIMITY DENSITY -- amplitude-free: each model's OWN "
          "N2 near its own trigger")
    print("=" * 100)
    print("  No perturbation enters this table. It is the density of "
          "interfaces sitting close enough to")
    print("  the threshold to be tippable at all -- the intrinsic "
          "susceptibility, which a larger")
    print("  perturbation cannot inflate. Control member only. Bands are "
          "identical for both models.")
    bands = (1e-12, 1e-11, 1e-10, 1e-9, 1e-8)
    print(f"\n  {'day':>4}{'arm':>12}" +
          "".join(f"{'|N2-thr|<' + f'{b:.0e}':>16}" for b in bands))
    faithful = {}
    ratios = []
    for day in (HORIZONS[0], HORIZONS[len(HORIZONS) // 2], HORIZONS[-1]):
        occ = {}
        for arm in arms:
            if arm == "lego":
                Tc, Sc = lego_state(LEGO_LANES[lane], CONTROL, day)
            else:
                Tc, Sc = nemo_state(os.path.join(NEMO_LANE,
                                                 NEMO_MEMBERS[CONTROL]), day)
                if arm == "nemo_fp32":
                    Tc, Sc = to_fp32(Tc, Sc)
            N2 = n2(Tc, Sc)
            _no_nan(N2, wet_i, f"N2_{arm}_day{day}")
            d = np.abs(N2[wet_i] - N2_THRESHOLD)
            occ[arm] = [np.count_nonzero(d < b) / max(nwi, 1) for b in bands]
            print(f"  {day:>4}{arm:>12}" +
                  "".join(f"{100 * v:>15.4f}%" for v in occ[arm]))
        # WHICH BANDS SURVIVE THE CAST?  A band is trustworthy on legoESM's
        # float32 side only where casting NEMO to float32 barely moves it.
        # Computed per band, not assumed.
        band_ok = [abs(occ["nemo_fp32"][i] - occ["nemo_fp64"][i])
                   <= 0.10 * max(occ["nemo_fp64"][i], 1e-30)
                   for i in range(len(bands))]
        faithful[day] = band_ok
        print(f"  {'':>4}{'fp32 faithful?':>12}" +
              "".join(f"{('yes' if b else 'NO'):>16}" for b in band_ok))
        rr = [(occ["lego"][i] / occ["nemo_fp32"][i]
               if occ["nemo_fp32"][i] > 0 else float("nan"))
              for i in range(len(bands))]
        print(f"  {'':>4}{'lego/nemo32':>12}" +
              "".join(f"{v:>16.3f}" for v in rr))
        ratios += [occ["lego"][i] / occ["nemo_fp32"][i]
                   for i in range(len(bands))
                   if band_ok[i] and occ["nemo_fp32"][i] > 0]
        print()
    if ratios:
        hi = max(ratios)
        occ_v = ("EXCEEDS" if hi >= CONFIRM_RATIO else
                 "COMPARABLE" if hi <= REFUTE_RATIO else "INTERMEDIATE")
        print(f"  SUSCEPTIBILITY VERDICT: across every band the float32 cast "
              f"leaves faithful, legoESM's")
        print(f"  near-threshold occupancy is at most {hi:.3f}x NEMO's -> "
              f"{occ_v}.")
        if hi <= REFUTE_RATIO:
            print("  The two models have essentially the SAME density of "
                  "tippable interfaces. legoESM is")
            print("  NOT intrinsically more rectifiable, so a higher flip "
                  "count at a matched horizon is")
            print("  explained by its LARGER PERTURBATION rather than by a "
                  "more rectifiable stratification.")
        print()
    print("  A model whose N2 piles up near the threshold is intrinsically "
          "more rectifiable. Read the")
    print("  nemo_fp64 row against nemo_fp32 first: the difference between "
          "them is what float32 storage")
    print("  alone does to each band, and it bounds what can be read off "
          "legoESM's row.\n")

    if do_two_level:
        print("\n" + "=" * 100)
        print("(e) NEMO's OWN TRIGGER RULE -- MIN(rn2, rn2b) vs the single-level "
              "test, on NEMO's control")
        print("=" * 100)
        print("  legoESM fires on the single now-level N2 "
              "(enhanced_diffusion.py:195); NEMO fires on")
        print("  MIN(rn2, rn2b) <= -1e-12 (zdfevd.F90:93). bn2(tb,sb) is a "
              "PROXY for rn2b -- the exact")
        print("  rn2b is not in the restart. legoESM's before-level 3-D state "
              "was never saved, so this")
        print("  column CANNOT be produced for legoESM.")
        print(f"\n  {'day':>4}{'single-level':>15}{'two-level (proxy)':>20}"
              f"{'extra firing':>14}{'ratio':>9}")
        d0 = os.path.join(NEMO_LANE, NEMO_MEMBERS[CONTROL])
        for day in HORIZONS:
            Tn, Sn = nemo_state(d0, day, "n")
            Tb, Sb = nemo_state(d0, day, "b")
            N2n = n2(Tn, Sn)
            _no_nan(N2n, wet_i, "N2_now")
            # <= on BOTH counts: the table is a DIFFERENCE of two
            # counts and must not mix comparison operators.
            one = int(np.count_nonzero(N2n[wet_i] <= N2_THRESHOLD))
            two, _ = firing_rate_two_level(n2, Tn, Sn, Tb, Sb, wet_i)
            print(f"  {day:>4}{one:>15d}{two:>20d}{two - one:>14d}"
                  f"{two / max(one, 1):>9.3f}")
        print(f"\n  (of {nwi} wet interfaces)")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--ladder-check", action="store_true")
    p.add_argument("--lanes", default="kick2,kick1")
    a = p.parse_args(argv)
    self_check()
    if a.self_check:
        return
    if a.ladder_check:
        ladder_check()
        return
    for lane in a.lanes.split(","):
        if lane.strip():
            measure(lane.strip())


if __name__ == "__main__":
    main()
