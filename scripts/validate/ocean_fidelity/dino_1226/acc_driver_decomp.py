#!/usr/bin/env python
"""#1455 SE: WHICH DRIVER's 90-day TIME-INTEGRAL diverges? -- ACC decomposed by
contributor on BOTH sides at the four matched twin times (days 0/30/60/90).

CONTEXT (SD, commit eaaa920a9): over the 90-day twin from a bit-identical NEMO
restart, NEMO's ACC spins up +3.19 Sv while legoESM tracks only ~55-60% of that
rate, reaching -1.557/-1.708/-1.740 Sv at day 90 for arms 1/2/3.  The shape is
an INTEGRATED per-step bias -- below every per-step probe's floor.  So the
question is no longer "which operator differs at one step" but "which DRIVER's
time-integral diverges while its instantaneous value agrees".

ROWS (each a separate physical contributor, per-time table on both sides):

  R1 THERMAL WIND     band meridional density contrast (upper/deep, the
                      acceptance gate's own metrics) and the bottom-referenced
                      baroclinic transport it predicts (A.thermal_wind).
  R2 TRANSPORT SPLIT  the SAME section the recorded ACC metric integrates, split
                      EXACTLY into a barotropic (bottom-referenced) part and the
                      shear above it.  Does the depth-mean lag, or the shear?
  R3 SURFACE FORCING  southern-band surface sigma MAX/MEAN (gate metrics): is the
                      surface buoyancy input integrating differently?
  R4 SURFACE CURRENT  the band's top-level zonal transport -- the only DAILY
                      series on the legoESM side (90 samples); NEMO at its
                      10-day restarts.
  R5 LATITUDE SPLIT   the SAME integrand restricted to the three latitude groups
                      the recorded metric sums over: south of the re-entrant
                      band (rows 0..J0-1), the band itself (J0..J1, the only
                      rows with all 52 longitudes wet), and north of it.  Added
                      because R2's whole-section barotropic/shear split turned
                      out to have huge cancelling parts (+0.74 / -2.29 Sv for a
                      -1.55 Sv total) and so cannot localise anything; the
                      latitude groups are additive and do not cancel.
  R5b LOCALISATION    per T-row day-90 transport inside the owning group, that
                      group's own barotropic/shear split at all four times, and
                      the per-longitude structure of the difference (is the
                      deficit broad-scale or a few-longitude feature?).

WEIGHTING / PROTOCOL -- identical for both models, nothing branches on model:
  * ACC + the R2 split : full 199-row section, e3t_1d reference thickness,
                         mask A.umask -- VERBATIM the recorded acc_full metric
                         (acc_thermal_wind.acc_full), so the split is additive to
                         the number the gate records.
  * reducer            : the recorded metric MEDIANs over longitudes 2..-2.  A
                         median is not additive, so the R2 split is reported
                         under the MEAN over the SAME longitudes (bt+bc == the
                         section transport EXACTLY), with the median total
                         printed next to it so the choice is visible.
  * R1/R3              : A.contrast_profile/A.depth_split/A.thermal_wind and the
                         gate's surface_sigma_south, on mesh_mask e3t_0/gdept_0
                         geometry with the tmask&land_mask wet mask -- unchanged.
  * NEMO side          : the RUN_90D_TWIN restarts, NOW level (tn/sn/un), kt
                         5760 + day*32; kt 5760 is the single-file IC restart
                         (the day-0 control the SD probe could not measure --
                         its glob required a per-rank suffix, so day 0 printed
                         '--' and its "day-0 == 0" control never actually ran;
                         corrected here).
  * legoESM side       : the gate2 arm npz 3-D snapshots via the gate's own
                         load_candidate (fp64 cast), daily surface u from the
                         same npz.

CONTROLS (all fatal, printed before any number is used):
  C1 dtype: every comparison array and the geometry behind it must be float64.
  C2 NaN on the wet mask is fatal on both sides.
  C3 day-0 identity: EVERY row must agree between lego and NEMO at day 0 to the
     fp32 snapshot quantum (the bridge is bit-identical).
  C4 the recorded day-90 ACC deficits 1.557/1.708/1.740 must be reproduced.
  C5 planted control for each NEW reduction (--self-test): a known barotropic
     perturbation must move the barotropic row by its analytic amount and leave
     the shear row EXACTLY unchanged, and a known T perturbation must move the
     density rows and leave the barotropic row EXACTLY unchanged.

Diagnosis only: reads recorded artifacts, writes nothing to packages/ or src/.

Run (fp64 required):
  JAX_ENABLE_X64=1 .venv/bin/python .../acc_driver_decomp.py ARM1.npz ARM2.npz ...
  JAX_ENABLE_X64=1 .venv/bin/python .../acc_driver_decomp.py --self-test
"""
import argparse
import glob
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))
import acc_thermal_wind as A            # noqa: E402  (recorded harness)
import acceptance_gate_90d as G         # noqa: E402  (the gate's loaders/metrics)
from rebuild_nemo_restart import rebuild  # noqa: E402

DAYS = (0, 30, 60, 90)
DAILY_DAYS = tuple(range(0, 91, 10))
RECORDED_D90 = {"arm1_pre": 1.557, "arm2_fixes": 1.708, "arm3_bn2": 1.740}
FP32_QUANTUM = 1e-5          # RELATIVE bound: fp32 storage of the snapshots

# The bottom-referenced split takes u at level (n_wet - 1); that is the deepest
# wet cell ONLY if every column's wet levels are a contiguous prefix.  Verified
# here rather than assumed (it holds for this mesh: 0 columns violate it).
assert np.array_equal(
    A.umask, np.arange(A.NZ)[None, None, :] < A.umask.sum(axis=2)[:, :, None]
), "umask columns are not contiguous from the surface -- u_bottom index is wrong"

_med = lambda a: float(np.median(np.asarray(a, dtype=np.float64)[2:-2]))   # noqa: E731
_avg = lambda a: float(np.mean(np.asarray(a, dtype=np.float64)[2:-2]))     # noqa: E731


# --------------------------------------------------------------- reductions ---
def section_bt_bc(u, wet_u, rows=None, e3=None):
    """EXACT split of the per-longitude section transport [Sv] into

        bt = u_bottom * H          (REFERENCE-LEVEL transport: what a
                                    depth-uniform field equal to the bottom
                                    value would carry -- NOT a true depth mean)
        bc = int (u - u_bottom) dz (the shear the thermal wind predicts)

    with bt + bc == the section transport IDENTICALLY (asserted by the caller).
    Default rows/e3 = ALL 199 rows and e3t_1d, i.e. the recorded acc_full
    weighting, so this splits the number the gate records.  u_bottom is the
    deepest wet cell-centre value (A.bc_bt_band's construction, generalised).
    """
    rows = slice(0, A.NY) if rows is None else rows
    us = np.asarray(u, dtype=np.float64)[rows]
    ws = wet_u[rows]
    e3s = (np.broadcast_to(np.asarray(A.e3t1d, dtype=np.float64), us.shape)
           if e3 is None else np.asarray(e3, dtype=np.float64)[rows])
    e2 = np.asarray(A.e2u_col, dtype=np.float64)[rows]
    col = ws.any(axis=2)
    kbot = np.where(col, ws.sum(axis=2) - 1, 0)
    ubot = np.where(col, np.take_along_axis(np.where(ws, us, 0.0),
                                            kbot[:, :, None], axis=2)[:, :, 0], 0.0)
    H = np.sum(np.where(ws, e3s, 0.0), axis=2)
    bt = np.einsum("ji,ji,j->i", ubot, H, e2) / 1e6
    bc = np.einsum("jik,j->i", np.where(ws, (us - ubot[:, :, None]) * e3s, 0.0), e2) / 1e6
    return bt, bc


def section_total(u, wet_u):
    """Per-longitude full-section transport [Sv], e3t_1d -- acc_full's integrand."""
    us = np.asarray(u, dtype=np.float64)
    return np.einsum("jik,k,j->i", np.where(wet_u, us, 0.0),
                     np.asarray(A.e3t1d, dtype=np.float64),
                     np.asarray(A.e2u_col, dtype=np.float64)) / 1e6


BAND = slice(A.J0, A.J1 + 1)
LAT_GROUPS = (("south of band", slice(0, A.J0)), ("BAND re-entrant", BAND),
              ("north of band", slice(A.J1 + 1, A.NY)))


def surface_transport(u_k0, wet_u, rows=BAND):
    """Top-level (k=0) zonal transport [Sv] over `rows` (default: the re-entrant
    channel band -- the only latitudes that can carry a net throughflow), same
    weights/reducer as acc_full.  u_k0 is (y, x) on NEMO u-columns."""
    us = np.asarray(u_k0, dtype=np.float64)[rows]
    w = wet_u[rows, :, 0]
    return _med(np.einsum("ji,j->i", np.where(w, us, 0.0),
                          np.asarray(A.e2u_col, dtype=np.float64)[rows])
                * float(A.e3t1d[0]) / 1e6)


def group_transport(u, wet_u, rows):
    """Full-section-weighted (e3t_1d) transport restricted to `rows` [Sv per lon].
    The three LAT_GROUPS sum EXACTLY to acc_full's integrand."""
    us = np.asarray(u, dtype=np.float64)[rows]
    return np.einsum("jik,k,j->i", np.where(wet_u[rows], us, 0.0),
                     np.asarray(A.e3t1d, dtype=np.float64),
                     np.asarray(A.e2u_col, dtype=np.float64)[rows]) / 1e6


# ------------------------------------------------------------------ loaders ---
def load_nemo(day, fields=("tn", "sn", "un")):
    """NEMO NOW-level state at `day` of the twin.  kt 5760 is the single-file IC
    restart; later kts are 16 per-rank tiles.  One glob covers both."""
    kt = G.KT_RESTART + day * G.STEPS_PER_DAY
    pat = f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart*.nc"
    if not glob.glob(pat):
        return None
    raw = rebuild(pat, list(fields))
    yxz = lambda a: np.moveaxis(np.asarray(a, dtype=np.float64), 0, -1)  # noqa: E731
    out = {}
    if "tn" in raw:
        out["T"] = yxz(raw["tn"])
    if "sn" in raw:
        out["S"] = yxz(raw["sn"])
    if "un" in raw:
        out["u"] = yxz(raw["un"])
    return out


def load_lego(npz, day):
    try:
        return G.load_candidate(npz, day)
    except SystemExit:
        return None


# ------------------------------------------------------------------- rows ----
def rowset(st, wet, *, with_density=True):
    """Every driver row for one state.  Reductions stated per key.

    The U-side mask is ALWAYS ``A.umask`` -- the recorded acc_full mask, applied
    identically to both models.  It is deliberately not a parameter: a per-model
    U-mask would be a protocol difference, i.e. a confound.
    """
    tot = section_total(st["u"], A.umask)          # acc_full's integrand, per lon
    bt, bc = section_bt_bc(st["u"], A.umask)
    assert np.allclose(bt + bc, tot, rtol=1e-9, atol=1e-7), "bt+bc != section total"
    out = {
        "acc_med": _med(tot),                      # == A.acc_full, the recorded metric
        "acc_avg": _avg(tot),
        "bt": _avg(bt),
        "bc": _avg(bc),
        "surf": surface_transport(np.asarray(st["u"], dtype=np.float64)[:, :, 0], A.umask),
    }
    # latitude-group split of the SAME integrand (additive under the mean reducer)
    grp = [group_transport(st["u"], A.umask, sl) for _, sl in LAT_GROUPS]
    assert np.allclose(sum(grp), tot, rtol=1e-12, atol=1e-9), "lat groups != section total"
    out.update({"g_south": _avg(grp[0]), "g_band": _avg(grp[1]), "g_north": _avg(grp[2])})
    # band-only split on the geometry the thermal wind uses (e3t_0), the
    # acc_thermal_wind sec.1 comparison -- B and D are then like-for-like
    # A.umask on BOTH sides (the gate's ACC mask) -- never branch on the model
    bcb, btb = A.bc_bt_band(np.asarray(st["u"], dtype=np.float64), A.umask)
    out.update({"bt_band": _avg(btb), "bc_band": _avg(bcb)})
    assert abs(out["acc_med"] - A.acc_full(st["u"], A.umask)) < 1e-9, "acc_med != A.acc_full"
    if with_density:
        rho = A.rho_of(st, wet)
        assert np.asarray(rho).dtype == np.float64, f"rho dtype {np.asarray(rho).dtype}"
        up, deep = A.depth_split(A.contrast_profile(rho, wet), wet)
        tw, _ = A.thermal_wind(rho, wet)
        smax, smean = G.surface_sigma_south(st, wet)
        out.update({"up": up, "deep": deep, "tw": _avg(tw), "tw_med": _med(tw),
                    "smax": smax, "smean": smean})
    return out


ROW_SPEC = [
    ("acc_med", "R0 ACC full-section [Sv]      (median lons 2..-2, e3t_1d)"),
    ("acc_avg", "R0 ACC same, MEAN reducer     (mean   lons 2..-2, e3t_1d)"),
    ("g_south", "R5 .. of which S of band [Sv] (mean, additive)"),
    ("g_band", "R5 .. of which BAND      [Sv] (mean, additive)"),
    ("g_north", "R5 .. of which N of band [Sv] (mean, additive)"),
    ("bt", "R2 .. BOTTOM-REF u_bot*H  [Sv] (mean, full sect, additive)"),
    ("bc", "R2 .. SHEAR above u_bot  [Sv] (mean, full sect, additive)"),
    ("bt_band", "R2b BAND bottom-ref [Sv]      (mean, band, e3t_0)"),
    ("bc_band", "R2b BAND shear      [Sv]      (mean, band, e3t_0)"),
    ("tw", "R1 thermal wind predicts B    (mean, band, e3t_0) [Sv]"),
    ("up", "R1 upper contrast <1400m [kg/m3]"),
    ("deep", "R1 deep  contrast >1400m [kg/m3]"),
    ("smax", "R3 S-band surf sigma MAX [kg/m3]"),
    ("smean", "R3 S-band surf sigma MEAN [kg/m3]"),
    ("surf", "R4 BAND top-level transp [Sv] (median, e3t_1d[0])"),
]


# ------------------------------------------------------------------ controls --
def check_finite(tag, st, wet):
    """NaN on the wet mask is fatal.  T/S use the T-mask, u uses the U-mask --
    a C-grid U-face can be dry where both its T-neighbours are wet."""
    for k, v in st.items():
        a = np.asarray(v)
        if a.ndim == 3 and a.shape[:2] == (A.NY, A.NX):
            m = A.umask if k == "u" else wet
            bad = int(np.sum(~np.isfinite(a[m])))
            if bad:
                raise SystemExit(f"FATAL: {tag} field {k} has {bad} non-finite wet cells")


def dtype_control():
    print("[C1 dtype] geometry + metric arrays (want float64):")
    for name in ("e3t0", "gdept0", "e3t1d", "gdept1d", "e2u_col"):
        a = np.asarray(getattr(A, name))
        print(f"    A.{name:9s} {a.dtype}")
        if a.dtype != np.float64:
            raise SystemExit(f"FATAL: A.{name} is {a.dtype}, not float64")


def self_test():
    """C5: planted controls for the NEW reductions (the R2 split and R4)."""
    dtype_control()
    nemo = load_nemo(90)
    if nemo is None:
        raise SystemExit("no NEMO day-90 restart for the self-test")
    wet = A.tmask
    base = rowset(nemo, wet)

    # (a) planted BAROTROPIC perturbation: u += du everywhere.
    du = 1e-3
    pert = {k: (v + du if k == "u" else v) for k, v in nemo.items()}
    p = rowset(pert, wet)
    H = np.sum(np.where(A.umask, np.broadcast_to(A.e3t1d, A.umask.shape), 0.0), axis=2)
    pred_bt = _avg(np.einsum("ji,j->i", H, np.asarray(A.e2u_col, dtype=np.float64)) * du / 1e6)
    pred_surf = _med(np.einsum("ji,j->i", A.umask[BAND, :, 0].astype(np.float64),
                               np.asarray(A.e2u_col, dtype=np.float64)[BAND])
                     * float(A.e3t1d[0]) * du / 1e6)
    got_bt, got_bc = p["bt"] - base["bt"], p["bc"] - base["bc"]
    got_surf = p["surf"] - base["surf"]
    print(f"[C5a planted barotropic du={du}] bt moved {got_bt:+.4f} Sv "
          f"(analytic {pred_bt:+.4f}) | shear moved {got_bc:+.3e} Sv (want 0) | "
          f"surf moved {got_surf:+.4f} (analytic {pred_surf:+.4f})")
    assert abs(got_bt - pred_bt) < 1e-6 * max(abs(pred_bt), 1.0), "bt != analytic"
    assert abs(got_bc) < 1e-9, "a pure barotropic shift moved the shear row"
    assert abs(got_surf - pred_surf) < 1e-6 * max(abs(pred_surf), 1.0), "surf != analytic"
    assert abs(pred_bt) > 1.0, "planted perturbation too small to be a control"

    # (b) planted DENSITY perturbation: -0.5 K on the southern half-band.
    jmid = (A.J0 + A.J1) // 2
    pert2 = {k: v.copy() for k, v in nemo.items()}
    pert2["T"][A.J0:jmid + 1] -= 0.5
    q = rowset(pert2, wet)
    print(f"[C5b planted T-0.5K south half-band] up {q['up'] - base['up']:+.5f} "
          f"deep {q['deep'] - base['deep']:+.5f} smean {q['smean'] - base['smean']:+.5f} "
          f"tw {q['tw'] - base['tw']:+.3f} Sv | bt moved {q['bt'] - base['bt']:+.3e} (want 0)")
    assert abs(q["up"] - base["up"]) > 10 * 1.1e-4, "density row did not move"
    assert abs(q["tw"] - base["tw"]) > 1.0, "thermal-wind row did not move"
    assert abs(q["bt"] - base["bt"]) < 1e-12, "a pure T perturbation moved the barotropic row"
    print("SELF-TEST PASS (both planted controls behave as predicted)")
    return 0


# -------------------------------------------------------------------- main ----
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arms", nargs="*", help="gate2 arm npz files (--save-3d output)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.arms:
        ap.error("give at least one arm npz (or --self-test)")

    dtype_control()
    print()

    # ---- load both sides at the matched times -------------------------------
    nemo = {d: load_nemo(d) for d in DAYS}
    missing = [d for d, v in nemo.items() if v is None]
    if missing:
        raise SystemExit(f"FATAL: no NEMO restart for days {missing}")
    lego = {os.path.basename(a).replace(".npz", ""): {d: load_lego(a, d) for d in DAYS}
            for a in args.arms}
    npz = {os.path.basename(a).replace(".npz", ""): np.load(a) for a in args.arms}

    # wet masks: the gate's -- A.tmask & land_mask for density, A.umask for ACC
    any_arm = next(iter(lego.values()))
    wet = A.tmask & (any_arm[0]["land_mask"] > 0.5)[:, :, None]
    for name, arm in lego.items():
        w = A.tmask & (arm[0]["land_mask"] > 0.5)[:, :, None]
        assert np.array_equal(w, wet), f"{name}: different land_mask -- not a controlled pair"
    print(f"[protocol] wet T-cells {int(wet.sum())} (tmask alone {int(A.tmask.sum())}); "
          f"band rows {A.J0}..{A.J1}; ACC mask A.umask ({int(A.umask.sum())} cells)")

    # ---- C2 NaN fatal --------------------------------------------------------
    for d in DAYS:
        check_finite(f"NEMO d{d}", nemo[d], wet)
        for name, arm in lego.items():
            check_finite(f"{name} d{d}", {k: v for k, v in arm[d].items()
                                          if k != "land_mask"}, wet)
    print("[C2 NaN] no non-finite wet cells on either side, all four times")

    # ---- C6 telescoping precondition behind the R1 contrast rows -------------
    # A.contrast_profile sums ADJACENT meridional differences across the band and
    # calls the result f(45S) - f(64S).  That telescoping identity holds ONLY where
    # every v-pair in the band is wet at that (lon, level).  J0/J1 are defined from
    # SURFACE wetness, so deep levels can have interior dry gaps -- where they do,
    # the "contrast" silently becomes a partial sum.  Measured, not assumed.
    _a, _b = slice(A.J0, A.J1), slice(A.J0 + 1, A.J1 + 1)
    _pair = wet[_a] & wet[_b]
    _used = _pair.any(axis=0)
    _gapped = _used & ~_pair.all(axis=0)
    print(f"\n[C6 telescoping] {int(_gapped.sum())} of {int(_used.sum())} (lon,level) "
          f"columns inside the band have an INTERIOR dry gap, so A.contrast_profile "
          f"is a partial sum there")
    if _gapped.any():
        _lev = sorted({int(k) for k in np.where(_gapped)[1]})
        print(f"    affected levels {_lev} = {[round(float(A.gdept1d[k]), 0) for k in _lev]} m"
              f" -- ALL below the {A.DEEP_M:.0f} m split, i.e. they bias the DEEP "
              f"contrast row only")
        print("    (this is a property of the RECORDED metric, identical on both sides; "
              "the deficits below are differences, in which it is common-mode)")

    # ---- compute every row ---------------------------------------------------
    R_nemo = {d: rowset(nemo[d], wet) for d in DAYS}
    R_lego = {n: {d: rowset(arm[d], wet) for d in DAYS} for n, arm in lego.items()}

    # ---- C3 day-0 identity ---------------------------------------------------
    print("\n[C3 day-0 identity] |lego - NEMO| at day 0, per row, as a RELATIVE "
          f"fraction of |NEMO's own value| (bound {FP32_QUANTUM:.0e}; the arms are\n"
          "    started from the same NEMO restart, so every row must be identical to "
          "the fp32 snapshot quantum)")
    bad = []
    for key, lab in ROW_SPEC:
        line = f"    {lab:48s}"
        for name in R_lego:
            d0 = abs(R_lego[name][0][key] - R_nemo[0][key])
            rel = d0 / max(abs(R_nemo[0][key]), 1e-30)
            line += f"  {name}:{d0:.1e}({rel:.0e})"
            if rel > FP32_QUANTUM:
                bad.append((name, key, d0, rel))
        print(line)
    if bad:
        print("    !! day-0 rows ABOVE the fp32 bound (NOT identical at t=0):")
        for n, k, v, r in bad:
            print(f"       {n} {k} abs {v:.3e} rel {r:.2e}")
        raise SystemExit("FATAL C3: the arms are not identical to NEMO at day 0 -- "
                         "every number below would be measured from a different "
                         "initial state, not from the 90-day divergence")

    # ---- C4 recorded day-90 reproduction ------------------------------------
    print("\n[C4 recorded day-90 ACC deficit -- SIGNED: the recorded values are "
          "legoESM BELOW NEMO]")
    failed = []
    for name in R_lego:
        d90 = R_lego[name][90]["acc_med"] - R_nemo[90]["acc_med"]
        rec = RECORDED_D90.get(name)
        ok = rec is not None and abs(d90 - (-rec)) < 2e-3      # signed, not |.|
        print(f"    {name:12s} {d90:+.3f} Sv vs recorded {-rec if rec else None}  "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)
    if failed:
        raise SystemExit(f"FATAL C4: {failed} do not reproduce the recorded day-90 "
                         "deficit -- this instrument is not the gate's, so nothing "
                         "below is comparable to the recorded campaign numbers")

    # ---- THE TABLE -----------------------------------------------------------
    for name in R_lego:
        print("\n" + "=" * 100)
        print(f"ARM {name}: driver rows, both sides, at the four matched times")
        print("=" * 100)
        print(f"{'row':50s}{'day':>5s}{'NEMO':>12s}{'lego':>12s}{'lego-NEMO':>12s}"
              f"{'frac of d90':>13s}")
        for key, lab in ROW_SPEC:
            d90 = R_lego[name][90][key] - R_nemo[90][key]
            for i, d in enumerate(DAYS):
                diff = R_lego[name][d][key] - R_nemo[d][key]
                frac = diff / d90 if abs(d90) > 1e-12 else np.nan
                print(f"{(lab if i == 0 else ''):50s}{d:5d}"
                      f"{R_nemo[d][key]:12.5f}{R_lego[name][d][key]:12.5f}"
                      f"{diff:+12.5f}{frac:+13.2f}")
            print()

        # additive attribution of the ACC deficit
        acc90 = R_lego[name][90]["acc_avg"] - R_nemo[90]["acc_avg"]
        bt90 = R_lego[name][90]["bt"] - R_nemo[90]["bt"]
        bc90 = R_lego[name][90]["bc"] - R_nemo[90]["bc"]
        tw90 = R_lego[name][90]["tw"] - R_nemo[90]["tw"]
        print(f"  ADDITIVE (mean reducer, exact): day-90 ACC deficit {acc90:+.3f} Sv"
              f"  =  bottom-ref {bt90:+.3f}  +  shear {bc90:+.3f}")
        print(f"    bottom-ref share {100 * bt90 / acc90:6.1f}%   "
              f"shear share {100 * bc90 / acc90:6.1f}%")
        gs = [R_lego[name][90][k] - R_nemo[90][k] for k in ("g_south", "g_band", "g_north")]
        print(f"  LATITUDE split (mean reducer, exact): {acc90:+.3f} Sv  =  S of band "
              f"{gs[0]:+.3f}  +  BAND {gs[1]:+.3f}  +  N of band {gs[2]:+.3f}")
        btb = R_lego[name][90]["bt_band"] - R_nemo[90]["bt_band"]
        bcb = R_lego[name][90]["bc_band"] - R_nemo[90]["bc_band"]
        print(f"  BAND-only (e3t_0) split: {btb + bcb:+.3f} Sv  =  bottom-ref {btb:+.3f}"
              f"  +  shear {bcb:+.3f};  thermal wind predicts the shear as {tw90:+.3f} Sv"
              f" (D/B ratio {tw90 / bcb if abs(bcb) > 1e-9 else np.nan:+.2f})")

    # ---- R5b: LOCALISE the south-of-band group (the row that owns the deficit) --
    print("\n" + "=" * 100)
    print("R5b SOUTH-OF-BAND LOCALISATION -- per T-row day-90 transport [Sv] "
          "(mean lons 2..-2, e3t_1d)\n     and the barotropic/shear split of the "
          "group at all four times (same weighting, additive)")
    print("=" * 100)
    print(f"{'row':>4s}{'lat':>8s}{'NEMO d0':>10s}{'NEMO d90':>10s}"
          + "".join(f"{n + ' d90':>16s}" for n in R_lego))
    for j in range(A.J0):
        rw = slice(j, j + 1)
        n0 = _avg(group_transport(nemo[0]["u"], A.umask, rw))
        n90 = _avg(group_transport(nemo[90]["u"], A.umask, rw))
        line = f"{j:4d}{A.gphit[j, 25]:8.1f}{n0:10.4f}{n90:10.4f}"
        for name in R_lego:
            line += f"{_avg(group_transport(lego[name][90]['u'], A.umask, rw)) - n90:+16.4f}"
        print(line)
    # Is the deficit a BROAD-SCALE bias or a few-longitude feature?  Compare the
    # spread of the per-longitude DIFFERENCE against NEMO's own spread across the
    # same longitudes 2..-2, day 90.  (No group's single-section transport is a
    # clean throughflow: every group has a 3-10 Sv longitude spread, so the
    # recorded metric is a MEDIAN over a highly longitude-variable quantity --
    # the difference is what is well posed, and it starts at exactly 0.)
    print("\n  per-longitude structure at day 90 [Sv], lons 2..-2:")
    print(f"    {'group':18s}{'wet cols':>9s}{'NEMO std':>10s}"
          + "".join(f"{n + ' mean/std':>22s}" for n in R_lego))
    for lab, sl in LAT_GROUPS:
        cols = int(np.median(A.tmask[sl, :, 0].sum(axis=1)))
        ng = group_transport(nemo[90]["u"], A.umask, sl)[2:-2]
        line = f"    {lab:18s}{cols:>9d}{float(np.std(ng)):10.3f}"
        for name in R_lego:
            dg = group_transport(lego[name][90]["u"], A.umask, sl)[2:-2] - ng
            line += f"{float(np.mean(dg)):12.3f}/{float(np.std(dg)):<9.3f}"
        print(line)

    print(f"\n{'':28s}{'day':>5s}{'NEMO':>12s}" + "".join(f"{n:>14s}" for n in R_lego))
    for lab, which in (("S-of-band BOTTOM-REF [Sv]", 0), ("S-of-band SHEAR      [Sv]", 1)):
        for d in DAYS:
            nv = _avg(section_bt_bc(nemo[d]["u"], A.umask, rows=slice(0, A.J0))[which])
            line = f"{(lab if d == 0 else ''):28s}{d:5d}{nv:12.4f}"
            for name in R_lego:
                lv = _avg(section_bt_bc(lego[name][d]["u"], A.umask,
                                        rows=slice(0, A.J0))[which])
                line += f"{lv - nv:+14.4f}"
            print(line)
        print()

    # ---- R4 daily series -----------------------------------------------------
    print("\n" + "=" * 100)
    print("R4 DAILY top-level zonal transport [Sv] (median lons 2..-2, e3t_1d[0]);"
          " NEMO only at its 10-day restarts")
    print("=" * 100)
    nemo_u = {}
    for d in DAILY_DAYS:
        s = load_nemo(d, fields=("un",))
        nemo_u[d] = (surface_transport(s["u"][:, :, 0], A.umask) if s is not None else None)
    hdr = f"{'day':>4s}{'NEMO':>10s}" + "".join(f"{n:>14s}" for n in R_lego)
    print(hdr)
    for d in DAILY_DAYS:
        row = f"{d:4d}" + (f"{nemo_u[d]:10.4f}" if nemo_u[d] is not None else f"{'--':>10s}")
        for name in R_lego:
            if d == 0:
                v = R_lego[name][0]["surf"]
            else:
                # daily npz surface u == u3d[:, 1:53, 0] (kamm_twin_90d.py), the
                # same columns load_candidate uses; day index d-1.
                v = surface_transport(np.asarray(npz[name]["u"][d - 1], dtype=np.float64),
                                      A.umask)
            row += f"{v:14.4f}" + ""
        print(row)
    # control: the daily array at a snapshot day must equal the 3-D snapshot's k=0
    for name in R_lego:
        for d in (30, 60, 90):
            a = surface_transport(np.asarray(npz[name]["u"][d - 1], dtype=np.float64), A.umask)
            b = R_lego[name][d]["surf"]
            assert abs(a - b) < 1e-9, f"{name} d{d}: daily u != 3-D snapshot k=0 ({a} vs {b})"
    print("[control] daily surface u reproduces the 3-D snapshot's k=0 transport "
          "exactly at days 30/60/90 -- the daily series is the same field")

    # per-arm daily deficit vs NEMO at the 10-day marks
    print("\nlego - NEMO at the 10-day marks [Sv]:")
    print(f"{'day':>4s}" + "".join(f"{n:>14s}" for n in R_lego))
    for d in DAILY_DAYS:
        if nemo_u[d] is None:
            continue
        row = f"{d:4d}"
        for name in R_lego:
            v = (R_lego[name][0]["surf"] if d == 0 else
                 surface_transport(np.asarray(npz[name]["u"][d - 1], dtype=np.float64), A.umask))
            row += f"{v - nemo_u[d]:+14.4f}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
