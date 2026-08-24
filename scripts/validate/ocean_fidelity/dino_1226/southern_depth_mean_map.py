#!/usr/bin/env python
"""#1455 SF: WHERE, per column, does the SUB-POLAR depth-mean flow spin up slower?

CONTEXT (SD/SE, commit b2ec54382): the "ACC deficit" is 76-85% owned by the
latitudes SOUTH of the re-entrant channel (T-rows 0..J0-1).  Inside that group
the bottom-referenced (u_bot*H) part carries 0 / -0.375 / -0.700 / -0.987 Sv at
days 0/30/60/90 while the shear part flattens after d60.  Density/thermal wind,
surface buoyancy forcing and the surface current are ruled out as drivers.

THIS PROBE goes from the group INTEGRAL to the per-COLUMN map, on both sides at
the four matched times, and asks what the map's STRUCTURE is -- not what causes
it.  Rows:

  M1 depth-mean zonal velocity  ubar = sum_k e3*u / H  per (row, lon), both
     sides, all four days, plus the day-30/60/90 growth per column.
  M2 depth-mean meridional velocity vbar (same construction on v-points).
     CAVEAT, established in review: every southern row is walled at BOTH ends, so
     the zonal integral of the depth-integrated meridional transport is ZERO by
     mass conservation (measured: 0.000-0.006 Sv against 3.5-44 Sv of gross
     throughput).  The MEAN of dvbar is therefore near-zero by construction and
     rules NOTHING out; only its structure carries information.
  M6 the same closure means the accumulated zonal transport from the wall IS the
     barotropic streamfunction of a closed circulation -- reported per row.
  M3 column depth H and bathymetry beside the bias: day-90 dubar binned by
     depth quartile and by distance north of the domain's southern wall, with
     BOTH an area-weighted mean dubar and the additive transport [Sv] each bin
     contributes to the recorded group deficit.
  M4 the SOUTHERN group's OWN meridional density contrast (A.contrast_profile /
     A.depth_split re-pointed at rows 0..J0-1), upper/deep, both sides, four
     days -- does the local density ALSO capture only ~55% of NEMO's change?
  M5 sign: is the group's absolute depth-mean flow westward on both sides, and
     is legoESM LESS westward (under-driven) or MORE (over-damped)?

WEIGHTING / PROTOCOL -- identical for both models, nothing branches on model:
  * u-columns   : mask A.umask, thickness e3t_1d (the RECORDED acc_full
                  weighting, so the per-column transports are ADDITIVE to the
                  committed group number).  H_ref = sum_k e3t_1d over A.umask.
                  The real partial-cell depth H_real = sum_k e3u_0 over umask is
                  reported alongside and is what the depth BINS use.
  * v-columns   : mesh_mask vmask, thickness e3v_0 (no recorded metric to match).
  * lego u      : G.load_candidate (faces 1..52 + the recorded seam fix).
  * lego v      : raw v3d_day{d} (200,52,36); the row offset against NEMO's vn is
                  MEASURED at day 0 (control V), never assumed.
  * NEMO        : RUN_90D_TWIN restarts, NOW level (tn/sn/un/vn), kt 5760+32*day.
  * reducer     : the recorded metric's MEAN over longitudes 2..-2 wherever a
                  number must be additive to the committed one.

CONTROLS (fatal, printed before any number is used):
  C1 dtype  : every geometry/state array float64 (D.dtype_control + our own).
  C2 NaN    : non-finite on the wet mask is fatal, both sides, all four days.
  C3 day-0  : per-column |dubar| and |dvbar| at day 0 must be at the fp32
              snapshot quantum (the arms start from the same NEMO restart).
  C4 RECONCILE (Rule 1e): summing OUR per-column transport map must reproduce
              (a) the committed g_south group deficits and (b) the committed
              bottom-referenced series 0/-0.375/-0.700/-0.987 via D's own
              reduction.  NOTE the two are DIFFERENT quantities -- (b) is
              u_bot*H, not a depth mean; see the printed reconciliation.
  V  v-offset: the lego v row offset is picked by day-0 agreement, and the
              runner-up offset must be clearly worse.
  P  planted : +1 mm/s on lego's u must move every wet dubar by exactly +1e-3
              and every binned mean by the same.

Diagnosis only: reads recorded artifacts, writes only its own npz/png.

Run (fp64):
  JAX_ENABLE_X64=1 .venv/bin/python .../southern_depth_mean_map.py ARM.npz [...] \
      --out-npz /path/maps.npz --out-png /path/dubar_d90.png
"""
import argparse
import glob
import os
import sys

import netCDF4 as nc
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))
import acc_thermal_wind as A            # noqa: E402  (recorded harness: mesh, masks, J0/J1)
import acceptance_gate_90d as G         # noqa: E402  (the gate's loaders)
import acc_driver_decomp as D           # noqa: E402  (the COMMITTED reductions)
from rebuild_nemo_restart import rebuild  # noqa: E402

DAYS = (0, 30, 60, 90)
SOUTH = slice(0, A.J0)                  # the group that owns the deficit
LONS = slice(2, -2)                     # the recorded metric's longitudes
FP32_REL = 1e-5

# committed (b2ec54382) south-of-band rows, arm1, days 0/30/60/90
COMMITTED_BOTREF = (0.0, -0.375, -0.700, -0.987)
NOISE_CONTRAST = 1.1e-4   # recorded noise floor of the contrast metric
COMMITTED_GSOUTH_D90 = {"arm1_pre": -1.320, "arm2_fixes": -1.297, "arm3_bn2": -1.483}

# ------------------------------------------------------------------ geometry --
_mm = nc.Dataset(f"{A.DINO}/RUN_TRAJ/mesh_mask.nc")
e3u0 = A.llz(_mm["e3u_0"][0])                      # (y,x,z) partial-cell u thickness
e3v0 = A.llz(_mm["e3v_0"][0])
vmask = A.llz(_mm["vmask"][0]) > 0.5
e1u = np.asarray(_mm["e1u"][0]).squeeze()
e2u_full = np.asarray(_mm["e2u"][0]).squeeze()
e2v_col = np.asarray(_mm["e2v"][0]).squeeze()[:, 25]
e3t1d = np.asarray(A.e3t1d, dtype=np.float64)
e2u_col = np.asarray(A.e2u_col, dtype=np.float64)   # the recorded metric's e2u

# H on the RECORDED weighting (e3t_1d) and on the real partial-cell geometry
H_ref = np.sum(np.where(A.umask, np.broadcast_to(e3t1d, A.umask.shape), 0.0), axis=2)
H_real = np.sum(np.where(A.umask, e3u0, 0.0), axis=2)
H_v = np.sum(np.where(vmask, e3v0, 0.0), axis=2)
# distance north of the domain's southern wall, at the T-row centres [km]
Y_KM = np.concatenate([[0.0], np.cumsum(e2v_col[:-1])]) / 1e3


# ----------------------------------------------------------------- reductions --
def depth_mean_u(u, e3=None):
    """ubar = sum_k e3*u / H over A.umask, per (row, lon) [m/s].  Dry columns NaN.

    Default e3 = e3t_1d, i.e. the thickness the recorded acc_full metric uses, so
    e2u*H_ref*ubar/1e6 is EXACTLY that metric's per-column integrand.
    """
    us = np.asarray(u, dtype=np.float64)
    w3 = np.broadcast_to(e3t1d, A.umask.shape) if e3 is None else np.asarray(e3, np.float64)
    H = np.sum(np.where(A.umask, w3, 0.0), axis=2)
    num = np.sum(np.where(A.umask, us * w3, 0.0), axis=2)
    return np.where(H > 0, num / np.where(H > 0, H, 1.0), np.nan)


def depth_mean_v(v):
    """vbar = sum_k e3v_0*v / H_v over vmask, per (row, lon) [m/s]."""
    vs = np.asarray(v, dtype=np.float64)
    num = np.sum(np.where(vmask, vs * e3v0, 0.0), axis=2)
    return np.where(H_v > 0, num / np.where(H_v > 0, H_v, 1.0), np.nan)


def col_transport(ubar):
    """Per-column contribution [Sv] to the recorded group number: e2u*H_ref*ubar,
    /1e6, and /nlon so that a plain sum over the map == the reported mean-reducer
    value.  Columns outside longitudes 2..-2 contribute 0 (the reducer drops
    them)."""
    nlon = A.NX - 4
    t = np.where(np.isfinite(ubar), np.nan_to_num(ubar), 0.0) * H_ref * e2u_col[:, None] / 1e6
    out = np.zeros_like(t)
    out[:, LONS] = t[:, LONS] / nlon
    return out


# -------------------------------------------------------------------- loaders --
def load_nemo(day):
    kt = G.KT_RESTART + day * G.STEPS_PER_DAY
    pat = f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart*.nc"
    if not glob.glob(pat):
        raise SystemExit(f"FATAL: no NEMO restart for day {day} ({pat})")
    raw = rebuild(pat, ["tn", "sn", "un", "vn"])
    yxz = lambda a: np.moveaxis(np.asarray(a, dtype=np.float64), 0, -1)  # noqa: E731
    return {"T": yxz(raw["tn"]), "S": yxz(raw["sn"]),
            "u": yxz(raw["un"]), "v": yxz(raw["vn"])}


def load_lego(npz_path, npz, day, voff):
    st = G.load_candidate(npz_path, day)
    st["v"] = np.asarray(npz[f"v3d_day{day}"], dtype=np.float64)[voff:voff + A.NY]
    return st


def pick_v_offset(npz, nemo0):
    """Control V: the (200,52,36) lego v array holds v-FACES; which 199 of them
    correspond to NEMO's vn rows is MEASURED at day 0, not assumed."""
    ref = np.where(vmask, nemo0["v"], 0.0)
    scores = {}
    for off in (0, 1):
        lv = np.asarray(npz["v3d_day0"], dtype=np.float64)[off:off + A.NY]
        scores[off] = float(np.max(np.abs(np.where(vmask, lv, 0.0) - ref)))
    best = min(scores, key=scores.get)
    other = 1 - best
    print(f"[control V] lego v row offset: max|lego-NEMO| at day 0 = "
          + ", ".join(f"off {o}: {s:.3e} m/s" for o, s in sorted(scores.items()))
          + f"  -> using offset {best}")
    if not (scores[best] < 1e-5 and scores[other] > 100 * max(scores[best], 1e-12)):
        raise SystemExit("FATAL control V: the v row offset is not decided by the "
                         "day-0 identity -- every vbar number below would be a "
                         "half-cell shifted comparison")
    return best


# -------------------------------------------------------------------- helpers --
def group_density_contrast(st, wet):
    """A.contrast_profile / A.depth_split RE-POINTED at the southern rows.

    Reuses the recorded machinery verbatim (it reads A.J0/A.J1 at call time);
    nothing is re-derived here.  Returns (upper, deep) contrast [kg/m3] taken
    across rows 0..J0-1 exactly as the recorded band version takes it across the
    channel."""
    j0, j1 = A.J0, A.J1
    try:
        A.J0, A.J1 = 0, j0 - 1
        assert (A.J0, A.J1) != (j0, j1), "the re-point did not change the row range"
        rho = A.rho_of(st, wet)
        prof = A.contrast_profile(rho, wet)
        up, deep = A.depth_split(prof, wet)
        # levels whose telescoping is UNBROKEN across every southern v-pair; below
        # the first break the "contrast" is a partial sum, so report both.
        a, b = slice(A.J0, A.J1), slice(A.J0 + 1, A.J1 + 1)
        pair = wet[a] & wet[b]
        # a LEVEL is clean when every longitude that participates has all of its
        # row-pairs wet there (the C6 test of the committed probe, per level).
        used = pair.any(axis=0)                       # (lon, lev) participates
        gapped = used & ~pair.all(axis=0)             # ...but with an interior gap
        unbroken = ~gapped.any(axis=0)                # (lev,)
        prof_ok = np.where(unbroken, prof, np.nan)
        if int(unbroken.sum()) < 2:
            # no clean levels exist south of the channel: depth_split of an empty
            # selection returns 0.0, a plausible-looking sentinel.  Return NaN.
            up_ok = deep_ok = float("nan")
        else:
            up_ok, deep_ok = A.depth_split(prof_ok, wet)
    finally:
        A.J0, A.J1 = j0, j1
    if (A.J0, A.J1) != (j0, j1):
        raise SystemExit("FATAL: band indices not restored")
    return up, deep, up_ok, deep_ok, int(unbroken.sum()), int(unbroken.size)


def binned(dub, bins, labels, tcol_diff, area, HH):
    """area-weighted mean dubar [mm/s] and additive transport [Sv] per bin."""
    rows = []
    for b, lab in zip(bins, labels):
        m = b & np.isfinite(dub)
        w = area[m]
        if m.sum() == 0 or np.sum(w) <= 0:
            # an empty bin returns NaN, never a plausible-looking 0.0 (a sentinel
            # of the right dtype passes every finite guard downstream)
            rows.append((lab, 0, np.nan, np.nan, np.nan, np.nan))
            continue
        mean = 1e3 * float(np.sum(dub[m] * w) / np.sum(w))
        rows.append((lab, int(m.sum()), mean, float(np.sum(tcol_diff[m])),
                     float(np.sum(HH[m] * w) / np.sum(w)),
                     float(np.sum(dub[m] * HH[m] * w) / np.sum(w))))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args(argv)

    D.dtype_control()
    for nm, a in (("e3u_0", e3u0), ("e3v_0", e3v0), ("H_ref", H_ref),
                  ("H_real", H_real), ("e2v", e2v_col)):
        print(f"    {nm:9s} {np.asarray(a).dtype}")
        if np.asarray(a).dtype != np.float64:
            raise SystemExit(f"FATAL: {nm} is not float64")

    nemo = {d: load_nemo(d) for d in DAYS}
    npz = {os.path.basename(p).replace(".npz", ""): np.load(p) for p in args.arms}
    paths = {os.path.basename(p).replace(".npz", ""): p for p in args.arms}
    voff = pick_v_offset(next(iter(npz.values())), nemo[0])
    for n, z in npz.items():
        if pick_v_offset(z, nemo[0]) != voff:
            raise SystemExit(f"FATAL: arm {n} needs a different v offset")
    lego = {n: {d: load_lego(paths[n], npz[n], d, voff) for d in DAYS} for n in npz}

    wet = A.tmask & (next(iter(lego.values()))[0]["land_mask"] > 0.5)[:, :, None]
    for n in lego:
        if not np.array_equal(A.tmask & (lego[n][0]["land_mask"] > 0.5)[:, :, None], wet):
            raise SystemExit(f"FATAL: arm {n} has a different land_mask -- not a "
                             "controlled pair")
    print(f"[protocol] southern group = T-rows 0..{A.J0 - 1} "
          f"({A.gphit[0, 25]:.1f}..{A.gphit[A.J0 - 1, 25]:.1f} lat); "
          f"{int(np.isfinite(depth_mean_u(nemo[0]['u'])[SOUTH]).sum())} wet u-columns; "
          f"{int((H_v[SOUTH] > 0).sum())} wet v-columns")

    # ---- C2 NaN fatal ------------------------------------------------------
    for d in DAYS:
        for tag, st in [("NEMO", nemo[d])] + [(n, lego[n][d]) for n in lego]:
            for k, m in (("u", A.umask), ("v", vmask), ("T", wet), ("S", wet)):
                bad = int(np.sum(~np.isfinite(np.asarray(st[k], np.float64)[m])))
                if bad:
                    raise SystemExit(f"FATAL C2: {tag} d{d} field {k}: {bad} non-finite wet cells")
    print("[C2 NaN] no non-finite wet u/v/T/S cells on either side, all four days")

    # ---- the maps ----------------------------------------------------------
    UB = {"NEMO": {d: depth_mean_u(nemo[d]["u"]) for d in DAYS}}
    VB = {"NEMO": {d: depth_mean_v(nemo[d]["v"]) for d in DAYS}}
    UBP = {"NEMO": {d: depth_mean_u(nemo[d]["u"], e3u0) for d in DAYS}}   # partial-cell e3
    for n in lego:
        UB[n] = {d: depth_mean_u(lego[n][d]["u"]) for d in DAYS}
        VB[n] = {d: depth_mean_v(lego[n][d]["v"]) for d in DAYS}
        UBP[n] = {d: depth_mean_u(lego[n][d]["u"], e3u0) for d in DAYS}

    # ---- C3 day-0 identity per column --------------------------------------
    print("\n[C3 day-0 identity per column, southern group]")
    for n in lego:
        du0 = (UB[n][0] - UB["NEMO"][0])[SOUTH]
        dv0 = (VB[n][0] - VB["NEMO"][0])[SOUTH]
        s_u = float(np.nanmax(np.abs(UB["NEMO"][0][SOUTH])))
        s_v = float(np.nanmax(np.abs(VB["NEMO"][0][SOUTH])))
        nu_, nv_ = int(np.isfinite(du0).sum()), int(np.isfinite(dv0).sum())
        if nu_ == 0 or nv_ == 0:
            raise SystemExit("FATAL C3: no finite columns to compare -- the control "
                             "would have passed vacuously")
        mu, mv = float(np.nanmax(np.abs(du0))), float(np.nanmax(np.abs(dv0)))
        print(f"    {n:12s} max|dubar| {mu:.3e} m/s (rel {mu / s_u:.1e})  "
              f"max|dvbar| {mv:.3e} m/s (rel {mv / s_v:.1e})")
        if mu / s_u > FP32_REL or mv / s_v > FP32_REL:
            raise SystemExit("FATAL C3: the arms are not identical to NEMO at day 0")

    # ---- C4 reconciliation against the COMMITTED numbers -------------------
    print("\n[C4 RECONCILE -- Rule 1e: our per-column map vs the committed group rows]")
    print(f"    {'arm':12s}{'day':>5s}{'sum(our map) [Sv]':>20s}"
          f"{'D.group_transport [Sv]':>24s}{'committed bot-ref [Sv]':>24s}")
    for n in lego:
        for d in DAYS:
            dub = UB[n][d] - UB["NEMO"][d]
            ours = float(np.nansum(col_transport(dub)[SOUTH]))
            ref = (D._avg(D.group_transport(lego[n][d]["u"], A.umask, SOUTH))
                   - D._avg(D.group_transport(nemo[d]["u"], A.umask, SOUTH)))
            bot = (D._avg(D.section_bt_bc(lego[n][d]["u"], A.umask, rows=SOUTH)[0])
                   - D._avg(D.section_bt_bc(nemo[d]["u"], A.umask, rows=SOUTH)[0]))
            com = COMMITTED_BOTREF[DAYS.index(d)] if n == "arm1_pre" else np.nan
            print(f"    {n:12s}{d:5d}{ours:20.4f}{ref:24.4f}"
                  f"{bot:16.4f} (committed {com:+.3f})" if n == "arm1_pre"
                  else f"    {n:12s}{d:5d}{ours:20.4f}{ref:24.4f}{bot:24.4f}")
            if abs(ours - ref) > 1e-9:
                raise SystemExit("FATAL C4: our per-column map does not sum to the "
                                 "committed group transport -- the reduction differs")
            if n == "arm1_pre" and abs(bot - com) > 2e-3:
                raise SystemExit(f"FATAL C4: bottom-ref d{d} {bot:.4f} != committed {com}")
            if d == 90 and n in COMMITTED_GSOUTH_D90:
                if abs(ours - COMMITTED_GSOUTH_D90[n]) > 2e-3:
                    raise SystemExit(
                        f"FATAL C4: {n} day-90 group deficit {ours:.4f} != the "
                        f"committed {COMMITTED_GSOUTH_D90[n]} -- our depth-mean map "
                        "is not measuring the committed quantity")
    print("    NOTE the two right-hand columns are DIFFERENT quantities: the committed\n"
          "    0/-0.375/-0.700/-0.987 series is BOTTOM-REFERENCED (u_bot*H), not a depth\n"
          "    mean.  Our depth-mean map sums to the FULL group transport (left column);\n"
          "    both are reproduced here, so the label -- not the arithmetic -- was loose.")

    # ---- P planted control --------------------------------------------------
    n0 = next(iter(lego))
    pert = {**lego[n0][90], "u": lego[n0][90]["u"] + 1e-3}
    dub_ref = UB[n0][90] - UB["NEMO"][90]
    dub_p = depth_mean_u(pert["u"]) - UB["NEMO"][90]
    shift = (dub_p - dub_ref)[SOUTH]
    shift = shift[np.isfinite(shift)]
    print(f"\n[P planted +1 mm/s on lego u] dubar shift min/max "
          f"{shift.min():.6e} / {shift.max():.6e} m/s (want exactly 1.000000e-03)")
    if np.max(np.abs(shift - 1e-3)) > 1e-15:
        raise SystemExit("FATAL P: the planted uniform shift did not move dubar by 1 mm/s")

    # ---- M5 sign ------------------------------------------------------------
    area_u = (e1u * e2u_full)
    area_v = np.asarray(_mm["e1v"][0]).squeeze() * np.asarray(_mm["e2v"][0]).squeeze()
    print("\n" + "=" * 96)
    print("M5 SIGN: area-weighted mean depth-mean flow in the southern group [mm/s] "
          "(negative = westward)")
    print("=" * 96)
    print(f"{'day':>5s}{'NEMO ubar':>12s}" + "".join(f"{n + ' ubar':>16s}" for n in lego)
          + f"{'NEMO vbar':>12s}" + "".join(f"{n + ' vbar':>16s}" for n in lego))
    print("  absolute group transport [Sv] (e3t_1d, mean lons 2..-2 -- the recorded "
          "integrand restricted to rows 0..J0-1; POSITIVE = eastward):")
    print(f"    {'day':>5s}{'NEMO':>12s}" + "".join(f"{n:>14s}" for n in lego))
    gt = {"NEMO": {d: D._avg(D.group_transport(nemo[d]["u"], A.umask, SOUTH)) for d in DAYS}}
    for n in lego:
        gt[n] = {d: D._avg(D.group_transport(lego[n][d]["u"], A.umask, SOUTH)) for d in DAYS}
    for d in DAYS:
        print(f"    {d:5d}{gt['NEMO'][d]:12.4f}" + "".join(f"{gt[n][d]:14.4f}" for n in lego))
    print(f"    {'d90-d0':>5s}{gt['NEMO'][90] - gt['NEMO'][0]:12.4f}"
          + "".join(f"{gt[n][90] - gt[n][0]:14.4f}" for n in lego))
    print(f"    {'cap%':>5s}{100.0:12.1f}"
          + "".join(f"{100 * (gt[n][90] - gt[n][0]) / (gt['NEMO'][90] - gt['NEMO'][0]):14.1f}"
                    for n in lego))
    print()
    for d in DAYS:
        def aw(f, msk, ar):
            m = np.isfinite(f[SOUTH]) & msk
            return 1e3 * float(np.sum(f[SOUTH][m] * ar[SOUTH][m]) / np.sum(ar[SOUTH][m]))
        mu = np.isfinite(UB["NEMO"][d][SOUTH])
        mv = np.isfinite(VB["NEMO"][d][SOUTH])
        line = f"{d:5d}{aw(UB['NEMO'][d], mu, area_u):12.3f}"
        line += "".join(f"{aw(UB[n][d], mu, area_u):16.3f}" for n in lego)
        line += f"{aw(VB['NEMO'][d], mv, area_v):12.3f}"
        line += "".join(f"{aw(VB[n][d], mv, area_v):16.3f}" for n in lego)
        print(line)

    # ---- M1/M2 per-column bias, growth --------------------------------------
    print("\n" + "=" * 96)
    print("M1/M2 per-column bias in the southern group (lego - NEMO), all wet columns")
    print("=" * 96)
    print(f"{'arm':12s}{'day':>5s}{'mean dubar':>12s}{'std':>9s}{'min':>9s}{'max':>9s}"
          f"{'|corr with d90|':>16s}{'mean dvbar':>12s}{'std dvbar':>11s}"
          f"{'|dvbar|/|dubar|':>16s}")
    for n in lego:
        d90u = (UB[n][90] - UB["NEMO"][90])[SOUTH]
        for d in DAYS:
            du = (UB[n][d] - UB["NEMO"][d])[SOUTH]
            dv = (VB[n][d] - VB["NEMO"][d])[SOUTH]
            mu, mv = np.isfinite(du), np.isfinite(dv)
            c = (np.corrcoef(du[mu & np.isfinite(d90u)],
                             d90u[mu & np.isfinite(d90u)])[0, 1] if d else np.nan)
            print(f"{n:12s}{d:5d}{1e3 * du[mu].mean():12.4f}{1e3 * du[mu].std():9.4f}"
                  f"{1e3 * du[mu].min():9.3f}{1e3 * du[mu].max():9.3f}{c:16.3f}"
                  f"{1e3 * dv[mv].mean():12.4f}{1e3 * dv[mv].std():11.4f}"
                  f"{np.abs(dv[mv]).mean() / max(np.abs(du[mu]).mean(), 1e-30):16.3f}")
        print()

    # ---- M2b zonal COHERENCE: is the bias jet-like (same sign every row) or
    #      rotational (a sign reversal within the group, i.e. a gyre)? --------
    print("=" * 96)
    print("M2b day-90 bias per T-row: zonal MEAN vs within-row STD [mm/s].  A zonally\n"
          "    coherent deficit has |mean| ~ std; incoherent scatter has |mean| << std.")
    print("=" * 96)
    for n in lego:
        du = 1e3 * (UB[n][90] - UB["NEMO"][90])[SOUTH]
        dv = 1e3 * (VB[n][90] - VB["NEMO"][90])[SOUTH]
        print(f"  arm {n}")
        print(f"    {'row':>4s}{'lat':>8s}{'<dubar>':>10s}{'std':>8s}"
              f"{'<dvbar>':>10s}{'std':>8s}")
        zu, zv = [], []
        for j in range(du.shape[0]):
            a, b = du[j][np.isfinite(du[j])], dv[j][np.isfinite(dv[j])]
            if a.size == 0:
                print(f"    {j:4d}{A.gphit[j, 25]:8.1f}{'(dry row)':>10s}")
                continue
            zu.append(a.mean()); zv.append(b.mean())
            print(f"    {j:4d}{A.gphit[j, 25]:8.1f}{a.mean():10.3f}{a.std():8.3f}"
                  f"{b.mean():10.3f}{b.std():8.3f}")
        fu, fv = du[np.isfinite(du)], dv[np.isfinite(dv)]
        print(f"    |zonal mean| / total std:  u {np.abs(zu).mean() / fu.std():.2f}"
              f"   v {np.abs(zv).mean() / fv.std():.3f}   |   rows with dubar<0: "
              f"{int(np.sum(np.asarray(zu) < 0))}/{len(zu)}")

    # ---- M3 binning ---------------------------------------------------------
    Hs = H_real[SOUTH]
    print("=" * 96)
    print("M3 day-90 dubar BINNED by column depth (real partial-cell H) and by "
          "distance north of the southern wall")
    print("=" * 96)
    for n in lego:
        dub = (UB[n][90] - UB["NEMO"][90])[SOUTH]
        tc = col_transport(UB[n][90] - UB["NEMO"][90])[SOUTH]
        # the transport reducer drops longitudes 0,1,-2,-1; restrict the population
        # statistics to the SAME columns so ncol/mean/H describe the transport's set
        inlon = np.zeros(dub.shape, dtype=bool)
        inlon[:, LONS] = True
        ok = np.isfinite(dub) & inlon
        # NOTE 37% of the southern columns sit at EXACTLY the flat abyssal depth, so
        # the upper quartile edge lands on it and a naive 4th quartile is empty.
        # Bins are therefore: three depth ranges below the flat bottom + the flat
        # bottom itself as its own (zero-spread) bin.
        hmax = float(np.nanmax(np.where(ok, Hs, np.nan)))
        q = np.nanpercentile(np.where(ok & (Hs > 0) & (Hs < hmax), Hs, np.nan), [33, 67])
        dbins = [ok & (Hs <= q[0]), ok & (Hs > q[0]) & (Hs <= q[1]),
                 ok & (Hs > q[1]) & (Hs < hmax), ok & (Hs >= hmax)]
        dlabels = [f"H <= {q[0]:.0f} m", f"{q[0]:.0f} < H <= {q[1]:.0f}",
                   f"{q[1]:.0f} < H < {hmax:.0f}", f"H == {hmax:.0f} m (flat abyss)"]
        ykm = np.broadcast_to(Y_KM[SOUTH][:, None], dub.shape)
        edges = np.linspace(0, ykm.max() + 1e-6, 5)
        ybins = [ok & (ykm >= edges[i]) & (ykm < edges[i + 1]) for i in range(4)]
        ylabels = [f"{edges[i]:.0f}-{edges[i+1]:.0f} km N of wall" for i in range(4)]
        print(f"\n  arm {n}   (total group deficit {float(np.nansum(tc)):+.4f} Sv)")
        print(f"    {'bin':34s}{'ncol':>6s}{'mean dubar [mm/s]':>20s}"
              f"{'transport [Sv]':>17s}{'share':>8s}{'mean H [m]':>12s}"
              f"{'<dubar*H> [m2/s]':>18s}")
        tot = float(np.nansum(tc))
        for lab, nc_, m_, t_, h_, p_ in (binned(dub, dbins, dlabels, tc, area_u[SOUTH], Hs)
                                         + binned(dub, ybins, ylabels, tc, area_u[SOUTH], Hs)):
            print(f"    {lab:34s}{nc_:6d}{m_:20.4f}{t_:17.4f}"
                  f"{100 * t_ / tot if abs(tot) > 1e-12 else np.nan:7.1f}%"
                  f"{h_:12.0f}{p_:18.4f}")
        # does the bias scale like 1/H (a stress/drag signature) or is it flat?
        m = ok & (Hs > 0)
        r_h = float(np.corrcoef(dub[m], Hs[m])[0, 1])
        r_inv = float(np.corrcoef(dub[m], 1.0 / Hs[m])[0, 1])
        print(f"    corr(dubar, H) = {r_h:+.3f} | corr(dubar, 1/H) = {r_inv:+.3f} | "
              f"top-5 columns by |transport| carry "
              f"{100 * float(np.sum(np.sort(np.abs(tc[ok]))[-5:])) / float(np.sum(np.abs(tc[ok]))):.1f}% "
              "of the summed |per-column transport| (localisation test)")

    # ---- M6 the band is CLOSED: the metric IS a streamfunction ---------------
    # Every one of these rows is walled at BOTH ends (verified below), so no
    # depth-integrated zonal through-flow is possible: the transport accumulated
    # from the Antarctic wall northward is the barotropic streamfunction of a
    # closed circulation.  Report it as such, per row, both sides.
    print("\n" + "=" * 96)
    print("M6 CLOSED-BAND STREAMFUNCTION: per-row depth-integrated zonal transport "
          "and its\n    cumulative sum from the southern wall [Sv, e3t_1d, mean lons "
          "2..-2]")
    print("=" * 96)
    walled = [int(A.NX - A.tmask[j, :, 0].sum()) for j in range(A.J0)]
    print(f"    dry surface columns per row (0 would mean re-entrant): {walled}")
    if min(walled) == 0:
        raise SystemExit("FATAL M6: a southern row is zonally UNBLOCKED -- the "
                         "streamfunction reading does not apply")
    rows_t = {"NEMO": [D._avg(D.group_transport(nemo[90]["u"], A.umask, slice(j, j + 1)))
                       for j in range(A.J0)]}
    rows_t0 = {"NEMO": [D._avg(D.group_transport(nemo[0]["u"], A.umask, slice(j, j + 1)))
                        for j in range(A.J0)]}
    for n in lego:
        rows_t[n] = [D._avg(D.group_transport(lego[n][90]["u"], A.umask, slice(j, j + 1)))
                     for j in range(A.J0)]
        rows_t0[n] = [D._avg(D.group_transport(lego[n][0]["u"], A.umask, slice(j, j + 1)))
                      for j in range(A.J0)]
    print(f"    {'row':>4s}{'lat':>8s}{'NEMO d0':>10s}{'NEMO d90':>10s}{'NEMO cum d90':>14s}"
          + "".join(f"{n + ' cum-diff':>18s}" for n in lego))
    cum_n = np.cumsum(rows_t["NEMO"])
    cum_l = {n: np.cumsum(rows_t[n]) for n in lego}
    for j in range(A.J0):
        print(f"    {j:4d}{A.gphit[j, 25]:8.1f}{rows_t0['NEMO'][j]:10.3f}"
              f"{rows_t['NEMO'][j]:10.3f}{cum_n[j]:14.3f}"
              + "".join(f"{cum_l[n][j] - cum_n[j]:18.3f}" for n in lego))
    # how much of the day-90 deficit is accumulated in the rows where NEMO's own
    # circulation barely changed (the wall limb) vs the rows that actually spin up?
    k = 5
    dn_wall = float(np.sum(rows_t["NEMO"][:k]) - np.sum(rows_t0["NEMO"][:k]))
    print(f"\n    NEMO's OWN 90-day change in the {k} rows nearest the wall: "
          f"{dn_wall:+.3f} Sv (vs {float(np.sum(rows_t['NEMO']) - np.sum(rows_t0['NEMO'])):+.3f} "
          "Sv for the whole group)")
    for n in lego:
        share = (cum_l[n][k - 1] - cum_n[k - 1]) / (cum_l[n][-1] - cum_n[-1])
        print(f"    {n:12s} deficit accumulated by row {k - 1}: "
              f"{cum_l[n][k - 1] - cum_n[k - 1]:+.3f} Sv = {100 * share:.0f}% of the "
              f"group total {cum_l[n][-1] - cum_n[-1]:+.3f} Sv")

    # ---- M4 southern density contrast --------------------------------------
    print("\n" + "=" * 96)
    print("M4 the SOUTHERN group's OWN meridional density contrast [kg/m3] "
          "(A.contrast_profile re-pointed at rows 0..J0-1)")
    print("=" * 96)
    print(f"{'day':>5s}{'NEMO up':>10s}{'NEMO deep':>11s}"
          + "".join(f"{n + ' up':>13s}{n + ' deep':>15s}" for n in lego))
    cd = {"NEMO": {d: group_density_contrast(nemo[d], wet) for d in DAYS}}
    for n in lego:
        cd[n] = {d: group_density_contrast(lego[n][d], wet) for d in DAYS}
    _, _, _, _, nok, ntot = cd["NEMO"][0]
    print(f"  [caveat] the telescoping that defines this contrast is UNBROKEN on only "
          f"{nok}/{ntot} levels;\n  below the first break the printed value is a "
          "PARTIAL sum, so this is a bathymetry-masked\n  density functional, not "
          "'the' meridional contrast.  The mask is identical on both sides\n  and in "
          "time, so the RATIO is still a controlled comparison.  'ok' columns below "
          "use\n  only the unbroken levels.")
    for d in DAYS:
        line = f"{d:5d}{cd['NEMO'][d][0]:10.5f}{cd['NEMO'][d][1]:11.5f}"
        line += "".join(f"{cd[n][d][0]:13.5f}{cd[n][d][1]:15.5f}" for n in lego)
        print(line)
    print("  same, UNBROKEN levels only (NaN = no clean level exists there):")
    for d in DAYS:
        line = f"{d:5d}{cd['NEMO'][d][2]:10.5f}{cd['NEMO'][d][3]:11.5f}"
        line += "".join(f"{cd[n][d][2]:13.5f}{cd[n][d][3]:15.5f}" for n in lego)
        print(line)
    print("\n  CHANGE from day 0 [kg/m3].  A capture RATIO is printed only where NEMO's "
          "own change\n  exceeds this metric's recorded noise floor "
          f"({NOISE_CONTRAST:.1e}); otherwise it is a divide-by-noise\n  and is "
          "suppressed.")
    print(f"{'day':>5s}{'NEMO d(up)':>12s}" + "".join(f"{n + ' d(up)':>14s}{'cap':>8s}" for n in lego)
          + f"{'NEMO d(deep)':>14s}" + "".join(f"{n + ' d(deep)':>16s}{'cap':>8s}" for n in lego))

    def _cap(l, n_):
        return f"{100 * l / n_:7.0f}%" if abs(n_) > NOISE_CONTRAST else f"{'noise':>8s}"
    for d in DAYS[1:]:
        nu = cd["NEMO"][d][0] - cd["NEMO"][0][0]
        nd = cd["NEMO"][d][1] - cd["NEMO"][0][1]
        line = f"{d:5d}{nu:12.5f}"
        for n in lego:
            line += f"{cd[n][d][0] - cd[n][0][0]:14.5f}{_cap(cd[n][d][0] - cd[n][0][0], nu)}"
        line += f"{nd:14.5f}"
        for n in lego:
            line += f"{cd[n][d][1] - cd[n][0][1]:16.5f}{_cap(cd[n][d][1] - cd[n][0][1], nd)}"
        print(line)
    print("  ARM SPREAD in the deep change at day 90: "
          + " ".join(f"{n}:{cd[n][90][1] - cd[n][0][1]:+.5f}" for n in lego)
          + "  <- compare with the arms' transport spread below")

    # ---- artifacts ----------------------------------------------------------
    if args.out_npz:
        out = {"H_ref": H_ref[SOUTH], "H_real": H_real[SOUTH], "H_v": H_v[SOUTH],
               "y_km": Y_KM[SOUTH], "lat": A.gphit[SOUTH, 25], "area_u": area_u[SOUTH]}
        for d in DAYS:
            out[f"ubar_NEMO_d{d}"] = UB["NEMO"][d][SOUTH]
            out[f"vbar_NEMO_d{d}"] = VB["NEMO"][d][SOUTH]
            out[f"ubar_pc_NEMO_d{d}"] = UBP["NEMO"][d][SOUTH]
            for n in lego:
                out[f"ubar_{n}_d{d}"] = UB[n][d][SOUTH]
                out[f"vbar_{n}_d{d}"] = VB[n][d][SOUTH]
                out[f"ubar_pc_{n}_d{d}"] = UBP[n][d][SOUTH]
        np.savez_compressed(args.out_npz, **out)
        print(f"\n[artifact] maps -> {args.out_npz}")
    if args.out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = next(iter(lego))
        dub = 1e3 * (UB[n][90] - UB["NEMO"][90])[SOUTH]
        fig, ax = plt.subplots(figsize=(9, 5))
        v = np.nanpercentile(np.abs(dub), 99)
        im = ax.pcolormesh(np.arange(A.NX), A.gphit[SOUTH, 25], dub,
                           cmap="RdBu_r", vmin=-v, vmax=v, shading="nearest")
        cs = ax.contour(np.arange(A.NX), A.gphit[SOUTH, 25], np.where(H_real[SOUTH] > 0,
                        H_real[SOUTH], np.nan), levels=[500, 1500, 2500, 3500],
                        colors="k", linewidths=0.6)
        ax.clabel(cs, fmt="%.0f")
        ax.set_xlabel("longitude index"); ax.set_ylabel("latitude")
        ax.set_title(f"day-90 depth-mean zonal velocity bias, {n} - NEMO [mm/s]\n"
                     "black = column depth [m]")
        fig.colorbar(im, ax=ax, label="mm/s")
        fig.tight_layout(); fig.savefig(args.out_png, dpi=130)
        print(f"[artifact] figure -> {args.out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
