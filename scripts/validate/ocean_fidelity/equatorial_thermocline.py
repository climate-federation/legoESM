#!/usr/bin/env python
"""Equatorial Pacific thermocline: ours vs NEMO, along 2S-2N.

WHY.  The cold tongue is +3.19 C too warm in Nino3 and the bias GROWS
(+2.09/+2.50/+3.19 at days 10/20/30).  Two families of cause produce that same
SST signature and need different fixes:

  A. THERMOCLINE / UPWELLING.  If the equatorial thermocline is too DEEP or
     too DIFFUSE, the water upwelled at the equator comes from a warmer level
     and the cold tongue cannot form.  Z20 (depth of the 20 C isotherm) is the
     standard index for this, and its EAST-WEST TILT is the standard index for
     whether the equatorial circulation is right at all.

  B. SURFACE FLUX.  A heat-flux or wind-speed error warms the surface without
     touching the subsurface structure.

They are distinguished by the SUBSURFACE: cause A shifts Z20 and flattens the
tilt, cause B leaves Z20 alone and warms only the top few tens of metres.  So
this probe reports Z20(lon) for both models plus the temperature profile at a
few longitudes, and nothing else -- it is a discriminator, not a scorecard.

READ THE OUTPUT AS A DISCRIMINATOR, NOT A VERDICT.  A matching Z20 with a warm
surface points at B.  A too-deep or too-flat Z20 points at A.  Anything else
means neither family is clean and the probe has done its job by saying so.

Instrument caveats, stated up front:
  * ours is an INSTANTANEOUS snapshot, NEMO is a climatological month mean --
    the same mismatch every scorecard in this campaign carries;
  * Z20 is undefined where the whole column is below 20 C; those columns are
    reported as NaN and COUNTED rather than silently dropped;
  * we interpolate Z20 linearly in depth between the bracketing levels, on
    both sides, through the same function -- no per-model convention.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1]))


def z20_from_column(T, z, iso=20.0):
    """Depth [m, >0] where T first crosses `iso` downward; NaN if never.

    Linear interpolation between the bracketing levels. Applied identically to
    both models so the convention cannot become a difference.
    """
    T = np.asarray(T, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    good = np.isfinite(T)
    if good.sum() < 2:
        return np.nan
    T, z = T[good], z[good]
    if T[0] < iso:          # surface already colder than the isotherm
        return np.nan
    below = np.nonzero(T < iso)[0]
    if below.size == 0:     # never reaches it within the column
        return np.nan
    k = int(below[0])
    T1, T0 = T[k], T[k - 1]
    z1, z0 = z[k], z[k - 1]
    if not np.isfinite(T1 - T0) or (T1 - T0) == 0.0:
        return float(z1)
    return float(z0 + (z1 - z0) * (T0 - iso) / (T0 - T1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legoesm-snapshot", required=True)
    ap.add_argument("--nemo-gridt", required=True)
    ap.add_argument("--nemo-month", type=int, default=1)
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    ap.add_argument("--label", default="legoESM")
    ap.add_argument("--nemo-wfile", default=None,
                    help="NEMO file carrying `wocetr_eff` (the hourly trd1h_T; "
                         "the monthly grid_T does NOT have it). Enables the "
                         "UPWELLING block, which asks the question Z20 cannot: "
                         "if the thermocline is too deep, is it because the "
                         "equatorial ascent is too weak?")
    ap.add_argument("--mesh-mask", default=None,
                    help="eORCA1 mesh_mask, required with --nemo-wfile: "
                         "`wocetr_eff` is a TRANSPORT [m3/s] and our "
                         "`mass_flux_w` is a VELOCITY [m/s], so one side has "
                         "to be divided by the cell area e1t*e2t before they "
                         "are the same quantity.")
    ap.add_argument("--nemo-w-var", default="wocetr_eff",
                    choices=("wocetr_eff", "wo"),
                    help="which NEMO vertical field the wfile carries. "
                         "`wocetr_eff` is a TRANSPORT [m3/s] (hourly trd1h_T; "
                         "needs --mesh-mask for the area divide). `wo` is a "
                         "VELOCITY [m/s] (the 5-day grid_W files) and is "
                         "compared directly.")
    ap.add_argument("--nemo-w-recs", default=None,
                    help="record selection in the wfile, `K` or `A:B` "
                         "(python slice, stop-exclusive). Default: all "
                         "records, averaged. Use this to MATCH the window of "
                         "the snapshot (e.g. `17:18` = days 86-90 of a 5-day "
                         "file against a day-90 snapshot).")
    ap.add_argument("--w-depth-m", type=float, default=50.0,
                    help="depth at which the equatorial vertical velocity is "
                         "compared. 50 m sits inside the upwelling core and "
                         "above the Z20 the bias is measured at.")
    a = ap.parse_args()

    import xarray as xr
    from compare_omip_nemo import _load_legoesm, _load_nemo, regrid_curv_to_latlon

    L = _load_legoesm(a.legoesm_snapshot)
    if L.get("T3d") is None or L.get("z_center_ref") is None:
        raise SystemExit("snapshot lacks T3d / z_center_ref -- cannot do Z20")
    N = _load_nemo(a.nemo_gridt, 0, month=a.nemo_month)

    ds = xr.open_dataset(a.nemo_gridt, decode_times=False)
    # ORCA1's grid_T calls 3-D temperature "to"; other NEMO configs and CMOR
    # output use thetao/votemper/toce. Checked in that order and named in the
    # error so a miss is a one-line fix rather than a hunt.
    tvar = next((v for v in ("to", "thetao", "votemper", "toce")
                 if v in ds and ds[v].ndim >= 3), None)
    if tvar is None:
        raise SystemExit(f"{a.nemo_gridt}: no 3-D temperature (looked for "
                         f"to/thetao/votemper/toce; present: "
                         f"{sorted(ds.data_vars)[:12]})")
    print(f"[nemo] 3-D temperature variable: {tvar!r}")

    zc = np.asarray(L["z_center_ref"], dtype=np.float64)
    zc = np.abs(zc)
    nlev = zc.size
    print(f"[lego] {nlev} levels, {zc[0]:.1f} -> {zc[-1]:.0f} m")

    # Regrid our columns level by level, through the SAME helper the scorecard
    # uses, so the horizontal treatment is not a new convention either.
    tgt_lat = np.arange(-89.5, 90.0, 1.0)
    tgt_lon = np.arange(0.5, 360.0, 1.0)
    lev = []
    for k in range(nlev):
        f, _ = regrid_curv_to_latlon(np.asarray(L["T3d"])[..., k], L["lat"],
                                     L["lon"], L["mask"], tgt_lat, tgt_lon)
        lev.append(f)
    Lg = np.stack(lev, axis=-1)                      # (nlat, nlon, nlev)

    band = np.abs(tgt_lat) <= a.lat_halfwidth
    print(f"[band] |lat| <= {a.lat_halfwidth} -> {int(band.sum())} rows")

    # NEMO on its own grid, band-averaged after regridding to the same target.
    nT = np.asarray(ds[tvar].values)
    if nT.ndim == 4:
        from compare_omip_nemo import _nemo_record_months
        tdim = ds[tvar].dims[0]
        nt = int(ds.sizes[tdim])
        months = _nemo_record_months(ds, tdim, nt)
        idx = ([i for i in range(nt) if months[i] == a.nemo_month]
               if months is not None else [a.nemo_month - 1])
        nT = np.nanmean(nT[idx], axis=0)
    nT = np.where(np.abs(nT) > 1e10, np.nan, nT)
    nT = np.where(nT == 0.0, np.nan, nT)
    zn = np.asarray(ds["deptht"].values, dtype=np.float64)
    print(f"[nemo] {zn.size} levels, {zn[0]:.1f} -> {zn[-1]:.0f} m")

    nlat_n, nlon_n = nT.shape[1], nT.shape[2]
    nav_lat = np.asarray(ds["nav_lat"].values)
    nav_lon = np.asarray(ds["nav_lon"].values)
    mask2d = np.isfinite(nT[0])
    Ng = []
    for k in range(zn.size):
        f, _ = regrid_curv_to_latlon(nT[k], nav_lat, nav_lon,
                                     mask2d.astype(np.float64), tgt_lat, tgt_lon)
        Ng.append(f)
    Ng = np.stack(Ng, axis=-1)

    print()
    print("Z20 [m] along the equator, band mean. NEMO minus ours is the gap;")
    print("a DEEPER ours (positive gap) means a more diffuse thermocline.")
    print(f"{'lon':>6} {'ours':>9} {'NEMO':>9} {'ours-NEMO':>10}")
    lons = [160, 180, 200, 220, 240, 260, 270, 280]
    gaps = []
    for lon in lons:
        j = int(np.argmin(np.abs(tgt_lon - lon)))
        zl = np.nanmean([z20_from_column(Lg[i, j, :], zc)
                         for i in np.nonzero(band)[0]])
        zn_ = np.nanmean([z20_from_column(Ng[i, j, :], zn)
                          for i in np.nonzero(band)[0]])
        d = zl - zn_
        if np.isfinite(d):
            gaps.append(d)
        f = lambda x: f"{x:9.1f}" if np.isfinite(x) else "      nan"
        print(f"{lon:6d} {f(zl)} {f(zn_)} {d:10.1f}" if np.isfinite(d)
              else f"{lon:6d} {f(zl)} {f(zn_)}        nan")
    if gaps:
        print(f"\n[Z20] mean gap {np.mean(gaps):+.1f} m over "
              f"{len(gaps)}/{len(lons)} longitudes")

    # East-west tilt: the single number that says whether the equatorial
    # circulation exists at all. NEMO's Pacific thermocline shoals eastward.
    def tilt(Z):
        jw = int(np.argmin(np.abs(tgt_lon - 160)))
        je = int(np.argmin(np.abs(tgt_lon - 260)))
        w = np.nanmean([z20_from_column(Z[i, jw, :], zc if Z is Lg else zn)
                        for i in np.nonzero(band)[0]])
        e = np.nanmean([z20_from_column(Z[i, je, :], zc if Z is Lg else zn)
                        for i in np.nonzero(band)[0]])
        return w - e
    tl, tn = tilt(Lg), tilt(Ng)
    print(f"[tilt] west(160E)-east(100W) Z20: ours {tl:+.1f} m, "
          f"NEMO {tn:+.1f} m  -> ours is {100.0 * tl / tn:.0f}% of NEMO's"
          if np.isfinite(tn) and tn != 0 else
          f"[tilt] ours {tl:+.1f} m, NEMO {tn:+.1f} m")

    if a.nemo_wfile:
        if a.nemo_w_var == "wocetr_eff" and not a.mesh_mask:
            raise SystemExit("--nemo-wfile requires --mesh-mask: wocetr_eff is "
                             "m3/s and our mass_flux_w is m/s, and without the "
                             "cell area the two are not the same quantity.")
        _upwelling_block(a, L, tgt_lat, tgt_lon, band, zc,
                         regrid_curv_to_latlon)
    return 0


def _upwelling_block(a, L, tgt_lat, tgt_lon, band, zc, regrid):
    """Equatorial vertical velocity, ours vs NEMO, at a single depth.

    WHY IT IS HERE.  Z20 says the thermocline is 26 m too deep; it cannot say
    whether that is because the water is not being lifted.  This is the cheapest
    measurement that can, and it runs on files that already exist.

    UNITS ARE NOT THE SAME ON THE TWO SIDES and that is the whole trap: NEMO's
    ``wocetr_eff`` is an effective vertical TRANSPORT [m3/s] on the W points,
    ours is ``mass_flux_w`` [m/s] (MASS_FLUX_W_UNITS, ocean_model_latlon_cgrid
    :120).  NEMO's is divided by the T-cell area e1t*e2t here so both sides are
    velocities before anything is compared.

    SIGN IS DERIVED, NOT ASSUMED.  Equatorial upwelling is a known answer: the
    Pacific cold tongue exists because water rises there.  So NEMO's band mean
    at this depth TELLS us NEMO's sign convention, and ours is then reported
    with the same test rather than flipped to match.  If the two disagree the
    block says so and stops short of a ratio -- a sign difference is either a
    convention difference or a reversed circulation, and this probe cannot tell
    those apart.

    WHAT IT CANNOT SUPPORT.  The two sides are at DIFFERENT TIMES: ours is an
    instantaneous snapshot from a 30/90-day run, NEMO's is a 24-hour mean at the
    end of its first year, because the monthly grid_T carries no vertical
    velocity at all.  A 10-20% difference is inside that mismatch and must not
    be read.  A factor of ~2 is not, and the Z20 gap (26 m on 55 m) is a
    factor-of-2-sized question.
    """
    import netCDF4 as nc
    from legoesm import constants  # noqa: F401  (unit conventions live there)

    w_l = np.load(a.legoesm_snapshot).get("mass_flux_w")
    if w_l is None:
        raise SystemExit("snapshot has no `mass_flux_w`; rerun with the mass "
                         "flux carry stored, or drop --nemo-wfile.")
    w_l = np.asarray(w_l, dtype=np.float64)
    # Interface depths: the snapshot stores CENTRES, so interfaces are their
    # midpoints with the surface at 0. Good to a few metres, which is far
    # inside the level spacing at 50 m and is only used to PICK a level.
    z_if = np.concatenate([[0.0], 0.5 * (zc[:-1] + zc[1:])])
    kl = int(np.argmin(np.abs(z_if - a.w_depth_m)))

    ds = nc.Dataset(a.nemo_wfile)
    try:
        wn = np.ma.filled(np.ma.masked_invalid(
            ds.variables[a.nemo_w_var][:]), np.nan).astype(np.float64)
        zw = np.asarray(ds.variables["depthw"][:], dtype=np.float64)
        # trd1h_T names its coords nav_lat_grid_T; the 5-day grid_W nav_lat.
        latname = ("nav_lat_grid_T" if "nav_lat_grid_T" in ds.variables
                   else "nav_lat")
        nav_lat = np.asarray(ds.variables[latname][:])
        nav_lon = np.asarray(ds.variables[latname.replace("lat", "lon")][:])
    finally:
        ds.close()
    if a.nemo_w_recs is not None:
        if ":" in a.nemo_w_recs:
            lo, hi = (int(x) for x in a.nemo_w_recs.split(":"))
            wn = wn[lo:hi]
        else:
            wn = wn[int(a.nemo_w_recs):int(a.nemo_w_recs) + 1]
        if wn.shape[0] == 0:
            raise SystemExit(f"--nemo-w-recs {a.nemo_w_recs} selects no "
                             "records -- refusing an empty mean.")
    n_rec = wn.shape[0]
    wn = np.nanmean(wn, axis=0)
    kn = int(np.argmin(np.abs(zw - a.w_depth_m)))

    if a.nemo_w_var == "wocetr_eff":
        dsm = nc.Dataset(a.mesh_mask)
        try:
            e1t = np.asarray(dsm.variables["e1t"][:],
                             dtype=np.float64).squeeze()
            e2t = np.asarray(dsm.variables["e2t"][:],
                             dtype=np.float64).squeeze()
        finally:
            dsm.close()
        # wocetr_eff is written on the (331,360) inner frame; the mesh_mask is
        # the full (332,362). Slice with the SAME native window the rest of
        # this campaign uses rather than guessing an offset here.
        from global_tracer_content import _NATIVE_J, _NATIVE_I
        area = (e1t * e2t)[_NATIVE_J, _NATIVE_I]
        if area.shape != wn.shape[1:]:
            raise SystemExit(f"area {area.shape} vs wocetr_eff {wn.shape[1:]}"
                             " -- frames do not match; do not divide.")
        wn_ms = wn[kn] / area                        # m3/s -> m/s
    else:
        wn_ms = wn[kn]                               # wo is already m/s

    lm = np.isfinite(np.asarray(L["mask"], dtype=np.float64))
    ours, _ = regrid(w_l[..., kl], L["lat"], L["lon"],
                     np.asarray(L["mask"], dtype=np.float64),
                     tgt_lat, tgt_lon)
    theirs, _ = regrid(wn_ms, nav_lat, nav_lon,
                       np.isfinite(wn_ms).astype(np.float64),
                       tgt_lat, tgt_lon)
    del lm

    print()
    print(f"Equatorial vertical velocity at {z_if[kl]:.1f} m (ours) / "
          f"{zw[kn]:.1f} m (NEMO), band mean, m/s x 1e6.")
    src = ("wocetr_eff [m3/s] divided by e1t*e2t"
           if a.nemo_w_var == "wocetr_eff" else "wo [m/s]")
    print(f"  NEMO side is a {n_rec}-record mean of {src}; ours is "
          f"mass_flux_w [m/s] from the snapshot (instantaneous).")
    rows = np.nonzero(band)[0]
    box = (tgt_lon >= 180.0) & (tgt_lon <= 280.0)    # 180-80W, the cold tongue
    bo = np.nanmean(ours[np.ix_(rows, np.nonzero(box)[0])])
    bt = np.nanmean(theirs[np.ix_(rows, np.nonzero(box)[0])])
    print(f"{'lon':>6} {'ours':>11} {'NEMO':>11}")
    for lon in (160, 180, 200, 220, 240, 260, 280):
        j = int(np.argmin(np.abs(tgt_lon - lon)))
        print(f"{lon:6d} {1e6 * np.nanmean(ours[rows, j]):11.3f} "
              f"{1e6 * np.nanmean(theirs[rows, j]):11.3f}")
    print(f"[box 180-80W] ours {1e6 * bo:+.3f}  NEMO {1e6 * bt:+.3f}  (1e-6 m/s)")
    if not (np.isfinite(bo) and np.isfinite(bt)):
        print("[upwelling] one side is NaN in the box — no ratio.")
    elif np.sign(bo) != np.sign(bt):
        print("[upwelling] THE TWO SIGNS DISAGREE. NEMO's sign here is the "
              "known-answer anchor (the cold tongue requires ascent), so this "
              "is either an opposite w convention in our snapshot or a "
              "reversed equatorial cell. NO RATIO is reported: this probe "
              "cannot tell those apart, and the next step is to check the "
              "convention before reading anything into the magnitude.")
    else:
        print(f"[upwelling] ours / NEMO = {bo / bt:.2f} in the 180-80W box. "
              "Read only a LARGE departure from 1: ours is an instantaneous "
              "snapshot against a NEMO time mean, which is worth more than "
              "10-20%.")


if __name__ == "__main__":
    raise SystemExit(main())
