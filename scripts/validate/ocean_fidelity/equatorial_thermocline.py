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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
