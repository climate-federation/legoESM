#!/usr/bin/env python
"""Basin-resolved MLD bias vs NEMO + equator-line artifact check (Pacific).

WHY (user report, 2026-08-25): the long180 run is faithful to NEMO almost
everywhere, EXCEPT the mixed-layer depth in the Pacific.  Candidate causes
split cleanly by spatial structure, so this probe prints, per basin x band:
area-weighted MLD bias/rmse/corr, plus two artifact detectors:

  * EQUATOR-LINE JUMP: the mean |MLD(j)-MLD(j+1)| across the equatorial rows
    versus the same statistic 5-10 degrees off-equator, per basin, computed
    on OUR field alone.  A grid-line discontinuity (equatorial viscosity
    band edge, forcing-interpolation seam, mask row) shows up as a jump
    ratio >> 1 in our field that NEMO's field does not carry.
  * NW-PACIFIC BOX (30-45N, 140E-180): the Kuroshio-extension deep-MLD
    tongue where the d30 map showed us deeper/broader than NEMO.

Reuses compare_omip_nemo's loaders/regridder and the campaign MLD instrument
(dsigma=0.01 wrt 10 m, matching NEMO mldr10_1).  CAVEAT stated with every
number: our MLD is computed from an INSTANTANEOUS snapshot; NEMO's mldr10_1
in GATEWAY is a 5-DAY MEAN of an online diagnostic.  MLD is the quantity
most exposed to that mismatch (short-timescale variability), and the
mismatch is largest where diurnal/synoptic MLD variance is largest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

BASINS = {          # lon ranges in [0, 360); lat clipped to +-65 (no Arctic)
    "pacific":  [(120.0, 290.0)],
    "atlantic": [(290.0, 360.0), (0.0, 20.0)],
    "indian":   [(20.0, 120.0)],
}
BANDS = {
    "SH_45S_23S": (-45.0, -23.0),
    "tropics_23S_23N": (-23.0, 23.0),
    "NH_23N_45N": (23.0, 45.0),
    "NH_45N_65N": (45.0, 65.0),
}


def _basin_mask(lat, lon, name):
    m = np.zeros(lat.shape, bool)
    for lo, hi in BASINS[name]:
        m |= (lon >= lo) & (lon < hi)
    return m & (np.abs(lat) <= 65.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legoesm-snapshot", required=True, action="append",
                    help="repeatable: N snapshots -> MLD computed PER "
                         "snapshot then AVERAGED (mean-of-diagnostic, "
                         "matching NEMO's 5-day-mean mldr10_1), the codex/"
                         "GLM protocol-mismatch discriminator")
    ap.add_argument("--nemo-gridt", required=True)
    ap.add_argument("--nemo-time-idx", type=int, required=True)
    ap.add_argument("--png", default=None, help="Pacific zoom diff figure")
    args = ap.parse_args()

    from compare_omip_nemo import (_load_legoesm, _load_nemo,
                                   regrid_curv_to_latlon)
    from legoesm.ocean.diagnostics import mixed_layer_depth

    N = _load_nemo(args.nemo_gridt, args.nemo_time_idx)
    if N.get("mld") is None:
        print("FATAL: NEMO file has no mldr10_1")
        return 1

    mlds, sss_l, sst_l = [], [], []
    for snap in args.legoesm_snapshot:
        L = _load_legoesm(snap)
        z_c = np.asarray(L["z_center_ref"], np.float64)
        Hb = np.asarray(L["H_bathy"], np.float64)
        wet = ((z_c[(None,) * Hb.ndim + (slice(None),)] < Hb[..., None])
               & (L["mask"][..., None] > 0.5)).astype(np.float64)
        mlds.append(np.asarray(mixed_layer_depth(
            L["T3d"], L["S3d"], z_c, delta_sigma=0.01,
            wet_mask=wet, bottom_depth=Hb)))
        sss_l.append(L["sss"])
        sst_l.append(L["sst"])
    mldL = np.mean(mlds, axis=0)      # mean-of-diagnostic, like mldr10_1
    L = dict(L, sss=np.mean(sss_l, axis=0), sst=np.mean(sst_l, axis=0))
    if len(mlds) > 1:
        print(f"# MLD averaged over {len(mlds)} snapshots "
              f"(mean-of-diagnostic; protocol matched to NEMO 5-day mean)")

    tgt_lat = np.arange(-89.5, 90.0, 1.0)
    tgt_lon = np.arange(0.5, 360.0, 1.0)
    mldL_g, ocL = regrid_curv_to_latlon(
        np.nan_to_num(mldL, nan=0.0), L["lat"], L["lon"],
        np.isfinite(mldL).astype(np.float64), tgt_lat, tgt_lon)
    mldN_g, ocN = regrid_curv_to_latlon(
        np.nan_to_num(N["mld"], nan=0.0), N["lat"], N["lon"],
        np.isfinite(N["mld"]).astype(np.float64), tgt_lat, tgt_lon)
    lat2 = tgt_lat[:, None] * np.ones_like(tgt_lon)[None, :]
    lon2 = np.ones_like(tgt_lat)[:, None] * tgt_lon[None, :]
    ocean = (ocL > 0.5) & (ocN > 0.5) & np.isfinite(mldL_g) & np.isfinite(mldN_g)
    aw = np.cos(np.deg2rad(lat2))

    print(f"# {args.legoesm_snapshot} vs rec {args.nemo_time_idx}")
    print("# CAVEAT: ours instantaneous snapshot; NEMO 5-day-mean online MLD.")
    print(f"{'basin':9s} {'band':17s} {'bias_m':>8s} {'rmse_m':>8s} "
          f"{'corr':>6s} {'lego':>7s} {'nemo':>7s}")
    for bs in BASINS:
        bm = _basin_mask(lat2, lon2, bs)
        for bd, (lo, hi) in BANDS.items():
            m = ocean & bm & (lat2 >= lo) & (lat2 < hi)
            if m.sum() < 20:
                continue
            w = aw[m]
            dL, dN = mldL_g[m], mldN_g[m]
            bias = float(((dL - dN) * w).sum() / w.sum())
            rmse = float(np.sqrt((((dL - dN) ** 2) * w).sum() / w.sum()))
            c = float(np.corrcoef(dL, dN)[0, 1])
            print(f"{bs:9s} {bd:17s} {bias:+8.1f} {rmse:8.1f} {c:6.2f} "
                  f"{(dL * w).sum() / w.sum():7.1f} "
                  f"{(dN * w).sum() / w.sum():7.1f}")

    # NW Pacific Kuroshio-extension box
    m = ocean & (lat2 >= 30) & (lat2 <= 45) & (lon2 >= 140) & (lon2 <= 180)
    w = aw[m]
    print(f"\nNW_Pacific_30N45N_140E180: bias "
          f"{(((mldL_g - mldN_g)[m]) * w).sum() / w.sum():+.1f} m  "
          f"lego {((mldL_g[m]) * w).sum() / w.sum():.1f}  "
          f"nemo {((mldN_g[m]) * w).sum() / w.sum():.1f}")

    # Equator-line jump detector, per basin, OUR field vs NEMO's
    def jump(field, rows):
        d = np.abs(np.diff(field, axis=0))          # (nlat-1, nlon)
        ok = ocean[:-1] & ocean[1:]
        sel = np.zeros_like(ok)
        sel[rows] = True
        v = d[ok & sel]
        return float(np.nanmean(v)) if v.size else np.nan
    eq_rows = np.where(np.abs(tgt_lat[:-1]) <= 2.0)[0]
    off_rows = np.where((np.abs(tgt_lat[:-1]) >= 5.0)
                        & (np.abs(tgt_lat[:-1]) <= 10.0))[0]
    print("\n# equator-line jump: mean |dMLD/drow| at |lat|<=2 over the same "
          "at 5-10 deg (ratio >> 1 only in OURS = grid-line artifact)")
    for bs in BASINS:
        bm = _basin_mask(lat2, lon2, bs)
        oc_b = ocean & bm
        both = np.zeros_like(ocean)
        both[:] = oc_b

        def bj(field, rows):
            d = np.abs(np.diff(np.where(oc_b, field, np.nan), axis=0))
            sel = d[rows]
            return float(np.nanmean(sel)) if np.isfinite(sel).any() else np.nan
        rL = bj(mldL_g, eq_rows) / max(bj(mldL_g, off_rows), 1e-9)
        rN = bj(mldN_g, eq_rows) / max(bj(mldN_g, off_rows), 1e-9)
        print(f"  {bs:9s} ours ratio {rL:5.2f}   nemo ratio {rN:5.2f}")

    # RESIDUAL REGRESSION (GLM discriminator, 2026-08-25): if the tropical
    # Pacific MLD residual is COLLOCATED with the SSS residual, the shallow
    # bias is a salinity barrier layer (precip/freshwater placement), not a
    # mixing-strength error; collocation with the SST residual instead
    # supports the entrainment/cold-tongue loop.  Zero-cost: both fields are
    # already in the snapshot and the NEMO record.
    sssL_g, _ = regrid_curv_to_latlon(L["sss"], L["lat"], L["lon"],
                                      L["mask"], tgt_lat, tgt_lon)
    sssN_g, _ = regrid_curv_to_latlon(N["sss"], N["lat"], N["lon"],
                                      N["mask"], tgt_lat, tgt_lon)
    sstL_g, _ = regrid_curv_to_latlon(L["sst"], L["lat"], L["lon"],
                                      L["mask"], tgt_lat, tgt_lon)
    sstN_g, _ = regrid_curv_to_latlon(N["sst"], N["lat"], N["lon"],
                                      N["mask"], tgt_lat, tgt_lon)
    dmld = mldL_g - mldN_g
    dsss = sssL_g - sssN_g
    dsst = sstL_g - sstN_g
    print("\n# residual collocation: corr(dMLD, dSSS) and corr(dMLD, dSST)"
          "\n#   strong NEGATIVE corr(dMLD,dSSS): fresh cap where MLD shallow"
          " = barrier-layer/precip placement"
          "\n#   strong POSITIVE corr(dMLD,dSST): warm where MLD shallow"
          " = entrainment loop")
    for bs in BASINS:
        bm = _basin_mask(lat2, lon2, bs)
        for bd, (lo, hi) in BANDS.items():
            m = (ocean & bm & (lat2 >= lo) & (lat2 < hi)
                 & np.isfinite(dsss) & np.isfinite(dsst))
            if m.sum() < 50:
                continue
            cs = float(np.corrcoef(dmld[m], dsss[m])[0, 1])
            ct = float(np.corrcoef(dmld[m], dsst[m])[0, 1])
            print(f"  {bs:9s} {bd:17s} corr(dMLD,dSSS)={cs:+5.2f}  "
                  f"corr(dMLD,dSST)={ct:+5.2f}  n={int(m.sum())}")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
        box = (slice(None), slice(None))
        sel_lat = (tgt_lat >= -50) & (tgt_lat <= 65)
        sel_lon = (tgt_lon >= 110) & (tgt_lon <= 295)
        ext = [110, 295, -50, 65]
        for ax, f, t, cm, vl in (
            (axes[0], mldL_g, "legoESM MLD", "viridis", (0, 250)),
            (axes[1], mldN_g, "NEMO MLD (5-day mean)", "viridis", (0, 250)),
            (axes[2], mldL_g - mldN_g, "diff (lego - NEMO)", "RdBu_r", (-120, 120)),
        ):
            ff = np.where(ocean, f, np.nan)[np.ix_(sel_lat, sel_lon)]
            im = ax.imshow(ff, origin="lower", extent=ext, cmap=cm,
                           vmin=vl[0], vmax=vl[1], aspect="auto")
            ax.set_title(t); fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"Pacific MLD: {Path(args.legoesm_snapshot[0]).name} "
                     f"vs GATEWAY rec {args.nemo_time_idx}")
        fig.savefig(args.png, dpi=110)
        print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
