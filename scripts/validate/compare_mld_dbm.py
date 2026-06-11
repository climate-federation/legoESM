"""Compare legoESM mixed-layer depth to the de Boyer Montegut (2022) climatology.

The OMIP MLD evaluation of Treguier et al. (2023, GMD 16:3849): legoESM MLD,
computed with the de Boyer Montegut method (potential-density threshold
Delta-sigma = 0.03 kg/m^3 referenced to 10 m), is scored against the dBM (2022)
observational climatology (``mld_dr003_ref10m.nc``, 1deg monthly) for a chosen
calendar month -- the NH winter maximum (March) and the SH winter maximum
(September) are the paper's headline maps.

Reuses ``ocean.diagnostics.mixed_layer_depth`` (the canonical diagnostic) and
``compare_omip_nemo.regrid_curv_to_latlon`` (the curvilinear regridder); no
re-derivation.  Requires a snapshot written with the MLD geometry (H_bathy +
z_center_ref); see ``run_omip_core2._save_snapshot``.

Reports global + regional (paper regions) bias / RMSE / median bias and a
legoESM | obs | diff map.  The paper's metric is inter-model spread, not a
single RMSE, so the regional biases are interpreted against its reported ranges
(winter subpolar +-50..100+ m; summer +-20..40 m; tropical 5N maximum).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse the curvilinear regridder + weighted stats from the NEMO comparison.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_omip_nemo import regrid_curv_to_latlon, _wstats  # noqa: E402

_DBM_DEFAULT = ("results/omip_nemo/obs_mld/mld_dr003_ref10m.nc")

# Paper regions (lat bands; the paper highlights subpolar winter + Southern Ocean).
_REGIONS = {
    "southern_ocean_S_of_45S": (-90.0, -45.0),
    "SH_midlat_45S_20S": (-45.0, -20.0),
    "tropics_20S_20N": (-20.0, 20.0),
    "NH_subtropics_20N_45N": (20.0, 45.0),
    "subpolar_N_of_45N": (45.0, 90.0),
}


def _load_snapshot_mld(snapshot: Path, delta_sigma: float):
    from legoesm.ocean.diagnostics import mixed_layer_depth
    s = np.load(snapshot)
    keys = set(getattr(s, "files", []))
    if "z_center_ref" not in keys or "H_bathy" not in keys:
        raise SystemExit(
            f"[mld-dbm] {snapshot} lacks z_center_ref/H_bathy geometry "
            "(written only by post-a92a21eb/7d914c8e runs) -- re-run to enable.")
    T = np.asarray(s["T"]); S = np.asarray(s["S"])
    z_c = np.asarray(s["z_center_ref"], dtype=np.float64)
    Hb = np.asarray(s["H_bathy"], dtype=np.float64)
    mask2d = np.asarray(s["land_mask"])
    wet = ((z_c[(None,) * Hb.ndim + (slice(None),)] < Hb[..., None])
           & (mask2d[..., None] > 0.5)).astype(np.float64)
    mld = np.asarray(mixed_layer_depth(
        T, S, z_c, delta_sigma=delta_sigma, wet_mask=wet, bottom_depth=Hb))
    return mld, np.asarray(s["lat_T"]), np.asarray(s["lon_T"]), mask2d


def _load_dbm_month(dbm_file: Path, month: int):
    import xarray as xr
    ds = xr.open_dataset(dbm_file, decode_times=False)
    mld = np.asarray(ds["mld_dr003"].isel(time=month - 1).values, dtype=np.float64)  # (lat,lon)
    lat = np.asarray(ds["lat"].values); lon = np.asarray(ds["lon"].values)
    omask = np.asarray(ds["mask"].values) > 0.5 if "mask" in ds else np.isfinite(mld)
    mld = np.where(omask, mld, np.nan)
    return mld, lat, lon


def _region_stats(L, N, area, tgt_lat):
    out = {}
    for name, (lo, hi) in _REGIONS.items():
        band = (tgt_lat >= lo) & (tgt_lat < hi)
        a = area * band[:, None]
        if a.sum() <= 0:
            out[name] = None
            continue
        out[name] = _wstats(L, N, a)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True,
                   help="legoESM snapshot .npz with MLD geometry (z_center_ref + H_bathy).")
    p.add_argument("--month", type=int, required=True,
                   help="Calendar month 1-12 (3 = NH winter max, 9 = SH winter max).")
    p.add_argument("--dbm-file", type=Path, default=Path(_DBM_DEFAULT))
    p.add_argument("--delta-sigma", type=float, default=0.03,
                   help="Density threshold [kg/m^3]; 0.03 = dBM/Treguier (default).")
    p.add_argument("--res-deg", type=float, default=1.0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--grid-label", type=str, default="legoESM")
    args = p.parse_args()
    if not (1 <= args.month <= 12):
        raise SystemExit(f"--month must be 1..12, got {args.month}")
    if abs(args.res_deg - 1.0) > 1e-9:
        raise SystemExit(
            "--res-deg must be 1.0: the dBM nearest-index alignment assumes the "
            "1deg cell centres of the dBM grid; a different target needs true "
            "interpolation of the obs.")
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)

    mldL, latL, lonL, maskL = _load_snapshot_mld(args.snapshot, args.delta_sigma)
    mldO, latO, lonO = _load_dbm_month(args.dbm_file, args.month)
    print(f"[load] legoESM MLD {mldL.shape} (dsigma={args.delta_sigma}), "
          f"dBM month {args.month} {mldO.shape}")

    r = args.res_deg
    tgt_lat = np.arange(-89.5, 90.0, r)
    tgt_lon = np.arange(0.5, 360.0, r)

    # legoESM MLD -> common 1deg grid via the curvilinear regridder.
    mldL_g, ocL = regrid_curv_to_latlon(
        np.nan_to_num(mldL, nan=0.0), latL, lonL,
        np.isfinite(mldL).astype(np.float64), tgt_lat, tgt_lon)
    # dBM is already 1deg lat-lon: shift lon (-179.5..179.5 -> 0.5..359.5) by a
    # column roll (no interpolation), align lat ascending.
    lonO360 = lonO % 360.0
    order = np.argsort(lonO360)
    mldO_s = mldO[:, order]
    lonO_s = lonO360[order]
    if latO[0] > latO[-1]:
        mldO_s = mldO_s[::-1, :]; latO = latO[::-1]
    # Nearest-index map onto tgt grid (both ~1deg; exact for matching centres).
    li = np.clip(np.searchsorted(latO, tgt_lat), 0, len(latO) - 1)
    ci = np.clip(np.searchsorted(lonO_s, tgt_lon), 0, len(lonO_s) - 1)
    mldO_g = mldO_s[np.ix_(li, ci)]
    ocO = np.isfinite(mldO_g)

    # Valid = both grids ocean AND both MLD finite (a bad regrid cell can't
    # poison the area-weighted stats).
    ocean = (ocL > 0.5) & ocO & np.isfinite(mldL_g) & np.isfinite(mldO_g)
    if not ocean.any():
        raise SystemExit("[mld-dbm] no overlapping ocean cells between legoESM and dBM "
                         "-- check the snapshot grid / month / masks.")
    area = (np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones_like(tgt_lon)[None, :]) * ocean

    stats = _wstats(mldL_g, mldO_g, area)
    logstats = _wstats(np.log1p(np.maximum(mldL_g, 0.0)),
                       np.log1p(np.maximum(mldO_g, 0.0)), area)
    fin = ocean & np.isfinite(mldL_g) & np.isfinite(mldO_g)
    med_bias = float(np.median((mldL_g - mldO_g)[fin])) if fin.any() else float("nan")
    bands = _region_stats(mldL_g, mldO_g, area, tgt_lat)

    report = {
        "snapshot": str(args.snapshot), "month": args.month,
        "dbm_file": str(args.dbm_file), "delta_sigma": args.delta_sigma,
        "reference": "de Boyer Montegut (2022) mld_dr003_ref10m; Treguier+ 2023 GMD 16:3849",
        "comparison_type": ("instantaneous perpetual-NYF month-N snapshot vs 1970-2021 "
                            "climatological month -- a skill indicator, not strict validation; "
                            "the paper's metric is inter-model spread (winter subpolar +-50..100 m, "
                            "summer +-20..40 m)"),
        "global": {**stats, "rmse_log1p_m": logstats["rmse"], "median_bias_m": med_bias},
        "regions": bands,
        "n_ocean_cells": int(ocean.sum()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[MLD vs dBM month {args.month}] global rmse {stats['rmse']:.1f} m  "
          f"bias {stats['bias']:+.1f} m  median-bias {med_bias:+.1f} m  corr {stats['corr']:.3f}")
    for bn, bs in bands.items():
        if bs:
            print(f"   {bn:24s} bias {bs['bias']:+7.1f} m  rmse {bs['rmse']:6.1f} m")

    # Maps (legoESM | obs | diff), capped at obs 99th pct so deep convection
    # tails don't wash out the scale.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cap = float(np.nanpercentile(np.where(ocean, mldO_g, np.nan), 99))
    Lm = np.where(ocean, np.minimum(mldL_g, cap), np.nan)
    Nm = np.where(ocean, np.minimum(mldO_g, cap), np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    for a, dat, ttl in [(ax[0], Lm, f"{args.grid_label} MLD"),
                        (ax[1], Nm, "dBM2022 MLD")]:
        im = a.pcolormesh(tgt_lon, tgt_lat, dat, vmin=0, vmax=cap, cmap="viridis", shading="auto")
        a.set_title(ttl); plt.colorbar(im, ax=a, shrink=0.8, label="m")
    dmax = float(np.nanpercentile(np.abs(Lm - Nm), 98))
    im = ax[2].pcolormesh(tgt_lon, tgt_lat, Lm - Nm, vmin=-dmax, vmax=dmax,
                          cmap="RdBu_r", shading="auto")
    ax[2].set_title(f"MLD diff ({args.grid_label}-dBM)"); plt.colorbar(im, ax=ax[2], shrink=0.8, label="m")
    fig.suptitle(f"{args.grid_label} vs dBM2022 MLD — month {args.month} (dsigma={args.delta_sigma})",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(out / f"MLD_maps_m{args.month:02d}.png", dpi=90)
    plt.close(fig)
    # Zonal-mean MLD (paper Fig-9 style).
    fig, a = plt.subplots(figsize=(6, 5))
    Lz = np.nanmean(np.where(ocean, mldL_g, np.nan), axis=1)
    Nz = np.nanmean(np.where(ocean, mldO_g, np.nan), axis=1)
    a.plot(Lz, tgt_lat, label=args.grid_label); a.plot(Nz, tgt_lat, label="dBM2022")
    a.set_xlabel("zonal-mean MLD [m]"); a.set_ylabel("lat"); a.legend()
    a.set_title(f"zonal-mean MLD — month {args.month}")
    fig.tight_layout(); fig.savefig(out / f"MLD_zonal_m{args.month:02d}.png", dpi=90)
    plt.close(fig)
    print(f"[done] report + maps -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
