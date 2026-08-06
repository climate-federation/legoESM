"""Convert the NEMO ORCA1 WOCE/Gouretski IC (curvilinear eORCA1 grid) into a
regular-1° WOA18-format NetCDF, so legoESM can initialise from the SAME
climatology NEMO uses (colleague faithfulness check, 2026-06-23).

NEMO ORCA1 (morays) initialises T/S from ``woce_temp_monthly_init_4p2.nc`` /
``woce_salt_monthly_init_4p2.nc`` — the WOCE-Argo (Gouretski) climatology on the
eORCA1 tripolar grid (2-D ``nav_lat``/``nav_lon``, 331×360, 75 NEMO levels,
monthly; vars ``contemp`` = CONSERVATIVE temperature, ``presalt`` = PRACTICAL
salinity).  legoESM's IC path (``init_ocean_from_woa`` → ``compute_woa_3d``)
consumes a REGULAR lat-lon WOA18 file (``t_an``/``s_an`` on ``WOA_DEPTHS``) and
regrids/vertical-interps to the model grid.  This script bridges the two with
ZERO model-code change: the output is a drop-in for ``--woa-init --woa-t/--woa-s``.

Pipeline (per requested month or the annual mean):
  1. read ``contemp``/``presalt`` + 2-D ``nav_lat``/``nav_lon`` + ``nav_lev``;
  2. horizontal regrid each native level curv→1° via unit-sphere kNN IDW
     (ocean source cells only; wrap/pole-safe — the same xyz-KDTree approach as
     ``run_omip_core2._regrid_curv_to_points``);
  3. vertical-interp each 1° column from the 75 NEMO levels to the WOA18
     standard depths (``WOA_DEPTHS``, the full 0–5500 m level set) so
     ``init_ocean_from_woa`` reads it as a standard WOA18 file (its source-depth
     axis is hardcoded to ``WOA_DEPTHS``).

CAVEATS (documented, surfaced at runtime):
  * ``contemp`` is CONSERVATIVE temperature (TEOS-10).  legoESM's EOS uses
    POTENTIAL temperature; with no ``gsw`` in the env we pass CT as PT — a
    ~0.05–0.2 °C IC offset (largest in warm, salty surface water).  Acceptable
    for a starting condition that then evolves; flagged for a future exact
    CT→PT conversion if needed.
  * The 1° / WOA-depth output coarsens the native ~1 m surface spacing to the
    WOA near-surface spacing (5–10 m); the mixed layer (≫5 m) is still well
    resolved, but the very-near-surface skin is smoothed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# eORCA1 INPUTS (the morays ORCA1 reference; same files NEMO ingests).
_NEMO_IC_DIR = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/"
                "ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions")
_DEFAULT_T = f"{_NEMO_IC_DIR}/woce_temp_monthly_init_4p2.nc"
_DEFAULT_S = f"{_NEMO_IC_DIR}/woce_salt_monthly_init_4p2.nc"


def _xyz_unit_sphere(lat_deg, lon_deg):
    """Unit-sphere Cartesian coords for (lat, lon) in degrees — wrap/pole-safe."""
    latr = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lonr = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    cl = np.cos(latr)
    return np.stack([cl * np.cos(lonr), cl * np.sin(lonr), np.sin(latr)], axis=-1)


def regrid_curv_to_latlon(field_lev, src_lat2d, src_lon2d, ocean2d,
                          tgt_lat1d, tgt_lon1d, k=4, max_deg=3.0):
    """kNN inverse-distance regrid of a curvilinear level field → regular 1° grid.

    ``field_lev`` (ny, nx) for ONE level; ``ocean2d`` (ny, nx) bool valid mask.
    Donors are restricted PER LEVEL to finite-and-ocean source cells, and to a
    great-circle ``max_deg`` locality cutoff (mirrors ``_regrid_curv_to_points``)
    so a target is NEVER filled by a far-away donor (no lateral extrapolation of
    deep-ocean water into shoaled/isolated columns).  Returns (n_lat, n_lon) with
    NaN where NO valid source lies within ``max_deg`` — the caller converts those
    to the WOA no-data sentinel so the downstream WOA flood-fill / land-mask /
    bathymetry masking handles them exactly as for a real WOA18 file.
    """
    from scipy.spatial import cKDTree
    valid = np.asarray(ocean2d) & np.isfinite(field_lev)
    out = np.full((tgt_lat1d.size, tgt_lon1d.size), np.nan)
    if not valid.any():
        return out
    src_xyz = _xyz_unit_sphere(src_lat2d[valid], src_lon2d[valid])
    vals = np.asarray(field_lev, dtype=np.float64)[valid]
    tree = cKDTree(src_xyz)
    LON, LAT = np.meshgrid(tgt_lon1d, tgt_lat1d)
    tgt_xyz = _xyz_unit_sphere(LAT.ravel(), LON.ravel())
    kk = int(min(k, vals.size))
    dist, idx = tree.query(tgt_xyz, k=kk)
    if kk == 1:
        dist = dist[:, None]; idx = idx[:, None]
    # chord length (Euclidean on the unit sphere) for the great-circle cutoff
    cutoff = 2.0 * np.sin(0.5 * np.deg2rad(max_deg))
    w = 1.0 / np.maximum(dist, 1e-12)
    w = np.where(dist <= cutoff, w, 0.0)             # drop donors beyond cutoff
    wsum = w.sum(axis=1)
    reached = wsum > 0.0                              # ≥1 donor within max_deg
    w[reached] /= wsum[reached, None]
    filled = np.einsum("nk,nk->n", w, vals[idx])
    flat = np.full(tgt_xyz.shape[0], np.nan)
    flat[reached] = filled[reached]
    return flat.reshape(tgt_lat1d.size, tgt_lon1d.size)


def build(woce_t, woce_s, out_dir, month=None):
    import xarray as xr
    from legoesm.ocean.init_woa import WOA_DEPTHS, interp_column_to_depths

    dt = xr.open_dataset(woce_t, decode_times=False)
    ds_ = xr.open_dataset(woce_s, decode_times=False)
    src_lat = np.asarray(dt["nav_lat"], dtype=np.float64)        # (ny, nx)
    src_lon = np.asarray(dt["nav_lon"], dtype=np.float64)
    src_z = np.abs(np.asarray(dt["nav_lev"], dtype=np.float64))  # (nz,) +down
    if not np.all(np.diff(src_z) > 0):
        raise ValueError(
            "nav_lev is not strictly ascending after abs(); np.interp requires "
            f"monotonic source depths. Got {src_z[:5]}...{src_z[-3:]}")
    T = np.asarray(dt["contemp"], dtype=np.float64)              # (t, z, y, x)
    S = np.asarray(ds_["presalt"], dtype=np.float64)
    if month is None:
        T = np.nanmean(np.where(np.abs(T) < 1e10, T, np.nan), axis=0)
        S = np.nanmean(np.where(np.abs(S) < 1e10, S, np.nan), axis=0)
        tag = "annual"
    else:
        if not (1 <= month <= 12):
            raise ValueError(f"--month must be 1..12 or omitted; got {month}")
        T = np.where(np.abs(T[month - 1]) < 1e10, T[month - 1], np.nan)
        S = np.where(np.abs(S[month - 1]) < 1e10, S[month - 1], np.nan)
        tag = f"m{month:02d}"
    nz = src_z.size
    print(f"[woce] source {T.shape} (nz={nz}) grid {src_lat.shape}; building {tag}")
    print("[woce] CAVEAT: contemp is CONSERVATIVE temp; passed as POTENTIAL temp "
          "(no gsw in env; ~0.05-0.2 degC IC offset).")

    tgt_lat = np.arange(-89.5, 90.0, 1.0)
    tgt_lon = np.arange(0.5, 360.0, 1.0)
    nwoa = WOA_DEPTHS.size
    # ocean mask = finite at the SURFACE native level (land is fill/NaN).
    ocean2d = np.isfinite(T[0])

    # 1) horizontal regrid each native level -> 1° ; 2) vertical -> WOA depths.
    Tll = np.stack([regrid_curv_to_latlon(T[k], src_lat, src_lon, ocean2d,
                                          tgt_lat, tgt_lon) for k in range(nz)])
    Sll = np.stack([regrid_curv_to_latlon(S[k], src_lat, src_lon, ocean2d,
                                          tgt_lat, tgt_lon) for k in range(nz)])
    # (nz, nlat, nlon) -> per-column vertical interp to WOA_DEPTHS
    Tw = np.empty((nwoa, tgt_lat.size, tgt_lon.size))
    Sw = np.empty_like(Tw)
    for j in range(tgt_lat.size):
        for i in range(tgt_lon.size):
            Tw[:, j, i] = interp_column_to_depths(Tll[:, j, i], src_z, WOA_DEPTHS)
            Sw[:, j, i] = interp_column_to_depths(Sll[:, j, i], src_z, WOA_DEPTHS)

    # No-data sentinel = NaN (NOT 0).  Targets with no valid donor within max_deg
    # (continents / isolated cells) stay NaN, exactly as a real WOA18 land cell.
    # The downstream path is NaN-aware: init_ocean_from_woa's _bilinear_2d EXCLUDES
    # NaN corners (so a coastal model cell straddling land+ocean gets the real
    # ocean value, NOT a 0-contaminated average), then fills any residual NaN with
    # T_fill/S_fill (init_woa.py:223-224).  Using 0 instead would let bilinear
    # average the sentinel into fresh coastal salinities that bypass the S<1
    # flood-fill (codex review).  So we leave NaN and let the validated WOA path
    # handle it.
    nodata = ~np.isfinite(Sw[0])                         # surface no-data (lat,lon)
    print(f"[woce] surface no-data (NaN) frac {nodata.mean():.3f} "
          "(NaN-aware downstream: bilinear excludes it, residual -> WOA fill)")

    def _rng(name, a):
        f = a[np.isfinite(a)]
        print(f"[woce] {name} (ocean): {f.min():.2f}..{f.max():.2f}")
    _rng("T", Tw); _rng("S", Sw)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    coords = dict(time=("time", [0.0]), depth=("depth", WOA_DEPTHS),
                  lat=("lat", tgt_lat), lon=("lon", tgt_lon))
    fT = out / f"woce_t_woa57_{tag}.nc"
    fS = out / f"woce_s_woa57_{tag}.nc"
    xr.Dataset({"t_an": (("time", "depth", "lat", "lon"), Tw[None])},
               coords=coords).to_netcdf(fT)
    xr.Dataset({"s_an": (("time", "depth", "lat", "lon"), Sw[None])},
               coords=coords).to_netcdf(fS)
    print(f"[woce] wrote {fT}\n[woce] wrote {fS}")
    print("[woce] USE: run_omip_core2.py --woa-init "
          f"--woa-t {fT} --woa-s {fS}")
    return fT, fS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--woce-t", default=_DEFAULT_T)
    p.add_argument("--woce-s", default=_DEFAULT_S)
    p.add_argument("--out-dir", default="data/woce_gouretski")
    p.add_argument("--month", type=int, default=None,
                   help="1..12 to extract a month; omit for the annual mean.")
    args = p.parse_args()
    build(args.woce_t, args.woce_s, args.out_dir, month=args.month)


if __name__ == "__main__":
    main()
