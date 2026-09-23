#!/usr/bin/env python
"""Does the coastal salinity gap look like MISPLACED river water or like a
DIFFERENT amount of water?

The two candidates left for FESOM's coastal surface-salinity excess both act
from t=0 and both concentrate at river mouths, so every horizontal statistic
confounds them: river cells carry the ocean's sharpest salinity gradients, so
"the initial condition interpolates badly here" and "the river water is put in
the wrong place here" predict the same map.

They do NOT predict the same COLUMN.

The tripole lane runs NEMO's ``ln_rnf_depth_ini``, which spreads a river's
freshwater down to a depth set by the local runoff climatology, up to about
150 m. The FESOM lane cannot -- the flag is on its allowlist's rejected list
and its ocean core carries no spread-depth map -- so the same river water
lands entirely in the surface cell. Both lanes receive the SAME runoff from
the SAME file. So if vertical placement is the whole story:

  * FESOM is FRESHER than the tripole at the surface and SALTIER just below,
    a dipole whose crossing sits inside the spreading depth, and
  * the column integral of the difference is near ZERO, because no water was
    added or removed, only moved.

If instead the lanes hold different amounts of salt in the column -- which is
what an initial-condition difference means -- the column integral does not
cancel.

That ratio is the discriminator, it needs no oracle, and because it compares
our two lanes directly it is immune to the salinity-convention difference
between the models and the NEMO output.

CONTROL, and it is the point of the whole design: the same two numbers are
computed on LOW-RUNOFF coastal columns, where neither the spreading lever nor
the river water exists. A dipole that shows up there too is not about rivers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_NATIVE_J = slice(0, 331)
_NATIVE_I = slice(1, 361)


def _xyz(lat_rad, lon_rad):
    c = np.cos(lat_rad)
    return np.stack([c * np.cos(lon_rad), c * np.sin(lon_rad),
                     np.sin(lat_rad)], axis=-1)


def node_to_cell_mean(node_val, node_lat, node_lon, cell_lat, cell_lon,
                      cell_wet):
    """Average FESOM node columns onto the tripole cells they fall in.

    Nearest-cell assignment, so a cell with no node nearest to it is left
    empty rather than being filled by a distant node -- the caller drops those
    cells from every stratum, which keeps both lanes on exactly the same set.
    """
    from scipy.spatial import cKDTree

    wet = np.asarray(cell_wet).ravel() > 0.5
    if not wet.any():
        raise SystemExit("target geometry has no wet cells")
    tgt = _xyz(np.deg2rad(np.asarray(cell_lat).ravel()[wet]),
               np.deg2rad(np.asarray(cell_lon).ravel()[wet]))
    src = _xyz(np.deg2rad(np.asarray(node_lat).ravel()),
               np.deg2rad(np.asarray(node_lon).ravel()))
    _, idx = cKDTree(tgt).query(src, k=1)
    idx = np.asarray(idx).ravel()

    nlev = node_val.shape[1]
    acc = np.zeros((int(wet.sum()), nlev), dtype=np.float64)
    cnt = np.zeros(int(wet.sum()), dtype=np.int64)
    good = np.isfinite(node_val)
    np.add.at(acc, idx, np.where(good, node_val, 0.0))
    np.add.at(cnt, idx, 1)
    out_flat = np.full((wet.size, nlev), np.nan)
    hit = cnt > 0
    filled = np.full((int(wet.sum()), nlev), np.nan)
    filled[hit] = acc[hit] / cnt[hit, None]
    out_flat[wet] = filled
    return out_flat.reshape(np.asarray(cell_lat).shape + (nlev,))


def dz_from_centers(z_center):
    """Layer thicknesses from level centres, positive, same length."""
    z = np.abs(np.asarray(z_center, dtype=np.float64))
    face = np.empty(z.size + 1)
    face[0] = 0.0
    face[1:-1] = 0.5 * (z[:-1] + z[1:])
    face[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    return np.diff(face)


def dipole_stats(dS, dz, z_center, spread_m):
    """Surface lobe, subsurface lobe, and the cancellation ratio.

    ``dS`` is FESOM minus tripole for one stratum, already averaged over the
    stratum's columns, shape (nlev,). Everything is restricted to the
    spreading depth, since that is the only depth range the lever acts on.
    """
    z = np.abs(np.asarray(z_center, dtype=np.float64))
    band = z <= spread_m
    if band.sum() < 3:
        raise SystemExit(f"fewer than 3 levels above {spread_m} m")
    w = dz[band]
    d = dS[band]
    ok = np.isfinite(d)
    if ok.sum() < 3:
        raise SystemExit("stratum has fewer than 3 finite levels")
    w, d, zb = w[ok], d[ok], z[band][ok]
    net = float((d * w).sum())
    gross = float((np.abs(d) * w).sum())
    top = zb <= dz[0] * 1.5          # the surface cell and its immediate pair
    return {
        "net": net,
        "gross": gross,
        # 0 = pure redistribution (water moved, none added); 1 = a net
        # freshwater/salt difference that no rearrangement can explain.
        "cancel": abs(net) / gross if gross > 0 else float("nan"),
        "surface": float(d[top].mean()) if top.any() else float("nan"),
        "subsurface": float(d[~top].mean()) if (~top).any() else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fesom", type=Path, required=True)
    p.add_argument("--tripole", type=Path, required=True)
    p.add_argument("--runoff-file", type=Path, required=True)
    p.add_argument("--spread-m", type=float, default=150.0,
                   help="NEMO rn_dep_max, the depth the biggest rivers spread "
                        "over; the band the lever can act in")
    p.add_argument("--river-pct", type=float, default=90.0,
                   help="percentile of the wet cells' NONZERO runoff above "
                        "which a column counts as a river mouth")
    p.add_argument("--low-pct", type=float, default=50.0,
                   help="percentile below which a column is the low-runoff "
                        "control")
    args = p.parse_args()

    import netCDF4 as nc

    F = np.load(args.fesom, allow_pickle=True)
    T = np.load(args.tripole, allow_pickle=True)
    for tag, d, keys in (("fesom", F, ("S", "lat_T", "lon_T")),
                         ("tripole", T, ("S", "lat_T", "lon_T", "land_mask",
                                         "z_center_ref"))):
        miss = [k for k in keys if k not in d.files]
        if miss:
            raise SystemExit(f"{tag} snapshot lacks {miss}")

    z_center = np.asarray(T["z_center_ref"], dtype=np.float64)
    if F["S"].shape[1] != z_center.size:
        raise SystemExit(
            f"level mismatch: FESOM has {F['S'].shape[1]}, tripole "
            f"{z_center.size} -- this probe assumes a shared vertical grid")
    dz = dz_from_centers(z_center)

    cell_lat = np.asarray(T["lat_T"], dtype=np.float64)
    cell_lon = np.asarray(T["lon_T"], dtype=np.float64)
    wet = np.asarray(T["land_mask"], dtype=np.float64) > 0.5

    fes = node_to_cell_mean(np.asarray(F["S"], dtype=np.float64),
                            np.asarray(F["lat_T"], dtype=np.float64),
                            np.asarray(F["lon_T"], dtype=np.float64),
                            cell_lat, cell_lon, wet)
    trp = np.asarray(T["S"], dtype=np.float64)
    if fes.shape != trp.shape:
        raise SystemExit(f"shape mismatch {fes.shape} vs {trp.shape}")
    dS = fes - trp

    with nc.Dataset(args.runoff_file) as f:
        rnf = np.nan_to_num(np.asarray(f.variables["sorunoff"][:],
                                       dtype=np.float64), nan=0.0).max(axis=0)
    rnf_full = np.zeros_like(cell_lat)
    rnf_full[_NATIVE_J, _NATIVE_I] = rnf
    if rnf.shape != rnf_full[_NATIVE_J, _NATIVE_I].shape:
        raise SystemExit(f"runoff {rnf.shape} does not fit the native slice")

    have = wet & np.isfinite(dS[..., 0])
    nz = rnf_full[have & (rnf_full > 0.0)]
    if nz.size == 0:
        raise SystemExit("no runoff on any usable column")
    hi = np.percentile(nz, args.river_pct)
    lo = np.percentile(nz, args.low_pct)

    # The HALO is the conservation test. Horizontal smoothing moves river
    # water out of the mouth and into the cells around it, so if extra
    # smoothing is what makes FESOM's mouths saltier, the same water must show
    # up as FRESHER somewhere nearby -- and the mouth-plus-halo total must
    # very nearly cancel. A saltier mouth with no fresher halo is not
    # smoothing; it is missing water.
    strata = [
        (f"river mouths (>p{args.river_pct:g})", have & (rnf_full > hi)),
        (f"halo (p{args.low_pct:g}-p{args.river_pct:g})",
         have & (rnf_full > lo) & (rnf_full <= hi)),
        (f"mouth+halo (>p{args.low_pct:g})", have & (rnf_full > lo)),
        (f"low runoff (<=p{args.low_pct:g})", have & (rnf_full > 0.0)
         & (rnf_full <= lo)),
    ]

    print(f"shared vertical grid: {z_center.size} levels, "
          f"spreading band z <= {args.spread_m:g} m "
          f"({int((np.abs(z_center) <= args.spread_m).sum())} levels)")
    print(f"{'stratum':28s} {'cols':>6s} {'surf dS':>9s} {'sub dS':>9s} "
          f"{'net':>10s} {'gross':>10s} {'cancel':>8s}")
    out = {}
    for name, sel in strata:
        n = int(sel.sum())
        if n == 0:
            raise SystemExit(f"stratum '{name}' is empty")
        prof = np.nanmean(dS[sel], axis=0)
        s = dipole_stats(prof, dz, z_center, args.spread_m)
        out[name] = s
        print(f"{name:28s} {n:6d} {s['surface']:+9.4f} {s['subsurface']:+9.4f} "
              f"{s['net']:+10.3f} {s['gross']:10.3f} {s['cancel']:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
