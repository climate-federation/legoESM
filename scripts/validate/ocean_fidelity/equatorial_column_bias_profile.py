#!/usr/bin/env python
"""Equatorial box: per-level temperature bias and 0-D m heat-content bias, legoESM vs NEMO.

The discriminator this exists for: an ADVECTIVE heat-supply defect (too little cold water
upwelled or advected into the box) changes the box's column heat content; a VERTICAL
redistribution defect (mixing) moves heat within the column and leaves the column integral
near NEMO's while the surface warms and the subsurface cools.  The per-level profile shows
which; the integral to depth D says how much heat is actually missing.

Grid-agnostic: reads a legoESM window-mean snapshot (T_mean, h_mean, lat_T, lon_T,
land_mask, optional cell_area) from the tripole (2-D horizontal) or MPAS (1-D cell list)
lane, and one NEMO 5-day record (to, so, e3t).

Conventions (read from the files, not assumed):
  * NEMO 3-D ``to`` holds CONSERVATIVE temperature and ``so`` ABSOLUTE salinity
    (diaar5.F90, ln_teos10); converted here to potential temperature with NEMO's own
    eos_pt_from_ct (``legoesm.ocean.eos.nemo_potential_temperature_from_conservative``),
    so both sides are potential temperature.
  * Our T_mean is potential temperature; h_mean the window-mean layer thickness.
  * NEMO record r is the mean over days 5r..5r+5; our snapshot_dayN the mean over N-5..N.
  * Area weights: tripole cell_area; NEMO e1t*e2t from the mesh.  The MPAS snapshot
    carries no cell area, so MPAS means are UNWEIGHTED (quasi-uniform SCVT); the tripole
    and NEMO are printed both weighted and unweighted so the size of that choice is visible.
"""
import argparse
import subprocess
import sys

import netCDF4 as nc
import numpy as np

from legoesm.ocean.eos import nemo_potential_temperature_from_conservative


def _box_mask(lat, lon, lon_lo, lon_hi, halfwidth):
    return (np.abs(lat) <= halfwidth) & (lon % 360.0 >= lon_lo) & (lon % 360.0 <= lon_hi)


def _overlap(h, depth):
    """Per-level thickness lying above ``depth`` given layer thicknesses h (..., nz)."""
    top = np.concatenate([np.zeros_like(h[..., :1]), np.cumsum(h, axis=-1)[..., :-1]], axis=-1)
    return np.clip(depth - top, 0.0, h)


def _summaries(T, h, wet, w, depths):
    """Box statistics. T, h, wet: (ncol, nz); w: (ncol,) area weights or ones."""
    out = {}
    prof = []
    for k in range(T.shape[1]):
        m = wet[:, k]
        prof.append(np.nan if not m.any() else float((T[m, k] * w[m]).sum() / w[m].sum()))
    out["profile"] = np.array(prof)
    for D in depths:
        ov = np.where(wet, _overlap(np.where(wet, h, 0.0), D), 0.0)
        col_depth = ov.sum(axis=1)
        full = col_depth >= D - 1e-6
        hc = (T * ov).sum(axis=1)
        out[f"mean0_{D:g}"] = float((hc[full] / D * w[full]).sum() / w[full].sum())
        out[f"ncol0_{D:g}"] = int(full.sum())
    return out


def load_ours(path, lon_lo, lon_hi, hw):
    z = np.load(path)
    lat = np.asarray(z["lat_T"], np.float64); lon = np.asarray(z["lon_T"], np.float64)
    T = np.asarray(z["T_mean"], np.float64); h = np.asarray(z["h_mean"], np.float64)
    if np.nanmax(np.abs(lat)) <= 0.5 * np.pi + 1e-6:  # MPAS stores radians
        lat = np.degrees(lat); lon = np.degrees(lon)
    # The snapshot's ``land_mask`` is 1 over OCEAN (despite its name): verified on the
    # base1006 tripole (1736 of 1740 nino3 cells flagged 1 carry ~27.5 C, the 4 flagged 0
    # carry exactly 0).  Re-checked here on every file so a convention change fails loudly.
    wet2d = np.asarray(z["land_mask"]) == 1
    sst = T[..., 0]
    if np.mean(sst[wet2d] == 0.0) > 0.01 or np.mean(sst[~wet2d] != 0.0) > 0.01:
        raise SystemExit(f"{path}: land_mask is not the ocean flag this probe assumes; refusing")
    box = _box_mask(lat, lon, lon_lo, lon_hi, hw) & wet2d
    area = np.asarray(z["cell_area"], np.float64) if "cell_area" in z.files else None
    zc = np.abs(np.asarray(z["z_center_ref"], np.float64))
    Tc = T[box]; hc = h[box]
    wet = np.isfinite(Tc) & (hc > 1e-3)
    return dict(T=np.where(wet, Tc, 0.0), h=hc, wet=wet,
                area=None if area is None else area[box], zc=zc, n=int(box.sum()))


def load_nemo(gridt, mesh, rec, lon_lo, lon_hi, hw):
    t = nc.Dataset(gridt); m = nc.Dataset(mesh)
    lat = np.asarray(t.variables["nav_lat"][:], np.float64); lon = np.asarray(t.variables["nav_lon"][:], np.float64)
    ny, nx = lat.shape
    gphit = np.asarray(m.variables["gphit"][0])
    # locate the file's window in the mesh (5-day files may be the full or the trimmed domain)
    tm0 = np.asarray(m.variables["tmask"][0, 0])
    for (oj, oi) in ((0, 0), (0, 1)):
        g = gphit[oj:oj + ny, oi:oi + nx]
        wet = tm0[oj:oj + ny, oi:oi + nx] == 1
        if g.shape == lat.shape and np.max(np.abs(g - lat)[wet & (np.abs(lat) < 60)]) < 1e-3:
            break
    else:
        raise SystemExit("NEMO file is not a recognised window of the mesh; refusing")
    e1t = np.asarray(m.variables["e1t"][0], np.float64)[oj:oj + ny, oi:oi + nx]
    e2t = np.asarray(m.variables["e2t"][0], np.float64)[oj:oj + ny, oi:oi + nx]
    tmask = np.asarray(m.variables["tmask"][0], np.float64)[:, oj:oj + ny, oi:oi + nx]

    def rd(v):
        x = np.asarray(t.variables[v][rec], np.float64)
        return np.where(np.isfinite(x) & (np.abs(x) < 1e15), x, np.nan)
    ct = rd("to"); sa = rd("so"); e3 = rd("e3t")
    th = np.asarray(nemo_potential_temperature_from_conservative(ct, sa), np.float64)
    box = _box_mask(lat, lon, lon_lo, lon_hi, hw) & (tmask[0] == 1)
    nz = ct.shape[0]
    Tc = th[:, box].T; hc = e3[:, box].T
    wet = (tmask[:nz, box].T == 1) & np.isfinite(Tc) & np.isfinite(hc)
    zc = np.asarray(t.variables["deptht"][:], np.float64).ravel()
    return dict(T=np.where(wet, Tc, 0.0), h=np.where(wet, hc, 0.0), wet=wet,
                area=(e1t * e2t)[box], zc=zc, n=int(box.sum()),
                ct_minus_theta=float(np.nanmean((ct - th)[0][box])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", action="append", required=True,
                    help="LABEL=path to a legoESM window-mean snapshot; repeatable")
    ap.add_argument("--nemo-gridt", required=True)
    ap.add_argument("--nemo-rec", type=int, required=True, help="5-day record r = days 5r..5r+5")
    ap.add_argument("--mesh-mask", default="/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc")
    ap.add_argument("--lon-lo", type=float, required=True); ap.add_argument("--lon-hi", type=float, required=True)
    ap.add_argument("--lat-halfwidth", type=float, required=True)
    ap.add_argument("--depths", default="10,20,50,100,200")
    ap.add_argument("--zmax-profile", type=float, default=160.0)
    a = ap.parse_args()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"[provenance] git {sha} | {' '.join(sys.argv)}")
    depths = [float(x) for x in a.depths.split(",")]
    box = (a.lon_lo, a.lon_hi, a.lat_halfwidth)
    ne = load_nemo(a.nemo_gridt, a.mesh_mask, a.nemo_rec, *box)
    print(f"[box] {a.lon_lo:g}-{a.lon_hi:g}E |lat|<={a.lat_halfwidth:g} | NEMO rec {a.nemo_rec} "
          f"(days {5 * a.nemo_rec}-{5 * a.nemo_rec + 5}) | NEMO cols {ne['n']} | surface CT-theta mean {ne['ct_minus_theta']:+.4f} K")
    res = {"NEMO_w": _summaries(ne["T"], ne["h"], ne["wet"], ne["area"], depths),
           "NEMO_u": _summaries(ne["T"], ne["h"], ne["wet"], np.ones(ne["n"]), depths)}
    zref = ne["zc"]
    ours = {}
    for spec in a.snapshot:
        lab, path = spec.split("=", 1)
        o = load_ours(path, *box)
        ours[lab] = o
        print(f"[ours] {lab}: {path} | cols {o['n']} | area weights {'yes' if o['area'] is not None else 'NO (unweighted)'}")
        if o["area"] is not None:
            res[lab + "_w"] = _summaries(o["T"], o["h"], o["wet"], o["area"], depths)
        res[lab + "_u"] = _summaries(o["T"], o["h"], o["wet"], np.ones(o["n"]), depths)
    print("\n## 0-D m depth-mean temperature [degC] (full-depth columns only; ncol in brackets)")
    for key, s in res.items():
        print(f"  {key:14s} " + "  ".join(f"0-{D:g}: {s[f'mean0_{D:g}']:7.3f} [{s[f'ncol0_{D:g}']}]" for D in depths))
    print("\n## bias vs NEMO, same weighting (ours_w vs NEMO_w; ours_u vs NEMO_u)")
    for key, s in res.items():
        if key.startswith("NEMO"):
            continue
        ref = res["NEMO_w"] if key.endswith("_w") else res["NEMO_u"]
        print(f"  {key:14s} " + "  ".join(f"0-{D:g}: {s[f'mean0_{D:g}'] - ref[f'mean0_{D:g}']:+7.3f}" for D in depths))
    print("\n## per-level box-mean bias vs NEMO [K] at NEMO's level centres (our levels matched by nearest centre)")
    keys = [k for k in res if not k.startswith("NEMO")]
    print("  depth_m  NEMO_w  " + "  ".join(f"{k:>12s}" for k in keys))
    for k, z in enumerate(zref):
        if z > a.zmax_profile:
            break
        row = f"  {z:7.1f}  {res['NEMO_w']['profile'][k]:6.3f}  "
        for key in keys:
            lab = key.rsplit("_", 1)[0]
            ko = int(np.argmin(np.abs(ours[lab]["zc"] - z)))
            ref = res["NEMO_w"] if key.endswith("_w") else res["NEMO_u"]
            row += f"  {res[key]['profile'][ko] - ref['profile'][k]:+12.3f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
