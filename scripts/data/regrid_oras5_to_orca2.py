#!/usr/bin/env python3
"""Map ORAS5 monthly means (ORCA025.L75) onto the ORCA2 model grid (30 levels).

Method (chosen by the user, details voted unanimously by author / Claude /
Argon on 2026-11-16):
- each ORCA025 T-cell goes to the NEAREST ORCA2 T-point on the unit sphere
  (the repo's first-order curvilinear locator);
- vertical mapping by layer-overlap thickness between ORCA025's 3D layers
  (1-D thicknesses, partial bottom cell e3t_ps) and ORCA2's own 3D layers
  (domcfg e3t_0, partial bottom cells), so heat and salt content are kept;
- weight = ORCA025 cell area (e1t*e2t) x overlap thickness x ORCA025 wet;
  SSH uses area x surface wet;
- an ORCA2 wet cell with no ORCA025 sample is NaN (excluded from the loss),
  ORCA025 water landing on ORCA2 land is dropped; both are counted.

Writes ``<out>/oras5_orca2_YYYYMM.nc`` (T, S on model levels, ssh) and
``<out>/weights.nc`` (per-cell weight sums, the counts as attributes).
Re-running skips months already written.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from legoesm.ocean.forcing.curvilinear_regrid import (  # noqa: E402
    layer_overlap_thickness,
    nearest_centre_index,
)

N_LEV_ORCA2 = 30
FIELDS = {"T": ("potential_temperature", "votemper"), "S": ("salinity", "vosaline")}
SSH = ("sea_surface_height", "sossheig")


def orca025_geometry(mesh_path):
    """lon, lat, area (ny, nx); wet (nz, ny, nx); layer tops/bottoms (nz, ny, nx)."""
    with netCDF4.Dataset(mesh_path) as m:
        g = lambda v: np.asarray(m[v][:], dtype=np.float64).squeeze()  # noqa: E731
        lon, lat = g("glamt"), g("gphit")
        area = g("e1t") * g("e2t")
        wet = np.asarray(m["tmask"][:]).squeeze() > 0
        gdepw, e3t = g("gdepw_0"), g("e3t_0")
        e3t_ps = g("e3t_ps")
        mbathy = np.asarray(m["mbathy"][:]).squeeze().astype(int)
    nz = gdepw.size
    top = np.broadcast_to(gdepw[:, None, None], wet.shape).copy()
    thick = np.broadcast_to(e3t[:, None, None], wet.shape).copy()
    jj, ii = np.nonzero(mbathy > 0)
    thick[mbathy[jj, ii] - 1, jj, ii] = e3t_ps[jj, ii]          # partial bottom cell
    if nz != wet.shape[0]:
        raise ValueError("ORCA025 level count disagrees between gdepw_0 and tmask")
    return lon, lat, area, wet, top, top + thick


def orca2_geometry(domcfg_path):
    """lon, lat (ny, nx); wet (nz, ny, nx); layer tops/bottoms (nz, ny, nx), z down."""
    with netCDF4.Dataset(domcfg_path) as d:
        lon, lat = np.asarray(d["glamt"][:], float), np.asarray(d["gphit"][:], float)
        e3t = np.asarray(d["e3t_0"][:], float)
        bottom = np.asarray(d["bottom_level"][:]).astype(int)
        top_level = np.asarray(d["top_level"][:]).astype(int)
    if bottom.max() > N_LEV_ORCA2:
        raise ValueError(f"ORCA2 has wet cells below level {N_LEV_ORCA2}")
    e3t = e3t[:N_LEV_ORCA2]
    k = np.arange(N_LEV_ORCA2)[:, None, None]
    wet = (k >= top_level[None] - 1) & (k < bottom[None])
    bot = np.cumsum(e3t, axis=0)
    return lon, lat, wet, bot - e3t, bot


def build_weights(src, dst):
    """Sparse (dst 3D x src 3D) and (dst 2D x src 2D) weights, plus counts."""
    slon, slat, sarea, swet, stop, sbot = src
    dlon, dlat, dwet, dtop, dbot = dst
    nz_s, ny_s, nx_s = swet.shape
    nz_d, ny_d, nx_d = dwet.shape
    n2s, n2d = ny_s * nx_s, ny_d * nx_d
    col = nearest_centre_index(slon, slat, dlon, dlat).ravel()       # dst column per src column
    dtop_c, dbot_c = dtop.reshape(nz_d, -1).T, dbot.reshape(nz_d, -1).T  # (n2d, nz_d)
    dwet_c = dwet.reshape(nz_d, -1).T
    rows, cols, vals = [], [], []
    dropped = 0.0
    total = 0.0
    for k in range(nz_s):
        c = np.nonzero(swet[k].ravel())[0]
        if c.size == 0:
            continue
        j = col[c]
        ov = layer_overlap_thickness(stop[k].ravel()[c][:, None], sbot[k].ravel()[c][:, None],
                                     dtop_c[j], dbot_c[j])[:, 0, :]           # (n_c, nz_d)
        vol = sarea.ravel()[c][:, None] * ov
        total += float(sarea.ravel()[c] @ (sbot[k].ravel()[c] - stop[k].ravel()[c]))
        keep = (vol > 0) & dwet_c[j]
        dropped_k = vol[(vol > 0) & ~dwet_c[j]].sum()
        dropped += float(dropped_k)
        ci, kd = np.nonzero(keep)
        rows.append(kd * n2d + j[ci])
        cols.append(k * n2s + c[ci])
        vals.append(vol[ci, kd])
    W3 = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(nz_d * n2d, nz_s * n2s))
    s = np.nonzero(swet[0].ravel())[0]
    keep2 = dwet[0].ravel()[col[s]]
    W2 = sp.csr_matrix((sarea.ravel()[s][keep2], (col[s][keep2], s[keep2])), shape=(n2d, n2s))
    counts = {"dropped_volume_fraction": dropped / total,
              "orca2_wet_cells": int(dwet.sum()),
              "orca2_wet_cells_without_data": int((dwet.ravel() & (np.asarray(W3.sum(1)).ravel() == 0)).sum()),
              "orca2_wet_surface_cells_without_ssh": int((dwet[0].ravel() & (np.asarray(W2.sum(1)).ravel() == 0)).sum())}
    return W3, W2, counts


def apply(W, x, wet_src, shape):
    """Weighted mean on the destination; NaN where no weight."""
    x = np.where(wet_src, np.asarray(x, dtype=np.float64), 0.0).ravel()
    num, den = W @ x, np.asarray(W.sum(1)).ravel()
    out = np.full(den.shape, np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out.reshape(shape)


def month_files(oras5_dir, subdir):
    out = {}
    for f in glob.glob(f"{oras5_dir}/{subdir}/*.nc"):
        m = re.search(r"_(\d{6})_", Path(f).name)
        if m:
            out[m.group(1)] = f
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--oras5", type=Path, required=True)
    p.add_argument("--mesh", type=Path, required=True, help="ORCA025 mesh_mask.nc")
    p.add_argument("--domcfg", type=Path, required=True, help="ORCA2 domain_cfg")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--months", nargs="*", default=None, help="YYYYMM subset")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    src, dst = orca025_geometry(args.mesh), orca2_geometry(args.domcfg)
    W3, W2, counts = build_weights(src, dst)
    print("weights:", counts, flush=True)
    dwet = dst[2]
    with netCDF4.Dataset(args.out / "weights.nc", "w") as ds:
        ds.createDimension("z", dwet.shape[0]); ds.createDimension("y", dwet.shape[1])
        ds.createDimension("x", dwet.shape[2])
        ds.createVariable("weight3d", "f8", ("z", "y", "x"))[:] = np.asarray(W3.sum(1)).reshape(dwet.shape)
        ds.createVariable("weight2d", "f8", ("y", "x"))[:] = np.asarray(W2.sum(1)).reshape(dwet.shape[1:])
        for k, v in counts.items():
            ds.setncattr(k, v)
    files = {name: month_files(args.oras5, sub) for name, (sub, _) in FIELDS.items()}
    files["ssh"] = month_files(args.oras5, SSH[0])
    months = sorted(set.intersection(*(set(v) for v in files.values())))
    if args.months:
        months = [m for m in months if m in set(args.months)]
    swet = src[3]
    for ym in months:
        target = args.out / f"oras5_orca2_{ym}.nc"
        if target.exists():
            continue
        out = {}
        for name, (_, var) in FIELDS.items():
            with netCDF4.Dataset(files[name][ym]) as ds:
                x = np.ma.filled(ds[var][0], np.nan)
            out[name] = apply(W3, x, swet & np.isfinite(x), dwet.shape)
        with netCDF4.Dataset(files["ssh"][ym]) as ds:
            x = np.ma.filled(ds[SSH[1]][0], np.nan)
        out["ssh"] = apply(W2, x, swet[0] & np.isfinite(x), dwet.shape[1:])
        tmp = target.with_suffix(".nc.part")
        with netCDF4.Dataset(tmp, "w") as ds:
            ds.createDimension("z", dwet.shape[0]); ds.createDimension("y", dwet.shape[1])
            ds.createDimension("x", dwet.shape[2])
            for name in ("T", "S"):
                ds.createVariable(name, "f4", ("z", "y", "x"))[:] = out[name]
            ds.createVariable("ssh", "f4", ("y", "x"))[:] = out["ssh"]
            ds.setncattr("source", "ORAS5 consolidated, regridded by scripts/data/regrid_oras5_to_orca2.py")
        tmp.replace(target)
        print("wrote", target.name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
