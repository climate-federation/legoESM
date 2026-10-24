#!/usr/bin/env python
"""Build bgc_init.npz (DIC, ALK, NO3, PO4, Si, O2 on the eORCA1 model grid) from
GLODAPv2.2016b mapped climatologies and WOA23 annual nutrients / oxygen.

    DIC  <- GLODAPv2.2016b.PI_TCO2.nc   (pre-industrial DIC = TCO2 - Cant)
    ALK  <- GLODAPv2.2016b.TAlk.nc
    NO3  <- woa23_all_n00_01.nc  (n_an)
    PO4  <- woa23_all_p00_01.nc  (p_an)
    Si   <- woa23_all_i00_01.nc  (i_an)
    O2   <- woa23_all_o00_01.nc  (o_an)

All sources are umol/kg on regular 1-degree grids; output is mol/m^3 with
rho = 1025 kg/m^3.  Pipeline per variable:
  1. horizontal fill of missing cells on every source level (nearest valid,
     iterative), then vertical fill below the deepest valid level (hold last);
  2. horizontal regrid source 1-degree -> eORCA1 T-points by unit-sphere kNN
     inverse-distance weights (k=4), wrap/pole safe — the same approach as
     scripts/data/build_woce_ic_latlon.py and run_omip_core2._regrid_curv_to_points;
  3. vertical linear interpolation onto the model's 75 reference level centres
     (from domain_cfg e3t_1d), clamped at both ends;
  4. land (tmask surface == 0) set to 0.

Also writes DIC_present (TCO2) for later present-day experiments.

Usage (CPU, login node or `shared`):
    $PY build_bgc_init_eorca1.py \
        --mesh   /work/bb1596/b384036/diffESM/legoesm_main_fresh/data/grids/eORCA1.2_mesh_mask.nc \
        --domain /work/bb1596/b384036/diffESM/data/nemo_orca1_inputs/orca1_inputs/data_repository/input_fields/domain_cfg.nc \
        --glodap /work/bb1596/b384036/data/bgc_obs/glodap/GLODAPv2.2016b_MappedClimatologies \
        --woa    /work/bb1596/b384036/data/bgc_obs/woa23 \
        --out    /work/bb1596/b384036/diffESM/data/bgc_init/bgc_init.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

RHO_SW = 1025.0
UMOLKG_TO_MOLM3 = RHO_SW * 1.0e-6


# ----------------------------------------------------------------------------- io
def _read_regular(path: Path, var: str):
    """Return (lat[ny], lon[nx] in [-180,180), depth[nz] positive down, data[nz,ny,nx] with NaN)."""
    ds = xr.open_dataset(path, decode_times=False)
    da = ds[var]
    if "time" in da.dims:
        da = da.isel(time=0)
    # depth axis: the dim that is not lat/lon
    dep_dim = [d for d in da.dims if d not in ("lat", "lon")][0]
    if dep_dim in ds.coords or dep_dim in ds.variables:
        depth = np.asarray(ds[dep_dim].values, dtype=np.float64)
    else:  # GLODAP: dim 'depth_surface' with a separate 1-D variable 'Depth'
        cand = [v for v in ds.variables if ds[v].dims == (dep_dim,)]
        depth = np.asarray(ds[cand[0]].values, dtype=np.float64)
    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    lon = np.asarray(ds["lon"].values, dtype=np.float64)
    data = np.asarray(da.transpose(dep_dim, "lat", "lon").values, dtype=np.float64)
    fill = da.attrs.get("_FillValue", None)
    if fill is not None:
        data = np.where(np.isclose(data, fill), np.nan, data)
    data = np.where(np.abs(data) > 1e19, np.nan, data)
    # wrap longitude to [-180, 180) and sort
    lon_w = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon_w)
    lon_w = lon_w[order]
    data = data[:, :, order]
    if lat[0] > lat[-1]:
        lat = lat[::-1]; data = data[:, ::-1, :]
    print(f"  {path.name}:{var}  nz={depth.size} ({depth.min():.0f}-{depth.max():.0f} m) "
          f"valid {np.isfinite(data).mean()*100:.1f}%  range {np.nanmin(data):.3g}..{np.nanmax(data):.3g}")
    return lat, lon_w, np.abs(depth), data


# ----------------------------------------------------------------------------- fill
def _fill_horizontal_nearest(field2d: np.ndarray, lat: np.ndarray, lon: np.ndarray, xyz_cache: dict):
    """Fill NaNs on one level from the nearest valid cell (unit-sphere kNN, k=1)."""
    valid = np.isfinite(field2d)
    if valid.all():
        return field2d
    if not valid.any():
        return field2d  # whole level empty; handled by vertical fill
    key = "xyz"
    if key not in xyz_cache:
        LON, LAT = np.meshgrid(lon, lat)
        xyz_cache[key] = _xyz(LAT.ravel(), LON.ravel())
    xyz = xyz_cache[key]
    tree = cKDTree(xyz[valid.ravel()])
    _, idx = tree.query(xyz[~valid.ravel()], k=1)
    out = field2d.copy().ravel()
    out[~valid.ravel()] = field2d.ravel()[valid.ravel()][idx]
    return out.reshape(field2d.shape)


def _fill_vertical_hold(data: np.ndarray):
    """Below the deepest valid level in each column hold the last valid value;
    levels that are entirely empty are filled from the level above."""
    nz = data.shape[0]
    out = data.copy()
    for k in range(1, nz):
        bad = ~np.isfinite(out[k])
        out[k][bad] = out[k - 1][bad]
    # top level NaNs (should not happen after horizontal fill): take from below
    for k in range(nz - 2, -1, -1):
        bad = ~np.isfinite(out[k])
        out[k][bad] = out[k + 1][bad]
    return out


# ----------------------------------------------------------------------------- regrid
def _xyz(lat_deg, lon_deg):
    la = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lo = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    c = np.cos(la)
    return np.stack([c * np.cos(lo), c * np.sin(lo), np.sin(la)], axis=-1)


def _horizontal_to_curvilinear(data, lat, lon, tgt_lat2d, tgt_lon2d, k=4):
    """IDW kNN from the regular grid to the target T-points, level by level."""
    LON, LAT = np.meshgrid(lon, lat)
    src = _xyz(LAT.ravel(), LON.ravel())
    tgt = _xyz(tgt_lat2d.ravel(), tgt_lon2d.ravel())
    tree = cKDTree(src)
    d, idx = tree.query(tgt, k=k)
    w = 1.0 / np.maximum(d, 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    nz = data.shape[0]
    out = np.empty((nz,) + tgt_lat2d.shape, dtype=np.float64)
    for kk in range(nz):
        flat = data[kk].ravel()
        out[kk] = (flat[idx] * w).sum(axis=1).reshape(tgt_lat2d.shape)
    return out


def _vertical_to_model(data_zyx, src_depth, tgt_depth):
    """Linear interpolation in depth onto tgt_depth, clamped; data (nz,ny,nx) -> (ny,nx,nt)."""
    nz, ny, nx = data_zyx.shape
    flat = data_zyx.reshape(nz, -1)
    out = np.empty((tgt_depth.size, flat.shape[1]))
    t = np.clip(tgt_depth, src_depth[0], src_depth[-1])
    j = np.searchsorted(src_depth, t, side="right") - 1
    j = np.clip(j, 0, nz - 2)
    f = (t - src_depth[j]) / np.maximum(src_depth[j + 1] - src_depth[j], 1e-9)
    for i in range(tgt_depth.size):
        out[i] = (1.0 - f[i]) * flat[j[i]] + f[i] * flat[j[i] + 1]
    return out.reshape(tgt_depth.size, ny, nx).transpose(1, 2, 0)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--domain", required=True, help="NEMO domain_cfg.nc with e3t_1d (75 levels)")
    ap.add_argument("--glodap", required=True, help="dir with GLODAPv2.2016b.*.nc")
    ap.add_argument("--woa", required=True, help="dir with woa23/{nitrate,phosphate,silicate,oxygen}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    mesh = xr.open_dataset(args.mesh, decode_times=False)
    lat2d = np.squeeze(np.asarray(mesh["nav_lat"].values if "nav_lat" in mesh else mesh["gphit"].values))
    lon2d = np.squeeze(np.asarray(mesh["nav_lon"].values if "nav_lon" in mesh else mesh["glamt"].values))
    tmask = np.squeeze(np.asarray(mesh["tmask"].values))
    land2d = (tmask[0] if tmask.ndim == 3 else tmask) < 0.5
    dom = xr.open_dataset(args.domain, decode_times=False)
    e3t = np.squeeze(np.asarray(dom["e3t_1d"].values, dtype=np.float64))
    z_w = np.cumsum(e3t)
    z_t = z_w - 0.5 * e3t                        # model reference level centres [m, positive down]
    print(f"target grid {lat2d.shape}, {z_t.size} levels ({z_t[0]:.1f}-{z_t[-1]:.0f} m), "
          f"wet T-points {(~land2d).sum()}")

    G = Path(args.glodap); W = Path(args.woa)
    sources = {
        "DIC":         (G / "GLODAPv2.2016b.PI_TCO2.nc", "PI_TCO2"),
        "DIC_present": (G / "GLODAPv2.2016b.TCO2.nc",    "TCO2"),
        "ALK":         (G / "GLODAPv2.2016b.TAlk.nc",    "TAlk"),
        "NO3":         (W / "nitrate"   / "woa23_all_n00_01.nc", "n_an"),
        "PO4":         (W / "phosphate" / "woa23_all_p00_01.nc", "p_an"),
        "Si":          (W / "silicate"  / "woa23_all_i00_01.nc", "i_an"),
        "O2":          (W / "oxygen"    / "woa23_all_o00_01.nc", "o_an"),
    }
    out = {}
    cache = {}
    for name, (path, var) in sources.items():
        if not path.exists():
            print(f"  !! {name}: {path} missing, skipped"); continue
        lat, lon, depth, data = _read_regular(path, var)
        for kk in range(data.shape[0]):
            data[kk] = _fill_horizontal_nearest(data[kk], lat, lon, cache if name.startswith("DIC") or name == "ALK" else {})
        data = _fill_vertical_hold(data)
        assert np.isfinite(data).all(), f"{name}: NaNs remain after fill"
        h = _horizontal_to_curvilinear(data, lat, lon, lat2d, lon2d, k=args.k)
        v = _vertical_to_model(h, depth, z_t) * UMOLKG_TO_MOLM3
        v[land2d] = 0.0
        out[name] = v.astype(np.float64)
        wet = ~land2d
        print(f"  -> {name}: surface wet mean {v[wet][:, 0].mean():.4e}  bottom-level wet mean "
              f"{v[wet][:, -1].mean():.4e} mol/m^3  (min {v[wet].min():.3e}, max {v[wet].max():.3e})")

    out["z_t"] = z_t
    out["_units"] = np.asarray("mol m-3")
    out["_source"] = np.asarray("GLODAPv2.2016b (PI_TCO2, TCO2, TAlk) + WOA23 annual (n,p,i,o _an); rho=1025")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}: {sorted(k for k in out if not k.startswith('_'))}")


if __name__ == "__main__":
    main()
