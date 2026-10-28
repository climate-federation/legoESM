#!/usr/bin/env python
"""Monthly surface deposition forcing for --marbl-forcing on the eORCA1 grid, from the
CESM2 / MARBL input set (POP gx1v6 T grid, 384 x 320):

    dust_flux [kg/m^2/s]      <- dst79gnx_gx1v6_090416.nc : DSTSF [kg/m^2/s]
    iron_flux [mmol Fe/m^2/s] <- solFe_scenario4_current_gx1v6_8gmol_cesm1_93_20161114.nc : DSTSF
                                 (dust flux pre-weighted by the Luo et al. solubility map; POP's
                                 iron_flux scale_factor 6.2668e4 nmol/cm^2/s per kg/m^2/s =
                                 3.5 % Fe by mass -> 0.62668 mol Fe / kg -> x 626.68 mmol/kg)
    nox_flux, nhy_flux [mmol N/m^2/s] <- ndep_ocn_1850_w_nhx_emis_gx1v6_c180803.nc :
                                 NOy_deposition, NHx_deposition [kg N/m^2/s]  (x 1e6 / 14.007)

Regridding: unit-sphere kNN inverse-distance (k=4) from POP ocean T-cells to eORCA1 T-points
(same approach as build_bgc_init_eorca1.py), land set to 0, then each field is rescaled by a
single global factor so that its area-integrated annual deposition equals the source's
(IDW is not conservative; the factor is printed and should be within a few % of 1).

Not included (need an interior-forcing hook in the adapter; Step 4): sediment Fe flux
(fesedflux_total_reduce_oxic_POP_gx1v7.c200618.nc, 3-D), hydrothermal Fe (feventflux), rivers
(river_nutrients_GNEWS2000_gx1v6_20170226.nc).

Usage (login node):
    /work/bb1596/b384036/b384036/envs/legoesm_main/bin/python \
        /work/bb1596/b384036/diffESM/legoesm_main_fresh/scripts/data/build_marbl_forcing_eorca1.py \
        --mesh /work/bb1596/b384036/diffESM/legoesm_main_fresh/data/grids/eORCA1.2_mesh_mask.nc \
        --src  /work/bb1596/b384036/data/bgc_obs/cesm_forcing \
        --out  /work/bb1596/b384036/diffESM/data/bgc_init/marbl_forcing_eorca1.nc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

SEC_PER_YEAR = 365.0 * 86400.0
M_FE_FRAC = 0.035                 # Fe mass fraction of dust (POP/MARBL convention)
M_FE = 55.845e-3                  # kg/mol
M_N = 14.007e-3                   # kg/mol

SOURCES = {
    # name: (file, variable, factor to the port's units, label of the global integral)
    "dust_flux": ("dst79gnx_gx1v6_090416.nc", "DSTSF", 1.0, "Tg dust/yr", 1e-9),
    "iron_flux": ("solFe_scenario4_current_gx1v6_8gmol_cesm1_93_20161114.nc", "DSTSF",
                  1.0 / M_FE * 1e3, "Gmol Fe/yr", 1e-3 * 1e-9),   # DSTSF here = soluble Fe mass flux [kg Fe/m^2/s]
    "nox_flux":  ("ndep_ocn_1850_w_nhx_emis_gx1v6_c180803.nc", "NOy_deposition", 1e3 / M_N, "Tg N/yr",
                  M_N * 1e-3 * 1e-9),
    "nhy_flux":  ("ndep_ocn_1850_w_nhx_emis_gx1v6_c180803.nc", "NHx_deposition", 1e3 / M_N, "Tg N/yr",
                  M_N * 1e-3 * 1e-9),
}


def _xyz(lat, lon):
    la, lo = np.deg2rad(np.asarray(lat, np.float64)), np.deg2rad(np.asarray(lon, np.float64))
    c = np.cos(la)
    return np.stack([c * np.cos(lo), c * np.sin(lo), np.sin(la)], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--src", required=True, help="directory with the CESM forcing files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    mesh = xr.open_dataset(args.mesh, decode_times=False)
    lat2d = np.squeeze(np.asarray(mesh["nav_lat"].values if "nav_lat" in mesh else mesh["gphit"].values))
    lon2d = np.squeeze(np.asarray(mesh["nav_lon"].values if "nav_lon" in mesh else mesh["glamt"].values))
    tmask = np.squeeze(np.asarray(mesh["tmask"].values))
    wet = (tmask[0] if tmask.ndim == 3 else tmask) > 0.5
    e1t = np.squeeze(np.asarray(mesh["e1t"].values)); e2t = np.squeeze(np.asarray(mesh["e2t"].values))
    area = e1t * e2t
    print(f"target grid {lat2d.shape}, wet T-points {wet.sum()}, ocean area {area[wet].sum()*1e-12:.2f} e6 km^2")
    tgt = _xyz(lat2d.ravel(), lon2d.ravel())

    out = {}
    src_dir = Path(args.src)
    tree_cache = {}
    for name, (fname, var, fac, label, to_label) in SOURCES.items():
        ds = xr.open_dataset(src_dir / fname, decode_times=False)
        da = ds[var]
        a = np.asarray(da.values, np.float64)                       # (12, Y, X)
        a = np.where(np.abs(a) > 1e30, np.nan, a)
        tlat = np.asarray(ds["TLAT"].values, np.float64); tlon = np.asarray(ds["TLONG"].values, np.float64)
        # source ocean cells: finite in all months (land is missing or 0 on the POP mask)
        valid = np.isfinite(a).all(0)
        if "KMT" in ds:
            valid &= np.asarray(ds["KMT"].values) > 0
        if "TAREA" in ds:
            sarea = np.asarray(ds["TAREA"].values, np.float64) * 1e-4   # cm^2 -> m^2
        else:
            sarea = tree_cache.get("tarea")
            if sarea is None:
                sarea = np.asarray(xr.open_dataset(src_dir / SOURCES["dust_flux"][0],
                                                   decode_times=False)["TAREA"].values, np.float64) * 1e-4
        tree_cache["tarea"] = sarea
        key = (fname, "tree")
        if key not in tree_cache:
            tree_cache[key] = (cKDTree(_xyz(tlat[valid], tlon[valid])), valid)
        tree, valid = tree_cache[key]
        d, idx = tree.query(tgt, k=args.k)
        w = 1.0 / np.maximum(d, 1e-12); w /= w.sum(1, keepdims=True)
        fld = np.empty((12,) + lat2d.shape)
        for m in range(12):
            flat = a[m][valid]
            fld[m] = (flat[idx] * w).sum(1).reshape(lat2d.shape)
        fld = np.where(wet[None], fld, 0.0) * fac
        # conservation: match the source's annual, area-integrated deposition over its ocean cells
        src_tot = float((np.nan_to_num(a[:, valid]).mean(0) * sarea[valid]).sum()) * fac * SEC_PER_YEAR
        tgt_tot = float((fld.mean(0) * area)[wet].sum()) * SEC_PER_YEAR
        scale = src_tot / tgt_tot if tgt_tot > 0 else 1.0
        fld *= scale
        out[name] = fld
        print(f"  {name:10s} <- {fname}:{var}  source total {src_tot*to_label:9.3f} {label}; "
              f"IDW total {tgt_tot*to_label:9.3f} -> rescaled x{scale:.4f};  "
              f"range {fld.min():.3e}..{fld.max():.3e}, annual mean over wet {fld.mean(0)[wet].mean():.3e}")

    ds_out = xr.Dataset(
        {k: (("time", "y", "x"), v.astype(np.float32)) for k, v in out.items()},
        coords={"time": np.arange(12), "nav_lat": (("y", "x"), lat2d.astype(np.float32)),
                "nav_lon": (("y", "x"), lon2d.astype(np.float32))},
        attrs={"source": "CESM2/MARBL gx1v6 forcing (ftp.cgd.ucar.edu/cesm/inputdata/ocn/pop/gx1v6/forcing): "
                         "dst79gnx (dust), solFe_scenario4 8gmol (soluble Fe), ndep_ocn_1850_w_nhx_emis (N dep)",
               "units": "dust_flux kg/m2/s; iron_flux mmol Fe/m2/s; nox_flux, nhy_flux mmol N/m2/s",
               "regrid": "unit-sphere kNN IDW k=4 from POP T cells, land=0, rescaled to the source global total",
               "months": "0..11 = Jan..Dec climatology"})
    ds_out["dust_flux"].attrs["units"] = "kg m-2 s-1"
    ds_out["iron_flux"].attrs["units"] = "mmol m-2 s-1"
    ds_out["nox_flux"].attrs["units"] = "mmol m-2 s-1"; ds_out["nhy_flux"].attrs["units"] = "mmol m-2 s-1"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(args.out)
    print(f"wrote {args.out}: {list(out)}  shape (12,) + {lat2d.shape}")


if __name__ == "__main__":
    main()
