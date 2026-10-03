#!/usr/bin/env python3
"""Global land effective momentum roughness from legoESM's surfdata + PFT tables.

Per (cell, PFT): canopy z0 = rz0m*htop, d = rd*htop, with the sparse-canopy
log blend toward the bare-ground 0.01 m (CLM5 FrictionVelocity/CanopyFluxes
form, egvf = (1-exp(-lt))/(1-exp(-2)), lt = min(LAI[+SAI], 2)); bare soil and
glacier z0 = 0.01 m, d = 0.  Weights = cos(lat) * (f_land*pft_frac or
f_glacier); lakes excluded.  Prints, per season and per LAI-only / LAI+SAI:
(1) neutral-drag effective z0 at a blending height z_b (Cd summed, d averaged),
(2) geometric mean of z0, (3) arithmetic mean of z0.

    land_effective_z0.py [--zb 50]
"""
import argparse

import numpy as np
import xarray as xr

from legoesm import constants
from legoesm.land.boundary_data._internals import pft_lookup_arrays, RZ0M_BARE

SURF = "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/data/legoesm_surfdata_c260716.nc"
Z0_BARE = 0.01            # m: CLM5 zlnd / legoESM _Z0MG_BARE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zb", type=float, default=50.0)
    a = ap.parse_args()
    k = constants.kappa_vk
    d = xr.open_dataset(SURF)
    lut = pft_lookup_arrays()
    rz0m = lut["rz0m"][:, None, None]
    rd = lut["rd"][:, None, None]
    veg = lut["is_veg"][:, None, None] > 0
    w_cell = np.cos(np.deg2rad(d.lat.values))[:, None]
    f_land = d.f_land.isel(year=0).values / 100.0
    f_gla = d.f_glacier.isel(year=0).values / 100.0
    pf = np.nan_to_num(d.pft_frac.isel(year=0).values / 100.0)
    for months, name in ((range(12), "annual"), ((11, 0, 1), "DJF"), ((5, 6, 7), "JJA")):
        for use_sai in (False, True):
            acc = {"cd": 0.0, "dw": 0.0, "lnz": 0.0, "z": 0.0, "w": 0.0}
            for m in months:
                lai = np.nan_to_num(d.monthly_lai.isel(month=m).values)
                sai = np.nan_to_num(d.monthly_sai.isel(month=m).values)
                hc = np.nan_to_num(d.monthly_height_top.isel(month=m).values)
                lt = np.clip(lai + (sai if use_sai else 0.0), 0.0, 2.0)
                egvf = (1 - np.exp(-lt)) / (1 - np.exp(-2.0))
                zc = np.maximum(hc * np.where(veg, rz0m, RZ0M_BARE), Z0_BARE)
                z0 = np.where(veg, np.exp(egvf * np.log(zc) + (1 - egvf) * np.log(Z0_BARE)), Z0_BARE)
                dd = np.where(veg, hc * rd * egvf, 0.0)
                w = w_cell[None] * f_land[None] * pf                     # (pft, lat, lon)
                arg = np.maximum((a.zb - dd) / z0, np.e ** 2)
                cd = (k / np.log(arg)) ** 2
                wg = w_cell * f_gla
                cdg = (k / np.log(a.zb / Z0_BARE)) ** 2
                acc["cd"] += (w * cd).sum() + (wg * cdg).sum()
                acc["dw"] += (w * dd).sum()
                acc["lnz"] += (w * np.log(z0)).sum() + (wg * np.log(Z0_BARE)).sum()
                acc["z"] += (w * z0).sum() + (wg * Z0_BARE).sum()
                acc["w"] += w.sum() + wg.sum()
            W = acc["w"]
            cd_eff, d_eff = acc["cd"] / W, acc["dw"] / W
            z0_eff = (a.zb - d_eff) * np.exp(-k / np.sqrt(cd_eff))
            print(f"{name:6s} {'LAI+SAI' if use_sai else 'LAI    '} z_b={a.zb:.0f} m: "
                  f"drag-effective z0={z0_eff:.3f} m (d_eff {d_eff:.2f} m, Cd {cd_eff:.4f}) | "
                  f"geometric mean {np.exp(acc['lnz'] / W):.3f} m | arithmetic mean {acc['z'] / W:.3f} m")


if __name__ == "__main__":
    main()
