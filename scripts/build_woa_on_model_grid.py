"""WOA observations on the legoESM 1-degree model grid (reference data set for the MARBL evaluation).

Reads the CMORised WOA climatologies of the ESMValTool OBS tree, averages them in time
(annual mean), interpolates them to the model grid (horizontal: linear, periodic in
longitude; vertical: linear to the model layer centres) and fills model ocean cells that
WOA does not cover (coasts, marginal seas, cells near the sea floor) from the nearest WOA
value at the same depth.  Cells that are not ocean in the model are NaN.

Output (one netCDF file, dims lat, lon, lev; lev = depth of the layer centres, m):
  NO3, PO4, SiO3, O2   mmol m-3      WOA 2018 annual (Oyr)
  T, S                 degC, psu     WOA 2013v2 monthly (Omon) -> annual mean
  wet                  1 = ocean cell of the model (land mask and depth of the sea floor)
  filled               1 = model ocean cell with no WOA value (filled from the nearest one)

With --ic, the MARBL initial state is compared with WOA (surface by latitude band,
volume means), to check the initial nutrients and oxygen.

Usage:
  python scripts/build_woa_on_model_grid.py \
     --grid /work/bb1596/b384036/diffESM/data/marbl_grid_1deg_mc.npz \
     --restart <any restart_dayNNNNNN.npz of the run (for the land mask and sea-floor depth)> \
     --out /work/bb1596/b384036/diffESM/data/woa_on_model_grid_1deg_mc.nc \
     --ic /work/bb1596/b384036/diffESM/data/marbl_ic_1deg_fe2_mc.nc
"""
import argparse
from pathlib import Path

import numpy as np

WOA = "/work/bd0854/DATA/ESMValTool2/OBS/Tier2/WOA"
SOURCES = {  # output name: (file, CMOR variable, kind)
    "NO3": (f"{WOA}/OBS6_WOA_clim_2018_Oyr_no3_200001-200012.nc", "no3", "conc"),
    "PO4": (f"{WOA}/OBS6_WOA_clim_2018_Oyr_po4_200001-200012.nc", "po4", "conc"),
    "SiO3": (f"{WOA}/OBS6_WOA_clim_2018_Oyr_si_200001-200012.nc", "si", "conc"),
    "O2": (f"{WOA}/OBS6_WOA_clim_2018_Oyr_o2_200001-200012.nc", "o2", "conc"),
    "T": (f"{WOA}/OBS_WOA_clim_2013v2_Omon_thetao_200001-200012.nc", "thetao", "temp"),
    "S": (f"{WOA}/OBS_WOA_clim_2013v2_Omon_so_200001-200012.nc", "so", "salt"),
}


def load_woa(path, var, latd, lond, z):
    """Annual mean of a CMOR WOA field on the model grid, (nlat, nlon, nlev); units as in the file."""
    import xarray as xr
    from scipy.interpolate import interp1d
    da = xr.open_dataset(path, decode_times=False)[var]
    units = da.attrs.get("units", "")
    tdim = [d for d in da.dims if d.lower().startswith("time")]
    if tdim:
        da = da.mean(tdim[0])
    latn = next(d for d in da.dims if d.lower().startswith("lat"))
    lonn = next(d for d in da.dims if d.lower().startswith("lon"))
    zn = next(d for d in da.dims if d not in (latn, lonn))
    da = da.transpose(zn, latn, lonn).sortby(latn)
    da = da.assign_coords({lonn: np.mod(da[lonn].values, 360.0)}).sortby(lonn)
    first, last = da.isel({lonn: [0]}), da.isel({lonn: [-1]})
    da = xr.concat([last.assign_coords({lonn: last[lonn].values - 360.0}), da,
                    first.assign_coords({lonn: first[lonn].values + 360.0})], dim=lonn)
    lq = np.mod(lond, 360.0); o = np.argsort(lq)
    hor = da.interp({latn: latd, lonn: lq[o]}).values[:, :, np.argsort(o)]       # (nz_woa, nlat, nlon)
    zw = np.abs(np.asarray(da[zn].values, float))
    if zw.max() > 1e5:                                                            # cm -> m
        zw = zw / 100.0
    out = interp1d(zw, hor, axis=0, bounds_error=False, fill_value=np.nan)(z)    # (nlev, nlat, nlon)
    return np.moveaxis(out, 0, -1), units, (float(zw.min()), float(zw.max()), zw.size)


def to_units(kind, a, units):
    u = units.replace(" ", "").lower()
    if kind == "conc":                                   # -> mmol m-3
        if ("molm-3" in u or "mol/m3" in u or "molm^-3" in u) and not any(p in u for p in ("mmol", "umol", "µmol")):
            return a * 1e3
        if "umol/kg" in u or "µmol/kg" in u or "umolkg-1" in u:
            return a * 1.025
        return a * 1e3 if np.nanmedian(a) < 0.1 else a
    if kind == "temp" and np.nanmedian(a) > 100:         # K -> degC
        return a - 273.15
    return a


def fill_nearest(a, wet):
    """Fill wet cells without a value from the nearest valid cell at the same level."""
    from scipy import ndimage
    out = a.copy(); filled = np.zeros(a.shape, bool)
    for k in range(a.shape[-1]):
        v = np.isfinite(a[..., k])
        need = wet[..., k] & ~v
        if not need.any():
            continue
        if v.any():
            idx = ndimage.distance_transform_edt(~v, return_distances=False, return_indices=True)
            out[..., k] = np.where(need, a[..., k][tuple(idx)], a[..., k])
        else:
            out[..., k] = np.where(need, out[..., k - 1], a[..., k])
        filled[..., k] = need
    return np.where(wet, out, np.nan), filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--restart", required=True, help="model restart, for land mask and sea-floor depth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ic", default=None, help="MARBL initial state to compare with WOA")
    a = ap.parse_args()
    import xarray as xr

    g = np.load(a.grid)
    lat, lon = np.asarray(g["lat"], float), np.asarray(g["lon"], float)
    if np.abs(lat).max() <= np.pi / 2 + 1e-6:
        lat, lon = np.rad2deg(lat), np.rad2deg(lon)
    z, zh = np.abs(np.asarray(g["z"], float)), np.abs(np.asarray(g["z_half"], float))
    r = np.load(a.restart)
    mask, H = r["land_mask"] > 0.5, r["H_bathy"]
    wet = mask[..., None] & (H[..., None] > zh[:-1][None, None, :] + 1e-6)
    dz = np.diff(zh)
    area = (np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size)))
    vol = area[..., None] * dz[None, None, :] * wet

    data, filled_any = {}, np.zeros(wet.shape, bool)
    for name, (path, var, kind) in SOURCES.items():
        if not Path(path).exists():
            print(f"{name}: MISSING {path}"); continue
        f, units, (z0, z1, nz) = load_woa(path, var, lat, lon, z)
        f = to_units(kind, f, units)
        f, filled = fill_nearest(f, wet)
        filled_any |= filled
        data[name] = f
        s = f[..., 0][wet[..., 0]]
        print(f"{name:4s} <- {Path(path).name} [{units}], WOA depths {z0:.0f}-{z1:.0f} m ({nz} levels): "
              f"surface mean {np.nanmean(s):8.3f}, range {np.nanmin(s):.3g}..{np.nanmax(s):.3g}, "
              f"volume mean {np.nansum(f * vol) / np.sum(vol):8.3f}, filled {100 * filled.sum() / wet.sum():.2f} % of ocean cells")

    ds = xr.Dataset({k: (("lat", "lon", "lev"), v.astype(np.float32)) for k, v in data.items()},
                    coords={"lat": lat, "lon": lon, "lev": z})
    ds["wet"] = (("lat", "lon", "lev"), wet.astype(np.int8))
    ds["filled"] = (("lat", "lon", "lev"), filled_any.astype(np.int8))
    ds.attrs.update(description="WOA climatologies on the legoESM 1-degree grid (annual means); "
                                "NO3/PO4/SiO3/O2 from WOA 2018 Oyr [mmol m-3], T/S from WOA 2013v2 Omon [degC, psu]; "
                                "model ocean cells without WOA data filled from the nearest value at the same depth "
                                "(build_woa_on_model_grid.py)",
                    grid=str(a.grid), mask_from=str(a.restart))
    ds.to_netcdf(a.out)
    print(f"wrote {a.out}")

    if a.ic:
        ic = xr.open_dataset(a.ic)
        bands = [(-90, -60), (-60, -45), (-45, -30), (-30, -10), (-10, 10), (10, 30), (30, 50), (50, 90)]
        print("\nMARBL initial state vs WOA, surface mean by latitude band (IC / WOA):")
        print("  band       " + "".join(f"{k:>18s}" for k in ("NO3", "PO4", "SiO3", "O2") if k in data))
        for lo, hi in bands:
            b = (lat >= lo) & (lat < hi)
            row = ""
            for k in ("NO3", "PO4", "SiO3", "O2"):
                if k not in data or k not in ic:
                    continue
                m = wet[b][:, :, 0]
                icv = np.asarray(ic[k].values, float)[b][..., 0][m]
                row += f"{np.nanmean(icv):8.2f} /{np.nanmean(data[k][b][..., 0][m]):7.2f} "
            print(f"  {lo:4d}..{hi:3d}  {row}")
        print("volume mean (IC / WOA):", ", ".join(
            f"{k} {np.nansum(np.where(wet, np.asarray(ic[k].values, float), 0) * vol) / np.sum(vol):.2f} / "
            f"{np.nansum(np.nan_to_num(data[k]) * vol) / np.sum(vol):.2f}" for k in ("NO3", "PO4", "SiO3", "O2")
            if k in data and k in ic))
        # sharp latitude steps in the IC surface fields (a seam shows up as a jump between neighbouring rows)
        for k in ("NO3", "Fe"):
            if k in ic:
                zm = np.nanmean(np.where(wet[..., 0], np.asarray(ic[k].values, float)[..., 0], np.nan), axis=1)
                j = np.nanargmax(np.abs(np.diff(zm)))
                print(f"IC {k}: largest jump in the zonal-mean surface value between rows "
                      f"{lat[j]:.1f} and {lat[j + 1]:.1f}: {zm[j]:.4g} -> {zm[j + 1]:.4g}")


if __name__ == "__main__":
    main()
