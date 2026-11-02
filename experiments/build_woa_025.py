"""Model-ready 1/4-degree WOA13v2 in the layout of the 1-degree ESMValTool files.
thetao: annual potential temperature (TEOS-10), 102 levels; so: monthly salinity to 1500 m, annual below."""
import numpy as np
import xarray as xr
import gsw

RAW = "/work/bb1596/b384036/diffESM/data/woa13v2_025deg_raw"
D = "/work/bb1596/b384036/diffESM/data"
TT = xr.open_dataset(f"{D}/woa13v2_thetao_annual_degC.nc", decode_times=False)
TS = xr.open_dataset("/work/bd0854/DATA/ESMValTool2/OBS/Tier2/WOA/OBS_WOA_clim_2013v2_Omon_so_200001-200012.nc",
                     decode_times=False)
print("template  T:", TT["thetao"].dims, TT["thetao"].shape, "  S:", TS["so"].dims, TS["so"].shape)


def raw(v, m):
    ds = xr.open_dataset(f"{RAW}/woa13_decav_{v}{m}_04v2.nc", decode_times=False)
    a = ds[f"{v}_an"].isel(time=0)
    return a.assign_coords(lon=a.lon % 360).sortby("lon")


t00, s00 = raw("t", "00"), raw("s", "00")
depth, lat, lon = t00.depth.values, t00.lat.values, t00.lon.values
LON, LAT = np.meshgrid(lon, lat)
print(f"raw: {len(depth)} levels to {depth[-1]:.0f} m, {len(lat)} x {len(lon)}, lon {lon[0]}..{lon[-1]}")

pt = np.full(t00.shape, np.nan, np.float32)
for k, z in enumerate(depth):
    p = gsw.p_from_z(-z, LAT)
    SA = gsw.SA_from_SP(s00.values[k], p, LON, LAT)
    pt[k] = gsw.pt0_from_t(SA, t00.values[k], p)
print("potential temperature done")

nm = raw("s", "01").sizes["depth"]
so = np.empty((12, len(depth), len(lat), len(lon)), np.float32)
so[:, nm:] = s00.values[nm:][None]
for m in range(12):
    sm = raw("s", f"{m + 1:02d}")
    assert np.allclose(sm.depth.values, depth[:nm])
    so[m, :nm] = sm.values
print(f"salinity: {nm} monthly levels + {len(depth) - nm} annual levels")


def coords(time):
    return {"time": time, "lev": ("lev", depth, TT["lev"].attrs),
            "lat": ("lat", lat, TT["lat"].attrs), "lon": ("lon", lon, TT["lon"].attrs)}


def bnds(ds, tpl):
    ds["time_bnds"] = tpl["time_bnds"]
    mid = 0.5 * (depth[1:] + depth[:-1])
    ds["lev_bnds"] = (("lev", "bnds"), np.stack([np.r_[0, mid], np.r_[mid, depth[-1]]], 1))
    ds["lat_bnds"] = (("lat", "bnds"), np.stack([lat - 0.125, lat + 0.125], 1))
    ds["lon_bnds"] = (("lon", "bnds"), np.stack([lon - 0.125, lon + 0.125], 1))
    return ds


T = bnds(xr.Dataset({"thetao": (("time", "lev", "lat", "lon"), pt[None], TT["thetao"].attrs)},
                    coords=coords(TT["time"])), TT)
T.attrs = dict(TT.attrs, title="WOA13v2 0.25deg decav, potential temperature (TEOS-10) for legoESM")
S = bnds(xr.Dataset({"so": (("time", "lev", "lat", "lon"), s00.values[None].astype("float32"), TS["so"].attrs)}, coords=coords(TS["time"])), TS)
S.attrs = dict(TS.attrs, title="WOA13v2 0.25deg decav, annual salinity for legoESM")
T.to_netcdf(f"{D}/woa13v2_thetao_annual_025deg.nc")
S.to_netcdf(f"{D}/woa13v2_so_annual_025deg.nc")
print("written", f"{D}/woa13v2_thetao_annual_025deg.nc", f"{D}/woa13v2_so_annual_025deg.nc")

for name, la, lo in (("Makassar N", 0.6, 118.25), ("Sulu/Basilan", 7.3, 122.0)):
    p = T["thetao"].isel(time=0).sel(lat=la, lon=lo, method="nearest")
    s0 = float(S["so"].isel(time=0, lev=0).sel(lat=la, lon=lo, method="nearest"))
    print(f"{name}: theta 0 m {float(p.isel(lev=0)):.2f}  100 m {float(p.sel(lev=100)):.2f}  S0 {s0:.2f}")
z = np.load("/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/results/omip_qdeg_bt4e11/latlon/720x1440/restart_day000005.npz")
ocean = z["land_mask"] > 0.5
w0 = T["thetao"].isel(time=0, lev=0).sel(lat=-89.875 + 0.25 * np.arange(720),
                                          lon=(0.25 * np.arange(1440) + 0.125) % 360, method="nearest").values
print(f"1/4-deg ocean cells with no WOA-0.25 surface value: {int((ocean & ~np.isfinite(w0)).sum())} "
      f"of {int(ocean.sum())}  (1-deg WOA: 26242)")
