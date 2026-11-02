"""Do WOA (1 deg) gaps under the 1/4-degree ocean get filled with level means? Check at the hotspots."""
import numpy as np
import xarray as xr

T = xr.open_dataset("/work/bb1596/b384036/diffESM/data/woa13v2_thetao_annual_degC.nc", decode_times=False)
print(T)
v = [k for k in T.data_vars if T[k].ndim >= 3][0]
t = T[v].squeeze()
latn = [d for d in t.dims if "lat" in d][0]; lonn = [d for d in t.dims if "lon" in d][0]
t = t.assign_coords({lonn: t[lonn] % 360}).sortby(lonn)
for name, la, lo in (("Makassar N", 0.6, 118.25), ("Sulu/Basilan", 7.3, 122.0), ("Makassar S", -2.0, 118.5)):
    p = t.sel({latn: la, lonn: lo}, method="nearest")
    print(f"\n{name}: WOA surface value {float(p.isel({p.dims[0]: 0})):.2f}  (nan = land in WOA)")
    near = t.isel({t.dims[0]: 0}).sel({latn: slice(la - 2, la + 2), lonn: slice(lo - 2, lo + 2)})
    print(np.round(near.values, 1))

z = np.load("/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/results/omip_qdeg_bt4e11/latlon/720x1440/restart_day000005.npz")
ocean = z["land_mask"] > 0.5
lat = -89.875 + 0.25 * np.arange(720); lon = 0.25 * np.arange(1440)
w0 = t.isel({t.dims[0]: 0}).interp({latn: lat, lonn: lon}, method="nearest").values
gap = ocean & ~np.isfinite(w0)
print(f"\n1/4-deg ocean cells with no WOA surface value: {gap.sum()} of {ocean.sum()}")
jj, ii = np.where(gap)
print("examples (lat, lon):", [(float(lat[a]), float(lon[b])) for a, b in list(zip(jj, ii))[::max(1, len(jj)//10)]])
