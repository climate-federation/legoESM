"""Model bathymetry and day-1 T vs raw ETOPO/WOA around the Mindanao hotspot (main, 1/4 deg)."""
import glob
import numpy as np
import xarray as xr

R = "/work/bb1596/b384036/diffESM/legoesm_main_2026-11/results/omip_qdeg_L60_day1/latlon/720x1440/restarts"
f = sorted(glob.glob(f"{R}/restart_day*.npz"))[-1]; print(f)
z = np.load(f)
H, m, T = z["H_bathy"], z["land_mask"] > 0.5, z["T"]
dz = np.loadtxt("/work/bb1596/b384036/diffESM/data/dz_L60_tanh_2m_5500m.txt"); zc = np.cumsum(dz) - dz / 2
lat = -89.875 + 0.25 * np.arange(720); lon = 0.25 * np.arange(1440)
J = lambda la: int(np.argmin(abs(lat - la))); I = lambda lo: int(np.argmin(abs(lon - lo)))

e = xr.open_dataset("/work/bb1596/b384036/diffESM/data/bathy/bathy_025deg_etopo.nc")["z"]
e = e.assign_coords(lon=e.lon % 360).sortby("lon")
lats = np.arange(9.125, 5.6, -0.25); lons = np.arange(120.75, 124.3, 0.25)
print("\nmodel depth (m, . = land) | raw ETOPO elevation (m) -- rows north to south")
print("       " + "".join(f"{lo:7.2f}" for lo in lons))
for la in lats:
    mod = "".join(f"{H[J(la), I(lo)]:7.0f}" if m[J(la), I(lo)] else "      ." for lo in lons)
    raw = "".join(f"{float(e.sel(lat=la, lon=lo, method='nearest')):7.0f}" for lo in lons)
    print(f"{la:6.2f} model {mod}\n{'':6s} raw   {raw}")

w = xr.open_dataset("/work/bb1596/b384036/diffESM/data/woa13v2_thetao_annual_025deg.nc", decode_times=False)["thetao"].isel(time=0)
for name, la, lo in (("hotspot day 2", 7.375, 123.0), ("Zamboanga shelf", 7.375, 122.5), ("Moro Gulf deep", 6.875, 123.5)):
    j, i = J(la), I(lo)
    print(f"\n{name} ({la}N {lo}E): model H {H[j, i] if m[j, i] else 0:.0f} m")
    print("  depth   model T(day1)   WOA T at this point")
    for k in [0, 10, 20, 25, 30, 35, 40]:
        if zc[k] < (H[j, i] if m[j, i] else 0):
            wv = float(w.sel(lev=zc[k], method="nearest").sel(lat=la, lon=lo + 0.125, method="nearest"))
            print(f"  {zc[k]:6.0f}   {T[j, i, k]:8.2f}        {wv:8.2f}")
