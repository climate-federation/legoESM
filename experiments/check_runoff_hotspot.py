"""Is the 1/4-degree Makassar hotspot a river-runoff lens? Profiles at the hotspot + runoff peak, 1/4 vs 1 degree."""
import numpy as np
import xarray as xr

R = "/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/results/omip_qdeg_bt4e11/latlon/720x1440/restart_day000010.npz"
D = "/work/bb1596/b384036/diffESM/data"
z = np.load(R)
u, S, T, eta, ocean = z["u"], z["S"], z["T"], z["eta"], z["land_mask"] > 0.5
lat = -89.875 + 0.25 * np.arange(720); lon = 0.25 * np.arange(1440)

s = np.abs(u).max(axis=2); j, i = np.unravel_index(np.argmax(s), s.shape); i %= 1440
print(f"hotspot lat {lat[j]:.2f} lon {lon[i]:.2f}   eta {eta[j, i]:+.2f} m")
print("level  u[m/s]   S[psu]   T[C]")
for k in range(8):
    print(f"{k:5d} {u[j, i, k]:+7.2f} {S[j, i, k]:8.2f} {T[j, i, k]:7.2f}")

sss = np.where(ocean, S[..., 0], np.nan)
k = np.nanargmin(sss); jm, im = np.unravel_index(k, sss.shape)
print(f"\nglobal min SSS {sss[jm, im]:.2f} psu at lat {lat[jm]:.2f} lon {lon[im]:.2f}")
reg = sss[(lat > -3)[:, None] & (lat < 3)[:, None] & ((lon > 116) & (lon < 121))[None, :]]
print(f"SSS in 3S-3N, 116-121E: min {np.nanmin(reg):.2f}  mean {np.nanmean(reg):.2f} psu")

for tag, f in (("0.25deg", "jra55_ryf9091_0p25deg_noleap.zarr"), ("1deg", "jra55_ryf9091_1deg_noleap.zarr")):
    ds = xr.open_zarr(f"{D}/{f}", decode_times=False)
    fr = ds["friver"].isel(time=slice(0, None, 80)).mean("time").load()
    box = fr.sel(lat=slice(-3, 3), lon=slice(116, 121))
    print(f"\nfriver {tag} ({fr.attrs.get('units', '?')}): global max {float(fr.max()):.2e}, "
          f"Makassar box max {float(box.max()):.2e}")
    m = box.where(box == box.max(), drop=True)
    print(f"  box peak at lat {float(m.lat[0]):.2f} lon {float(m.lon[0]):.2f}")
