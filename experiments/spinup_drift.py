"""Year-by-year T/S bias vs WOA13v2 0.25deg (volume-weighted, per depth band) from legoESM restarts.
usage: python spinup_drift.py RUN_DIR Y0 Y1 [OUT.txt]"""
import glob
import re
import sys
import numpy as np
import xarray as xr

rdir, y0, y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = sys.argv[4] if len(sys.argv) > 4 else None
D = "/work/bb1596/b384036/diffESM/data"
zz = np.load("/work/bb1596/b384036/simulation_mareike/z_levels.npz")
zc, dz = zz["zc"], zz["dz"]
nlat, nlon, nz = 720, 1440, len(zc)
lat = -89.875 + 0.25 * np.arange(nlat)


def woa(path, var):
    a = xr.open_dataset(path, decode_times=False)[var].isel(time=0)       # (lev, lat, lon), lon 0.125..359.875
    lev = a["lev"].values; a = a.values.astype("float32")
    a = 0.5 * (np.roll(a, 1, axis=2) + a)                                 # -> model lon 0, 0.25, ...
    out = np.empty((nlat, nlon, nz), "float32")
    for k, z in enumerate(zc):                                            # linear in depth
        j = min(np.searchsorted(lev, z), len(lev) - 1); i = max(j - 1, 0)
        w = 0.0 if j == i else (z - lev[i]) / (lev[j] - lev[i])
        out[..., k] = (1 - w) * a[i] + w * a[j]
    return out


Tw = woa(f"{D}/woa13v2_thetao_annual_025deg.nc", "thetao")
Sw = woa(f"{D}/woa13v2_so_annual_025deg.nc", "so")

files = {int(re.search(r"day(\d+)", f).group(1)): f for f in glob.glob(f"{rdir}/restart_day*.npz")}
z0 = np.load(files[min(files)])
H, ocean = z0["H_bathy"], z0["land_mask"] > 0.5
wet = ocean[..., None] & (zc[None, None, :] < H[..., None]) & np.isfinite(Tw) & np.isfinite(Sw)
vol = np.cos(np.radians(lat))[:, None, None] * dz[None, None, :] * wet
bands = [("0-200", 0, 200), ("200-1000", 200, 1000), ("1000-2000", 1000, 2000), (">2000", 2000, 9e9)]
surf = wet[..., 0] & (np.abs(lat)[:, None] <= 60)
aw = np.cos(np.radians(lat))[:, None] * surf

lines = ["year  n |  SST bias  SSS bias (60S-60N) | " +
         " | ".join(f"{b:>9s} T bias/RMSE  S bias" for b, _, _ in bands)]
for y in range(y0, y1 + 1):
    days = [d for d in sorted(files) if 365 * (y - 1) < d <= 365 * y]
    if not days:
        continue
    T = np.zeros((nlat, nlon, nz), "float64"); S = np.zeros_like(T)
    for d in days:
        z = np.load(files[d]); T += z["T"]; S += z["S"]
    T /= len(days); S /= len(days)
    dT, dS = np.where(wet, T - Tw, 0.0), np.where(wet, S - Sw, 0.0)
    sst = (dT[..., 0] * aw).sum() / aw.sum(); sss = (dS[..., 0] * aw).sum() / aw.sum()
    row = f"{y:4d} {len(days):2d} | {sst:+8.2f}  {sss:+8.3f}                | "
    cells = []
    for _, a, b in bands:
        m = ((zc >= a) & (zc < b))[None, None, :]
        v = vol * m; V = v.sum()
        cells.append(f"   {(dT * v).sum() / V:+6.2f} / {np.sqrt((dT**2 * v).sum() / V):5.2f}  {(dS * v).sum() / V:+6.3f}")
    lines.append(row + " | ".join(cells))
    print(lines[-1], flush=True)

txt = "\n".join(lines)
print("\n" + txt)
if out:
    open(out, "w").write(txt + "\n"); print("written", out)
