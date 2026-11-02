"""Most extreme points in the last finite state before a BLOWUP. usage: python inspect_blowup.py FILE.npz"""
import sys
import numpy as np

z = np.load(sys.argv[1])
H, ocean = z["H_bathy"], z["land_mask"] > 0.5
nlat, nlon = H.shape
r = 180.0 / nlat
latc = -90 + r / 2 + r * np.arange(nlat); latf = -90 + r * np.arange(nlat + 1)
lonc = r * np.arange(nlon);               lonf = r * np.arange(nlon + 1) - r / 2
pos = {"u": (latc, lonf), "v": (latf, lonc), "w": (latc, lonc), "eta": (latc, lonc), "T": (latc, lonc)}


def top(name, a, n=6):
    a = np.asarray(a, dtype="float64")
    fin = np.isfinite(a)
    print(f"\n== {name} shape {a.shape}  non-finite {int((~fin).sum())}  max|.| {np.nanmax(np.abs(a)):.3g}")
    b = np.where(fin, np.abs(a), np.inf)
    la, lo = pos[name]
    for k in np.argsort(b.ravel())[::-1][:n]:
        idx = np.unravel_index(k, a.shape)
        j, i = idx[0], idx[1]
        lev = f" level {idx[2]:2d}" if a.ndim == 3 else ""
        print(f"  {a[idx]:+10.3g} at lat {la[j]:7.2f} lon {lo[i]:7.2f}{lev}  H {H[min(j, nlat-1), i % nlon]:6.0f} m")


for k in ("u", "v", "w", "eta"):
    top(k, z[k])
T = np.where(ocean[..., None] & (np.arange(z["T"].shape[2]) < 35), z["T"], np.nan)
bad = np.isfinite(T) & ((T < -3) | (T > 35))
print(f"\n== T outside [-3, 35] C in ocean cells: {int(bad.sum())}")
if bad.any():
    top("T", np.where(bad, z["T"], 0.0))
