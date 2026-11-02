"""Locate the fastest velocities in a legoESM restart and show the local bathymetry."""
import sys
import numpy as np

z = np.load(sys.argv[1])
u, v, H, ocean = z["u"], z["v"], z["H_bathy"], z["land_mask"] > 0.5
nlat, nlon = H.shape
res = 180.0 / nlat
lat_c = -90 + res / 2 + res * np.arange(nlat)          # cell-centre latitudes
lon_c = res * np.arange(nlon)                           # cell-centre longitudes (0, 0.25, ...)

for name, a in (("u", u), ("v", v)):
    s = np.abs(a).max(axis=2)
    for k in np.argsort(s.ravel())[::-1][:8]:
        j, i = np.unravel_index(k, s.shape)
        lev = int(np.argmax(np.abs(a[j, i])))
        jc, ic = min(j, nlat - 1), i % nlon
        lat = lat_c[jc] if name == "u" else -90 + res * j
        lon = lon_c[ic] - res / 2 if name == "u" else lon_c[ic]
        print(f"{name} {s[j, i]:6.2f} m/s  level {lev:2d}  lat {lat:7.2f}  lon {lon:7.2f}  H {H[jc, ic]:7.1f} m")

# depth map (m, '  .' = land) around the fastest u point
s = np.abs(u).max(axis=2)
j, i = np.unravel_index(np.argmax(s), s.shape)
print(f"\nDepth around lat {lat_c[min(j, nlat-1)]:.2f}, lon {lon_c[i % nlon]:.2f} (rows north to south)")
print("        " + "".join(f"{lon_c[(i+di) % nlon]:7.2f}" for di in range(-7, 8)))
for dj in range(7, -8, -1):
    jj = j + dj
    row = "".join(f"{H[jj, (i+di) % nlon]:7.0f}" if ocean[jj, (i+di) % nlon] else "      ." for di in range(-7, 8))
    print(f"{lat_c[jj]:7.2f} {row}")
