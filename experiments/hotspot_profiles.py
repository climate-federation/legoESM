"""u,v,T profiles at the fastest points of a restart; usage: python hotspot_profiles.py FILE.npz"""
import sys
import numpy as np

z = np.load(sys.argv[1])
u, v, T, H = z["u"], z["v"], z["T"], z["H_bathy"]
uc = 0.5 * (u[:, :-1] + u[:, 1:]); vc = 0.5 * (v[:-1] + v[1:])
spd = np.sqrt(uc**2 + vc**2)
lat = -89.875 + 0.25 * np.arange(720); lon = 0.25 * np.arange(1440)
surf = spd[..., 0].copy()
for n in range(2):                                   # two separate hotspots
    j, i = np.unravel_index(np.nanargmax(surf), surf.shape)
    print(f"\nhotspot {n+1}: lat {lat[j]:.2f} lon {lon[i]:.2f}  H {H[j, i]:.0f} m")
    print("level   speed     u       v      T")
    for k in range(8):
        print(f"{k:5d} {spd[j, i, k]:7.2f} {uc[j, i, k]:+7.2f} {vc[j, i, k]:+7.2f} {T[j, i, k]:6.2f}")
    surf[max(0, j-20):j+21, max(0, i-20):i+21] = 0     # mask 5 deg around it
