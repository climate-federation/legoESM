"""New initial iron for the MARBL initial state (writes a copy of the IC file).

The first IC used 0.2 nM Fe at the surface everywhere, which removes iron limitation in
the HNLC regions (Southern Ocean, equatorial and subarctic Pacific): NPP ~110 PgC/yr in
the first month.  Here surface Fe follows surface nitrate, as observed (GEOTRACES):

    Fe_surf = 0.05 + 0.15 * exp(-NO3_surf / 2)      [nM]   (0.20 nM where NO3 ~ 0, 0.05 nM in HNLC)
    Fe(z)   = Fe_surf + (0.60 - Fe_surf) * 0.5 * (1 + tanh((z - 500) / 200))

All other tracers are copied unchanged.  Units in the file: mmol/m3 (1 nM = 1e-3 mmol/m3).

Usage:
  python scripts/rebuild_marbl_fe_ic.py --ic /work/bb1596/b384036/diffESM/data/marbl_ic_1deg.nc \
      --out /work/bb1596/b384036/diffESM/data/marbl_ic_1deg_fe2.nc
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fe-deep-nM", type=float, default=0.60)
    a = ap.parse_args()
    import xarray as xr

    ds = xr.open_dataset(a.ic).load()
    z = np.asarray(ds["lev"].values, float)                       # positive depth [m]
    no3_s = np.asarray(ds["NO3"].values, float)[..., 0]
    wet = no3_s > 0
    fe_s = 0.05 + 0.15 * np.exp(-np.maximum(no3_s, 0.0) / 2.0)     # nM
    w = 0.5 * (1.0 + np.tanh((z - 500.0) / 200.0))
    fe = (fe_s[..., None] + (a.fe_deep_nM - fe_s[..., None]) * w[None, None, :]) * 1e-3   # mmol/m3
    old = np.asarray(ds["Fe"].values, float)
    land = old == 0
    fe = np.where(land, 0.0, fe)
    ds["Fe"] = (ds["Fe"].dims, fe.astype(ds["Fe"].dtype))
    ds.attrs["description"] = ds.attrs.get("description", "") + "; Fe: NO3-dependent surface value (rebuild_marbl_fe_ic.py)"
    ds.to_netcdf(a.out)
    print(f"surface Fe [nM]: old mean {old[..., 0][wet].mean() * 1e3:.3f}, new mean {fe[..., 0][wet].mean() * 1e3:.3f}, "
          f"new range {fe[..., 0][wet].min() * 1e3:.3f}..{fe[..., 0][wet].max() * 1e3:.3f}")
    for lo, hi in [(-90, -50), (-10, 10), (10, 30), (40, 60)]:
        latd = np.asarray(ds["lat"].values, float)
        b = (latd >= lo) & (latd < hi)
        print(f"  lat {lo:4d}..{hi:3d}: surface NO3 {no3_s[b][wet[b]].mean():6.2f}, Fe {fe[b][..., 0][wet[b]].mean() * 1e3:.3f} nM")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
