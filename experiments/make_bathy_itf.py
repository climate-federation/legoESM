"""Copy bathy_1deg_etopo.nc and impose realistic Indonesian Throughflow geometry."""
import numpy as np, xarray as xr
SRC = "/work/bb1596/b384036/diffESM/data/bathy/bathy_1deg_etopo.nc"
DST = "/work/bb1596/b384036/diffESM/data/bathy/bathy_1deg_etopo_itf.nc"
ds = xr.open_dataset(SRC); z = ds["z"].values.copy()           # elevation, positive up
lat = ds["lat"].values; lon = ds["lon"].values % 360
J = lambda la: int(np.argmin(np.abs(lat - la))); I = lambda lo: int(np.argmin(np.abs(lon - (lo % 360))))
def show(tag):
    print(f"\n{tag}: depth [m] (0 = land), lat -12.5..-1.5, lon 112.5..131.5E")
    print("lat    " + " ".join(f"{lo:5.0f}" for lo in np.arange(112.5, 132.5, 1.0)))
    for la in np.arange(-1.5, -13.5, -1.0):
        print(f"{la:5.1f} " + " ".join(f"{max(0, int(-z[J(la), I(lo)])) if np.isfinite(z[J(la), I(lo)]) else 0:5d}"
                                      for lo in np.arange(112.5, 132.5, 1.0)))
show("BEFORE")
LAND = 10.0
for lo in np.arange(113.5, 123.5, 1.0): z[J(-8.5), I(lo)] = LAND          # Bali..Flores, closes Lombok
for lo in (124.5, 125.5, 126.5):        z[J(-9.5), I(lo)] = LAND          # Timor
for lo in np.arange(123.5, 127.5, 1.0): z[J(-8.5), I(lo)] = max(z[J(-8.5), I(lo)], -1500.0)  # Ombai/Wetar
for lo in np.arange(124.5, 127.5, 1.0): z[J(-10.5), I(lo)] = max(z[J(-10.5), I(lo)], -1900.0) # Timor Passage
for la in (-3.5, -4.5, -5.5):
    for lo in np.arange(116.5, 119.5, 1.0): z[J(la), I(lo)] = max(z[J(la), I(lo)], -700.0)    # Makassar/Dewakang sill
show("AFTER")
out = ds.copy(); out["z"].values[:] = z
out.attrs["history"] = "ITF fix: Lesser Sunda barrier, Timor land, Ombai<=1500 m, Timor Passage<=1900 m, Makassar sill<=700 m"
out.to_netcdf(DST); print("\nwritten", DST)
