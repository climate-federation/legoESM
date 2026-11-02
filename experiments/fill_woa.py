"""Fill WOA land/below-floor gaps: (1) copy deepest valid value downward, (2) nearest valid value on the same level.
usage: python fill_woa.py IN.nc OUT.nc"""
import sys
import numpy as np
import xarray as xr
from scipy import ndimage

src, dst = sys.argv[1], sys.argv[2]
ds = xr.open_dataset(src, decode_times=False)
var = [k for k in ds.data_vars if ds[k].ndim >= 3][0]
da = ds[var]
dims = da.dims
levd = [d for d in dims if d in ("lev", "depth", "z")][0]
latd = [d for d in dims if "lat" in d][0]
lond = [d for d in dims if "lon" in d][0]
order = [d for d in dims if d not in (levd, latd, lond)] + [levd, latd, lond]
a = da.transpose(*order).values.astype("float64")
lead = a.shape[:-3]
a = a.reshape((-1,) + a.shape[-3:])
n0 = int(np.isnan(a).sum())
for k in range(1, a.shape[1]):
    a[:, k] = np.where(np.isnan(a[:, k]), a[:, k - 1], a[:, k])
nlon = a.shape[-1]; W = nlon // 2
for t in range(a.shape[0]):
    for k in range(a.shape[1]):
        f = a[t, k]; bad = np.isnan(f)
        if not bad.any() or bad.all():
            continue
        fp = np.concatenate([f[:, -W:], f, f[:, :W]], axis=1)
        idx = ndimage.distance_transform_edt(np.isnan(fp), return_distances=False, return_indices=True)
        a[t, k] = fp[idx[0], idx[1]][:, W:W + nlon]
print(f"{var}: filled {n0} of {a.size} values; remaining NaN {int(np.isnan(a).sum())}")
out = ds.copy()
out[var] = (order, a.reshape(lead + a.shape[-3:]).astype("float32"), da.attrs)
out[var] = out[var].transpose(*dims)
out.attrs["history"] = ds.attrs.get("history", "") + f"; gaps filled (downward + nearest) by fill_woa.py from {src}"
out.to_netcdf(dst)
print("written", dst)
