"""Put the MARBL initial state on a new land mask / bathymetry.
Cells that are ocean in the model (land_mask & H > layer top) but empty in the IC are filled
from the nearest valid cell at the same level; cells that are not ocean are set to 0.
Validity is taken from DIC (> 0 in every real ocean cell).  Also writes a grid npz with the new mask/H.
Usage: python fill_marbl_ic_newmask.py IC_IN RESTART GRID_IN IC_OUT GRID_OUT"""
import sys
import numpy as np
import xarray as xr
from scipy import ndimage

ic_in, rst, grid_in, ic_out, grid_out = sys.argv[1:6]
r = np.load(rst); g = dict(np.load(grid_in))
zh = np.abs(np.asarray(g["z_half"], float))
mask = r["land_mask"] > 0.5; H = r["H_bathy"]
wet = mask[..., None] & (H[..., None] > zh[:-1][None, None, :] + 1e-6)

ds = xr.open_dataset(ic_in).load()
ref = np.asarray(ds["DIC"].values, float)
valid = np.isfinite(ref) & (ref > 1.0)
need = wet & ~valid
print(f"model ocean cells {int(wet.sum())}, valid in IC {int((wet & valid).sum())}, "
      f"to fill {int(need.sum())} ({100 * need.sum() / wet.sum():.2f} %), IC cells now land/below floor {int((valid & ~wet).sum())}")
print("cells to fill per level:", need.sum((0, 1)).tolist())
idx = []
for k in range(wet.shape[-1]):
    v = valid[..., k]
    idx.append(None if not v.any() else ndimage.distance_transform_edt(~v, return_distances=False, return_indices=True))
for name in ds.data_vars:
    a = np.asarray(ds[name].values, float)
    if a.shape != wet.shape:
        continue
    out = a.copy()
    for k in range(wet.shape[-1]):
        if idx[k] is None:                       # no valid cell at this level: copy the level above
            out[..., k] = out[..., k - 1]
        else:
            filled = a[..., k][tuple(idx[k])]
            out[..., k] = np.where(need[..., k], filled, a[..., k])
    out = np.where(wet, out, 0.0)
    ds[name] = (ds[name].dims, out.astype(ds[name].dtype))
ds.attrs["description"] = ds.attrs.get("description", "") + "; filled onto the bathy_1deg_etopo_mc mask (fill_marbl_ic_newmask.py)"
ds.to_netcdf(ic_out)
g["mask"] = mask.astype(np.asarray(g["mask"]).dtype)
for key in ("H", "H_bathy", "depth", "bathymetry"):
    if key in g:
        g[key] = H.astype(np.asarray(g[key]).dtype); print("grid: replaced", key)
np.savez(grid_out, **g)
print(f"wrote {ic_out}\nwrote {grid_out} (keys: {sorted(g)})")
