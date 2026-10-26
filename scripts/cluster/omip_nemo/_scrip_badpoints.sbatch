#!/bin/bash
#SBATCH --job-name=scripbad
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scripbad_%j.out
# A LINEAR field reproduces to 2e-08 MEDIAN, so the convention is right, yet
# ~2.5% of points are badly wrong. Identify that subset exactly rather than
# waving at it: is it the partially-populated stencils (non-zero count < 4)?
set -uo pipefail
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -u - <<'PY'
import numpy as np, xarray as xr
W = "/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/"
MESH = "/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc"
CORE = ("/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/"
        "core2_nyf_mod/nyf.zarr")
d = xr.open_dataset(W + "weights_coreII_2_eORCA1.4.2_bilinear.nc", decode_times=False)
src = np.stack([np.asarray(d[f"src{k:02d}"]) for k in (1,2,3,4)]).astype(np.int64)
wgt = np.stack([np.asarray(d[f"wgt{k:02d}"]) for k in (1,2,3,4)])
s = xr.open_zarr(CORE, consolidated=False)
slat = np.asarray(s["lat"]); slon = np.asarray(s["lon"])
m = xr.open_dataset(MESH, decode_times=False)
dlon = np.asarray(m["glamt"]).squeeze()[0:331, 1:361]
dlat = np.asarray(m["gphit"]).squeeze()[0:331, 1:361]

# (dropped a stray broadcast line; only `flat` is needed)
flat = (slat[:, None]/90.0 + 0*slon[None, :]).ravel(order="C")
out = (flat[np.clip(src-1, 0, flat.size-1)] * wgt).sum(0)
truth = dlat/90.0
err = np.abs(out - truth)
nz = (np.abs(wgt) > 0).sum(axis=0)
bad = err > 1e-3
print(f"bad points: {bad.sum()} of {err.size} ({100*bad.mean():.3f}%)")
for k in (1,2,3,4):
    sel = nz == k
    if sel.any():
        print(f"  stencils with {k} non-zero weights: {sel.sum():7d}  "
              f"bad among them: {100*bad[sel].mean():6.2f}%  "
              f"max err {err[sel].max():.3e}")
full = nz == 4
print(f"\nFULL 4-point stencils only: max err {err[full].max():.3e}, "
      f"median {np.median(err[full]):.3e}")
print("VERDICT:", "the error lives ENTIRELY in partial stencils"
      if err[full].max() < 1e-3 else
      "NOT explained by partial stencils -- something else is wrong")
# where are the remaining bad full-stencil points, if any?
rem = bad & full
if rem.any():
    print(f"  remaining bad FULL-stencil points: {rem.sum()}")
    print(f"  their lat range: {dlat[rem].min():.2f} .. {dlat[rem].max():.2f}")
    print(f"  their j (row) range: {np.where(rem.any(axis=1))[0].min()} .. "
          f"{np.where(rem.any(axis=1))[0].max()} of {rem.shape[0]}")
PY
echo DONE
