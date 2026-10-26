#!/bin/bash
#SBATCH --job-name=scripw
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scripw_%j.out
#
# VALIDATE THE ORACLE'S WEIGHT FILES BEFORE USING A SINGLE WEIGHT.
#
# The user chose exact reproduction over method-equivalence, so we are about to
# remap forcing through NEMO's own SCRIP weights. A wrong index base or a
# transposed destination produces a PLAUSIBLE-LOOKING field, not an error, which
# is precisely the failure class this repo keeps hitting. So: prove the
# convention on a field whose answer is known analytically, before trusting it
# on real forcing.
#
# WHAT IS ALREADY ESTABLISHED (from the file's own history attribute, not from
# arithmetic): it was produced by
#     ncks -F --abc -d lon,2,361 -d lat,1,331
# i.e. the INTERIOR of a 362x332 eORCA1 grid -- drop both cyclic-overlap
# columns and the north-fold row. Our mesh is exactly 362x332. map_method says
# "Bicubic remapping"; note its title string says "bilin", which is a
# copy-paste artifact in the ORACLE'S OWN file -- trust map_method.
#
# PRE-REGISTERED, each a pass/fail:
#   1. Weights per destination point sum to 1 (within 1e-6) wherever the
#      stencil is populated. A partition of unity is what keeps a constant
#      field constant; if it fails, normalization is not 'fracarea' as claimed.
#   2. src indices are 1-BASED and lie within the CORE-II source size.
#      Off-by-one here silently shifts the whole forcing field one cell.
#   3. Remapping an ANALYTIC field through the real weights reproduces it.
#      Using f = cos(lat)*cos(lon) on the source grid, the remapped field must
#      match the analytic value at the destination cell centres taken from the
#      mesh (glamt/gphit interior) to better than 1e-3. THIS is the test that
#      catches a transpose: a transposed destination still sums to 1 and still
#      has valid indices, and only an analytic comparison exposes it.
#   4. The bilinear file uses at most 4 non-zero triples per point; the
#      bicubic file uses more. If the "bilinear" file also used 16, the two
#      files would not differ in the way their names claim.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

$PY -u - <<'PY'
import numpy as np, xarray as xr

W = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/"
     "INPUTS/")   # the RUN_GATEWAY copies are BROKEN SYMLINKS
MESH = "/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc"
CORE = ("/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/"
        "core2_nyf_mod")

def load(path):
    d = xr.open_dataset(path, decode_times=False)
    n = sum(1 for v in d.variables if v.startswith("src"))
    src = np.stack([np.asarray(d[f"src{k:02d}"]) for k in range(1, n + 1)])
    wgt = np.stack([np.asarray(d[f"wgt{k:02d}"]) for k in range(1, n + 1)])
    return src, wgt, dict(d.attrs), n

for name in ("weights_coreII_2_eORCA1.4.2_bicubic.nc",
             "weights_coreII_2_eORCA1.4.2_bilinear.nc"):
    src, wgt, attrs, n = load(W + name)
    nz = (np.abs(wgt) > 0).sum(axis=0)
    print(f"\n=== {name}")
    print(f"  triples={n}  dst shape={src.shape[1:]}  map_method={attrs.get('map_method')}")
    print(f"  src index min={src.min()} max={src.max()}")
    print(f"  weight-sum: min={wgt.sum(axis=0).min():.6f} "
          f"max={wgt.sum(axis=0).max():.6f}")
    print(f"  non-zero triples per point: min={nz.min()} max={nz.max()} "
          f"median={int(np.median(nz))}")

# --- the analytic test, on the BILINEAR file (simplest to reason about) -----
src, wgt, attrs, n = load(W + "weights_coreII_2_eORCA1.4.2_bilinear.nc")
ny, nx = src.shape[1:]

# Source grid: read the CORE-II axes the model itself uses.
# The forcing cache is a ZARR store (nyf.zarr), not netCDF.
s = xr.open_zarr(CORE + "/nyf.zarr", consolidated=False)
print("\nforcing vars:", list(s.data_vars)[:8])
latname = [c for c in list(s.coords) + list(s.dims) if "lat" in c.lower()][0]
lonname = [c for c in list(s.coords) + list(s.dims) if "lon" in c.lower()][0]
slat = np.asarray(s[latname]); slon = np.asarray(s[lonname])
print(f"source grid: lat {slat.shape} [{slat[0]:.3f},{slat[-1]:.3f}]  "
      f"lon {slon.shape} [{slon[0]:.3f},{slon[-1]:.3f}]")
S_ny, S_nx = slat.size, slon.size
print(f"src index max {src.max()} vs source size {S_ny*S_nx} "
      f"-> 1-based fits: {src.max() <= S_ny*S_nx}")

# Analytic source field, flattened the way SCRIP expects (C order, lat-major).
F = (np.cos(np.deg2rad(slat))[:, None]
     * np.cos(np.deg2rad(slon))[None, :])
flat = F.ravel(order="C")
idx = np.clip(src.astype(np.int64) - 1, 0, flat.size - 1)  # 1-based -> 0-based
out = (flat[idx] * wgt).sum(axis=0)

m = xr.open_dataset(MESH, decode_times=False)
glamt = np.asarray(m["glamt"]).squeeze()      # (332, 362)
gphit = np.asarray(m["gphit"]).squeeze()
# The ncks slice: Fortran lon 2..361, lat 1..331 -> python [0:331, 1:361]
dlon = glamt[0:331, 1:361]
dlat = gphit[0:331, 1:361]
print(f"destination interior slice: {dlat.shape} (weights say {src.shape[1:]})")
truth = np.cos(np.deg2rad(dlat)) * np.cos(np.deg2rad(dlon))

good = np.abs(wgt).sum(axis=0) > 0.5
err = np.abs(out - truth)[good]
print(f"\nANALYTIC TEST over {good.sum()} populated points:")
print(f"  max |remapped - analytic| = {err.max():.3e}")
print(f"  median                    = {np.median(err):.3e}")
print("VERDICT:", "CONVENTION CONFIRMED" if err.max() < 1e-3
      else "CONVENTION WRONG -- do not use these weights yet")
PY
echo DONE

# --- WHERE is the error? Added after the first run: median 5.2e-05 says the
# convention is right, max 1.08 says something is wrong SOMEWHERE. Relaxing the
# threshold would be unclamping a limiter; locate it instead.
PY2=$(cat <<'PYEOF'
import numpy as np, xarray as xr
W = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/")
MESH = "/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc"
CORE = ("/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/"
        "core2_nyf_mod/nyf.zarr")
d = xr.open_dataset(W + "weights_coreII_2_eORCA1.4.2_bilinear.nc",
                    decode_times=False)
src = np.stack([np.asarray(d[f"src{k:02d}"]) for k in (1, 2, 3, 4)])
wgt = np.stack([np.asarray(d[f"wgt{k:02d}"]) for k in (1, 2, 3, 4)])
s = xr.open_zarr(CORE, consolidated=False)
slat = np.asarray(s["lat"]); slon = np.asarray(s["lon"])
F = np.cos(np.deg2rad(slat))[:, None] * np.cos(np.deg2rad(slon))[None, :]
flat = F.ravel(order="C")
out = (flat[np.clip(src.astype(np.int64) - 1, 0, flat.size - 1)] * wgt).sum(0)
m = xr.open_dataset(MESH, decode_times=False)
dlon = np.asarray(m["glamt"]).squeeze()[0:331, 1:361]
dlat = np.asarray(m["gphit"]).squeeze()[0:331, 1:361]
truth = np.cos(np.deg2rad(dlat)) * np.cos(np.deg2rad(dlon))
err = np.abs(out - truth)
bad = err > 1e-3
print(f"points over 1e-3: {bad.sum()} of {err.size} ({100*bad.mean():.3f}%)")
if bad.any():
    print(f"  their latitudes: {dlat[bad].min():.2f} .. {dlat[bad].max():.2f}")
    print(f"  source lat range: {slat.min():.3f} .. {slat.max():.3f}")
    outside = (dlat[bad] > slat.max()) | (dlat[bad] < slat.min())
    print(f"  fraction of bad points OUTSIDE the source latitude range: "
          f"{100*outside.mean():.1f}%")
    inside = ~bad & (dlat <= slat.max()) & (dlat >= slat.min())
    print(f"  max error at points INSIDE the source range: "
          f"{err[(dlat <= slat.max()) & (dlat >= slat.min())].max():.3e}")
PYEOF
)
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -u -c "$PY2"
