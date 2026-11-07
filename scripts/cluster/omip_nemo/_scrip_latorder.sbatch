#!/bin/bash
#SBATCH --job-name=scriplat
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scriplat_%j.out
# MY PROBE WAS THE THING AT FAULT (again). The first analytic field was
# cos(lat)*cos(lon), which is SYMMETRIC in latitude and therefore CANNOT detect
# a latitude-order mismatch between our ascending source array and whatever
# ordering the weights were built against. Re-run with an ANTISYMMETRIC field,
# sin(lat), and try BOTH source orderings. Whichever gives a small error is the
# ordering the weights assume.
set -uo pipefail
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -u - <<'PY'
import numpy as np, xarray as xr
W = "/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/"
MESH = "/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc"
CORE = ("/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/"
        "core2_nyf_mod/nyf.zarr")
d = xr.open_dataset(W + "weights_coreII_2_eORCA1.4.2_bilinear.nc",
                    decode_times=False)
src = np.stack([np.asarray(d[f"src{k:02d}"]) for k in (1, 2, 3, 4)]).astype(np.int64)
wgt = np.stack([np.asarray(d[f"wgt{k:02d}"]) for k in (1, 2, 3, 4)])
s = xr.open_zarr(CORE, consolidated=False)
slat = np.asarray(s["lat"]); slon = np.asarray(s["lon"])
m = xr.open_dataset(MESH, decode_times=False)
dlon = np.asarray(m["glamt"]).squeeze()[0:331, 1:361]
dlat = np.asarray(m["gphit"]).squeeze()[0:331, 1:361]
inside = (dlat <= slat.max()) & (dlat >= slat.min())

def score(field_fn, lat_axis, label):
    F = field_fn(lat_axis[:, None], slon[None, :])
    flat = F.ravel(order="C")
    out = (flat[np.clip(src - 1, 0, flat.size - 1)] * wgt).sum(0)
    truth = field_fn(dlat, dlon)
    err = np.abs(out - truth)[inside]
    print(f"  {label:34s} median {np.median(err):.3e}   max {err.max():.3e}")
    return np.median(err)

fields = {
    "sin(lat)               ": lambda la, lo: np.sin(np.deg2rad(la)) + 0*lo,
    "sin(lat)*cos(lon)      ": lambda la, lo: np.sin(np.deg2rad(la))*np.cos(np.deg2rad(lo)),
    "lat/90 (linear, signed)": lambda la, lo: la/90.0 + 0*lo,
}
for name, fn in fields.items():
    print(f"\n{name}")
    a = score(fn, slat, "source as-is (ascending lat)")
    b = score(fn, slat[::-1], "source with lat REVERSED")
    print(f"  -> {'AS-IS' if a < b else 'REVERSED'} matches the weights")
PY
echo DONE
