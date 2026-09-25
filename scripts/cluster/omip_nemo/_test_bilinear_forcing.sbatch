#!/bin/bash
#SBATCH --job-name=t_bilin
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/t_bilin_%j.out
# Gate for the bilinear tripole forcing fix: new sampler tests + the scan-lane
# equivalence tests + a stripe demo image from the REAL forcing cache.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

$PY -m pytest -q tests/ocean/unit/test_forcing_bilinear_sampling.py \
  tests/ocean/unit/test_scm_column_twins.py \
  tests/ocean/unit/test_omip2_forcing_scan.py \
  tests/ocean/unit/test_mass_flux_store.py 2>&1 | tail -5

echo "=== stripe demo: real CORE-II u10 record 0, nearest vs bilinear on eORCA1 rows ==="
$PY - <<'PY'
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from legoesm.ocean.forcing.core2 import load_core2_nyf
from legoesm.ocean.coupler.omip2_applicator import (
    _nn_interp_to_points, _bilinear_interp_to_points)
import netCDF4 as nc

f = load_core2_nyf()
ds = nc.Dataset("data/grids/eORCA1.2_mesh_mask.nc")
lat_T = np.asarray(ds["gphit"][0] if "gphit" in ds.variables else ds["nav_lat"][:])
lon_T = np.asarray(ds["glamt"][0] if "glamt" in ds.variables else ds["nav_lon"][:])
# one mid-Pacific meridional line of T points
jcol = np.argmin(np.abs(((lon_T[100] % 360) - 180)))
la = lat_T[:, jcol]; lo = lon_T[:, jcol] % 360
sel = (la > -60) & (la < 60)
u = np.asarray(f.u10[0])
nn = _nn_interp_to_points(u, f.lat, f.lon, la[sel], lo[sel])
bi = _bilinear_interp_to_points(u, f.lat, f.lon, la[sel], lo[sel])
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                       constrained_layout=True)
ax[0].plot(la[sel], nn, label="nearest (old)", lw=0.8)
ax[0].plot(la[sel], bi, label="bilinear (new)", lw=0.8)
ax[0].set_ylabel("u10 [m/s]"); ax[0].legend()
ax[1].plot(la[sel][1:], np.abs(np.diff(nn)), label="|row diff| nearest", lw=0.8)
ax[1].plot(la[sel][1:], np.abs(np.diff(bi)), label="|row diff| bilinear", lw=0.8)
ax[1].set_xlabel("latitude"); ax[1].set_ylabel("|d u10 / d row|"); ax[1].legend()
fig.suptitle("CORE-II u10 sampled onto eORCA1 T rows at ~180E: the stairstep is the band source")
fig.savefig("results/omip_nemo/forcing_stripes_nn_vs_bilinear.png", dpi=110)
zn = int(np.sum(np.abs(np.diff(nn)) < 1e-12)); zb = int(np.sum(np.abs(np.diff(bi)) < 1e-12))
print(f"exact-duplicate adjacent rows: nearest={zn}  bilinear={zb}  (of {sel.sum()-1})")
PY
echo DONE
