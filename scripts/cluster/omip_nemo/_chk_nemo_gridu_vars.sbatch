#!/bin/bash
#SBATCH --job-name=chkU
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/chkU_%j.out
#
# WHAT IS ACTUALLY IN NEMO's grid_U, AND WHICH MESH METRICS EXIST?
#
# Inventory before scoping, not after. grid_U may carry a plain velocity
# (uo, m/s) or an EFFECTIVE TRANSPORT (uocetr_eff, already integrated over
# thickness and face width) -- multiplying the second by a thickness again
# would silently double-count and produce a plausible-looking wrong number.
# NEMO's CF labels have already lied once in this campaign, so the units
# attribute is printed next to every candidate and the magnitude is printed
# too, because a magnitude discriminates m/s from m3/s instantly.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge
cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

GU=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_U.nc
MESH=/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc

$PY -u - "$GU" "$MESH" <<'PY'
import sys
import numpy as np
import netCDF4 as nc

gu, mesh = sys.argv[1], sys.argv[2]
ds = nc.Dataset(gu)
print(f"=== {gu}")
print("time records:", len(ds.dimensions["time_counter"]))
for n, v in ds.variables.items():
    if v.ndim < 3:
        continue
    u = getattr(v, "units", "?")
    ln = getattr(v, "long_name", "?")
    a = np.ma.masked_invalid(v[5] if v.shape[0] > 5 else v[0])
    nz = int(a.count())
    mx = float(np.abs(a).max()) if nz else float("nan")
    print(f"  {n:16s} {str(v.shape):22s} units={u:12s} |max|={mx:11.4g} "
          f"finite={nz:9d}  {ln[:46]}")
ds.close()

m = nc.Dataset(mesh)
print(f"\n=== {mesh}")
for n in ("e1u", "e2u", "e3u_0", "e3t_0", "umask", "e1t", "e2t",
          "glamu", "gphiu", "glamt", "gphit"):
    if n in m.variables:
        print(f"  {n:8s} present shape={m.variables[n].shape}")
    else:
        print(f"  {n:8s} ABSENT")
m.close()
PY
echo "=== rc=$? ==="
