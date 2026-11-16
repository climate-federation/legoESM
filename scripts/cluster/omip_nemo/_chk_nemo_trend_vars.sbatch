#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=trdvars
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trdvars_%j.out
#
# WHICH TRACER TRENDS DID NEMO ACTUALLY WRITE, AND ARE THEY NON-EMPTY?
#
# The operator-resolved tracer-variance budget needs a per-operator theta
# tendency. Two things already suggest the budget cannot be built as
# specified: `compare_tendencies_nemo` loads only ttrd_zdf/strd_zdf, and both
# it and `nemo_trend_closure` record that ttrd_tot is ENABLED IN THE XIOS
# DEFS BUT EMPTY IN EVERY FILE. A variable that exists and is all-zero is the
# trap here -- a budget built on it would close trivially and mean nothing --
# so this reports the finite fraction and the magnitude of EVERY trend
# variable, not just its presence.
#
# No conclusions from names alone: a trend that is present but identically
# zero is reported as UNUSABLE.
set -uo pipefail
cd /burg-archive/glab/users/pg2328/_omip_merge || exit 1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TRD=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD
for f in "$TRD"/keep_run1_1h_trd.nc "$TRD"/keep_run1_1d_trd.nc; do
  [ -r "$f" ] || { echo "MISSING $f"; continue; }
  echo "############ $f"
  $PY -u - "$f" <<'PY'
import sys, numpy as np, netCDF4 as nc
ds = nc.Dataset(sys.argv[1])
print("dims:", {k: len(v) for k, v in ds.dimensions.items()})
names = sorted(ds.variables)
print("all variables:", names)
print()
for v in names:
    var = ds.variables[v]
    if var.ndim < 3:
        continue
    x = var[0]
    x = x.filled(np.nan) if np.ma.isMaskedArray(x) else np.asarray(x)
    x = np.asarray(x, dtype=np.float64)
    fin = np.isfinite(x)
    nz = int(np.count_nonzero(np.where(fin, x, 0.0)))
    amax = float(np.nanmax(np.abs(x))) if fin.any() else float("nan")
    tag = "UNUSABLE(all-zero)" if nz == 0 else "ok"
    print(f"{v:16s} shape={var.shape} finite={fin.mean():6.3f} "
          f"nonzero={nz:10d} max|x|={amax:.6e}  {tag}")
ds.close()
PY
done
