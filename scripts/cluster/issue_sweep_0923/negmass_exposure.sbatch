#!/bin/bash
#SBATCH --job-name=negmass_exp
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/negmass_exposure.%j.out
# #1029 follow-up, user decision "hard error, then pick a fix": the fix has to
# be picked from numbers, so measure what each candidate coordinate forbids --
# both as a surface pressure and as an ELEVATION, and against a real elevation
# dataset as a fraction of the planet's AREA (not a cell count; a lat-lon cell
# count is not a global fraction).
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
ETOPO=/burg-archive/glab/users/pg2328/legoESM_moistwt/data/bathymetry/etopo_1deg.nc
for N in 40 30 32; do
  echo "############################## nlev = $N"
  if [ -f "$ETOPO" ]; then
    "$PY" -u scripts/validate/hybrid_negative_layer_mass_exposure.py \
        --nlev "$N" --elevation-file "$ETOPO"
  else
    echo "elevation file missing: $ETOPO -- thresholds only"
    "$PY" -u scripts/validate/hybrid_negative_layer_mass_exposure.py --nlev "$N"
  fi
done
echo DONE
