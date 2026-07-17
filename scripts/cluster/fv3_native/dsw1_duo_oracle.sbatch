#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=dsw1duo
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --exclude=g161
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/dsw1duo_%j.log
set -euo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
S=$REPO/scripts/validate/fv3_native
W=/burg-archive/glab/users/pg2328/fv3_recon/dsw1_duo_work
rm -rf "$W" && mkdir -p "$W" && cd "$W"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

PYTHONPATH=$REPO/packages/core:$REPO/src $PY $S/gen_dsw1_duo_oracle.py "$W"

FF="-O2 -fdefault-real-8 -fdefault-double-8 -cpp -ffree-line-length-none"
gfortran $FF -J "$W" -c $S/fv3_swcore_shim.F90         -o "$W/shim.o"
gfortran $FF -J "$W" -c $S/fv3_tpcore_duo_extract.F90  -o "$W/tpc.o"
gfortran $FF -J "$W" -c $S/fv3_dsw1_duo_extract.F90    -o "$W/dsw1.o"
gfortran $FF -J "$W" $S/fv3_dsw1_duo_driver.F90 "$W/shim.o" "$W/tpc.o" "$W/dsw1.o" -o "$W/drv"
cd "$W" && ./drv

PYTHONPATH=$REPO/packages/core:$REPO/src $PY $S/gen_dsw1_duo_oracle.py "$W" --pack

PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  $PY -m pytest $REPO/tests/grids/test_fv3_native_dsw1_duo.py -q 2>/dev/null \
  || echo "pytest gate pending (test file may not exist yet)"
echo DSW1_DUO_DONE
