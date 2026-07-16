#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=d2a2c
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --exclude=g161
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/d2a2c_%j.log
set -euo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
S=$REPO/scripts/validate/fv3_native
W=/burg-archive/glab/users/pg2328/fv3_recon/d2a2c_work
rm -rf "$W" && mkdir -p "$W" && cd "$W"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

PYTHONPATH=$REPO/packages/core:$REPO/src $PY $S/gen_d2a2c_oracle.py "$W"

FF="-O2 -fdefault-real-8 -fdefault-double-8 -cpp -ffree-line-length-none"
gfortran $FF -J "$W" -c $S/fv3_swcore_shim.F90        -o "$W/shim.o"
gfortran $FF -J "$W" -c $S/fv3_swcore_extract.F90     -o "$W/ext.o"
gfortran $FF -J "$W" -c $S/fv3_d2a2c_duo_extract.F90  -o "$W/d2a2c.o"
gfortran $FF -J "$W" $S/fv3_d2a2c_driver.F90 "$W/shim.o" "$W/ext.o" "$W/d2a2c.o" -o "$W/drv"
cd "$W" && ./drv

PYTHONPATH=$REPO/packages/core:$REPO/src $PY $S/gen_d2a2c_oracle.py "$W" --pack

# postflight: the committed gate must pass on the freshly-packed fixture
PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  $PY -m pytest $REPO/tests/grids/test_fv3_native_d2a2c_duo.py -q
echo D2A2C_DONE
