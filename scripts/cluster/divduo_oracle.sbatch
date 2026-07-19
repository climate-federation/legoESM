#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=divduo
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/divduo_%j.log
set -euo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
S=$REPO/scripts/validate/fv3_native
W=/burg-archive/glab/users/pg2328/fv3_recon/divduo_work
rm -rf $W && mkdir -p $W && cd $W
PYTHONPATH=$REPO/packages/core:$REPO/src \
  /burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
  $S/gen_divduo_oracle.py $W
FF="-O2 -fdefault-real-8 -fdefault-double-8 -cpp -ffree-line-length-none"
gfortran $FF -J $W -c $S/fv3_swcore_shim.F90 -o $W/shim.o
gfortran $FF -J $W -c $S/fv3_swcore_extract.F90 -o $W/ext.o
gfortran $FF -J $W $S/fv3_divduo_driver.F90 $W/shim.o $W/ext.o -o $W/drv
cd $W && ./drv
# pack the committed fixture from the Fortran output (reproducible);
# --pack refuses to write if input.txt disagrees with the staged arrays
PYTHONPATH=$REPO/packages/core:$REPO/src \
  /burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
  $S/gen_divduo_oracle.py $W --pack
# postflight: the committed gate must pass on the freshly-packed fixture
# (bit-exact port-vs-Fortran + input-hash provenance) — the honest
# reproducibility check, no side npy needed
PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/src \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  /burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
  -m pytest $REPO/tests/grids/test_fv3_native_duo_phase4c.py -q
echo DIVDUO_DONE
