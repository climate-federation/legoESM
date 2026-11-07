#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=zm_pblh_oracle
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Build CAM6 zm_conv.F90 (cam_cesm2_1_rel_60) with driver.F90 + stubs.F90 and
# regenerate tests/atmosphere/oracle_fixtures/zm_pblh_cam.npz.
# Run from the repo root:  sbatch --wait scripts/validate/oracle/zm_pblh/build.sh
# Env: REPO (default: submit dir), CAM (CAM checkout), PY (python with legoesm deps).
set -euo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
CAM=${CAM:-/burg/glab/users/pg2328/oracles/CAM}
PY=${PY:-/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python}
HERE=$REPO/scripts/validate/oracle/zm_pblh
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
export PYTHONPATH=$(ls -d $REPO/packages/* | tr '\n' ':')$REPO
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1
echo "CAM SHA: $(git -C $CAM rev-parse HEAD)"
gfortran --version | head -1
cd $W
python3 $HERE/patch_oracle.py $CAM/src/physics/cam/zm_conv.F90 zm_conv_patched.F90
python3 $HERE/patch_wv_sat.py $CAM/src/physics/cam/wv_sat_methods.F90 wv_sat_methods.F90
FFLAGS="-O2 -cpp -ffree-line-length-none -fno-fast-math -DPCOLS_VAL=16 -DPVER_VAL=32"
gfortran $FFLAGS -c wv_sat_methods.F90
gfortran $FFLAGS -c $HERE/stubs.F90
gfortran $FFLAGS -c zm_conv_patched.F90
gfortran $FFLAGS -c $HERE/driver.F90
gfortran -o zm_driver.x wv_sat_methods.o stubs.o zm_conv_patched.o driver.o
echo BUILD_OK
$PY $HERE/make_fixture.py $W/zm_driver.x $W $CAM
echo FIXTURE_DONE
