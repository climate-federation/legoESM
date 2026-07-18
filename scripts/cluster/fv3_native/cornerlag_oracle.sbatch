#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cornerlag
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/cornerlag_%j.log
set -euo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
command -v gfortran

$PY $REPO/scripts/validate/fv3_native/gen_cornerlag_oracle.py \
    --n 12 --ng 3 --build-dir "$TMPDIR/cornerlag_build"

$PY -m pytest $REPO/tests/grids/test_fv3_native_ext_vector.py -q -x \
    -k "cornerlag"
echo CORNERLAG_DONE
