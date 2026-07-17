#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=extproj
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --exclude=g161
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/extproj_%j.log
set -euo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
command -v gfortran

$PY $REPO/scripts/validate/fv3_native/gen_extproj_oracle.py \
    --n 12 --ng 4 --build-dir "$TMPDIR/extproj_build"

# post-flight: the committed pytest gate must pass on the fresh fixture
$PY -m pytest $REPO/tests/grids/test_fv3_native_ext_vector.py -q -x
echo EXTPROJ_DONE
