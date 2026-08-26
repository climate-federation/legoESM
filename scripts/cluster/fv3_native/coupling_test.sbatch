#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3couple
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --exclude=g055,g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/couple_%j.log
# update_dwinds_phys_duo: loop-ref parity, jax twin, jit==eager, grad.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
cd $REPO
echo "SHA=$(git rev-parse HEAD)  dirty=$(git status --porcelain | wc -l)"
$PY -m pytest tests/grids/test_fv3_physics_coupling.py tests/grids/test_c2l_ord4_face.py -q 2>&1 | tail -20
