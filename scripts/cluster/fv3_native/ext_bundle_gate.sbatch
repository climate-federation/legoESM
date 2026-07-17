#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=extbundle
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --exclude=g161
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/extbundle_%j.log
set -euo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

echo "=== ruff ==="
$PY -m ruff check \
    $REPO/packages/core/legoesm/grids/fv3_native_ext_vector.py \
    $REPO/packages/core/legoesm/grids/fv3_native_gridstruct.py \
    $REPO/packages/core/legoesm/core/fv3_native_duo_stepper.py \
    $REPO/tests/grids/test_fv3_native_ext_vector.py \
    $REPO/scripts/validate/fv3_native/ablate_ext_bundle.py

echo "=== new ext-vector tests ==="
$PY -m pytest $REPO/tests/grids/test_fv3_native_ext_vector.py -x -q

echo "=== duo regression (stepper + stages) ==="
$PY -m pytest $REPO/tests/grids -k "fv3_native and (duo or dsw or ext)" -q

echo "=== C12 ablation (interim vs ext bundle) ==="
$PY $REPO/scripts/validate/fv3_native/ablate_ext_bundle.py --n 12

echo EXTBUNDLE_DONE
