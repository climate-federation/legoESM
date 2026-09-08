#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3dztwin
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/fv3_dztwin_%j.log
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
[ "$DIRTY" -eq 0 ] || { echo "PREFLIGHT FAILED: dirty ($DIRTY)"; exit 3; }
echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
cd "$REPO"
$PY scripts/validate/fv3_update_dz_d_twin_localiser.py 2>&1 | tail -60
echo "rc=${PIPESTATUS[0]}"
