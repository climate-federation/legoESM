#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cornermc
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/corner_mc_%j.log
set -uo pipefail
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_fv3jax}
[ -e "$REPO/.git" ] || { echo "PREFLIGHT FAILED: REPO=$REPO"; exit 5; }
DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
[ "$DIRTY" -eq 0 ] || { echo "PREFLIGHT FAILED: dirty ($DIRTY)"; exit 3; }
echo "host: $(hostname)"; echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
cd "$REPO"
$PY scripts/validate/fv3_native/corner_metric_compare.py
rc=$?
echo "  rc=$rc"
[ "$rc" -eq 0 ] || { echo "CORNER_MC_FAILED rc=$rc"; exit "$rc"; }
echo CORNER_MC_DONE
