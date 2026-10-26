#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=clubbunit
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/clubbunit_%j.out
set +e
: "${C_WT:?set C_WT}"
WT=$C_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -m pytest -q tests/atmosphere/test_clubb_variance_floor.py 2>&1 | tail -18
echo "unit rc=${PIPESTATUS[0]}"
$PY -m pytest -q tests/ -k "clubb" 2>&1 | tail -18
echo "suite rc=${PIPESTATUS[0]}"
echo "DONE $(date +%T)"
