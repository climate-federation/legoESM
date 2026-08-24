#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=simplelw
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/simplelw_%j.out
set +e
# Idealized + faithfulness tests for the shared Stevens (2005) simple longwave
# extracted out of run_dycoms_les.py, plus the repo-wide gates that a new
# physics module has to clear: physics contract, param spec, inline-coefficient
# and hardcoded-constant ratchets.
: "${LW_WT:?set LW_WT to the pinned worktree to run from}"
WT=$LW_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=============== simple_lw idealized tests ==============="
$PY -m pytest -q tests/atmosphere/test_simple_lw.py 2>&1 | tail -30
echo "simple_lw rc=${PIPESTATUS[0]}"

echo "=============== repo gates a new physics module must clear ==============="
$PY -m pytest -q \
    tests/test_physics_contracts.py \
    tests/test_param_specs.py \
    tests/test_no_inline_physics_coeffs.py \
    tests/test_no_hardcoded_constants.py 2>&1 | tail -30
echo "gates rc=${PIPESTATUS[0]}"
echo "DONE $(date +%T)"

echo "=============== joint-aggregation tests ==============="
$PY -m pytest -q tests/unit/test_scm_les_joint_aggregation.py 2>&1 | tail -20
echo "aggregation rc=${PIPESTATUS[0]}"
