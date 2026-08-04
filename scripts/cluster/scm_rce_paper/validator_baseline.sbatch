#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=val_base
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_val_baseline_%j.out
set +e
# CONTROL: run the SHIPPED convection validator at cf/main (my change absent)
# to establish whether the Test-4 magnitude failures for edmf / emanuel /
# tiedtke pre-exist this branch.
WT=/burg/glab/users/pg2328/scmrce_baseline
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)  (= cf/main baseline)"
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
JAX_ENABLE_X64=1 $PY scripts/validate/validate_convection_physics.py 2>&1 | tail -45
echo "validator rc=${PIPESTATUS[0]}"
echo "=== ratchet control (files my branch never touches) ==="
JAX_ENABLE_X64=1 $PY -m pytest tests/test_no_hardcoded_constants.py -q 2>&1 | tail -8
echo ALLDONE
