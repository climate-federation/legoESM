#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --job-name=deardq
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/deardq_%j.out
set +e
WT=/burg-archive/glab/users/pg2328/legoESM; cd "$WT" || exit 1
export PYTHONPATH=$WT/src:$(ls -d "$WT"/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -m pytest -q tests/atmosphere/hydrostatic/unit/test_smagorinsky_deardorff.py -v 2>&1 | tail -20
echo "rc=${PIPESTATUS[0]}"
echo "DONE $(date +%T)"
