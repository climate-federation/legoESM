#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=scmrce_td
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_test_drivers_%j.out
set +e
WT=/burg/glab/users/pg2328/scmrce_wt5
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=================== DRIVER CLI TESTS ==================="
JAX_ENABLE_X64=1 $PY -m pytest \
  tests/unit/test_scm_rce_convection_intercomparison_cli.py \
  tests/unit/test_train_scm_rce_params_cli.py -q 2>&1 | tail -35
echo "=================== EXISTING DRIVER REGRESSION ==================="
JAX_ENABLE_X64=1 $PY -m pytest tests/ -q -k "scm_rce" 2>&1 | tail -20
echo ALLDONE
