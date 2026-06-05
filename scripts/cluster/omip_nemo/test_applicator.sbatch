#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_test
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/test_%j.out

# Validate the omip2_applicator changes (tripole branch + wind-stress sign fix).
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -x -q \
  tests/ocean/unit/test_omip2_applicator.py 2>&1 | tail -45
echo "pytest_rc=${PIPESTATUS[0]}"
echo "=== DONE ==="
