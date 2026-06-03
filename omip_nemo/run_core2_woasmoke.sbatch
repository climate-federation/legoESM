#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_woasm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/woasm_%j.out

# Smoke-test WOA18-initialised tripole run (faithful IC). Validates (a)
# init_ocean_from_woa produces the right (332,362,20) field on the tripole grid,
# (b) the realistic T/S gradients stay stable at dt=600 (rest state was gentle;
# climatology IC can excite the PGF). 10-day, daily diag.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; nvidia-smi --query-gpu=name --format=csv,noheader 2>&1|head -1; date
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run_omip_core2.py \
  --smoke --dt 600 --woa-init --diag-every-days 1 \
  --output results/omip_nemo/legoesm_tripole_woa_smoke 2>&1 | tail -40
echo "woasm_rc=${PIPESTATUS[0]}"
echo "=== DONE ==="
