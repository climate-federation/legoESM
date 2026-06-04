#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=pcell_val
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/pcellval_%j.out

# VALIDATION of the partial-cell fix on the FULL forced runner (CORE-II in-step
# forcing). Plain z* blew up by day 0.5 (job 8106193). Does --partial-cell give
# a stable, physical multi-month WOA cold-start run?
#   ARM A: --partial-cell, NO drag, 90d -> does max|u| saturate (~physical) or grow?
#   ARM B: --partial-cell + light spin-up drag (tau10d, 30d, released) -> stable?
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader

echo; echo "### ARM A: tripole --partial-cell, WOA cold-start, 90d, NO drag"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run/run_omip_core2.py \
  --grid tripole --woa-init --partial-cell --years 0.25 --dt 600 \
  --diag-every-days 2 --output results/omip_nemo/pcell_nodrag 2>&1
echo "[armA_rc]=$?"

echo; echo "### ARM B: tripole --partial-cell + light drag (tau10/30d), 90d"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run/run_omip_core2.py \
  --grid tripole --woa-init --partial-cell --years 0.25 --dt 600 \
  --spinup-drag-tau-days 10 --spinup-drag-days 30 \
  --diag-every-days 2 --output results/omip_nemo/pcell_drag 2>&1
echo "[armB_rc]=$?"

echo; echo "=== DONE ==="; date
