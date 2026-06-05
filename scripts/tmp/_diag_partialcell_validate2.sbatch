#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=pcell_v2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/pcellv2_%j.out

# CLEAN re-validation of the partial-cell fix (compute_woa_3d mask now
# partial-cell-aware, codex round 1 fix). Fine diag (0.5d) to see the velocity
# trajectory. adcroft vs smc03 (smc03 now ACTIVE under partial-cell).
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader

echo; echo "### ARM A: --partial-cell adcroft, WOA cold-start, 30d, diag 0.5d"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run/run_omip_core2.py \
  --grid tripole --woa-init --partial-cell --pgf-scheme adcroft \
  --years 0.0833 --dt 600 --diag-every-days 0.5 \
  --output results/omip_nemo/pcellv2_adcroft 2>&1 | grep -vE "Constant folding|This isn|If you|reduce-window|slow_operation|^$"
echo "[armA_rc]=${PIPESTATUS[0]}"

echo; echo "### ARM B: --partial-cell smc03, WOA cold-start, 30d, diag 0.5d"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run/run_omip_core2.py \
  --grid tripole --woa-init --partial-cell --pgf-scheme smc03 \
  --years 0.0833 --dt 600 --diag-every-days 0.5 \
  --output results/omip_nemo/pcellv2_smc03 2>&1 | grep -vE "Constant folding|This isn|If you|reduce-window|slow_operation|^$"
echo "[armB_rc]=${PIPESTATUS[0]}"
echo; echo "=== DONE ==="; date
