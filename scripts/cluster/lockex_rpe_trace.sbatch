#!/bin/bash
# Per-step RPE trace, one arm per array task. Arm = grid[:tracer_adv].
# Default arms (0-3) are the controls; override with ARMS env:
#   sbatch --array=0-3 scripts/cluster/lockex_rpe_trace.sbatch
#   ARMS="latlon:fct2 tripole:fct2 mpas:upwind latlon" sbatch --array=0-3 \
#       --export=ALL,ARMS scripts/cluster/lockex_rpe_trace.sbatch
#SBATCH --job-name=rpe-trace
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/rpetrace_%A_%a.out
#SBATCH --error=logs/rpetrace_%A_%a.out
set -euo pipefail
REPO=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM
cd "$REPO"; mkdir -p logs
ARMS=(${ARMS:-fesom mpas latlon tripole})
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
GRID=${ARM%%:*}
ADV=""
[[ "$ARM" == *:* ]] && ADV="--tracer-adv ${ARM#*:}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "=== rpe trace | $ARM | EXTRA=${EXTRA:-} | $(date -Is) ==="
srun .venv/bin/python scripts/validate/lockex_rpe_trace.py --grid "$GRID" --days 1.0 $ADV ${EXTRA:-}
echo "=== COMPLETED $ARM rc=$? $(date -Is) ==="
