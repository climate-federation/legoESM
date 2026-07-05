#!/bin/bash
#SBATCH --job-name=wb_sweep_s1
#SBATCH --output=results/wb_sweep_stage1/slurm_logs/sweep_s1_%A_%a.out
#SBATCH --error=results/wb_sweep_stage1/slurm_logs/sweep_s1_%A_%a.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00

# SLURM array entry: each task trains one combo from the Stage-1 manifest at
# T63 + RRTMGP with the multi-step forecast loss.
#     sbatch --array=0-N scripts/run/_wb_sweep_stage1_runner.sbatch
# where N+1 = number of combos in the manifest.

set -euo pipefail

WORKTREE="/burg-archive/glab/users/pg2328/legoESM_wbforecast"
cd "$WORKTREE"
# worktree-pinned env (jn2808 conda python + worktree packages on PYTHONPATH)
source "$WORKTREE/scripts/cluster/wb_forecast/env.sh"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

MANIFEST="config/wb/sweep/stage1/manifest.json"
SUITE_PATH=$("$WB_PYTHON" -c "
import json, sys
m = json.load(open('${MANIFEST}'))
print(m['combos'][int(sys.argv[1])]['suite'])
" "${SLURM_ARRAY_TASK_ID}")

mkdir -p results/wb_sweep_stage1/slurm_logs

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} -> suite=${SUITE_PATH}"
"$WB_PYTHON" scripts/run/run_aimip.py \
    --suite "${SUITE_PATH}" \
    --variants classical \
    --resume
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done"
