#!/bin/bash
#SBATCH --job-name=wb_smoke_s1
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_wbforecast/results/wb_sweep_stage1/slurm_logs/smoke_%j.out
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# Tiny smoke of the Stage-1 pipeline: baseline combo with --smoke (T21/8lev/
# 1 epoch) validates that the config parses, RRTMGP + all 5 schemes compile,
# training runs, and a scorecard is written -- BEFORE the full T63 array.
set -euo pipefail
WORKTREE=/burg-archive/glab/users/pg2328/legoESM_wbforecast
cd "$WORKTREE"
source scripts/cluster/wb_forecast/env.sh
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
mkdir -p results/wb_sweep_stage1/slurm_logs
echo "[$(date)] WB Stage-1 smoke: baseline combo --smoke"
"$WB_PYTHON" scripts/run/run_aimip.py \
    --suite config/wb/sweep/stage1/combo_baseline/suite.yaml \
    --variants classical \
    --smoke
echo "[$(date)] smoke done; scorecard:"
find results -name aimip_scorecard.json -newermt '-1 hour' 2>/dev/null | head
