#!/usr/bin/env bash
#SBATCH --job-name=aimip_t106_column_nn
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=results/aimip_multistep_t106_001/column_nn_run/slurm-%j.out
#SBATCH --error=results/aimip_multistep_t106_001/column_nn_run/slurm-%j.err
#
# T106 (~1.1 deg) multi-step + CRPS training, column_nn variant only.
# Runs in parallel with the other two t106 variants on a separate
# GPU.

set -euo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
mkdir -p results/aimip_multistep_t106_001/column_nn_run

PYTHON=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HOME=${HOME:-/burg-archive/home/pg2328}

echo "[t106-column_nn] $(date)  host=$(hostname)  GPU info:"
nvidia-smi -L 2>&1 | head -5 || echo "  (nvidia-smi unavailable)"

echo "[t106-column_nn] Suite: config/aimip/t106/aimip_suite_column_nn.yaml"
$PYTHON -u scripts/run_aimip.py \
    --suite config/aimip/t106/aimip_suite_column_nn.yaml

echo "[t106-column_nn] Scorecard:"
ls -la results/aimip_multistep_t106_001/column_nn_run/aimip_scorecard.json || true
cat results/aimip_multistep_t106_001/column_nn_run/aimip_scorecard.json 2>/dev/null | head -60 || true

echo "[t106-column_nn] Done $(date)"
