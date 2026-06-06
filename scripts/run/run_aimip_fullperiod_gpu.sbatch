#!/usr/bin/env bash
#SBATCH --job-name=aimip_fullperiod
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=results/aimip_fullperiod_001/slurm-%j.out
#SBATCH --error=results/aimip_fullperiod_001/slurm-%j.err
#
# AIMIP full-period training across 2010-2016 (~98 days per variant)
# with early stopping.  Uses the architectures selected by the prior
# arch sweep (best column_nn + sfno) and the multi-step CRPS loss.
#
# Variants run sequentially; per-variant wall clock is ~1.5-3 h with
# early stopping, so total ~5-9 h.  12 h walltime gives margin for
# ERA5 data-load + JIT compile per variant.

set -euo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
mkdir -p results/aimip_fullperiod_001

PYTHON=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HOME=${HOME:-/burg-archive/home/pg2328}

echo "[full-period] $(date)  host=$(hostname)"
nvidia-smi -L 2>&1 | head -5 || echo "  (nvidia-smi unavailable)"

$PYTHON -u scripts/run/run_aimip.py \
    --suite config/aimip/aimip_suite_fullperiod.yaml

echo "[full-period] Scorecard:"
ls -la results/aimip_fullperiod_001/aimip_scorecard.json || true
echo "[full-period] Done $(date)"
