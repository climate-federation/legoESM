#!/usr/bin/env bash
#SBATCH --job-name=aimip_t106_classical
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=results/aimip_multistep_t106_001/classical_run/slurm-%j.out
#SBATCH --error=results/aimip_multistep_t106_001/classical_run/slurm-%j.err
#
# T106 (~1.1 deg) multi-step + CRPS training, classical variant only.
# Runs in parallel with run_aimip_t106_column_nn.sbatch and
# run_aimip_t106_sfno.sbatch on separate GPUs; each writes its own
# per-variant scorecard so there are no concurrent JSON merges.
#
# Submit all three with:
#   sbatch scripts/run_aimip_t106_classical.sbatch
#   sbatch scripts/run/run_aimip_t106_column_nn.sbatch
#   sbatch scripts/run/run_aimip_t106_sfno.sbatch

set -euo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
mkdir -p results/aimip_multistep_t106_001/classical_run

PYTHON=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HOME=${HOME:-/burg-archive/home/pg2328}

echo "[t106-classical] $(date)  host=$(hostname)  GPU info:"
nvidia-smi -L 2>&1 | head -5 || echo "  (nvidia-smi unavailable)"

echo "[t106-classical] Suite: config/aimip/t106/aimip_suite_classical.yaml"
$PYTHON -u scripts/run/run_aimip.py \
    --suite config/aimip/t106/aimip_suite_classical.yaml

echo "[t106-classical] Scorecard:"
ls -la results/aimip_multistep_t106_001/classical_run/aimip_scorecard.json || true
cat results/aimip_multistep_t106_001/classical_run/aimip_scorecard.json 2>/dev/null | head -60 || true

echo "[t106-classical] Done $(date)"
