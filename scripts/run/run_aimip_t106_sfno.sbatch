#!/usr/bin/env bash
#SBATCH --job-name=aimip_t106_sfno
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=results/aimip_multistep_t106_001/sfno_physics_run/slurm-%j.out
#SBATCH --error=results/aimip_multistep_t106_001/sfno_physics_run/slurm-%j.err
#
# T106 (~1.1 deg) multi-step + CRPS training, sfno_physics variant
# only.  Uses the downsized SFNO overlay in
# config/aimip/t106/variant_sfno_physics.yaml (embed_dim=32,
# n_blocks=2) so the model fits in 24 GiB.  Extra walltime + memory
# vs the other two variants because SFNO gradients are the largest.

set -euo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
mkdir -p results/aimip_multistep_t106_001/sfno_physics_run

PYTHON=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HOME=${HOME:-/burg-archive/home/pg2328}

echo "[t106-sfno] $(date)  host=$(hostname)  GPU info:"
nvidia-smi -L 2>&1 | head -5 || echo "  (nvidia-smi unavailable)"

echo "[t106-sfno] Suite: config/aimip/t106/aimip_suite_sfno.yaml"
$PYTHON -u scripts/run_aimip.py \
    --suite config/aimip/t106/aimip_suite_sfno.yaml

echo "[t106-sfno] Scorecard:"
ls -la results/aimip_multistep_t106_001/sfno_physics_run/aimip_scorecard.json || true
cat results/aimip_multistep_t106_001/sfno_physics_run/aimip_scorecard.json 2>/dev/null | head -60 || true

echo "[t106-sfno] Done $(date)"
