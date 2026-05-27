#!/usr/bin/env bash
#SBATCH --job-name=aimip_arch_sweep
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=results/aimip_arch_sweep_001/slurm-%j.out
#SBATCH --error=results/aimip_arch_sweep_001/slurm-%j.err
#
# Architecture-size sweep for column_nn and sfno_physics variants
# under the multi-step CRPS training config.  ~20-30 min per (variant,
# size) entry on a single RTX 8000; 8 entries -> ~3-4 h.  10 h walltime
# leaves headroom for ERA5 data-load variability.

set -euo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
mkdir -p results/aimip_arch_sweep_001

PYTHON=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HOME=${HOME:-/burg-archive/home/pg2328}

echo "[arch-sweep] $(date)  host=$(hostname)"
nvidia-smi -L 2>&1 | head -5 || echo "  (nvidia-smi unavailable)"
$PYTHON -u scripts/run_aimip_arch_sweep.py \
    --base config/aimip/aimip_era5_multistep.yaml \
    --output-dir results/aimip_arch_sweep_001

echo "[arch-sweep] Done $(date)"
ls -la results/aimip_arch_sweep_001/aimip_arch_sweep.json 2>&1 || true
