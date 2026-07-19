#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=aimip_sfno_smk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=results/aimip_latlon/logs/%x_%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/scripts/cluster/scaling_ginsburg/_env.sh"
cd "$REPO"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
mkdir -p results/aimip_latlon/logs
OUT="results/aimip_latlon/sfno_smoke_${SLURM_JOB_ID}"
JAX_PLATFORMS=cuda "$PY_GPU" -u scripts/run/run_aimip_latlon.py \
  --smoke --variants sfno --sfno-embed 64 --sfno-blocks 2 --nn-residual-scale 1e-6 --output-dir "$OUT" 2>&1 | tail -45
echo "=== scorecard ==="; cat "$OUT/aimip_latlon_scorecard.json" 2>/dev/null
