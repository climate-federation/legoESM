#!/usr/bin/env bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=uni_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/logs/smoke_%j.out
# Unified-path smoke: run the AIMIP suite dispatch end-to-end at smoke scale
# (T21/8lev/1epoch) for the variant ladder, through the SAME code path the
# scale launchers use. Gate before any scale submission (plan Task 10).
#   sbatch scripts/cluster/unified_training/smoke_ginsburg.sbatch
set -euo pipefail

source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/_env.sh
cd "$UT_REPO"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SUITE="${SUITE:-config/aimip/ace2/suite.yaml}"
VARIANTS="${VARIANTS:-classical,column_nn,sfno_physics}"

echo "[uni-smoke] $(date) suite=$SUITE variants=$VARIANTS"
"$UT_PYTHON" -u scripts/run/run_aimip.py --suite "$SUITE" --smoke --variants "$VARIANTS"
echo "[uni-smoke] done $(date)"
