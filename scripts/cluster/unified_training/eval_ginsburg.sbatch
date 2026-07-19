#!/usr/bin/env bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=uni_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/logs/eval_%j.out
# Unified-path evaluation on Ginsburg: WB2 scorecard (+plot) for a trained
# campaign. Single-process; run AFTER the train chain completes. The eval
# prefers the EMA checkpoint when present (campaign_driver, D3).
#   sbatch --export=ALL,CONFIG=config/wb/campaign/spectral_t63.yaml,MODES=physics,neural_gcm,sfno,OUT=results/wb_campaign \
#     scripts/cluster/unified_training/eval_ginsburg.sbatch
set -euo pipefail

source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/_env.sh
cd "$UT_REPO"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CONFIG="${CONFIG:-config/wb/campaign/spectral_t63.yaml}"
MODES="${MODES:-physics,neural_gcm,sfno}"
OUT="${OUT:-results/wb_campaign}"

echo "[uni-eval] $(date) config=$CONFIG modes=$MODES out=$OUT"
"$UT_PYTHON" -u scripts/run/run_weatherbench_campaign.py \
  --config "$CONFIG" --modes "$MODES" --stages eval,plot --out-root "$OUT"
echo "[uni-eval] done $(date)"
