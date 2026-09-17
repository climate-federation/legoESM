#!/usr/bin/env bash
# One WeatherBench training link (single GPU). The trainer resumes from the
# newest complete epoch, so a run longer than one walltime is chained by
# resubmitting this same script with the same OUT.
#
#   sbatch --export=ALL,LEGOESM_REPO=$PWD,\
#     CONFIG=config/wb/campaign/spectral_t63.yaml,MODE=physics,OUT=$PWD/results/wb_run \
#     scripts/cluster/wb_forecast/train_link.sbatch
#
# Env: CONFIG (required), OUT (required), MODE (physics|neural_gcm|sfno,
# default physics), EPOCHS (optional). The training core is the spectral one
# that every config/wb/campaign deck selects. The multi-node data-parallel
# launcher is scripts/cluster/levante/train_wb.slurm; this is the chainable
# one-GPU link.
#SBATCH --job-name=wbtrain
#SBATCH --partition=gpu
#SBATCH --account=bd1083_gpu
#SBATCH --constraint=a100_80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=wbtrain_%j.out
set -uo pipefail
# Resolve the checkout once and hand it to _env.sh, which otherwise falls back
# to a guessed clone path and runs the wrong tree.
export LEGOESM_REPO="${LEGOESM_REPO:-${SLURM_SUBMIT_DIR:?set LEGOESM_REPO or submit from the repo}}"
source "$LEGOESM_REPO/scripts/cluster/scaling_levante/_env.sh" || exit 1
cd "$REPO" || exit 1
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# The single ~45 GiB scratch arena fragments without the async allocator
# (job 27093614 died of it while the smokes, which set it, were fine).
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# EPOCHS is optional; keep it one argument so a stray value cannot inject flags.
epoch_args=()
[ -n "${EPOCHS:-}" ] && epoch_args=(--epochs "$EPOCHS")

"$PY" -u scripts/run/train_weatherbench_scale.py \
  --config "${CONFIG:?set CONFIG}" \
  --mode "${MODE:-physics}" --training-core spectral \
  --out "${OUT:?set OUT}" --resume \
  "${epoch_args[@]}"
train_exit=$?
echo "[wbtrain] exit=$train_exit $(date)"
exit "$train_exit"
