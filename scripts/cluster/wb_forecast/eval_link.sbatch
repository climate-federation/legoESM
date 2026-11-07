#!/usr/bin/env bash
# Score ONE WeatherBench checkpoint (single GPU) and, optionally, dump the
# forecast and ERA5 fields the maps are drawn from.
#
#   sbatch --export="ALL,LEGOESM_REPO=$PWD,CONFIG=config/wb/campaign/spectral_t63_amip.yaml,CKPT=$PWD/results/wb_amip/epoch_0011.eqx,OUT=$PWD/results/wb_amip/eval/scorecard_epoch0011.json,FIELDS=$PWD/results/wb_amip/eval/fields_epoch0011.npz" \
#     scripts/cluster/wb_forecast/eval_link.sbatch
#
# The --export list is ONE quoted argument (see train_link.sbatch: split across
# a backslash it reaches sbatch as the script name).
#
# Env: CONFIG, CKPT, OUT (required); MODE (default physics), LEADS (default
# 24,72), YEAR (default: the deck's first eval year), N_INITS (default 8),
# FIELDS (optional .npz of the case-mean forecast + ERA5 fields, which
# scripts/plot/plot_wb_forecast_maps.py turns into maps). The deck MUST be the
# one the checkpoint was trained with: the eval builds the model from it.
#SBATCH --job-name=wbeval
#SBATCH --partition=gpu
#SBATCH --account=bb1596_gpu
#SBATCH --constraint=a100_80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=wbeval_%j.out
set -uo pipefail
export LEGOESM_REPO="${LEGOESM_REPO:-${SLURM_SUBMIT_DIR:?set LEGOESM_REPO or submit from the repo}}"
source "$LEGOESM_REPO/scripts/cluster/scaling_levante/_env.sh" || exit 1
cd "$REPO" || exit 1
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Each optional value stays ONE argument so a stray value cannot inject flags.
opt_args=()
[ -n "${YEAR:-}" ] && opt_args+=(--eval-year "$YEAR")
[ -n "${FIELDS:-}" ] && opt_args+=(--save-fields "$FIELDS")

"$PY" -u scripts/validate/run_weatherbench_eval.py \
  --config "${CONFIG:?set CONFIG}" \
  --checkpoint "${CKPT:?set CKPT}" \
  --mode "${MODE:-physics}" --training-core spectral \
  --leads "${LEADS:-24,72}" --n-inits "${N_INITS:-8}" \
  --out "${OUT:?set OUT}" \
  "${opt_args[@]}"
eval_exit=$?
echo "[wbeval] exit=$eval_exit $(date)"
exit "$eval_exit"
