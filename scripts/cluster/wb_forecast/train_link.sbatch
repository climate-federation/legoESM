#!/usr/bin/env bash
# One WeatherBench training link (single GPU). The trainer resumes from the
# newest complete epoch, so a run longer than one walltime is chained by
# resubmitting this same script with the same OUT.
#
#   sbatch --export="ALL,LEGOESM_REPO=$PWD,CONFIG=config/wb/campaign/spectral_t63.yaml,MODE=physics,OUT=$PWD/results/wb_run" \
#     scripts/cluster/wb_forecast/train_link.sbatch
#
# The --export list is ONE quoted argument: broken across lines with a
# backslash it reached sbatch as a second argument and sbatch took CONFIG=...
# for the script name.
#
# Env: CONFIG (required), OUT (required), MODE (physics|neural_gcm|sfno,
# default physics), EPOCHS, LR, WARMUP_STEPS, VAL_WINDOWS (optional).
# VAL_WINDOWS scores that many held-out windows from the deck's eval years after
# every epoch, so the run has a generalisation curve and not only a training
# one; it must be the SAME on every link of a chain, like the two below.
# LR/WARMUP_STEPS
# override the deck for THIS run without editing it, which is how a campaign
# arm departs from a deck that other arms share -- config/wb/campaign/
# spectral_t63.yaml is read by the column-MLP and SFNO arms too, so moving its
# lr to suit one arm silently moves theirs. Every link of one chain must pass
# the SAME values: the resume fingerprint covers the schedule, so a link that
# changes them refuses to continue from the previous link's optimizer state.
# The training core is the spectral one that every config/wb/campaign deck
# selects. The multi-node data-parallel launcher is
# scripts/cluster/levante/train_wb.slurm; this is the chainable one-GPU link.
#SBATCH --job-name=wbtrain
#SBATCH --partition=gpu
#SBATCH --account=bb1596_gpu
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

# Optional overrides; each value stays ONE argument so a stray value cannot
# inject flags.
opt_args=()
[ -n "${EPOCHS:-}" ] && opt_args+=(--epochs "$EPOCHS")
[ -n "${LR:-}" ] && opt_args+=(--lr "$LR")
[ -n "${WARMUP_STEPS:-}" ] && opt_args+=(--warmup-steps "$WARMUP_STEPS")
[ -n "${VAL_WINDOWS:-}" ] && opt_args+=(--val-windows "$VAL_WINDOWS")

"$PY" -u scripts/run/train_weatherbench_scale.py \
  --config "${CONFIG:?set CONFIG}" \
  --mode "${MODE:-physics}" --training-core spectral \
  --out "${OUT:?set OUT}" --resume \
  "${opt_args[@]}"
train_exit=$?
echo "[wbtrain] exit=$train_exit $(date)"
exit "$train_exit"
