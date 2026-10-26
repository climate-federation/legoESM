#!/bin/bash -l
# Does a neighbour halo exchange get slower when there are more ranks, at a
# FIXED payload per rank?
#
# The lat-lon lane's measured communication term doubled between 32 and 64
# devices while per-device halo bytes stayed constant, and neither cutting the
# message count nor cutting the bytes moved it. Both independent reviews of
# that budget said the same thing: the step-minus-no-comm difference charges
# barrier and straggler wait as well as transport, so it cannot be read as
# communication until an isolated exchange -- real payload, real neighbours,
# no model compute -- has been timed at the same rank counts.
#
# Falsifiable, written before submit. Take the fitted per-message latency and
# the time at the production payload:
#   FLAT across 32/64/128 ranks  -> the model's growth is wait inside the step,
#                                   and load balance is the lever, not messages.
#   GROWS with rank count        -> the fabric or the collective library is the
#                                   term, and topology-aware placement is.
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
# Required from --export: NDEV
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
export LEGOESM_XLA_OVERLAP=0
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false --xla_gpu_enable_nccl_comm_splitting=false"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ppermute_ladder"
mkdir -p "$OUTDIR"
# Same NCCL environment as the ladder the model curves were measured with, so
# the two are comparable rather than two different machines.
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
GPN=4; NODES=$(( (NDEV + GPN - 1) / GPN )); NTPN=$(( NDEV < GPN ? NDEV : GPN ))
echo "=== ppermute ladder ndev=$NDEV nodes=$NODES $(date +%T) git=$(git rev-parse --short HEAD) ==="
timeout 1200 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
    --ntasks-per-node="$NTPN" --gpus-per-node="$NTPN" \
    --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_ppermute_microbench.py \
    --multicontroller --n-devices "$NDEV" --n-iters 50 --n-reps 64 \
    --out "$OUTDIR/pp_n${NDEV}.jsonl" || { echo "ARM FAILED"; exit 1; }
echo "DONE"
