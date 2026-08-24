#!/bin/bash -l
#SBATCH --job-name=rag_hang_discrim
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=rag_hang_discrim.%j.log
# ===========================================================================
# Which variable hangs the s6@8 wide+ragged parity gate (job 27039322:
# 15-min silent hang after setup, before any bench banner)?
# One-variable arms, 10-min timeout each, NCCL INIT logs on:
#   a) wide=1 ragged=1, no extra XLA flags   — the COMBINATION alone
#   b) a + multi-host decomposer flag        — MH flag
#   c) a + one-shot kernel flag              — OS flag
#   d) a + both                              — reproduction control (hung)
#   e) wide=1 ragged=0 ppermute, no flags    — parity-gate config sanity
# First arm that hangs names the trigger; if (a) hangs, the flags are
# exonerated and the wide+ragged COMBINATION is the defect (it has never
# had multi-host parity coverage). If only (d) hangs, flag interaction.
# If all pass, the 27039322 hang was node/fabric — resubmit the ladder.
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/rag_discrim_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
BASE_XLA="--xla_gpu_shard_autotuning=false"
MH="--xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer=true"
OS="--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=true"
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT

arm () { # tag ragged extra_flags
  local TAG=$1 RAG=$2 EXTRA=${3:-}
  echo "=== arm $TAG ragged=$RAG xla='$EXTRA' $(date +%T) ==="
  rm -f "$OUTDIR/${TAG}.jsonl"
  env -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS -u NCCL_P2P_NET_CHUNKSIZE \
      XLA_FLAGS="$BASE_XLA $EXTRA" \
      LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO="$RAG" \
      JAX_ENABLE_X64=0 \
    timeout 600 srun --export=ALL --nodes=2 --ntasks=8 --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices 8 --subdivision 6 --nlev 8 \
      --steps 4 --warmup 1 --lloyd 0 --partition-method sfc \
      --halo-strategy ppermute --parity-gate \
      --out "$OUTDIR/${TAG}.jsonl" \
    && echo "arm $TAG PASSED $(date +%T)" \
    || echo "arm $TAG FAILED/HUNG $(date +%T)"
  sleep 10
}

arm a_combo 1
arm b_mh 1 "$MH"
arm c_os 1 "$OS"
arm d_both 1 "$MH $OS"
arm e_ppermute 0

echo "DONE"; exit 0
