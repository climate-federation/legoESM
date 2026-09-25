#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=atm_mpas_nccl_sweep2_%j.log
# Second NCCL sweep for the MPAS-atmosphere wide halo at the plateau, around the
# adopted 32-channel pin.  Sweep 1 (atm_mpas_nccl_sweep, 2026-09-20, 128 GPUs):
# ch4 5.94 / ch8 5.57 / ch16 5.02 / ch32 4.65 ms, ch8+512K chunk 5.27 — the
# curve was still falling at 32 channels, and NCCL 2.30 allows 64.  LL128 was a
# null at 64 GPUs (mpas_nccl_ladder_s9_64).  Every arm here = the 32-channel pin
# plus ONE change; the pin is repeated as the noise floor.
#
# PRE-REGISTERED:
#   number produced : steady step ms per arm at 128 GPUs (and 32 as the control
#                     that a plateau lever must not regress).
#   CONFIRMS        : some arm beats the ch32 pin by >5% at 128 with the repeat
#                     within 2% -> adopt (ASK first: it is a lane default).
#   REFUTES         : nothing beats the pin by >2% -> NCCL transport knobs are
#                     exhausted; the remaining atmosphere lever is bytes per
#                     exchange (fields / halo depth), not the fabric.
#   why not cheaper : env-only; the effect exists only where the wire dominates.
#
#   sbatch -p gpu --constraint=a100_80 --gpus-per-node=4 -N 32 atm_mpas_nccl_sweep2.sbatch

set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export JAX_PLATFORMS=cuda,cpu
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/atm_mpas_nccl_sweep2}"
mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
NODES="${SLURM_JOB_NUM_NODES:-1}"; PPN=4
echo "nodes=$NODES outdir=$OUTDIR sha=$(git rev-parse --short HEAD)"
rc=0

arm() {  # arm <nd> <label> <env assignments...>
  local nd="$1" label="$2"; shift 2
  local nn=$(( (nd + PPN - 1) / PPN ))
  (( nn > NODES )) && { echo "[skip] d$nd needs $nn nodes"; return 0; }
  local tag="mpas9_d${nd}_${label}"
  grep -q '"valid": true' "$OUTDIR/${tag}.jsonl" 2>/dev/null && { echo "[have] $tag"; return 0; }
  rm -f "$OUTDIR/${tag}.jsonl"
  echo "=== arm $tag $(date '+%F %T')  env: $*"
  if ! env "$@" timeout --kill-after=30 1200 srun --export=ALL --nodes="$nn" --ntasks="$nd" \
        --ntasks-per-node="$PPN" --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
        "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
        --multicontroller --n-devices "$nd" --subdivision 9 --nlev 26 --lloyd 0 \
        --wide-halo --partition-method sfc --steps 60 --warmup 10 \
        --out "$OUTDIR/${tag}.jsonl" < /dev/null; then
    echo "!!! FAILED $tag"; rc=1
    [[ -e "$OUTDIR/${tag}.jsonl" ]] && mv -f "$OUTDIR/${tag}.jsonl" "$OUTDIR/${tag}.failed.jsonl"
  fi
  sleep 5
}

CH=131072

P="NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32"
for ND in ${ND_LIST:-128 32}; do
  arm "$ND" ch32          $P NCCL_P2P_NET_CHUNKSIZE=131072                       # the pin
  arm "$ND" ch64          NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64 NCCL_P2P_NET_CHUNKSIZE=131072
  arm "$ND" ch32_ck512k   $P NCCL_P2P_NET_CHUNKSIZE=524288
  arm "$ND" ch32_buf16m   $P NCCL_P2P_NET_CHUNKSIZE=131072 NCCL_BUFFSIZE=16777216
  arm "$ND" ch32_simple   $P NCCL_P2P_NET_CHUNKSIZE=131072 NCCL_PROTO=Simple
  arm "$ND" ch32_rep      $P NCCL_P2P_NET_CHUNKSIZE=131072                       # noise floor
done
echo "=== DONE rc=$rc outdir=$OUTDIR ==="
exit "$rc"
