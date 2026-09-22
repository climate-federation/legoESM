#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=atm_mpas_nccl_sweep_%j.log

# NCCL transport sweep for the MPAS-atmosphere wide halo at the plateau.
# The attribution probe (atm_mpas_halo_attrib.sbatch, 2026-09-20) put the wire
# at 3.2 of 5.6 ms at 128 GPUs, of which 2.66 ms is proportional to payload —
# for ~3.4 MB per device per exchange, i.e. ~1.3 GB/s effective per GPU,
# ~5 GB/s per node against a 25 GB/s HDR200 link.  The bytes are moving far
# below link rate.  The ladder pins NCCL_MIN/MAX_NCHANNELS=8 and
# NCCL_P2P_NET_CHUNKSIZE=131072 (chosen at 64 GPUs in August); this sweeps
# them at the plateau, everything else fixed.
#
# PRE-REGISTERED:
#   number produced : steady step ms per arm at 32 and 128 GPUs.
#   CONFIRMS        : some arm beats the 8-channel baseline by >10% at 128.
#   REFUTES         : no arm within noise (repeat baseline) beats it by >3%;
#                     then the bytes are limited elsewhere (PCIe/NIC sharing,
#                     per-pair message size) and the lever is round count.
#   why not cheaper : env-only, but the effect only exists where the wire
#                     dominates (>=64 GPUs).
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/atm_mpas_nccl_sweep}"
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
for ND in ${ND_LIST:-128 32}; do
  arm "$ND" ch8       NCCL_MIN_NCHANNELS=8  NCCL_MAX_NCHANNELS=8  NCCL_P2P_NET_CHUNKSIZE=$CH   # ladder baseline
  arm "$ND" ch16      NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16 NCCL_P2P_NET_CHUNKSIZE=$CH
  arm "$ND" ch32      NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32 NCCL_P2P_NET_CHUNKSIZE=$CH
  arm "$ND" ch4       NCCL_MIN_NCHANNELS=4  NCCL_MAX_NCHANNELS=4  NCCL_P2P_NET_CHUNKSIZE=$CH
  arm "$ND" ch8_bigck NCCL_MIN_NCHANNELS=8  NCCL_MAX_NCHANNELS=8  NCCL_P2P_NET_CHUNKSIZE=524288
  arm "$ND" chdef     NCCL_P2P_NET_CHUNKSIZE=$CH                                            # NCCL's own channel choice
  arm "$ND" ch8_rep   NCCL_MIN_NCHANNELS=8  NCCL_MAX_NCHANNELS=8  NCCL_P2P_NET_CHUNKSIZE=$CH   # noise floor
done
echo "=== DONE rc=$rc outdir=$OUTDIR ==="
exit "$rc"
