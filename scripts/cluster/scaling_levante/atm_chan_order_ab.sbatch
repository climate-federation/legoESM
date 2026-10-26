#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=32
#SBATCH --time=02:30:00
#SBATCH --output=atm_chan_order_ab_%j.log

# Does 64 NCCL channels really beat 32 on the atmosphere lane at 128 GPUs, or
# did the sweep's fixed arm order hand the later arm a warmer machine?
#
# The sweep that raised the question (job 27623741) ran 32 channels first and
# 64 second, every time, so channel count was confounded with elapsed time.
# It also lost its repeat arm to the walltime, and the noise estimate I put in
# its place was circular: three arms that differ in a knob which "did nothing"
# are only a noise estimate if the knob provably reaches the executed path,
# and their 0.01 ms range is one display rounding increment, not a spread.
# Both reviewers rejected that substitution and asked for reversed order.
#
# So: four arms, ALTERNATING and starting with the CANDIDATE, which is the
# order that disadvantages it if drift is what produced the effect.
#
#   numbers it produces : steady median ms/step for 64 and 32 channels, twice
#                         each, interleaved, one allocation
#   CONFIRMS            : both 64-channel arms beat both 32-channel arms, and
#                         the gap exceeds the within-setting spread by 3x
#   REFUTES             : the settings overlap, or the ordering explains the
#                         gap (each setting's second arm faster than its
#                         first by about the between-setting gap)
#   why not cheaper     : the confound is in the ORDER, so it can only be
#                         removed by running the order the other way round in
#                         the same allocation. No offline test sees it.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export JAX_PLATFORMS=cuda,cpu
ENV_SH="${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
[[ -f "$ENV_SH" ]] || ENV_SH="${SUBMIT_DIR}/_env.sh"
[[ -f "$ENV_SH" ]] || { echo "cannot find _env.sh from $SUBMIT_DIR" >&2; exit 2; }
source "$ENV_SH"
cd "$REPO"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/atm_chan_order_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"
ND="${ND:-128}"; PPN=4; NN=$(( ND / PPN ))
(( NN == ${SLURM_JOB_NUM_NODES:?} )) || {
  echo "ERROR: ND=$ND wants $NN nodes, allocation has $SLURM_JOB_NUM_NODES" >&2
  exit 2; }
echo "outdir=$OUTDIR nd=$ND sha=$(git rev-parse --short HEAD)"
rc=0

arm() {  # arm <label> <channels>
  local label="$1" ch="$2"
  local tag="mpas9_d${ND}_${label}"
  rm -f "$OUTDIR/${tag}.jsonl"
  echo "=== arm $tag $(date '+%F %T') channels=$ch"
  if ! env NCCL_MIN_NCHANNELS="$ch" NCCL_MAX_NCHANNELS="$ch" \
           NCCL_P2P_NET_CHUNKSIZE=131072 \
       timeout --kill-after=30 1500 \
       srun --export=ALL --nodes="$NN" --ntasks="$ND" --ntasks-per-node="$PPN" \
            --gpus-per-node=4 --gpu-bind=none --cpu-bind=cores \
            --cpus-per-task="$(( ${SLURM_CPUS_ON_NODE:?} / PPN ))" \
            --kill-on-bad-exit=1 \
       "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
         --multicontroller --n-devices "$ND" --subdivision 9 --nlev 26 \
         --lloyd 0 --wide-halo --partition-method sfc --steps 60 --warmup 10 \
         --out "$OUTDIR/${tag}.jsonl" < /dev/null; then
    echo "!!! FAILED $tag"; rc=1
  fi
  sleep 5
}

# Candidate FIRST, then alternating: if a warming machine is what produced the
# effect, this order works against the candidate and the effect disappears.
arm ch64_a 64
arm ch32_a 32
arm ch64_b 64
arm ch32_b 32

echo "=== DONE rc=$rc outdir=$OUTDIR ==="
exit $rc
