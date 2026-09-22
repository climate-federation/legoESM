#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=32
#SBATCH --time=02:30:00
#SBATCH --output=ocean_ch_settle_%j.log

# Settle the ocean lane's NCCL channel pin at 128 GPUs (2026-09-22).
#
# WHY.  The shared ladder pins 32 NCCL channels for BOTH unstructured-mesh
# lanes.  At 128 GPUs the atmosphere lane runs that setting fine and it is
# the fastest of {4,8,16,32} there (4.65 ms/step, job 27577271), while the
# ocean lane at the same count produced NO output and was killed three
# times: at the 1200 s arm limit (job 27582584) and twice at 2400 s
# (job 27589437, both diag arms).  The 8-channel ocean arm completes twice
# in the same allocation, 36.825 and 36.922 ms/step.
#
# Both reviewers (codex MEASURE-FIRST, GLM ACCEPT) agreed on the same
# settling job and on two corrections to the diagnosis:
#   * the NCCL bootstrap "error 6" lines in the diag log postdate the
#     SIGKILL storm and are a teardown artifact, NOT the cause;
#   * the live alternative is that the arm is merely SLOW TO COMPILE, not
#     hung.  The rank dump puts every rank past compile inside
#     execute_sharded, which argues against it, so this job kills it
#     outright by turning the persistent compile cache ON: a warm 32-ch arm
#     has ~zero compile left to blame.
# The two failing jobs shared 16 of their 32 nodes, so a sick-node /
# fabric-episode confound is NOT dead either; arm 1 is the fabric canary.
#
#   numbers it produces : steady_median_ms at 128 GPUs for the atmosphere
#                         canary and for ocean 8 / 16 / 8-repeat / 32 ch
#   CONFIRMS lane-specific runtime hang : canary completes, ocean 8 and 16
#                         complete WARM, ocean 32 still produces nothing
#                         within 3600 s with a warm cache
#   REFUTES  (it was compile) : warm ocean 32 completes
#   REFUTES  (it was the fabric) : the canary itself fails
#   pin 16 iff            : arm 3 completes AND beats arm 4 by more than
#                         the 8-vs-8 repeat spread (0.26% measured) + 1%;
#                         otherwise the ocean lane pins 8
#   why not cheaper      : the failure only appears at 128 GPUs, so it
#                         cannot be reproduced on a smaller allocation, and
#                         a single allocation is what removes the node
#                         confound that two separate jobs left open.
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ocean_ch_settle}"
mkdir -p "$OUTDIR"

# The persistent compile cache is the discriminator, so it is ON here and
# ONLY here: the ladder keeps compiling cold so its compile_ms stays
# comparable with every receipt already collected.
export JAX_COMPILATION_CACHE_DIR="$OUTDIR/xlacache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

ND=128; PPN=4; NN=$(( ND / PPN ))
echo "nodes=${SLURM_JOB_NUM_NODES:-?} nodelist=${SLURM_JOB_NODELIST:-?}"
echo "outdir=$OUTDIR sha=$(git rev-parse --short HEAD)"
rc=0

arm() {  # arm <tag> <limit_s> <bench> <extra bench args...> -- <env assignments...>
  local tag="$1" limit="$2" bench="$3"; shift 3
  local args=() envs=()
  while (( $# )); do [[ "$1" == "--" ]] && { shift; break; }; args+=("$1"); shift; done
  envs=("$@")
  rm -f "$OUTDIR/${tag}.jsonl"
  echo "=== arm $tag $(date '+%F %T')  limit=${limit}s  env: ${envs[*]}"
  local t0=$SECONDS
  if ! env "${envs[@]}" timeout --kill-after=30 "$limit" \
        srun --export=ALL --nodes="$NN" --ntasks="$ND" --ntasks-per-node="$PPN" \
             --gpus-per-node=4 --gpu-bind=none --cpu-bind=cores \
             --cpus-per-task="$(( ${SLURM_CPUS_ON_NODE:?} / PPN ))" \
             --kill-on-bad-exit=1 \
             "$PY" "$bench" --multicontroller --n-devices "$ND" \
             "${args[@]}" --out "$OUTDIR/${tag}.jsonl" < /dev/null; then
    echo "!!! FAILED $tag after $(( SECONDS - t0 ))s"; rc=1
    [[ -e "$OUTDIR/${tag}.jsonl" ]] && mv -f "$OUTDIR/${tag}.jsonl" "$OUTDIR/${tag}.failed.jsonl"
  else
    echo "--- ok $tag in $(( SECONDS - t0 ))s"
  fi
  sleep 5
}

CH=131072
OC=(--subdivision 9 --nlev 40 --lloyd 0 --block-steps 20 --blocks 3)
AT=(--subdivision 9 --nlev 26 --lloyd 0 --wide-halo --partition-method sfc --steps 60 --warmup 10)

# 1. fabric canary: the atmosphere lane at the same 32 channels that wedge
#    the ocean lane.  If THIS fails the allocation is suspect and nothing
#    below is interpretable.
arm canary_atm_ch32 900 scripts/bench/bench_mpas_spmd_scaling.py "${AT[@]}" \
    -- NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32 NCCL_P2P_NET_CHUNKSIZE=$CH
# 2. ocean baseline, cold cache: the known-good setting, and it populates
#    the compile cache the warm arms below reuse.
arm ocean_ch8_cold 1800 scripts/bench/bench_ocean_mpas_spmd_scaling.py "${OC[@]}" \
    -- NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=$CH
# 3. the decision arm.
arm ocean_ch16_warm 1800 scripts/bench/bench_ocean_mpas_spmd_scaling.py "${OC[@]}" \
    -- NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16 NCCL_P2P_NET_CHUNKSIZE=$CH
# 4. noise floor for arm 3.
arm ocean_ch8_warm_rep 1800 scripts/bench/bench_ocean_mpas_spmd_scaling.py "${OC[@]}" \
    -- NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=$CH
# 5. pure diagnosis: 32 channels with a warm cache and a long leash.
arm ocean_ch32_warm 3600 scripts/bench/bench_ocean_mpas_spmd_scaling.py "${OC[@]}" \
    -- NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32 NCCL_P2P_NET_CHUNKSIZE=$CH

echo "=== DONE rc=$rc outdir=$OUTDIR ==="
exit "$rc"
