#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=strong_ladder_%j.log

# Strong-scaling ladder: lat-lon + MPAS, float32/float64, one 32-node allocation.
# One JSONL per arm: $OUTDIR/${grid}_${res}_d${ND}_x${X64}.jsonl
# Precision is set by JAX_ENABLE_X64 (0/1); the benches set_policy from it so
# the fp64 arm is real fp64 STATE (not just an fp64 label on fp32 state).

set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"   # sets $REPO and $PY
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
cd "$REPO"

OUTDIR="$SCRATCH/legoesm_scaling/strong_ladder_j${SLURM_JOB_ID}"; mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
GPN=4
NODES="${SLURM_JOB_NUM_NODES:-32}"

rc=0

# run_arm <grid:latlon|mpas> <res:a|b> <nlat|sub> <nd> <x64>
run_arm() {
  local grid="$1" res="$2" nrs="$3" nd="$4" x64="$5"
  local nn=$(( (nd + GPN - 1) / GPN ))
  if (( nn > NODES )); then
    echo "[skip] ${grid}_${res} d${nd}: needs ${nn} nodes (>${NODES})"
    return 0
  fi
  local tag="${grid}_${res}_d${nd}_x${x64}"
  local prec=float32; [[ "$x64" == 1 ]] && prec=float64
  export JAX_ENABLE_X64="$x64"

  # Per-lane env: pinned per grid, nothing inherited between arms.
  unset NCCL_P2P_NET_CHUNKSIZE LEGOESM_LATLON_PACKED_EXCHANGE \
        LEGOESM_LATLON_SPMD_FUSED_HALO LEGOESM_MESH_CACHE_DIR \
        NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
  local bench
  local args=(--multicontroller --n-devices "$nd")
  if [[ "$grid" == latlon ]]; then
    export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
    export LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1
    bench="scripts/bench/bench_atm_latlon_spmd_scaling.py"
    args+=(--mode strong --n-lat "$nrs" --n-lon $((2 * nrs)) --nlev 26)
  else
    export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
    export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    bench="scripts/bench/bench_mpas_spmd_scaling.py"
    args+=(--subdivision "$nrs" --nlev 26 --lloyd 0 --wide-halo --partition-method sfc)
  fi
  args+=(--steps 60 --warmup 10 --out "$OUTDIR/${tag}.jsonl")

  echo "=== arm ${tag} prec=${prec} nd=${nd} nn=${nn} $(date '+%F %T')"
  if ! timeout --kill-after=30 900 srun --export=ALL --nodes="$nn" --ntasks="$nd" \
             --ntasks-per-node="$GPN" --gpus-per-node="$GPN" --gpu-bind=none \
             --kill-on-bad-exit=1 \
             "$PY" "$bench" "${args[@]}"; then
    echo "!!! FAILED ${tag} (continuing)"
    rc=1
  fi
  sleep 5
}

# Matrix as flattened 5-tuples (grid res nlat|sub nd x64).
ARMS=()
add_grid() {  # add_grid <grid> <res> <nlat|sub> <nd...>
  local grid="$1" res="$2" nrs="$3"; shift 3
  local nd x64
  for nd in "$@"; do
    for x64 in 0 1; do
      ARMS+=("$grid" "$res" "$nrs" "$nd" "$x64")
    done
  done
}
add_grid latlon a 512  1 2 4 8 16 32 64
add_grid latlon b 2048 8 16 32 64 128
add_grid mpas    a 6    1 2 4 8 16 32 64
add_grid mpas    b 9    8 16 32 64 128

# Discarded warmup arm (JIT/NCCL warm-up only).
run_arm latlon a 512 1 0
rm -f "$OUTDIR/latlon_a_d1_x0.jsonl"

echo "### forward pass"
n=$(( ${#ARMS[@]} / 5 ))
for ((i = 0; i < n; i++)); do
  s=$((i * 5)); run_arm "${ARMS[@]:s:5}"
done

echo "### reverse pass"
for ((i = n - 1; i >= 0; i--)); do
  s=$((i * 5)); run_arm "${ARMS[@]:s:5}"
done

echo "=== DONE outdir=$OUTDIR ==="
exit "$rc"
