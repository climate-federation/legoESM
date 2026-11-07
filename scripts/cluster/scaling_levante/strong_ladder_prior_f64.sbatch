#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=strong_ladder_prior_f64_%j.log

# GENUINE f64 arms for the prior scaling figure's LARGE-mesh configs, to overlay
# (thin f64) on the reused prior f32 receipts (thick). Small/coarse meshes are
# EXCLUDED (they cannot strong-scale). Receipts named {series}_f64_n{ND}.jsonl
# in the SAME ladder2n dir the prior f32 receipts live in, so plot_ladder_2n
# (extended) globs both. All arms x64=1 (real fp64 STATE via the set_policy fix).

set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
cd "$REPO"
OUTDIR="${LADDER2N_DIR:-$SCRATCH/legoesm_scaling/ladder2n}"; mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_scaling/_jit_priorf64_j${SLURM_JOB_ID}"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
GPN=4
NODES="${SLURM_JOB_NUM_NODES:-32}"
export JAX_ENABLE_X64=1   # every arm is fp64

rc=0
# run_arm <series> <nd>
run_arm() {
  local series="$1" nd="$2"
  local nn=$(( (nd + GPN - 1) / GPN ))
  if (( nn > NODES )); then echo "[skip] ${series}_f64_n${nd}: needs ${nn} nodes (>${NODES})"; return 0; fi
  local out="$OUTDIR/${series}_f64_n${nd}.jsonl"
  # per-lane NCCL recipe
  unset NCCL_P2P_NET_CHUNKSIZE LEGOESM_LATLON_PACKED_EXCHANGE LEGOESM_LATLON_SPMD_FUSED_HALO
  local bench args=()
  case "$series" in
    mpas_s9)        export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
                    bench=scripts/bench/bench_mpas_spmd_scaling.py
                    args=(--subdivision 9  --nlev 26 --lloyd 0 --wide-halo --partition-method sfc) ;;
    mpas_s9_metis)  export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
                    bench=scripts/bench/bench_mpas_spmd_scaling.py
                    args=(--subdivision 9  --nlev 26 --lloyd 0 --wide-halo --partition-method metis --reorder-for "$nd") ;;
    mpas_s10)       export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
                    bench=scripts/bench/bench_mpas_spmd_scaling.py
                    args=(--subdivision 10 --nlev 26 --lloyd 0 --wide-halo --partition-method sfc) ;;
    latlon)         export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
                    export LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1
                    bench=scripts/bench/bench_atm_latlon_spmd_scaling.py
                    args=(--mode strong --n-lat 2048 --n-lon 4096 --nlev 26) ;;
    latlon_4096)    export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
                    export LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1
                    bench=scripts/bench/bench_atm_latlon_spmd_scaling.py
                    args=(--mode strong --n-lat 4096 --n-lon 8192 --nlev 26) ;;
  esac
  echo "=== arm ${series}_f64_n${nd} nn=${nn} $(date '+%F %T')"
  if ! timeout --kill-after=30 1500 srun --export=ALL --nodes="$nn" --ntasks="$nd" \
             --ntasks-per-node="$GPN" --gpus-per-node="$GPN" --gpu-bind=none --kill-on-bad-exit=1 \
             "$PY" "$bench" --multicontroller --n-devices "$nd" "${args[@]}" \
               --steps 60 --warmup 10 --out "$out"; then
    echo "!!! FAILED ${series}_f64_n${nd} (continuing)"; rc=1
  fi
  sleep 5
}

# (series, device list) matching the prior figure's ranges (LARGE meshes only).
ARMS=()
add() { local s="$1"; shift; local nd; for nd in "$@"; do ARMS+=("$s" "$nd"); done; }
add mpas_s9       1 2 4 8 16 32 64 128
add mpas_s9_metis 1 2 4 8 16 32 64 128
add mpas_s10          2 4 8 16 32 64 128
add latlon        1 2 4 8 16 32 64 128
add latlon_4096       4 8 16 32 64 128

echo "### forward"; n=$(( ${#ARMS[@]} / 2 ))
for ((i=0;i<n;i++)); do s=$((i*2)); run_arm "${ARMS[@]:s:2}"; done
echo "### reverse"
for ((i=n-1;i>=0;i--)); do s=$((i*2)); run_arm "${ARMS[@]:s:2}"; done

echo "=== DONE outdir=$OUTDIR ==="
exit "$rc"
