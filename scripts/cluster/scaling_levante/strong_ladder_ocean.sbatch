#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=strong_ladder_ocean_%j.log

# Ocean-latlon SPMD strong-scaling ladder: f32 / f64 / mixed, one allocation.
# One JSONL per arm: $OUTDIR/oceanll_${res}_d${ND}_${prec}.jsonl  (path contains
# "ocean" so the plotter tags the grid ocean-latlon; precision read from the row).
# Precision: JAX_ENABLE_X64 (0/1) + LEGOESM_OCEAN_MIXED=1 for the mixed arm.
# ocean-MPAS is NOT here: it is mpi4jax rank-local (no SPMD voronoi ocean).

set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"   # sets $REPO and $PY
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
cd "$REPO"
OUTDIR="$SCRATCH/legoesm_scaling/strong_ladder_ocean_j${SLURM_JOB_ID}"; mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
GPN=4
NODES="${SLURM_JOB_NUM_NODES:-16}"
NLEV=20

rc=0
# run_arm <res:a|b> <nlat> <nd> <prec:f32|f64|mixed>
run_arm() {
  local res="$1" nlat="$2" nd="$3" prec="$4"
  local nn=$(( (nd + GPN - 1) / GPN ))
  if (( nn > NODES )); then echo "[skip] oceanll_${res} d${nd}: needs ${nn} nodes (>${NODES})"; return 0; fi
  local tag="oceanll_${res}_d${nd}_${prec}"
  unset LEGOESM_OCEAN_MIXED
  case "$prec" in
    f32)   export JAX_ENABLE_X64=0 ;;
    f64)   export JAX_ENABLE_X64=1 ;;
    mixed) export JAX_ENABLE_X64=1 LEGOESM_OCEAN_MIXED=1 ;;
  esac
  echo "=== arm ${tag} nd=${nd} nn=${nn} $(date '+%F %T')"
  if ! timeout --kill-after=30 900 srun --export=ALL --nodes="$nn" --ntasks="$nd" \
             --ntasks-per-node="$GPN" --gpus-per-node="$GPN" --gpu-bind=none \
             --kill-on-bad-exit=1 \
             "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
               --multicontroller --n-devices "$nd" --mode strong \
               --n-lat "$nlat" --n-lon $((2 * nlat)) --nlev "$NLEV" \
               --steps 40 --warmup 8 --out "$OUTDIR/${tag}.jsonl"; then
    echo "!!! FAILED ${tag} (continuing)"; rc=1
  fi
  sleep 5
}

ARMS=()
add() { local res="$1" nlat="$2"; shift 2; local nd p; for nd in "$@"; do for p in f32 f64 mixed; do ARMS+=("$res" "$nlat" "$nd" "$p"); done; done; }
add a 96  1 2 4 8 16 32 64
add b 192 8 16 32 64 128

# discarded warmup
run_arm a 96 1 f32; rm -f "$OUTDIR/oceanll_a_d1_f32.jsonl"

echo "### forward"; n=$(( ${#ARMS[@]} / 4 ))
for ((i=0;i<n;i++)); do s=$((i*4)); run_arm "${ARMS[@]:s:4}"; done
echo "### reverse"
for ((i=n-1;i>=0;i--)); do s=$((i*4)); run_arm "${ARMS[@]:s:4}"; done

echo "=== DONE outdir=$OUTDIR ==="
exit "$rc"
