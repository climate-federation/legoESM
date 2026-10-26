#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=24
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=strong_ladder_cube_%j.log

# Cube tiled-SPMD strong-scaling ladder: f32 / f64, one allocation.
# n_devices = 6*kt^2 (kt=2->24, kt=3->54, kt=4->96). Fixed resolution across kt
# = strong scaling (more tiles, same global grid). Cube is analytic (no mesh
# prewarm). One JSONL per arm: $OUTDIR/cube_c${N}_kt${KT}_x${X64}.jsonl
# (path has "cube" so the plotter tags grid=cube; precision from the row).

set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
cd "$REPO"
OUTDIR="$SCRATCH/legoesm_scaling/strong_ladder_cube_j${SLURM_JOB_ID}"; mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
GPN=4
NODES="${SLURM_JOB_NUM_NODES:-24}"

rc=0
# run_arm <N-resolution> <kt> <x64>
run_arm() {
  local N="$1" kt="$2" x64="$3"
  local nd=$(( 6 * kt * kt ))
  local nn=$(( (nd + GPN - 1) / GPN ))
  if (( nn > NODES )); then echo "[skip] cube_c${N}_kt${kt}: needs ${nn} nodes (>${NODES})"; return 0; fi
  if (( N % kt != 0 )); then echo "[skip] cube_c${N}_kt${kt}: N not divisible by kt"; return 0; fi
  local tag="cube_c${N}_kt${kt}_x${x64}"
  export JAX_ENABLE_X64="$x64"
  echo "=== arm ${tag} nd=${nd} nn=${nn} $(date '+%F %T')"
  if ! timeout --kill-after=30 1200 srun --export=ALL --nodes="$nn" --ntasks="$nd" \
             --ntasks-per-node="$GPN" --gpus-per-node="$GPN" --gpu-bind=none \
             --kill-on-bad-exit=1 \
             "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
               --multicontroller --kt "$kt" --resolution "$N" --nlev 26 \
               --steps 40 --warmup 8 --out "$OUTDIR/${tag}.jsonl"; then
    echo "!!! FAILED ${tag} (continuing)"; rc=1
  fi
  sleep 5
}

ARMS=()
add() { local N="$1"; shift; local kt x; for kt in "$@"; do for x in 0 1; do ARMS+=("$N" "$kt" "$x"); done; done; }
add 96  2 3 4
add 192 2 3 4

# warmup discard
run_arm 96 2 0; rm -f "$OUTDIR/cube_c96_kt2_x0.jsonl"

echo "### forward"; n=$(( ${#ARMS[@]} / 3 ))
for ((i=0;i<n;i++)); do s=$((i*3)); run_arm "${ARMS[@]:s:3}"; done
echo "### reverse"
for ((i=n-1;i>=0;i--)); do s=$((i*3)); run_arm "${ARMS[@]:s:3}"; done

echo "=== DONE outdir=$OUTDIR ==="
exit "$rc"
