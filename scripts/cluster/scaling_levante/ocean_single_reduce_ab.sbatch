#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=ocean_single_reduce_ab_%j.log

# Ocean barotropic-PCG reduction-count A/B: standard (2 reductions/iter) vs
# single_reduce Chronopoulos-Gear (1/iter). Ocean is reduction-bound, so if the
# reduction count is the scaling limiter this cuts the >=64-GPU step. f64,
# implicit-CN + fixed-iter PCG (--force-pcg) so both arms solve the SAME system.
# One JSONL per arm: $OUTDIR/ocean_pcg_{variant}_d{ND}.jsonl

set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
export JAX_ENABLE_X64=1
cd "$REPO"
OUTDIR="$SCRATCH/legoesm_scaling/ocean_single_reduce_j${SLURM_JOB_ID}"; mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
GPN=4
NODES="${SLURM_JOB_NUM_NODES:-16}"

rc=0
# run_arm <variant:standard|single_reduce> <nd>
run_arm() {
  local variant="$1" nd="$2"
  local nn=$(( (nd + GPN - 1) / GPN ))
  if (( nn > NODES )); then echo "[skip] pcg_${variant}_d${nd}: needs ${nn} nodes"; return 0; fi
  echo "=== arm ocean_pcg_${variant}_d${nd} nn=${nn} $(date '+%F %T')"
  if ! timeout --kill-after=30 900 srun --export=ALL --nodes="$nn" --ntasks="$nd" \
             --ntasks-per-node="$GPN" --gpus-per-node="$GPN" --gpu-bind=none --kill-on-bad-exit=1 \
             "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
               --multicontroller --n-devices "$nd" --mode strong \
               --n-lat 192 --n-lon 384 --nlev 20 \
               --baro-solver implicit_cn --force-pcg --pcg-fixed-iters 40 \
               --pcg-variant "$variant" \
               --steps 40 --warmup 8 --out "$OUTDIR/ocean_pcg_${variant}_d${nd}.jsonl"; then
    echo "!!! FAILED pcg_${variant}_d${nd} (continuing)"; rc=1
  fi
  sleep 5
}

# warmup discard
run_arm standard 8; rm -f "$OUTDIR/ocean_pcg_standard_d8.jsonl"

ARMS=()
add() { local v="$1"; shift; local nd; for nd in "$@"; do ARMS+=("$v" "$nd"); done; }
add standard      8 16 32 64
add single_reduce 8 16 32 64

echo "### forward"; n=$(( ${#ARMS[@]} / 2 ))
for ((i=0;i<n;i++)); do s=$((i*2)); run_arm "${ARMS[@]:s:2}"; done
echo "### reverse"
for ((i=n-1;i>=0;i--)); do s=$((i*2)); run_arm "${ARMS[@]:s:2}"; done

echo "=== DONE outdir=$OUTDIR ==="
exit "$rc"
