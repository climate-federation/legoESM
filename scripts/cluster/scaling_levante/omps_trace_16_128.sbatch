#!/bin/bash -l
#SBATCH --job-name=omps_trace
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=omps_trace_%j.log
# ===========================================================================
# WHAT GROWS IN THE OCEAN MPAS STEP FROM 16 TO 128 GPUs?
#
# Weak ladder row (20480 cells/device, 40 levels, poly-4 PCG 20 iterations,
# 8 NCCL channels): 8.3 ms at 2 GPUs -> 32.9 ms at 128 (25% weak efficiency).
# The 4-GPU trace says ~58% collectives but cannot show what grows. Same bench,
# same per-device work, kernel traces of ranks 0-3 at 16 and 128 GPUs, plus an
# untraced timing arm at each size (tracing perturbs timing).
#
# READ: analyze_jax_trace_gaps.py --top-ops / gap view at --steps = blocks x
# block-steps (60); the family whose per-step time grows most from 16 to 128
# is the lever. No verdict is pre-registered on a single family; the two sizes
# are the comparison.
# ===========================================================================
set -uo pipefail
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export LEGOESM_REPO="${LEGOESM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=0
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "$LEGOESM_REPO/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || exit 2
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
"$PY" -c 'import legoesm.parallel.voronoi_partition as m; print("provenance", m.__file__)' || { echo "PY cannot import legoesm"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/omps_trace_j${SLURM_JOB_ID}"
[ -e "$OUTDIR" ] && { echo "REFUSING: $OUTDIR exists" >&2; exit 1; }
mkdir -p "$OUTDIR"
echo "sha $(git rev-parse --short HEAD) outdir $OUTDIR"
arm() {  # arm <nd> <tag> [extra...]
  local nd="$1" tag="$2"; shift 2; local nn=$(( (nd + 3) / 4 ))
  echo "=== $tag nd=$nd $(date +%T)"
  timeout --kill-after=30 1200 srun --export=ALL --nodes="$nn" --ntasks="$nd" \
      --ntasks-per-node=4 --gpus-per-node=4 --cpu-bind=cores --cpus-per-task=64 \
      --kill-on-bad-exit=1 "$PY" scripts/bench/bench_ocean_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$nd" --mode weak --cells-per-device 20480 --nlev 40 \
      --lloyd 0 --block-steps 20 --blocks 3 --out "$OUTDIR/$tag.jsonl" "$@" \
      < /dev/null > "$OUTDIR/$tag.log" 2>&1 || echo "ARM FAILED $tag rc=$?"
  tail -n 2 "$OUTDIR/$tag.log"
}
arm 16 time_d16
arm 16 trace_d16 --profile-dir "$OUTDIR/trace_d16"
arm 128 time_d128
arm 128 trace_d128 --profile-dir "$OUTDIR/trace_d128"
echo OMPS_TRACE_DONE
