#!/bin/bash -l
#SBATCH --job-name=ll_trace
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --exclude=l50100,l50000
#SBATCH --output=ll_trace.%j.log
# ===========================================================================
# LAT-LON LANE ATTRIBUTION (directive order: lat-lon after MPAS).
# The lane sits at ratio ~1.4 vs its comm-model bound after fused halo
# + overlap (LL2048@64: 5.278 ms). Same instrument that cracked MPAS:
# XLA-profiler thunk timelines, ranks 0-3, one steady scan block —
# splits the step into in-collective / compute / gap and, with the
# MPAS lens, asks the two receipted questions first:
#   * padding — lat-lon pencil halos are shape-uniform across ranks
#     (regular grid), predicted inflation ~1.0; the trace's collective
#     duration vs wire estimate checks it.
#   * skew — per-pair arrival offsets (partner matching is trivial:
#     fixed neighbour pairs on a pencil grid).
# Protocol: the fused+overlap production arm of atm_ll64_combine_ab
# (LL2048@64, f32, steps 12 / warmup 3).
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll_trace_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

echo "=== LL2048@64 traced (ranks 0-3) ==="
JAX_ENABLE_X64=0 srun --ntasks=64 --ntasks-per-node=4 \
    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
    --multicontroller --n-devices 64 --mode strong \
    --n-lat 2048 --n-lon 4096 --nlev 26 --steps 12 --warmup 3 \
    --profile-dir "$OUTDIR/traces" \
    --out "$OUTDIR/traced.jsonl" || { echo "TRACED ARM FAILED"; rc=1; }

echo "=== TRACE ATTRIBUTION ==="
set +e
"$PY" scripts/bench/analyze_jax_trace_gaps.py \
  --trace-root "$OUTDIR/traces" --out "$OUTDIR/trace_summary.json"
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
