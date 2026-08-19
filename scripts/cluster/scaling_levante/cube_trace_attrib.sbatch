#!/bin/bash -l
#SBATCH --job-name=cube_trace
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=12
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --exclude=l50100,l50000
#SBATCH --output=cube_trace.%j.log
# ===========================================================================
# CUBE LANE RE-ATTRIBUTION under the count-is-free law. The lane's
# standing claim — "81% of the closed-loop C768@24 step is no-kernel
# gaps ~155 us per sequential collective slot" (nsys, job 26829838) —
# predicts a LAUNCH floor; but the cross-lane law (MPAS wide 0.970,
# lat-lon mixed 0.983) showed collective COUNT is nearly free on this
# stack, which contradicts a per-slot floor story. Same instrument as
# the other lanes (XLA profiler, ranks 0-3, 4 steady steps): splits the
# 18.9 ms step into in-collective / compute / genuine gap — the XLA
# profiler records what nsys silently dropped.
# Protocol: the banked C768 closed-loop receipt (26826851): kt=2 @24,
# 2 ranks/node (host-residency wall), dt=30, pinned 12-node placement.
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cube_trace_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

echo "=== C768 kt2 @24 closed-loop, ranks 0-3 traced ==="
JAX_ENABLE_X64=0 srun --nodes=12 --ntasks=24 --ntasks-per-node=2 \
  --distribution=block:block --cpu-bind=cores --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt 2 --resolution 768 --nlev 60 --steps 12 --warmup 3 --dt 30 \
    --closed-loop --multicontroller \
    --profile-dir "$OUTDIR/traces" \
    --out "$OUTDIR/traced.jsonl" || { echo "TRACED ARM FAILED"; rc=1; }

echo "=== TRACE ATTRIBUTION ==="
set +e
"$PY" scripts/bench/analyze_jax_trace_gaps.py \
  --trace-root "$OUTDIR/traces" --out "$OUTDIR/trace_summary.json"
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
