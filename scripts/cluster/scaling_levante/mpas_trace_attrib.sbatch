#!/bin/bash -l
#SBATCH --job-name=mpas_trace
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --exclude=l50100,l50000
#SBATCH --output=mpas_trace.%j.log
# ===========================================================================
# WHERE DO 10 ms GO? XLA-profiler thunk-timeline attribution of the
# s9@64 default step, ranks 0-3 (one node, shared clock).
#
# Every source-level lever shaves the same ~3% (wide halo 0.970, async
# window 0.964, graphs null, overlap null) while compute is 2% and
# bandwidth ~13%. The remaining candidates are (a) inter-rank arrival
# skew drained inside collectives vs (b) genuinely slow transport. The
# discriminator (GLM consult): per-occurrence collective START spread
# across same-node ranks vs the collective's duration —
#   spread ~= duration - wire  -> SKEW is the cost (desync/rebalance
#                                 levers next);
#   spread << duration         -> transport itself slow (NCCL pathology,
#                                 isolated nccl-tests next).
# Built-in calibration: matched collective END times must coincide.
# Also delivered per rank: gap share (launch/idle) + kernel-class mix —
# the numbers the nsys route kept silently dropping.
#
# Plus the GLM system-tail sampler: per-node GPU clock + throttle
# reasons each second — one throttled GPU caps every step (and is the
# prime suspect for today's l50000 CUBIN failures).
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/trace_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

# GPU clock/throttle sampler on every node (1 Hz, background).
srun --ntasks=16 --ntasks-per-node=1 --overlap bash -c '
  nvidia-smi --query-gpu=index,clocks.sm,clocks_throttle_reasons.active \
    --format=csv,noheader -l 1 > '"$OUTDIR"'/clocks_$(hostname).csv 2>&1 &
  echo sampler up on $(hostname); sleep 120; kill %1 2>/dev/null' &
SAMPLER_PID=$!

WIDE="${WIDE:-0}"
echo "=== s9@64 wide=$WIDE step, ranks 0-3 traced ==="
LEGOESM_MPAS_WIDE_HALO=$WIDE LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
  srun --nodes=16 --ntasks=64 --ntasks-per-node=4 --gpus-per-node=4 \
    --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    --multicontroller --n-devices 64 --subdivision 9 --nlev 26 \
    --steps 12 --warmup 3 --lloyd 0 --partition-method sfc \
    --reorder-for 64 --profile-dir "$OUTDIR/traces" \
    --out "$OUTDIR/traced.jsonl" || { echo "TRACED ARM FAILED"; rc=1; }
wait "$SAMPLER_PID" 2>/dev/null

echo "=== throttle sampler summary ==="
grep -h -v 'Not Active\|0x0000000000000000' "$OUTDIR"/clocks_*.csv \
  2>/dev/null | sort | uniq -c | sort -rn | head -10 || true
echo "(no lines above = no throttle events observed)"

echo "=== TRACE ATTRIBUTION ==="
set +e
"$PY" scripts/bench/analyze_jax_trace_gaps.py \
  --trace-root "$OUTDIR/traces" --out "$OUTDIR/trace_summary.json"
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
