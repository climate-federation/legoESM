#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=ocean_trace_attrib_%j.log

# ===========================================================================
# WHERE DOES THE OCEAN STEP GO?  Thunk-timeline attribution of the ocean
# unstructured-mesh step, ranks 0-3 (one node, shared clock).
#
# The atmosphere lane has been budgeted twice; the ocean lane never has,
# and it is the expensive one: 36.8 ms per step at 128 GPUs against the
# atmosphere's 4.65.  Its free-surface solve runs a fixed twenty
# preconditioned-conjugate-gradient iterations, each carrying two halo
# exchanges and two global reductions, so about eighty collectives sit
# inside one step.  Nobody has measured what share of the step that is.
#
#   numbers it produces : per rank, the share of the step in kernels, in
#                         collectives and in gap (launch or idle); the
#                         per-occurrence collective start spread across
#                         the four same-node ranks; and the collective
#                         duration histogram by class.
#   CONFIRMS solver-bound : collectives are at least half the step and the
#                         start spread is small next to their duration,
#                         i.e. the transport is the cost.
#   CONFIRMS skew-bound   : start spread is comparable to the duration,
#                         i.e. ranks arrive at different times and the
#                         collective is draining the imbalance.
#   CONFIRMS compute-bound: kernels dominate and the collectives are thin;
#                         the solver's iteration count is then the lever,
#                         not its transport.
#   REFUTES nothing by itself; this is an attribution, and its own
#                         calibration is that matched collective END times
#                         must nearly coincide across the four ranks.  If
#                         they do not, the clock alignment is invalid and
#                         no spread number may be quoted.
#   why not cheaper      : the split cannot be inferred from step time, and
#                         nsys silently drops the halo collectives on this
#                         stack, which cost this campaign a wrong budget
#                         once already.
#
#   ND=64  sbatch -p gpu --gpus-per-node=4 -N 16 ocean_trace_attrib.sbatch
#   ND=128 sbatch -p gpu --gpus-per-node=4 -N 32 ocean_trace_attrib.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
ENV_SH="${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
[[ -f "$ENV_SH" ]] || ENV_SH="${SUBMIT_DIR}/_env.sh"
[[ -f "$ENV_SH" ]] || { echo "cannot find _env.sh from $SUBMIT_DIR" >&2; exit 2; }
source "$ENV_SH"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ocean_trace_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR sha=$(git rev-parse --short HEAD)"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

ND="${ND:-128}"; PPN=4; NN=$(( ND / PPN ))
(( NN == ${SLURM_JOB_NUM_NODES:?} )) || {
  echo "ERROR: ND=$ND wants $NN nodes, allocation has $SLURM_JOB_NUM_NODES" >&2
  exit 2; }
# The ocean lane wedges at 32 NCCL channels beyond 64 GPUs (three arms,
# two jobs), so this capture pins the setting that is measured to complete.
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072

# The split is an attribution OF THIS transport configuration and may not be
# quoted against a receipt taken at another channel count.
env | grep -E '^(NCCL|XLA|JAX|LEGOESM)_' | sort > "$OUTDIR/capture_env.txt"

echo "=== ocean subdiv-9 at $ND GPUs, ranks 0-3 traced ==="
srun --export=ALL --nodes="$NN" --ntasks="$ND" --ntasks-per-node="$PPN" \
     --gpus-per-node=4 --gpu-bind=none --cpu-bind=cores \
     --cpus-per-task="$(( ${SLURM_CPUS_ON_NODE:?} / PPN ))" \
     --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_ocean_mpas_spmd_scaling.py \
    --multicontroller --n-devices "$ND" --subdivision 9 --nlev 40 --lloyd 0 \
    --block-steps 20 --blocks 3 \
    --profile-dir "$OUTDIR/traces" \
    --out "$OUTDIR/traced.jsonl" < /dev/null || { echo "TRACED ARM FAILED"; rc=1; }

echo "=== TRACE ATTRIBUTION ==="
set +e
# Without the schedule-matched partner map the analyzer falls back to pairing
# collectives by overlap, which pairs ranks that never exchanged anything; its
# own help says that form is not quotable. The bench writes the map next to
# the traces.
PM=("--partner-map" "$OUTDIR/traces/partner_map.json")
[[ -f "$OUTDIR/traces/partner_map.json" ]] || {
  echo "WARNING: no partner map was written; arrival-skew numbers from this"
  echo "         capture are NOT quotable. Kernel/collective/gap shares still are."
  PM=(); }
# RETRACTED: an earlier version of this comment said the refusal's printed
# counts would reveal the ocean's period. They do not. The count is degenerate
# — two fills with three reductions per step gives the same total as four
# fills with one — and the model hard-codes one reduction per step, so no
# setting of the two options here can describe an ocean step at all.
#
# The analyzer will therefore refuse, which is the correct outcome and the
# reason it is still run: the refusal is the evidence that the attribution is
# not available, and it now exits non-zero so this cannot read as success.
# Making it available needs the model driven by event NAMES rather than by a
# fixed period, which is a change to the analyzer, not a flag here.
# The census is the POINT of this capture for now: it prints the collective
# names in trace order and reports whether the sequence has an exact period.
# That is how the ocean's step gets learned rather than guessed, and it is the
# thing no amount of reasoning about the solver could supply.
"$PY" scripts/bench/analyze_jax_trace_gaps.py \
  --trace-root "$OUTDIR/traces" "${PM[@]}" --steps 60 --name-census 80 \
  --out "$OUTDIR/trace_summary.json"
# The skew half is expected to REFUSE until the model is name-driven, and it
# now exits non-zero when it does. That is the capture working, so do not let
# it mark the job failed.
rcv=$?; [ "$rcv" -eq 3 ] && { echo "(skew refused as expected; census above is
the deliverable)"; rcv=0; }
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc outdir=$OUTDIR"; exit $rc
