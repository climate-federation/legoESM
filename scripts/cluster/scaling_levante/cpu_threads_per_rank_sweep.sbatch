#!/bin/bash -l
#SBATCH --job-name=cpu_thr
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=cpu_thr_%j.log
# ===========================================================================
# WHY DOES ADDING NODES MAKE THE CPU ATMOSPHERE SLOWER?
#
# THE REGRESSION. Icosahedral s9, float32, 4 ranks per node, one code
# revision (f1a38ee4), one job:
#
#     ranks     16      32      64     128
#     ms       919    1005     822    1240
#
# Going from 64 to 128 ranks -- twice the nodes -- costs 51% MORE time. The
# earlier ladder, before every rank was given its full core share, went
# 7175 -> 4176 -> 2519 -> 2223 over the same rank counts: five to eight times
# slower everywhere, but monotone. Absolute speed improved enormously and the
# SCALING regressed to negative.
#
# THE HYPOTHESIS, and it is only that. Each rank now takes its full share of
# the node, which at 4 ranks per node is 64 hardware threads, at EVERY rank
# count. That share was chosen by a sweep on ONE node with a large local
# problem (subdivision 7, 4 ranks: 4x32 threads 172 ms, 8x16 195, 16x8 199,
# 2x64 211). At 128 ranks on subdivision 9 each rank holds about 20k cells,
# so those same 64 threads have roughly 300 cells each. A thread count tuned
# where the local problem was large, then applied where it is small, is the
# shape that produces exactly this curve.
#
# TRUNCATED 2026-09-23, on its own pre-registered rule. A one-node test
# varied the MESH instead of the rank count, so cells per rank moved from
# 164k down to 2.6k -- an eighth of what a rank holds at 128 ranks -- with the
# partition and layout held fixed. More cores won at EVERY size, by a wide
# margin and monotonically (at the smallest mesh: 46.6 ms on 8 cores, 28.5 on
# 16, 19.8 on 32, 17.0 on 64). The production share was never beaten, so its
# penalty cannot rise as the local problem shrinks: the LOCAL-SIZE mechanism
# is dead.
#
# What survives is the part one node cannot see -- at 128 ranks the local
# problem is small AND the inter-node exchange is expensive at once. So this
# keeps the two shares that bracket the question and drops the rest.
#
# WHAT IS MEASURED. Step time at a FIXED 128 ranks on a FIXED 32 nodes, with
# only the per-rank core share varied: 64 (current) and 32. Two
# 16-rank arms at the current and the best-looking share are included as a
# control, because the claim is not "64 is wrong" but "the right number
# depends on the local problem size" -- if the optimum does not move between
# 16 and 128 ranks, the explanation is wrong even if the sweep finds a faster
# setting.
#
# FALSIFIABILITY, WRITTEN BEFORE SUBMIT.
#   CONFIRMS something the one-node test could not see: 32 beats 64 at 128
#     ranks by more than the repeat spread, even though more cores won at
#     every local problem size on a single node. That would mean the cost is
#     in the interaction between a small local problem and the inter-node
#     exchange, not in the thread pool alone.
#   REFUTES: 64 is fastest at 128 ranks, or every share lands inside the
#     repeat spread. Then the thread pool is not the cause and the next
#     suspects are the partition and halo at 128 ranks -- the schedule's own
#     round count, and whether the 128-rank partition has a worse maximum
#     degree than the 64-rank one.
#   UNDEFINED, and reported as such rather than forced either way: a faster
#     setting at 128 ranks whose optimum does NOT move at 16 ranks. That
#     would be a real speed-up with the stated mechanism refuted.
#
# THE KNOB IS LIVE, checked before this job was written rather than assumed.
# XLA's CPU worker pool is sized from the AFFINITY MASK, not the node: under
# taskset at 4, 8 and 32 cores the process carries exactly 4, 8 and 32
# tf_XLAEigen workers while os.cpu_count() stays 256. Each arm re-records that
# census at its own share, and a share whose worker count does not match is
# reported INVALID rather than fed into a comparison -- a sweep over a dead
# knob varies pinning alone and still moves timings, which would look like
# information.
#
# ARMS ARE INTERLEAVED, NOT GROUPED, and each share is run twice. A ladder
# run in a fixed order cannot separate a setting from a drift in the machine,
# and without repeats there is no spread to compare a difference against.
#
# NOTHING IS ADOPTED BY THIS JOB. The per-rank core share is a production
# setting; this measures it and reports. Changing it is a separate decision.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/cpu_threads_per_rank_sweep.sbatch
# ===========================================================================
set -uo pipefail
# Without this a walltime kill leaves the log looking like a
# shorter sweep that finished, rather than one that was cut off.
trap 'echo "CPU_THR_INCOMPLETE: killed at $(date +%T)"' TERM
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

PPN=4
SUB=9
OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/cpu_thr_j${SLURM_JOB_ID}}"
if [ -e "$OUTDIR" ]; then
    echo "REFUSING: $OUTDIR exists; two jobs in one directory make its rows" \
         "unattributable." >&2
    exit 1
fi
mkdir -p "$OUTDIR"
echo "provenance: $($PY -c 'import legoesm.parallel.sharded_dynamics as m; print(m.__file__)')"
echo "sha=$(git -C "$REPO" rev-parse --short HEAD) outdir=$OUTDIR"

run_arm() {   # $1=ranks  $2=cpus-per-task  $3=rep
    local nd=$1 cpt=$2 rep=$3 nn=$(( ($1 + PPN - 1) / PPN ))
    local tag="nd${nd}_cpt${cpt}_r${rep}"
    if [ "$nn" -gt "${SLURM_JOB_NUM_NODES:?}" ]; then
        echo "SKIP $tag: needs $nn nodes"; return 1
    fi
    echo "=== $tag nodes=$nn $(date +%T) ==="
    # One rank is enough to establish what the share does to the pool, and it
    # runs under the arm's own srun flags.
    srun --export=ALL --nodes=1 --ntasks=1 --cpu-bind=cores \
        --cpus-per-task="$cpt" "$PY" \
        "$REPO/scripts/bench/thread_pool_census.py" \
        "$OUTDIR/threads_cpt${cpt}.json" >/dev/null 2>&1
    timeout --kill-after=30 900 srun --export=ALL --nodes="$nn" --ntasks="$nd" \
        --ntasks-per-node="$PPN" --cpu-bind=cores --cpus-per-task="$cpt" \
        --kill-on-bad-exit=1 \
        "$PY" scripts/bench/bench_mpas_spmd_scaling.py --multicontroller \
        --n-devices "$nd" --subdivision "$SUB" --nlev 26 --lloyd 0 \
        --wide-halo --partition-method sfc --steps 12 --warmup 3 \
        --out "$OUTDIR/${tag}.jsonl"
    local rc=$?
    [ "$rc" -ne 0 ] && echo "ARM FAILED $tag rc=$rc"
    return $rc
}

# Interleaved: every share is visited once before any is revisited, so a
# machine that slows down partway through penalises all shares equally
# instead of the ones that happened to run late.
want=0; ok=0
for REP in 1 2; do
    for CPT in 64 32; do
        want=$(( want + 1 )); run_arm 128 "$CPT" "$REP" && ok=$(( ok + 1 ))
    done
    # The control: does the best share at 16 ranks differ from 128?
    for CPT in 64; do
        want=$(( want + 1 )); run_arm 16 "$CPT" "$REP" && ok=$(( ok + 1 ))
    done
done

echo "=== sweep done: $ok of $want arms ==="
if [ "$ok" -lt "$want" ]; then
    # A sweep missing settings cannot say which setting is best, and a
    # partial one read as complete is how a wrong default gets adopted.
    echo "CPU_THR_INCOMPLETE: $(( want - ok )) arm(s) missing"; exit 1
fi
$PY "$SUBMIT_DIR/scripts/bench/_summarise_thread_sweep.py" "$OUTDIR"
