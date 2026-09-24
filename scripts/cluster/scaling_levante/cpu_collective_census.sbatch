#!/bin/bash -l
#SBATCH --job-name=cpu_census
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=cpu_census_%j.log
# ===========================================================================
# HOW MANY OF EACH COLLECTIVE DOES A CPU STEP ACTUALLY CONTAIN?
#
# This is the other half of the CPU attribution. The companion job
# (cpu_gloo_latency_ladder.sbatch) measures what one neighbour exchange and
# one global reduction COST at each rank count. Neither answers anything
# alone: a cost curve without counts cannot say which operation dominates a
# lane, and counts without costs cannot either. Together they decompose a
# lane's step time as
#
#     T(P)  ~=  N_reduce(P) * c_reduce(P)  +  N_halo(P) * c_halo(P)  +  local
#
# and every term on the right is then measured rather than assumed.
#
# WHAT IT IS FOR. The icosahedral atmosphere at a FIXED 5k cells per rank
# costs 22.55 ms on 2 ranks and 683.68 ms on 512 -- 30x for the same local
# work. The equivalent GPU lane goes 4.09 -> 7.32 ms from 2 to 128. If the
# counts on the left of the CPU range are the same as on the right, the
# growth is entirely in the per-operation cost; if they grow, part of it is
# the model asking for more communication and the transport is being blamed
# for it.
#
# EXECUTED COUNTS, NOT STATIC ONES. The first version of this job counted
# collectives in the compiled module via an XLA dump, and both reviewers
# rejected it for the same reason: a collective inside a loop body appears
# ONCE in the module and runs as many times as the loop iterates. The ocean
# barotropic solver runs twenty iterations per step, so a static count would
# have understated its reductions by twenty and the decomposition would have
# blamed the halo.
#
# So the counts come from a PROFILER CAPTURE, where each event is one
# EXECUTION. Verified on this stack before writing this job: a program doing
# 3 steps x 5 loop iterations x 4 devices of ppermute and psum records exactly
# 60 events of each. Dividing by devices and steps gives executions per step.
#
# That required a fix to the analyzer, which is part of this change. CPU
# captures name collectives with the JAX primitive names (psum, pmax,
# all_gather) while GPU captures use HLO names (all-reduce, collective-
# permute). Only ppermute overlapped, and the name pattern rejected the
# underscore in all_gather outright -- so every CPU capture reported ZERO
# reductions, silently, which is precisely the reading this job exists to
# test. An unknown name does not raise; it reads as a confident zero.
#
# A CPU CAPTURE IS THE RIGHT INSTRUMENT HERE, and the note in the GPU census
# that "a CPU proxy gives the wrong count" does not apply: that warns against
# using CPU counts to describe a GPU executable, which combines collectives
# differently (396 vs 123 on the cube lane, 3.2x). The lane under attribution
# here IS the CPU one.
#
# THE POINTS ARE THE WEAK-SCALING POINTS BEING EXPLAINED, exactly: 5120 cells
# per rank at every rank count, which for the icosahedral mesh is subdivision
# 6 at 8 ranks, 7 at 32 and 8 at 128. Censusing a different shape would
# describe a model nobody ran.
#
# NO THRESHOLD IS PRE-REGISTERED because this job makes no causal claim on its
# own. What is registered is the table: each lane, each rank count, the count
# of each collective family, reported beside the cost curves and not before
# them.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/cpu_collective_census.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

# Pin the collective-affecting flag: an inherited overlap setting enables
# pipelined p2p and changes the very count being read.
export LEGOESM_XLA_OVERLAP=0

PPN=4
OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/cpu_census_j${SLURM_JOB_ID}}"
if [ -e "$OUTDIR" ]; then
    echo "REFUSING: $OUTDIR already exists; two jobs in one directory make" \
         "its modules unattributable." >&2
    exit 1
fi
mkdir -p "$OUTDIR"
echo "provenance: $($PY -c 'import legoesm.parallel.sharded_dynamics as m; print(m.__file__)')"
echo "sha=$(git -C "$REPO" rev-parse --short HEAD) outdir=$OUTDIR"

# Every rank captures: the count that matters is per rank, and a capture
# from rank 0 alone cannot show whether ranks differ in how much they
# communicate. Traces land on shared scratch, not node-local /tmp, or they
# vanish with the allocation.
STEPS=6

CELLS_PER_RANK=5120
# (rank count, icosahedral subdivision) pairs that give exactly 5120 cells per
# rank: 10*4^s+2 cells over P ranks.
# Overridable so the SAME instrument can take a strong-scaling pair. The
# default is the weak ladder (constant cells per rank); the CPU regression
# needs 64 and 128 ranks on ONE mesh, which is ARMS="64:9 128:9".
ARMS="${ARMS:-8:6 32:7 128:8}"
want=0; ok=0

run_arm() {   # $1=lane  $2=nd  $3=subdivision
    local lane=$1 nd=$2 sub=$3 nn=$(( ($2 + PPN - 1) / PPN ))
    want=$(( want + 1 ))
    if [ "$nn" -gt "${SLURM_JOB_NUM_NODES:?}" ]; then
        echo "SKIP $lane nd=$nd: needs $nn nodes, have $SLURM_JOB_NUM_NODES"
        return
    fi
    local tag="${lane}_nd${nd}" trace="$OUTDIR/trace_${lane}_nd${nd}"
    mkdir -p "$trace"
    local args
    if [ "$lane" = atm ]; then
        args=(scripts/bench/bench_mpas_spmd_scaling.py --multicontroller
              --n-devices "$nd" --subdivision "$sub" --nlev 26 --lloyd 0
              --wide-halo --partition-method sfc
              --steps "$STEPS" --warmup 2 --profile-dir "$trace")
    else
        # The barotropic solver's settings are SPELLED OUT, not inherited: its
        # iteration count is most of the ocean step and it has changed once
        # already, so a capture at a different setting would count a different
        # model's collectives. 20 iterations with the degree-4 polynomial
        # preconditioner is what the plotted rows are pinned to.
        args=(scripts/bench/bench_ocean_mpas_spmd_scaling.py --multicontroller
              --n-devices "$nd" --nlev 40 --lloyd 0 --mode weak
              --cells-per-device "$CELLS_PER_RANK"
              --block-steps "$STEPS" --blocks 1
              --pcg-fixed-iters 20 --pcg-precond poly --pcg-poly-sweeps 4
              --profile-dir "$trace")
    fi
    echo "=== $tag nodes=$nn $(date +%T) ==="
    if ! timeout --kill-after=30 1200 srun --export=ALL --nodes="$nn" \
            --ntasks="$nd" --ntasks-per-node="$PPN" --cpu-bind=cores \
            --cpus-per-task="$(( ${SLURM_CPUS_ON_NODE:?} / PPN ))" \
            --kill-on-bad-exit=1 \
            "$PY" "${args[@]}" --out "$OUTDIR/${tag}.jsonl"; then
        echo "ARM FAILED $tag rc=$?"
        return
    fi
    # --expect-ranks makes a capture that lost ranks refuse rather than report
    # the survivors' counts as the lane's.
    if ! $PY scripts/bench/analyze_jax_trace_gaps.py --trace-root "$trace" \
             --time-by-family --steps "$STEPS" --expect-ranks "$nd" \
             --out "$OUTDIR/census_${tag}.json"; then
        echo "CENSUS FAILED $tag: the run finished but its capture could not" \
             "be counted, so this arm contributes nothing."
        return
    fi
    # KNOWN ANSWER, checked in the job rather than trusted. The ocean lane is
    # pinned to twenty fixed barotropic iterations above, each of which ends
    # in a global reduction, so its capture must find AT LEAST twenty per
    # step. Anything less means the capture is undercounting -- which is the
    # exact failure this census was rebuilt to avoid, and it would land as a
    # confident table rather than an error.
    if [ "$lane" = oce ]; then
        if ! $PY - "$OUTDIR/census_${tag}.json" "$STEPS" <<'FLOOR'; then
import json, sys
d = json.load(open(sys.argv[1]))
steps = int(sys.argv[2])
per_step = d.get("families", {}).get("all-reduce", {}).get("instructions", 0) / steps
floor = 20.0
print(f"[floor] ocean reductions per step per rank: {per_step:.1f} "
      f"(pinned solver does {floor:.0f})")
if per_step < floor:
    sys.exit(f"CENSUS UNDERCOUNTS: {per_step:.1f} < {floor:.0f} reductions "
             f"per step. The solver is pinned to {floor:.0f} fixed "
             f"iterations, so this capture is not seeing every execution.")
FLOOR
            return
        fi
    fi
    ok=$(( ok + 1 ))
}

for pair in $ARMS; do
    run_arm atm "${pair%%:*}" "${pair##*:}"
    run_arm oce "${pair%%:*}" "${pair##*:}"
done

echo "=== census done: $ok of $want arms counted ==="
if [ "$ok" -lt "$want" ]; then
    # A census missing rank counts cannot say whether counts GROW with ranks,
    # which is the only question it was submitted to answer.
    echo "CPU_CENSUS_INCOMPLETE: $(( want - ok )) arm(s) missing"
    exit 1
fi
$PY "$SUBMIT_DIR/scripts/bench/_summarise_cpu_census.py" "$OUTDIR"
