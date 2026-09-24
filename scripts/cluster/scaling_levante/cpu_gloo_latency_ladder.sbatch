#!/bin/bash -l
#SBATCH --job-name=gloo_lat
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=gloo_lat_%j.log
# ===========================================================================
# HOW DO THE TWO COLLECTIVES COST GROW AS RANKS ARE ADDED, ON gloo?
#
# WHY THIS EXISTS. The CPU lanes are the worst scaling in the whole matrix.
# The icosahedral atmosphere at a FIXED 5k cells per rank costs 22.55 ms on 2
# ranks and 683.68 ms on 512 -- 30x the time for the same local work, about
# 1.53x per doubling. The same model on GPU/NCCL goes 4.09 -> 7.32 ms from 2
# to 128. Something in the CPU transport is charging per rank.
#
# A CLAIM RETRACTED BEFORE IT COST ANYTHING. The first version of this job
# argued that because the ocean MPAS CPU lane degrades at a near-identical
# 1.60x per doubling while running roughly thirty times more global reductions
# per step, reductions could be ruled out. That inference is WRONG and both
# reviewers rejected it: for costs of the form T = n * R(P), the reduction
# COUNT sets the amplitude and cancels out of the per-doubling slope entirely.
# Two lanes with thirty-fold different reduction counts are EXPECTED to share
# a slope under a reduction-dominated model. Nothing was eliminated. Worse,
# gloo's all-reduce is itself a ring algorithm upstream, so its cost grows
# with the rank count by construction -- it was the leading suspect, not the
# excluded one.
#
# So this job measures BOTH sides of the contrast instead of assuming one.
#
# WHAT IS MEASURED. scripts/bench/bench_ppermute_microbench.py, at each rank
# count, once per collective:
#   * ppermute on a ring -- the halo fill's neighbour exchange. It talks to
#     the same TWO partners at every deployment size, so growth here cannot
#     be explained by the operation having more work to do.
#   * allreduce -- the solvers' global reduction, which genuinely does have
#     more participants as ranks are added.
# Both arms run on the same ranks, the same nodes and the same hour, so their
# curves are comparable. Each verifies it actually did its job before timing
# (the ring must deliver the right partner's index; the reduction must equal
# a known sum), because a collective that silently moved nothing still times
# beautifully. Host dispatch is subtracted by the 1-rep vs n-rep difference.
#
# WHAT A RESULT BUYS, and what it does NOT. With both curves on the same
# ranks, a lane's step time can be decomposed against its own counts of each
# collective. On its own this measures the COST OF THE OPERATIONS, not the
# lane: it cannot separate fabric transport from runtime bookkeeping or
# progress handling, and a rank count is not a communication pattern -- with
# four contiguous ranks per node most ring edges stay inside a node, which
# the production halo's pattern does not guarantee. Read it as the cost curve
# of two primitives, and attribute a lane only with its own counts alongside.
#
# NO PRE-REGISTERED THRESHOLD ON A RATIO, deliberately. The earlier version
# gated on "ppermute latency grows >=1.3x per doubling"; a bar on one arm of
# a two-arm contrast just re-runs the same mistake with a number attached.
# What is registered instead is the COMPARISON: both arms, every rank count,
# reported side by side with their growth per doubling, and no causal
# statement about any lane from this job alone.
#
# WHY NOT OFFLINE. gloo's cost at 256 processes over 64 nodes is a property
# of the transport and the fabric; it cannot be computed, only measured. This
# is the compute partition, not GPU hours.
#
# LAYOUT MATCHES PRODUCTION: 4 ranks per node, each with its full core share,
# because that is what every CPU ladder arm uses.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/cpu_gloo_latency_ladder.sbatch
# ===========================================================================
set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
# _env.sh defaults REPO to /work/bd1083/$USER/legoESM, which is NOT this
# worktree; without this the job would time a different tree than the one it
# stamps a sha from.
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"   # $REPO, $PY, PYTHONPATH

PPN=4
OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/gloo_lat_j${SLURM_JOB_ID}}"
if [ -e "$OUTDIR" ]; then
    echo "REFUSING: $OUTDIR already exists; a second job writing into one" \
         "directory makes its rows unattributable." >&2
    exit 1
fi
mkdir -p "$OUTDIR"

# The probe must come from THIS tree, not from whatever an installed venv
# resolves; a sweep was once read off main by accident.
echo "provenance: $($PY -c 'import legoesm.parallel.sharded_dynamics as m; print(m.__file__)')"
echo "sha=$(git -C "$REPO" rev-parse --short HEAD) ppn=$PPN outdir=$OUTDIR"

# TRIMMED to 128 ranks on 32 nodes 2026-09-23: the 64-node request sat
# unscheduled for hours while three other jobs queued behind it, and the
# question it serves is the 64 -> 128 kink, which 8 through 128 brackets. The
# 256-rank arm was the only one needing 64 nodes.
#
# LARGEST FIRST, and the wall clock covers the worst case: twelve arms at the
# 900 s per-arm timeout is three hours, which a two-hour job cannot hold. If
# something still eats the budget, the arms lost are the cheap small ones
# that can be re-run in a single node-hour, not the 64-node arms that are the
# whole reason for the allocation.
LADDER="${LADDER:-128 64 32 16 8}"
# The mpi_* arms are the SAME ring and reduction on mpi4jax (one single-process
# JAX program per rank, no jax.distributed) -- the transport the one CPU lane
# that scales 62x actually uses. Running both transports in ONE job on the
# SAME ranks is what makes the comparison controlled.
COLLECTIVES="${COLLECTIVES:-ppermute allreduce mpi_sendrecv mpi_allreduce}"
# Caps the sweep's largest message. Both 8-rank gloo arms outran the 900 s
# per-arm cap at 8 MiB (~75 ms per message at 0.11 GB/s x 64 reps x 50
# iters) and lost their receipts; 4 MiB keeps five sizes in the bandwidth fit.
MAX_KIB="${MAX_KIB:-4096}"
# mpi4py/mpi4jax need the OpenMPI they were built against; the module system
# does not prepend it from a batch shell (memory: mpi-tests-on-levante). It is
# applied to the mpi_* arms ONLY: the gcc dir carries a libstdc++ that jaxlib
# could pick up, and the gloo arms have to stay comparable to earlier runs.
MPI_LIBS="/sw/spack-levante/gcc-13.4.0-c4mgfk/lib64:/sw/spack-levante/openmpi-4.1.2-mnmady/lib"
want=0; ok=0
for ND in $LADDER; do
    NN=$(( (ND + PPN - 1) / PPN ))
    for C in $COLLECTIVES; do
        want=$(( want + 1 ))
        if [ "$NN" -gt "${SLURM_JOB_NUM_NODES:?}" ]; then
            # Counted in `want` but not attempted: a ladder short of the rank
            # counts it names cannot answer a question about a trend across
            # them, and an arm skipped for want of nodes must not read as an
            # arm that ran.
            echo "SKIP nd=$ND $C: needs $NN nodes, have $SLURM_JOB_NUM_NODES"
            continue
        fi
        echo "=== nd=$ND $C nodes=$NN $(date +%T) ==="
        if timeout --kill-after=30 900 srun --export=ALL --nodes="$NN" \
                --ntasks="$ND" --ntasks-per-node="$PPN" --cpu-bind=cores \
                --cpus-per-task="$(( ${SLURM_CPUS_ON_NODE:?} / PPN ))" \
                --kill-on-bad-exit=1 \
                env $( [[ "$C" == mpi_* ]] && echo "LD_LIBRARY_PATH=$MPI_LIBS:${LD_LIBRARY_PATH:-}" ) \
                "$PY" "$REPO/scripts/bench/bench_ppermute_microbench.py" \
                $( [[ "$C" == mpi_* ]] || echo "--multicontroller --n-devices $ND" ) \
                --collective "$C" --max-kib "$MAX_KIB" \
                --out "$OUTDIR/${C}_nd${ND}.jsonl"; then
            ok=$(( ok + 1 ))
        else
            echo "ARM FAILED nd=$ND $C rc=$?"
        fi
    done
done

cat "$OUTDIR"/*.jsonl > "$OUTDIR/all.jsonl" 2>/dev/null
echo "=== ladder done: $ok of $want arms produced a row ==="
if [ "$ok" -lt "$want" ]; then
    echo "GLOO_LAT_INCOMPLETE: $(( want - ok )) arm(s) missing"
    exit 1
fi
$PY - "$OUTDIR/all.jsonl" "$(echo $LADDER | wc -w)" "$(echo $COLLECTIVES | wc -w)" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
expect = int(sys.argv[2]) * int(sys.argv[3])
if len(rows) != expect:
    # The shell counted arms that exited zero; this counts rows that exist.
    # A probe that returns success without appending would otherwise be
    # summarised as a complete sweep.
    sys.exit(f"GLOO_LAT_INCOMPLETE: {len(rows)} rows, expected {expect}")
by = {}
for r in rows:
    by.setdefault(r["collective"], []).append(r)
for name in sorted(by):
    rs = sorted(by[name], key=lambda r: r["n_devices"])
    print(f"\n{name}")
    print(f"{'ranks':>6} {'latency_us':>11} {'bw_GB/s':>9} {'x per doubling':>15}")
    prev = None
    for r in rs:
        g = "" if prev is None else f"{r['latency_us'] / prev:.2f}"
        print(f"{r['n_devices']:>6} {r['latency_us']:>11.2f} "
              f"{r['bandwidth_gbs']:>9.2f} {g:>15}")
        prev = r["latency_us"]
    if len(rs) >= 2:
        span = rs[-1]["latency_us"] / rs[0]["latency_us"]
        d = len(rs) - 1
        print(f"  {rs[0]['n_devices']} -> {rs[-1]['n_devices']} ranks: "
              f"x{span:.2f} over {d} doublings = x{span ** (1 / d):.2f} each")
print("\nThese are the cost curves of two primitives. Attributing a lane's "
      "step time needs the lane's own count of each, measured separately.")
print("GLOO_LAT_DONE")
PYEOF
