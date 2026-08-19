#!/bin/bash -l
#SBATCH --job-name=mpas_lev_scan
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=mpas_lev_scan.%j.log
# ===========================================================================
# HOW WIDE IS THE 26-LEVEL TRAP, AND WHAT SHAPE IS IT?
#
# Measured on one GPU, same mesh of 163,842 cells, 60 timed steps:
#   13 levels   7.500 ms      26 levels  20.245 ms      52 levels  13.820 ms
# Twice the work at 52 levels runs 32% FASTER than at 26. Neighbouring mesh
# sizes show nothing like it. Communication, ghost cells, exchange schedule
# and partitioner are all separately measured and all cleared, and removing
# every host synchronisation changes it by 1%. It is the array shape.
#
# Two mechanisms fit, and they predict DIFFERENT shapes for this scan.
#
#   Cache straddle. One state array at 163,842 cells is 16.3 MB at 26
#   levels, and the live set with the 27-level staggered partner is about
#   33 MB against this GPU's 40 MB last-level cache. That is the worst
#   place to be: too big to reuse, too small to stream. At 13 levels it
#   fits; at 52 it misses cleanly and prefetches. This predicts a BROAD,
#   SMOOTH hump centred where the live set is near cache size.
#
#   Address aliasing or memory-partition camping. A row is 4 bytes times
#   the level count: 104 bytes at 26 levels, which is 8-byte but not
#   16-byte aligned, so only every other row can be loaded as a 128-bit
#   vector; 208 bytes at 52 levels is 16-byte aligned throughout. Certain
#   strides also collapse hot lines onto few cache sets or few memory
#   channels. This predicts a SHARP, NON-MONOTONE response: 26 bad while
#   27 or 28 are fine.
#
# The scan discriminates them by shape alone, which is why it sweeps one
# level at a time through the suspect region rather than doubling.
#
# A PRIORI GATE, written before submit, on cost per cell per level:
#   SHARP (aliasing / camping): 26 levels is at least 1.5x its immediate
#     neighbours at 25 and 27. Then the fix is a padding of the level or
#     cell dimension, and it must be verified as a BAND of good paddings
#     rather than one lucky value.
#   BROAD (cache straddle): the excess spans several level counts and
#     peaks where the live set is nearest 40 MB. Then padding will not
#     help and the fix is to shrink the live set -- fewer temporaries, or
#     splitting the step.
#   NEITHER: the response is flat here, and the 26-versus-52 result came
#     from something outside this range; report that rather than a cause.
#
# One allocation, arms interleaved, so node and compilation-cache state are
# held fixed across the whole map -- the review's objection to comparing
# level counts across separate jobs.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_level_stride_scan.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/lev_scan_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
rc=0

SUBDIV="${SUBDIV:-7}"
LEVELS="${LEVELS:-22 24 25 26 27 28 30 32 40}"

run_lev () {  # nlev tag
  local NL=$1 TAG=$2
  echo "=== single GPU, subdivision $SUBDIV, $NL levels [$TAG] $(date +%T) ==="
  JAX_ENABLE_X64=0 \
    timeout 1200 srun --export=ALL --nodes=1 --ntasks=1 \
      --ntasks-per-node=1 --gpus-per-node=1 --gpu-bind=none \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices 1 --subdivision "$SUBDIV" --nlev "$NL" \
      --steps 60 --warmup 5 --lloyd 0 --partition-method sfc --reorder-for 1 \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 5
}

run_lev 26 warmup_discard
for REP in a b; do
  for NL in $LEVELS; do
    run_lev "$NL" "l${NL}_${REP}"
  done
done

echo "=== RECEIPTS ==="
if [ "$rc" -ne 0 ]; then
  echo "ABORT: an arm failed (rc=$rc); no verdict"
  exit "$rc"
fi
LEVELS="$LEVELS" "$PY" - "$OUTDIR" <<'EOF'
import json, os, sys

outdir = sys.argv[1]
LEVELS = [int(x) for x in os.environ["LEVELS"].split()]


def med(tag):
    path = os.path.join(outdir, tag + ".jsonl")
    lines = [l for l in open(path).read().strip().split("\n")
             if l.startswith("{")]
    rec = json.loads(lines[-1])
    return rec["steady_median_ms"], rec["n_cells"]


rows = {}
for nl in LEVELS:
    try:
        a, cells = med(f"l{nl}_a")
        b, _ = med(f"l{nl}_b")
    except Exception as exc:
        print(f"ABORT: {nl} levels missing ({exc}); no verdict")
        sys.exit(4)
    ms = (a + b) / 2
    rows[nl] = (ms, abs(a - b) / ms, 1e6 * ms / cells / nl, cells)

cells = rows[LEVELS[0]][3]
print(f"subdivision cells: {cells:,}")
print(f"{'nlev':>5} {'row bytes':>10} {'live MB':>9} {'ms/step':>9} "
      f"{'ps/cell/lev':>12} {'spread':>8}")
for nl in LEVELS:
    ms, spread, ns, _ = rows[nl]
    # One state array plus its staggered partner, the live set the cache
    # sees; 4 bytes per float32 value.
    live = cells * (nl + (nl + 1)) * 4 / 1e6
    print(f"{nl:>5} {4*nl:>10} {live:9.1f} {ms:9.3f} {1000*ns:12.1f} "
          f"{spread:7.1%}")

# Shape of the response decides the mechanism.
per = {nl: rows[nl][2] for nl in LEVELS}
base = min(per.values())
excess = {nl: per[nl] / base for nl in LEVELS}
print()
print("excess over the cheapest level count, per cell per level:")
for nl in LEVELS:
    bar = "#" * int(round(40 * (excess[nl] - 1)))
    print(f"  {nl:>3}: {excess[nl]:5.2f}x {bar}")

if 26 in per:
    nbrs = [per[n] for n in (25, 27) if n in per]
    if nbrs:
        ratio = per[26] / min(nbrs)
        print(f"\n26 levels is {ratio:.2f}x its immediate neighbours")
        wide = sum(1 for nl in LEVELS if excess[nl] >= 1.3)
        if ratio >= 1.5:
            print("VERDICT: SHARP — a single level count is pathological, so "
                  "the cause is stride alignment or address aliasing and a "
                  "padding of the level or cell dimension is the fix. Verify "
                  "as a BAND of good paddings, never one value.")
        elif wide >= 3:
            print("VERDICT: BROAD — the excess spans several level counts, "
                  "which is the cache-straddle signature. Padding will not "
                  "help; shrink the live set instead.")
        else:
            print("VERDICT: neither shape — the response is flat through "
                  "this range, so the 26-versus-52 result comes from outside "
                  "it. Report that, do not name a cause.")
EOF
gate_rc=$?
[ "$gate_rc" -ne 0 ] && rc=$gate_rc
echo "DONE rc=$rc"; exit $rc
