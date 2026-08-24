#!/bin/bash -l
#SBATCH --job-name=mpas_size_curve
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=mpas_size_curve.%j.log
# ===========================================================================
# IS THE PER-CELL COST OF THE LOCAL STEP FLAT IN MESH SIZE?
#
# Everything else is now measured. On the icosahedral lane, communication
# is 0.04-0.11 ms of the step at 8, 16 and 32 GPUs -- zero. The ghost ring
# and the exchange schedule are on trend at every device count. Yet the
# cost per local cell per step is
#
#     1 GPU  66.3 ns    8 GPUs  62.2 ns    16 GPUs  97.8 ns    32 GPUs  67.9 ns
#
# and the graph partitioner, which buys 19.3% at 8 GPUs, buys only 5.5% at
# 16 -- so it is not the partition either.
#
# The remaining explanation that costs nothing to test: the local kernels
# themselves are not linear in problem size. This job measures exactly
# that, on ONE GPU, with no devices, no collectives and no partition in
# the picture: the same step at four mesh sizes.
#
#   subdivision 6      40,962 cells
#   subdivision 7     163,842 cells   <- close to the 16-GPU local region
#   subdivision 8     655,362 cells
#   subdivision 9   2,621,442 cells   <- the ladder's single-GPU point
#
# A PRIORI GATE (written before submit). Take nanoseconds per cell per
# step at each size.
#   CONFIRMS a kernel-size effect: the curve is non-monotonic, and in
#     particular subdivision 7 costs at least 25% more per cell than both
#     of its neighbours -- mirroring what the 16-GPU configuration shows
#     at a similar local size. The defect is then in the compiled kernels
#     at that shape and is reproducible on a single GPU, which makes it
#     cheap to chase.
#   REFUTES it: per-cell cost is flat or monotonically decreasing with
#     size. The 16-GPU penalty is then NOT a local-size effect, and the
#     next instrument is a kernel-level profile diff at 16 versus 32
#     devices -- which is now meaningful, because the collectives a
#     profiler cannot see cost nothing on this lane.
#
# Same build, same flags and the same 60-step protocol as the ladder.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_single_gpu_size_curve.sbatch
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/size_curve_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
rc=0

# SCAN=1 times the steady window with NO per-step host round-trip: the
# steps are dispatched back to back with one sync at the end. Against the
# default per-step loop it separates GPU work from host dispatch and
# per-step synchronisation, which is the remaining candidate for the
# per-cell penalty at 163,842 cells.
SCAN_FLAG=""
[ "${SCAN:-0}" = "1" ] && SCAN_FLAG="--timed-scan"
# Vertical levels. Sweeping this separates a per-CELL cost -- neighbour
# lookups over the unstructured connectivity, which do not care how many
# levels ride on each cell -- from a per-ELEMENT cost in the dense column
# work. If the penalty at 163,842 cells is the same fraction at every
# level count it is per-cell; if it moves with the level count it is the
# array shape and the compiler's tiling of it.
NLEV="${NLEV:-26}"

run_size () {  # subdivision tag
  local SD=$1 TAG=$2
  echo "=== single GPU, subdivision $SD scan=${SCAN:-0} [$TAG] $(date +%T) ==="
  JAX_ENABLE_X64=0 \
    timeout 2400 srun --export=ALL --nodes=1 --ntasks=1 \
      --ntasks-per-node=1 --gpus-per-node=1 \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices 1 --subdivision "$SD" --nlev "$NLEV" \
      --steps 60 --warmup 5 --lloyd 0 --partition-method sfc \
      --reorder-for 1 $SCAN_FLAG \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 10
}

# Discarded warm-up arm, then each size twice, interleaved, so a drift in
# the node cannot be read as a size effect.
echo "nlev=$NLEV scan=${SCAN:-0}"
run_size 7 warmup_discard
for REP in a b; do
  for SD in 6 7 8 9; do
    run_size "$SD" "s${SD}_${REP}"
  done
done

echo "=== RECEIPTS ==="
if [ "$rc" -ne 0 ]; then
  echo "ABORT: an arm failed (rc=$rc); no verdict"
  exit "$rc"
fi
"$PY" - "$OUTDIR" <<'EOF'
import json, os, sys

outdir = sys.argv[1]
SIZES = (6, 7, 8, 9)


def med(tag):
    path = os.path.join(outdir, tag + ".jsonl")
    lines = [l for l in open(path).read().strip().split("\n")
             if l.startswith("{")]
    rec = json.loads(lines[-1])
    return rec["steady_median_ms"], rec["n_cells"]


rows = {}
for sd in SIZES:
    try:
        a_ms, cells = med(f"s{sd}_a")
        b_ms, _ = med(f"s{sd}_b")
    except Exception as exc:
        print(f"ABORT: subdivision {sd} missing ({exc}); no verdict")
        sys.exit(4)
    ms = (a_ms + b_ms) / 2
    rows[sd] = (cells, ms, abs(a_ms - b_ms) / ms, 1e6 * ms / cells)

import os as _os
print(f"mode: {'async scan (no per-step host sync)' if _os.environ.get('SCAN') == '1' else 'per-step host loop'}"
      f", nlev {_os.environ.get('NLEV', '26')}")
print(f"{'subdiv':>7} {'cells':>12} {'ms/step':>9} {'ns/cell':>9} {'spread':>8}")
for sd in SIZES:
    cells, ms, spread, ns = rows[sd]
    print(f"{sd:>7} {cells:>12,} {ms:9.3f} {ns:9.1f} {spread:7.1%}")

ns6, ns7, ns8, ns9 = (rows[sd][3] for sd in SIZES)
neighbours = min(ns6, ns8)
print()
print(f"subdivision 7 costs {ns7 / neighbours:.2f}x the cheaper of its "
      f"neighbours per cell")
if ns7 >= 1.25 * neighbours:
    print("VERDICT: CONFIRMS — the local kernels are not linear in problem "
          "size, and the penalty lands near the local size the 16-GPU "
          "configuration holds. Reproducible on one GPU.")
elif ns7 <= 1.05 * neighbours:
    print("VERDICT: REFUTES — per-cell cost is flat in size, so the 16-GPU "
          "penalty is not a local-size effect; profile the kernels at 16 "
          "versus 32 devices next")
    sys.exit(6)
else:
    print("VERDICT: intermediate — a size effect exists but is too small to "
          "carry the 47% penalty on its own")
    sys.exit(3)
EOF
gate_rc=$?
[ "$gate_rc" -ne 0 ] && rc=$gate_rc
echo "DONE rc=$rc"; exit $rc
