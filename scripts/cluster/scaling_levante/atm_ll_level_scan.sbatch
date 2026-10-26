#!/bin/bash -l
#SBATCH --job-name=ll_lev_scan
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ll_lev_scan.%j.log
# ===========================================================================
# DOES THE LEVEL-COUNT CLIFF EXIST ON THE LAT-LON LANE TOO?
#
# On the icosahedral lane, one GPU, float32, 163,842 cells, the cost per
# cell per level varies threefold with the level count and in no pattern a
# rule can express: 16, 20, 32, 36, 40 and 52 levels are cheap; 18, 21, 22
# through 31, and 34 are about three times more expensive. The step is 22.6
# ms at 30 levels and 8.5 ms at 32 -- more work, a third of the time. Byte
# alignment and cache capacity were both tested against that table and both
# refuted.
#
# Both lanes run 26 levels in production and share a compiler, so the
# obvious question is whether the lat-lon lane pays the same tax. It is
# worth asking here because that lane has a separate unexplained term: at
# 128 devices its local work is 45% above perfect scaling while at 32 it is
# 0.2% above, and nothing yet accounts for it.
#
# One GPU, no devices and no collectives, so nothing about the answer can be
# communication. One allocation, arms interleaved, two replicates.
#
# A PRIORI GATE, on cost per column per level:
#   CONFIRMS a shared cliff: 26 levels costs at least 1.5x the cheapest
#     count measured here, and the cheap and expensive sets resemble the
#     icosahedral ones. Then it is a property of the compiler or the GPU
#     rather than of either dynamical core, and the vertical-coordinate
#     warning should quote both lanes.
#   REFUTES it: the response is flat within about 20%. Then the icosahedral
#     table is specific to that core's kernels, the warning must say so, and
#     the lat-lon 45% local-work excess needs a different explanation.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/atm_ll_level_scan.sbatch
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll_lev_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
rc=0

NLAT="${NLAT:-1024}"; NLON=$((2 * NLAT))
LEVELS="${LEVELS:-16 20 22 26 30 32 36 40}"

run_lev () {  # nlev tag
  local NL=$1 TAG=$2
  echo "=== 1 GPU, LL${NLAT}x${NLON}, $NL levels [$TAG] $(date +%T) ==="
  JAX_ENABLE_X64=0 \
    timeout 1200 srun --export=ALL --nodes=1 --ntasks=1 \
      --ntasks-per-node=1 --gpus-per-node=1 --gpu-bind=none \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --n-devices 1 --mode strong --n-lat "$NLAT" --n-lon "$NLON" \
      --nlev "$NL" --steps 60 --warmup 10 \
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
    return rec["steady_median_ms"], rec["n_lat"] * rec["n_lon"]


rows = {}
for nl in LEVELS:
    try:
        a, cols = med(f"l{nl}_a")
        b, _ = med(f"l{nl}_b")
    except Exception as exc:
        print(f"ABORT: {nl} levels missing ({exc}); no verdict")
        sys.exit(4)
    ms = (a + b) / 2
    rows[nl] = (ms, abs(a - b) / ms, 1e9 * ms / cols / nl)

print(f"{'nlev':>5} {'ms/step':>9} {'ps/col/lev':>11} {'spread':>8}")
for nl in LEVELS:
    ms, spread, ps = rows[nl]
    print(f"{nl:>5} {ms:9.3f} {ps:11.0f} {spread:7.1%}")

cheapest = min(r[2] for r in rows.values())
print()
for nl in LEVELS:
    print(f"  {nl:>3}: {rows[nl][2] / cheapest:5.2f}x the cheapest measured")
ratio26 = rows[26][2] / cheapest if 26 in rows else float("nan")
spread = max(r[2] for r in rows.values()) / cheapest
print(f"\n26 levels costs {ratio26:.2f}x the cheapest; full spread "
      f"{spread:.2f}x")
if ratio26 >= 1.5:
    print("VERDICT: CONFIRMS — the lat-lon lane pays the same level-count "
          "tax, so it is not specific to one dynamical core")
elif spread <= 1.2:
    print("VERDICT: REFUTES — the lat-lon lane is flat in level count, so "
          "the icosahedral table is specific to that core's kernels and the "
          "warning must say so")
    sys.exit(6)
else:
    print("VERDICT: partial — some structure but 26 is not the expensive "
          "case here; report the table, claim nothing shared")
    sys.exit(3)
EOF
gate_rc=$?
[ "$gate_rc" -ne 0 ] && rc=$gate_rc
echo "DONE rc=$rc"; exit $rc
