#!/bin/bash -l
#SBATCH --job-name=halo_coll
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=halo_coll.%j.log
# De-risk microbench for the MPAS halo-fill collective swap (coloured
# sequential ppermute rounds -> one ragged_all_to_all), BEFORE any
# schedule-builder refactor. Synthetic ring-offset graph at production
# degree, 364 KiB/pair (~s9 halo). Falsifiability, written BEFORE
# submit:
#   CONFIRMS the lever: ragged per-fill <= 0.5x ppermute at nd=8/16
#     (grouped P2P concurrency + single launch beat rounds x latency).
#   REFUTES: ragged >= ppermute (XLA serializes the grouped sends, or
#     the zero-size slices cost more than they save).
# Offline impossible: XLA:CPU has no ragged-all-to-all thunk (verified
# 2026-08-09, "not supported by XLA:CPU ThunkEmitter").
# Each strategy VERIFIES exact against the analytic oracle in-process
# (per addressable shard) before timing; a mismatch aborts the leg.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
# Hermeticity: the overlap scheduler flags change how XLA schedules the
# two collective kinds DIFFERENTLY — pin OFF before sourcing _env.sh
# (codex round-1 MAJOR). The bench records the effective value per row.
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/halo_coll_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0

# Leg 1: nd=4 single-process (pure NVLink; uniform arm included)
srun --nodes=1 --ntasks=1 --gpus-per-node=4 --gpu-bind=none \
  "$PY" scripts/bench/bench_halo_collectives.py \
    --n-devices 4 --offsets 1,-1,2 \
    --out "$OUTDIR/nd4.jsonl" || { echo "nd4 FAILED"; rc=1; }

# Leg 2: nd=8 multicontroller (2 nodes, IB in play)
srun --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-node=4 \
    --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_halo_collectives.py \
    --multicontroller --n-devices 8 --offsets 1,-1,2,-2,3,-3 \
    --out "$OUTDIR/nd8.jsonl" || { echo "nd8 FAILED"; rc=1; }

# Leg 3: nd=16 multicontroller (4 nodes, production degree 10)
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-node=4 \
    --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_halo_collectives.py \
    --multicontroller --n-devices 16 --offsets 1,-1,2,-2,5,-5,7,-7,6,-6 \
    --out "$OUTDIR/nd16.jsonl" || { echo "nd16 FAILED"; rc=1; }

# Receipt validation bound to each leg's expected config (codex round-1:
# a validator that accepts any row validates nothing).
check_leg() {  # file leg nd offsets
  "$PY" - "$1" "$2" "$3" "$4" <<'EOF' || return 1
import json, math, sys
path, leg, nd, offs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
want_offsets = [int(t) for t in offs.split(",")]
try:
    rows = [json.loads(l) for l in open(path)]
except Exception as e:
    print(f"{leg}: MISSING/INVALID -> {e}"); sys.exit(1)
byname = {r["strategy"]: r for r in rows}
if set(byname) != {"ppermute", "ragged", "uniform"}:
    print(f"{leg}: wrong strategy set {sorted(byname)}"); sys.exit(1)
if len(rows) != 3:
    print(f"{leg}: {len(rows)} rows (stale append?)"); sys.exit(1)
for r in rows:
    ok = (r["component"] == "halo_collectives_microbench"
          and r["n_devices"] == nd and r["offsets"] == want_offsets
          and r["verified_exact"] is True
          and math.isfinite(r["per_fill_us"]) and r["per_fill_us"] > 0
          and r["env_overlap"] == "0")
    if not ok:
        print(f"{leg}: row fails binding: {r}"); sys.exit(1)
pp, rg = byname["ppermute"]["per_fill_us"], byname["ragged"]["per_fill_us"]
un = byname["uniform"]["per_fill_us"]
nr = byname["ppermute"]["n_collectives_per_fill"]
print(f"{leg}: ppermute({nr}r) {pp:9.1f} us | ragged {rg:9.1f} us | "
      f"ratio {rg/pp:.3f} | uniform {un:9.1f} us")
EOF
}
check_leg "$OUTDIR/nd4.jsonl"  nd4  4  "1,-1,2" || rc=1
check_leg "$OUTDIR/nd8.jsonl"  nd8  8  "1,-1,2,-2,3,-3" || rc=1
check_leg "$OUTDIR/nd16.jsonl" nd16 16 "1,-1,2,-2,5,-5,7,-7,6,-6" || rc=1
echo "DONE rc=$rc"; exit $rc
