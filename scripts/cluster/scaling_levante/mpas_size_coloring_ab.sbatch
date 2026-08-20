#!/bin/bash -l
#SBATCH --job-name=szcol_ab
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --exclude=l50100,l50000
#SBATCH --output=szcol_ab.%j.log
# ===========================================================================
# SIZE-AWARE COLOURING production A/B/A2 at s9@64.
#
# Lever: LEGOESM_MPAS_SIZE_COLORING=1 — same 11 rounds, pairs grouped
# by payload size, padded wire bytes 512 -> 378 MB/fill (-26%,
# inflation 2.93 -> 2.16; offline receipt job 26856854). The lane is
# node-NIC-bandwidth-bound on the PADDED bytes (3 x 512 MB / 16 nodes
# / 7.1 ms measured in-collective = ~13.5 GB/s/node ~ HDR200
# practical), so the byte cut predicts step ~10.3 -> ~8.5 ms.
# Transfers/results are BIT-IDENTICAL (only wire grouping changes).
#
# GATE DEVIATION, stated: the offline gate said ">=30% or rework"; the
# local search converged at -26% against the size distribution's floor.
# -26% predicts a -17% step effect, 8x the 2% drift band — the A/B is
# the cheaper decider vs further objective work.
#
# A priori: CONFIRMS <= 0.88; REFUTES >= 0.97 (then bytes are not the
# binding term after all and the NIC-saturation model needs rework).
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/szcol_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

echo "=== parity gate: s6@4 size-coloring=1 ==="
LEGOESM_MPAS_SIZE_COLORING=1 LEGOESM_MPAS_WIDE_HALO=0 \
LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
  srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4 \
    --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    --multicontroller --n-devices 4 --subdivision 6 --nlev 8 \
    --steps 4 --warmup 1 --lloyd 0 --partition-method sfc \
    --halo-strategy ppermute --parity-gate \
    --out "$OUTDIR/parity.jsonl" \
  || { echo "SIZE-COLORING PARITY GATE FAILED — arms not run"; exit 5; }

run_arm () {  # sz tag
  local SZ=$1 TAG=$2
  echo "=== s9@64 size-coloring=$SZ [$TAG] ==="
  LEGOESM_MPAS_SIZE_COLORING=$SZ LEGOESM_MPAS_WIDE_HALO=0 \
  LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
    srun --nodes=16 --ntasks=64 --ntasks-per-node=4 --gpus-per-node=4 \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices 64 --subdivision 9 --nlev 26 \
      --steps 12 --warmup 3 --lloyd 0 --partition-method sfc \
      --reorder-for 64 \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
}
run_arm 0 off_a
run_arm 1 on_b
run_arm 0 off_a2

echo "=== RECEIPTS ==="
set +e
"$PY" - "$OUTDIR" <<'EOF'
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
t = {}
for tag in ("off_a", "on_b", "off_a2"):
    rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    assert len(rows) == 1, (tag, len(rows))
    r = rows[0]
    ok = (r["n_devices"] == 64 and r["platform"] == "gpu"
          and r["halo_strategy_effective"] == "ppermute"
          and not r["wide_halo"]
          and math.isfinite(r["steady_median_ms"]))
    if not ok:
        print(f"{tag}: binding fail"); sys.exit(1)
    t[tag] = r["steady_median_ms"]
    print(f"{tag}: {t[tag]:9.2f} ms")
drift = abs(t["off_a"] - t["off_a2"]) / t["off_a"]
ratio = t["on_b"] / min(t["off_a"], t["off_a2"])
print(f"A/A2 drift {100*drift:.1f}% | size-coloring/off ratio {ratio:.3f}")
if ratio <= 0.88:
    print("VERDICT: CONFIRMED — padded bytes were a binding term; land "
          "default-on + extend to s10/128 and re-plot ladders")
elif ratio >= 0.97:
    print("VERDICT: REFUTED — bytes not binding; NIC-saturation model "
          "needs rework"); sys.exit(2)
else:
    print("VERDICT: intermediate — reprice"); sys.exit(3)
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
