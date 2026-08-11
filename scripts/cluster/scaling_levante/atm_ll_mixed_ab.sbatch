#!/bin/bash -l
#SBATCH --job-name=ll_mixed_ab
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50100,l50000
#SBATCH --output=ll_mixed_ab.%j.log
# ===========================================================================
# MIXED-BOUNDARY FUSED PAD production A/B/A2 at LL2048@64 (commit
# 1fd9f22f0): the PPM fold pad rides the entry exchange — 2 band
# exchanges/stage -> 1, expected 13 -> 7 collectives/step.
# A priori: CONFIRMS <= 0.90 (the ~0.8-1.0 ms class shows against the
# 6.25 ms step); REFUTES >= 0.98.
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll_mixed_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

run_arm () {  # mx tag
  local MX=$1 TAG=$2
  echo "=== LL2048@64 entry-mixed=$MX [$TAG] ==="
  LEGOESM_LATLON_ENTRY_MIXED=$MX JAX_ENABLE_X64=0 \
    srun --ntasks=64 --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices 64 --mode strong \
      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 12 --warmup 3 \
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
    r = [json.loads(l) for l in open(out / f"{tag}.jsonl")][0]
    assert r["n_devices"] == 64 and math.isfinite(r["time_per_step_ms"]), tag
    t[tag] = r["time_per_step_ms"]
    print(f"{tag}: {t[tag]:9.3f} ms")
drift = abs(t["off_a"] - t["off_a2"]) / t["off_a"]
ratio = t["on_b"] / min(t["off_a"], t["off_a2"])
print(f"A/A2 drift {100*drift:.1f}% | mixed/off ratio {ratio:.3f}")
if ratio <= 0.90:
    print("VERDICT: CONFIRMED — land default-on, retrace, re-plot")
elif ratio >= 0.98:
    print("VERDICT: REFUTED"); sys.exit(2)
else:
    print("VERDICT: intermediate — reprice"); sys.exit(3)
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
