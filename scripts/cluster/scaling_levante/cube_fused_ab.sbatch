#!/bin/bash -l
#SBATCH --job-name=cube_fused
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=12
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50100
#SBATCH --output=cube_fused.%j.log
# Fused wave-A halo for the TILED cube lane (LEGOESM_CUBE_TILED_FUSED_HALO=1):
# ONE exchange per dtype group instead of per-field — 8->3 exchange
# calls/stage on the dry-sigma step. CPU receipts: bit-identical outputs
# + >=1.5x CP cut (test_fused_halo_bit_identical_and_fewer_collectives).
# Priced: collectives = ~19% of the C768@24 closed-loop step (18.93 ms,
# job 26826851, 2rpn dt30 — THIS job matches that protocol exactly).
# Falsifiability BEFORE submit, ratio = t(fused)/min(t(off), t(off2)):
#   CONFIRMS: <= 0.90 (a >=10% step win; the 19% collective share can
#     yield at most ~13% if the cut is 2.7x and latency-bound).
#   REFUTES: >= 0.98 (collectives already hidden/overlapped; lever dead
#     on GPU despite the CP cut).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cube_fused_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

run_arm () {  # env tag
  local FH=$1 TAG=$2
  echo "=== C768 kt2 @24 closed, fused=$FH [$TAG] ==="
  LEGOESM_CUBE_TILED_FUSED_HALO=$FH JAX_ENABLE_X64=0 \
    srun --nodes=12 --ntasks=24 --ntasks-per-node=2 \
      --distribution=block:block --cpu-bind=cores --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
      --kt 2 --resolution 768 --nlev 60 --dt 30 --steps 12 --warmup 3 \
      --closed-loop --multicontroller \
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
t, cps = {}, {}
for tag in ("off_a", "on_b", "off_a2"):
    rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows"); sys.exit(1)
    r = rows[0]
    ok = (r["n_devices"] == 24 and r.get("closed_loop") is True
          and math.isfinite(r["steady_median_ms"])
          and r["steady_median_ms"] > 0)
    if not ok:
        print(f"{tag}: binding fail"); sys.exit(1)
    t[tag] = r["steady_median_ms"]
    cps[tag] = r.get("hlo_collective_permutes")
    print(f"{tag}: {t[tag]:8.2f} ms  CPs={cps[tag]}")
# independent evidence: fused module must hold fewer CPs
if cps["on_b"] is not None and cps["off_a"] is not None:
    if not cps["on_b"] < cps["off_a"]:
        print(f"CP census does not show the cut: {cps}"); sys.exit(1)
drift = abs(t["off_a"] - t["off_a2"]) / t["off_a"]
ratio = t["on_b"] / min(t["off_a"], t["off_a2"])
print(f"A/A2 drift {100*drift:.1f}% | fused/off ratio {ratio:.3f}")
if ratio <= 0.90:
    print("VERDICT: CONFIRMED")
elif ratio >= 0.98:
    print("VERDICT: REFUTED"); sys.exit(2)
else:
    print("VERDICT: intermediate"); sys.exit(3)
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
