#!/bin/bash -l
#SBATCH --job-name=scan_ab
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --exclude=l50100
#SBATCH --output=scan_ab.%j.log
# ===========================================================================
# HOST-SYNC/PIPELINING discriminator: per-step Python loop + block_until_ready vs ONE jit(lax.scan) window (s9@64).
#
# Lever: LEGOESM_MPAS_WIDE_HALO=1 — ONE depth-9 halo fill per step,
# whole SSP-RK3 body inside shard_map, 33 -> 11 sequential collectives
# per step. Offline part 1 (job 26845329): depth-9 comm graph colours
# to the SAME 11 rounds, local extents only +17%. Consult verdicts
# (GLM-5.2 + codex, 2026-08-10) rank this the top structural lever on
# the latency-floor-bound lane; step model predicts ~4.9-6 ms vs the
# 10.2 ms baseline.
#
# Falsifiability BEFORE submit (a priori gate):
#   CONFIRMS: wide/off ratio <= 0.70 (the 33->11 round cut shows up at
#     the measured ~155-310 us/collective floor).
#   REFUTES:  ratio >= 0.90 (the fill's 3x payload or the in-shard RK
#     restructuring eats the round cut — then census the compiled wide
#     step's CP count before any further work).
#   Intermediate: real but shallower — census + repricing.
# Protocol: matches the s9 ladder rows byte-for-byte except the env
# flag (s9 lloyd-0 prewarmed mesh, sfc, reorder-for 64, f32,
# steps 12 / warmup 3, banked A/B/A2 with drift control).
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_wide_halo_s9_64.sbatch
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/scan_ab_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

run_arm () {  # extra_flag tag
  local XFLAG=$1 TAG=$2
  echo "=== s9@64 timing-mode $XFLAG [$TAG] ==="
  LEGOESM_MPAS_WIDE_HALO=0 LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
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
WANT = {"loop_a": False, "scan_b": True, "loop_a2": False}
t = {}
for tag, want_scan in WANT.items():
    try:
        rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}"); sys.exit(1)
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows"); sys.exit(1)
    r = rows[0]
    ok = (r.get("timed_scan") == want_scan
          and r.get("halo_strategy_effective") == "ppermute"
          and r.get("n_devices") == 64 and r.get("platform") == "gpu"
          and math.isfinite(r["steady_median_ms"])
          and r["steady_median_ms"] > 0)
    if not ok:
        print(f"{tag}: binding fail timed_scan={r.get('timed_scan')}"); sys.exit(1)
    t[tag] = (r["scan_median_ms"] if want_scan else r["steady_median_ms"])
    print(f"{tag}: {t[tag]:9.2f} ms")
drift = abs(t["off_a"] - t["off_a2"]) / t["off_a"]
ratio = t["on_b"] / min(t["off_a"], t["off_a2"])
print(f"A/A2 drift {100*drift:.1f}% | wide/off ratio {ratio:.3f}")
if ratio <= 0.70:
    print("VERDICT: CONFIRMED — wide halo cuts the latency floor")
elif ratio >= 0.90:
    print("VERDICT: REFUTED — census the wide step's CP count"); sys.exit(2)
else:
    print("VERDICT: intermediate — census + reprice"); sys.exit(3)
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
