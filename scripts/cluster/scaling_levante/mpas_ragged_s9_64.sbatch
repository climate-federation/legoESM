#!/bin/bash -l
#SBATCH --job-name=ragged_s9
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --exclude=l50100
#SBATCH --output=ragged_s9.%j.log
# SCALE receipt for the merged ragged halo fill (PR #1534): s9@64 A/B.
# At s8@16 the ratio was 0.686 (-31% full step); the microbench trend
# (0.475 per-fill at nd16, improving with scale) predicts >= that at 64.
# Falsifiability BEFORE submit:
#   CONFIRMS scale-out: ratio <= 0.75 at s9@64 (halo share grows with
#     device count at fixed mesh; ragged advantage should not shrink).
#   REFUTES / caps the lever: ratio >= 0.9 (something rank-dependent
#     eats the grouped collective — then census the schedule at 64).
# Protocol: s9 lloyd-0 synthetic mesh (prewarmed cache), sfc,
# reorder-for 64, f32, steps 12/warmup 3 — matches the s9 ladder rows.
# Banked-arm structure: A (off) then B (on) then A2 (off drift control);
# each s8@16 arm cost ~50 min at 655k cells — s9 is 2.6M cells, so
# budget 3 h and accept a banked continuation if walltime bites.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ragged_s9_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

# Device count from the allocation, and the collective channel count pinned:
# sixteen channels is worth 9-10% on this lane at 64 and 128 GPUs, so an arm
# run at whatever NCCL happens to choose is not comparable to one that is.
NODES="${SLURM_JOB_NUM_NODES:-16}"; GPN=4; NDEV=$((NODES * GPN))
CHANNELS="${CHANNELS:-16}"
# Twelve steps gave 7.7% spread on IDENTICAL arms in this campaign; sixty
# gave 0.3%. The old arm count here predates that lesson.
STEPS="${STEPS:-65}"
echo "ragged A/B at ${NDEV} GPUs, ${CHANNELS} channels, ${STEPS} steps"

run_arm () {  # ragged tag
  local RG=$1 TAG=$2
  echo "=== s9@${NDEV} ragged=$RG channels=$CHANNELS [$TAG] $(date +%T) ==="
  printf 'RAGGED=%s CHANNELS=%s\n' "$RG" "$CHANNELS" > "$OUTDIR/${TAG}.env"
  LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_HALO_BALLAST=1 \
  LEGOESM_MPAS_HALO_NOCOMM=0 LEGOESM_MPAS_HALO_NOSTAGE=0 \
  LEGOESM_MPAS_RAGGED_HALO=$RG JAX_ENABLE_X64=0 \
  NCCL_MIN_NCHANNELS=$CHANNELS NCCL_MAX_NCHANNELS=$CHANNELS \
  NCCL_P2P_NET_CHUNKSIZE=131072 \
    timeout --kill-after=30 1500 srun --export=ALL --nodes="$NODES" \
      --ntasks="$NDEV" --ntasks-per-node="$GPN" --gpus-per-node="$GPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --subdivision 9 --nlev 26 \
      --steps "$STEPS" --warmup 5 --lloyd 0 --partition-method sfc \
      --reorder-for "$NDEV" \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 15
}
# Palindrome, two repetitions each: a plain off/on/off gives the two settings
# different mean time-positions, so a drift on the allocation lands on one of
# them rather than on both.
run_arm 0 warmup_discard
run_arm 0 off_a
run_arm 1 on_a
run_arm 1 on_b
run_arm 0 off_b

echo "=== RECEIPTS ==="
set +e
"$PY" - "$OUTDIR" <<'EOF'
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
WANT = {"off_a": "ppermute", "on_b": "ppermute_ragged", "off_a2": "ppermute"}
t = {}
for tag, eff in WANT.items():
    try:
        rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}"); sys.exit(1)
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows"); sys.exit(1)
    r = rows[0]
    ok = (r.get("halo_strategy_effective") == eff
          and r.get("n_devices") == 64 and r.get("platform") == "gpu"
          and r.get("n_processes") == 64
          and math.isfinite(r["steady_median_ms"])
          and r["steady_median_ms"] > 0)
    if not ok:
        print(f"{tag}: binding fail {r.get('halo_strategy_effective')} "
              f"{r.get('n_devices')}"); sys.exit(1)
    t[tag] = r["steady_median_ms"]
    print(f"{tag}: {t[tag]:9.2f} ms")
drift = abs(t["off_a"] - t["off_a2"]) / t["off_a"]
ratio = t["on_b"] / min(t["off_a"], t["off_a2"])
print(f"A/A2 drift {100*drift:.1f}% | ragged/off ratio {ratio:.3f}")
if ratio <= 0.75:
    print("VERDICT: CONFIRMED at scale")
elif ratio >= 0.9:
    print("VERDICT: capped — census the 64-dev schedule"); sys.exit(2)
else:
    print("VERDICT: intermediate (0.75-0.9) — real win, shallower than s8@16"); sys.exit(3)
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
