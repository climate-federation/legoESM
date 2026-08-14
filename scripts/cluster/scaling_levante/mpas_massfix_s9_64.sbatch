#!/bin/bash -l
#SBATCH --job-name=wh_massfix
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --exclude=l50100
#SBATCH --output=wh_massfix.%j.log
# ===========================================================================
# WHAT DOES THE ONE GLOBAL REDUCTION COST AT 64 GPUs?
#
# The measured subdiv-9@64 step (job 26942819, 6.820 ms) splits as
#   kernel with no halo at all   3.460 ms  51 %
#   halo wire payload            1.815 ms  27 %
#   skew / waiting               1.230 ms  18 %
#   on-device halo staging       0.315 ms   5 %
# and the no-halo arm itself grew from 2.725 ms at 4 devices to 3.460 ms at
# 64 on the SAME per-device load -- +0.735 ms in a program that exchanges no
# halo at all. The step performs exactly one global collective: the mass
# fixer's fp64 allreduce over two stacked area-weighted p_s fields
# (sharded_dynamics.py, "Global mass fixer"). That is the candidate.
#
# --no-fix-mass removes it and nothing else. Mass is then unpinned, so this
# is a TIMING arm only and the conservation gate is not used.
#
# A PRIORI GATE:
#   CONFIRMS  fix - nofix in [0.55, 0.95] ms -> the global reduction is the
#             +0.735 ms that appears between 4 and 64 devices, and the lever
#             is to defer or overlap it (it is ONE collective, and a single
#             collective is measured to hide ~79 % behind independent
#             compute when the latency-hiding scheduler is on).
#   REFUTES   fix - nofix <= 0.25 ms -> the reduction is not the cost; the
#             +0.735 ms is jitter and cross-node skew, and the lever is
#             load balance instead.
#   Intermediate: both contribute; reprice before acting.
#
# The 4-device arm runs in the same job at the SAME per-device load so the
# device-count difference is measured, not carried from an earlier run.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_massfix_s9_64.sbatch
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/massfix_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

run_arm () {  # nodes gpn subdiv nofix tag
  local ND=$1 GPN=$2 SUB=$3 NOFIX=$4 TAG=$5
  local NDEV=$((ND * GPN))
  local FLAG=""; [ "$NOFIX" = 1 ] && FLAG="--no-fix-mass"
  echo "=== s${SUB}@${NDEV} nofix=$NOFIX [$TAG] ==="
  LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
    srun --nodes="$ND" --ntasks="$NDEV" --ntasks-per-node="$GPN" \
      --gpus-per-node="$GPN" --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --subdivision "$SUB" --nlev 26 \
      --steps 60 --warmup 5 --lloyd 0 --partition-method sfc \
      --reorder-for "$NDEV" $FLAG \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 15
}

run_arm 16 4 9 0 warm_discard
run_arm 16 4 9 0 fix64_a
run_arm 16 4 9 1 nofix64_a
run_arm 16 4 9 0 fix64_b
run_arm 16 4 9 1 nofix64_b
# Same per-device load on ONE node: 40,961 cells x 26 levels per device.
run_arm  1 4 7 0 fix4_a
run_arm  1 4 7 1 nofix4_a

echo "=== RECEIPTS ==="
"$PY" - "$OUTDIR" <<'EOF'
import json, os, sys
d = sys.argv[1]
med = {}
for tag in ("fix64_a", "fix64_b", "nofix64_a", "nofix64_b",
            "fix4_a", "nofix4_a"):
    try:
        lines = [l for l in open(os.path.join(d, tag + ".jsonl"))
                 .read().strip().split("\n") if l.startswith("{")]
        med[tag] = json.loads(lines[-1])["steady_median_ms"]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}")
missing = [t for t in ("fix64_a", "fix64_b", "nofix64_a", "nofix64_b",
                       "fix4_a", "nofix4_a") if t not in med]
if missing:
    print(f"ABORT: arms missing {missing}; no verdict")
    sys.exit(4)
for t, v in med.items():
    print(f"{t:>10}: {v:7.3f} ms")
f64 = (med["fix64_a"] + med["fix64_b"]) / 2
n64 = (med["nofix64_a"] + med["nofix64_b"]) / 2
sp_f = abs(med["fix64_a"] - med["fix64_b"]) / f64
sp_n = abs(med["nofix64_a"] - med["nofix64_b"]) / n64
d64 = f64 - n64
d4 = med["fix4_a"] - med["nofix4_a"]
print(f"@64  fix {f64:.3f}  nofix {n64:.3f}  cost {d64:+.3f} ms "
      f"(spreads {sp_f:.1%}/{sp_n:.1%})")
print(f"@4   fix {med['fix4_a']:.3f}  nofix {med['nofix4_a']:.3f}  "
      f"cost {d4:+.3f} ms")
print(f"growth of the reduction 4 -> 64 devices: {d64 - d4:+.3f} ms")
if abs(d64) < 0.5 * max(sp_f, sp_n) * f64:
    print("INCONCLUSIVE: the difference is inside the arm spread")
elif 0.55 <= d64 <= 0.95:
    print("VERDICT: CONFIRMS -- the global reduction is the device-count "
          "term; defer or overlap it")
elif d64 <= 0.25:
    print("VERDICT: REFUTES -- not the reduction; chase skew/jitter")
    sys.exit(6)
else:
    print("VERDICT: intermediate -- reprice before acting")
EOF
echo "DONE rc=$rc"
exit $rc
