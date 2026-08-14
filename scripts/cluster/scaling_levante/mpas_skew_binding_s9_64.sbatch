#!/bin/bash -l
#SBATCH --job-name=wh_skew
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:55:00
#SBATCH --exclude=l50100
#SBATCH --output=wh_skew.%j.log
# ===========================================================================
# IS THE 1.23 ms "SKEW" RANK-ARRIVAL VARIANCE, AND DOES GPU/NIC BINDING
# REMOVE IT?  Both reviewers' first-ranked items, one job.
#
# Budget under test (job 26942819, subdiv-9@64): skew/waiting 1.230 ms =
# full - nocomm - wire. Every rank runs an identical program on identical
# shapes, so the candidates are extrinsic: rank->GPU/NIC/NUMA binding
# (never controlled -- every arm so far ran --gpu-bind=none), clock/power
# variance, host jitter. The bench now records per_rank_median_ms and
# per_rank_spread_ms in each receipt, so this job measures BOTH the
# spread and whether binding moves the median.
#
# A PRIORI GATE:
#   spread(none) >= 0.8 ms          -> rank arrival variance is real and
#                                      is most of the 1.23 ms.
#   spread(none) <= 0.2 ms          -> the 1.23 ms is INSIDE the
#                                      collectives (round serialisation),
#                                      not rank variance; binding is moot.
#   closest vs none median <= -0.3  -> CONFIRMS binding lever (codex
#                                      predicted -0.60 ms; threshold set
#                                      at half).
#   |closest - none| < 0.15         -> binding REFUTED.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_skew_binding_s9_64.sbatch
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/skew_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

run_arm () {  # bind tag
  local BIND=$1 TAG=$2
  echo "=== s9@64 wide=1 gpu-bind=$BIND [$TAG] ==="
  LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
    srun --nodes=16 --ntasks=64 --ntasks-per-node=4 --gpus-per-node=4 \
      --gpu-bind="$BIND" --cpu-bind=cores --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices 64 --subdivision 9 --nlev 26 \
      --steps 60 --warmup 5 --lloyd 0 --partition-method sfc \
      --reorder-for 64 \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 15
}

run_arm none warm_discard
run_arm none none_a
run_arm closest closest_a
run_arm none none_b
run_arm closest closest_b

echo "=== RECEIPTS ==="
"$PY" - "$OUTDIR" <<'EOF'
import json, os, sys
d = sys.argv[1]
rec = {}
for tag in ("none_a", "none_b", "closest_a", "closest_b"):
    try:
        lines = [l for l in open(os.path.join(d, tag + ".jsonl"))
                 .read().strip().split("\n") if l.startswith("{")]
        r = json.loads(lines[-1])
        rec[tag] = (r["steady_median_ms"],
                    r["metadata"]["extra"].get("per_rank_spread_ms"),
                    r["metadata"]["extra"].get("per_rank_median_ms"))
    except Exception as e:
        print(f"{tag}: MISSING -> {e}")
missing = [t for t in ("none_a", "none_b", "closest_a", "closest_b")
           if t not in rec]
if missing:
    print(f"ABORT: arms missing {missing}; no verdict")
    sys.exit(4)
for t, (m, sp, _) in rec.items():
    print(f"{t:>10}: median {m:7.3f} ms   per-rank spread {sp} ms")
none = (rec["none_a"][0] + rec["none_b"][0]) / 2
clos = (rec["closest_a"][0] + rec["closest_b"][0]) / 2
sp_none = max(rec["none_a"][1] or 0, rec["none_b"][1] or 0)
print(f"binding effect (closest - none): {clos - none:+.3f} ms")
print(f"max per-rank spread under none: {sp_none:.3f} ms")
if sp_none >= 0.8:
    print("VERDICT spread: rank arrival variance is REAL (most of the "
          "1.23 ms)")
elif sp_none <= 0.2:
    print("VERDICT spread: variance is INSIDE the collectives; "
          "binding moot")
else:
    print("VERDICT spread: intermediate")
if clos - none <= -0.3:
    print("VERDICT binding: CONFIRMS the lever")
elif abs(clos - none) < 0.15:
    print("VERDICT binding: REFUTED")
else:
    print("VERDICT binding: intermediate")
# worst/best rank map for the topology follow-up
import numpy as np
pr = rec["none_a"][2]
if pr:
    a = np.asarray(pr)
    print(f"none_a rank medians: min {a.min():.3f} @rank {a.argmin()}, "
          f"max {a.max():.3f} @rank {a.argmax()}")
EOF
echo "DONE rc=$rc"
exit $rc
