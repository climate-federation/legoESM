#!/bin/bash -l
#SBATCH --job-name=oc_etafl
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=oc_etafl.%j.log
# Ocean-MPAS rank-count term, round 4: cut the SCAN's reduction count
# without touching its halo epochs. --eta-floor-iters 3->1 drops the
# eta-floor clamp from 60 to 20 batched allreduces/step (epochs
# unchanged at 35). Attribution state: scan share of D = 30-41%
# (substeps 10->5 halves BOTH epochs and reductions); this arm splits
# reductions from epochs WITHIN the scan.
# Falsifiability BEFORE submit, D(iters) = t(s8@128) - t(s7@32):
#   REDUCTIONS MATTER: D(1) clearly below D(3) (~2/3 of the scan's
#     sync points removed; segment-length model predicts a smaller but
#     nonzero drop since segments merge);
#   REFUTED: D(1) ~= D(3) => the scan share is all halo epochs;
#     the deferred-redistribution surgery would buy nothing and is
#     CANCELLED. Positivity holds by construction (final maximum);
#     conservation residual absorbed by the step fixer (on).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/oc_etafl_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
COMMON="--mode strong --nlev 10 --precision float64 --partition-method geometric \
  --steps 12 --warmup 2 --block-steps 8 --blocks 2 --probe-steps 3 \
  --halo-refresh auto --barotropic-solver explicit_substep --n-substeps 10"

run_arm () {  # subdiv np nodes iters tag
  local SUBDIV=$1 NP=$2 NODES=$3 IT=$4 TAG=$5
  echo "--- ocean mpas s${SUBDIV} np${NP} eta-floor-iters=${IT} [${TAG}] ---"
  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" \
      --ntasks-per-node=32 --distribution=block:cyclic --cpu-bind=cores \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_ocean_mpas_scaling.py \
      --subdivision "$SUBDIV" $COMMON --eta-floor-iters "$IT" \
      --out "$OUTDIR/${TAG}.jsonl" < /dev/null \
    || { echo "${TAG} FAILED"; rc=1; }
}
run_arm 7  32 1 3 s7np32_it3
run_arm 8 128 4 3 s8np128_it3
run_arm 7  32 1 1 s7np32_it1
run_arm 8 128 4 1 s8np128_it1

echo "=== RECEIPTS ==="
set +e
"$PY" - "$OUTDIR" <<'EOF'
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
WANT = {"s7np32_it3": (32, 3), "s8np128_it3": (128, 3),
        "s7np32_it1": (32, 1), "s8np128_it1": (128, 1)}
t = {}
for tag, (nr, it) in WANT.items():
    rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows"); sys.exit(1)
    r = rows[0]
    ex = r.get("metadata", {}).get("extra", {})
    ok = (math.isfinite(r["steady_median_ms"]) and r["steady_median_ms"] > 0
          and r.get("metadata", {}).get("n_ranks") == nr
          and ex.get("eta_floor_clamp_iters") == it
          and ex.get("n_barotropic_substeps") == 10)
    if not ok:
        print(f"{tag}: binding fail {ex.get('eta_floor_clamp_iters')=}"); sys.exit(1)
    t[tag] = r["steady_median_ms"]
    print(f"{tag}: {t[tag]:9.2f} ms")
D3 = t["s8np128_it3"] - t["s7np32_it3"]
D1 = t["s8np128_it1"] - t["s7np32_it1"]
print(f"D(iters=3)={D3:7.2f}  D(iters=1)={D1:7.2f}  D1/D3={D1/D3:.3f}")
print(f"absolute: s8@128 {t['s8np128_it3']:.2f} -> {t['s8np128_it1']:.2f} ms "
      f"({100*(1-t['s8np128_it1']/t['s8np128_it3']):.1f}% step win at 128)")
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
