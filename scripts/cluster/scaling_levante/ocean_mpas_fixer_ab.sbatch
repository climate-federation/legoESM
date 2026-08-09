#!/bin/bash -l
#SBATCH --job-name=oc_fixab
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=oc_fixab.%j.log
# Ocean-MPAS CPU rank-count term, attribution round 2: the NON-scan
# reductions. State after jobs 26819741/26819928 (replicated):
#   D(sub) = t(s8@128) - t(s7@32) at 5120 cells/rank, geometric:
#   D5/D10 = 0.795 / 0.851 -> scan-scaling share 30-41%, the rest
#   (~25 ms) OUTSIDE the substep scan; placement/fabric REFUTED
#   (s7@32 spread over 4 nodes ran FASTER, 0.947).
# The census names exactly 5 non-scan reductions/step, ALL gated on
# use_conservation_fixer (3 batched fixer reductions + 2 UNBATCHED
# scalar global_sum_mpi at ocean_model_mpas.py:1026-1075).
# 2x2 = {s7@32, s8@128} x {fixer on, off}, substeps 10, protocol pinned.
# Falsifiability, BEFORE submit, on D_on - D_off:
#   CONFIRMS reductions-term: D_off clearly below D_on (the 5
#     reductions carry a rank-count cost; then the fix is BATCHING the
#     5 into 1, a small production change).
#   REFUTES: D_off ~= D_on (the non-scan term is NOT these reductions;
#     remaining candidates: the 5 non-scan halo epochs' per-message
#     growth, local halo-size asymmetry).
#   The fixer-on pair doubles as replicate #3 of the baseline delta.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/oc_fixab_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
COMMON="--mode strong --nlev 10 --precision float64 --partition-method geometric \
  --steps 12 --warmup 2 --block-steps 8 --blocks 2 --probe-steps 3 \
  --halo-refresh auto --barotropic-solver explicit_substep --n-substeps 10"

run_arm () {  # subdiv np nodes fixer tag
  local SUBDIV=$1 NP=$2 NODES=$3 FIX=$4 TAG=$5
  echo "--- ocean mpas s${SUBDIV} np${NP} fixer=${FIX} [${TAG}] ---"
  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" \
      --ntasks-per-node=32 --distribution=block:cyclic --cpu-bind=cores \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_ocean_mpas_scaling.py \
      --subdivision "$SUBDIV" $COMMON --conservation-fixer "$FIX" \
      --out "$OUTDIR/${TAG}.jsonl" < /dev/null \
    || { echo "${TAG} FAILED"; rc=1; }
}

run_arm 7  32 1 on  s7np32_fixon
run_arm 8 128 4 on  s8np128_fixon
run_arm 7  32 1 off s7np32_fixoff
run_arm 8 128 4 off s8np128_fixoff

echo "=== RECEIPTS ==="
"$PY" - "$OUTDIR" <<'EOF' || rc=1
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
WANT = {"s7np32_fixon": (32, "on"), "s8np128_fixon": (128, "on"),
        "s7np32_fixoff": (32, "off"), "s8np128_fixoff": (128, "off")}
t = {}
for tag, (nr, fx) in WANT.items():
    try:
        rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}"); sys.exit(1)
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows"); sys.exit(1)
    r = rows[0]
    ex = r.get("metadata", {}).get("extra", {})
    ok = (math.isfinite(r["steady_median_ms"]) and r["steady_median_ms"] > 0
          and r.get("metadata", {}).get("n_ranks") == nr
          and ex.get("conservation_fixer") == fx
          and ex.get("n_barotropic_substeps") == 10
          and ex.get("barotropic_solver") == "explicit_substep")
    if not ok:
        print(f"{tag}: row fails binding: {ex.get('conservation_fixer')=} "
              f"{r.get('metadata',{}).get('n_ranks')=}"); sys.exit(1)
    t[tag] = r["steady_median_ms"]
    print(f"{tag}: {t[tag]:9.2f} ms")
D_on = t["s8np128_fixon"] - t["s7np32_fixon"]
D_off = t["s8np128_fixoff"] - t["s7np32_fixoff"]
print(f"D_on = {D_on:.2f} ms (replicate #3; r1/r2 gave 42.50/40.43)")
print(f"D_off = {D_off:.2f} ms  ->  reductions' share of the delta = "
      f"{D_on - D_off:.2f} ms ({100*(D_on-D_off)/D_on:.0f}%)")
EOF
echo "DONE rc=$rc"; exit $rc
