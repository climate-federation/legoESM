#!/bin/bash -l
#SBATCH --job-name=oc_sub2x2
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=oc_sub2x2.%j.log
# Ocean-MPAS CPU rank-count term (1.65x @4x ranks, matched 5120
# cells/rank, geometric — jobs 26508258/26508336): STAGE-LEVEL
# DISCRIMINATOR. The 2026-08-09 comm census (call-path confirmed):
# one step = 35 halo epochs + 65 allreduces, of which the 10-substep
# explicit barotropic scan holds 30 epochs + 60 reductions (eta-floor
# clamp = 3 allreduces x 2 sites x 10 substeps). The atm lane (3 epochs
# + 1 reduction) has NO rank-count term. Hypothesis: the term lives in
# the barotropic scan's comm.
# 2x2 = {s7@32, s8@128} x {substeps 10, 5}, all geometric, all else
# byte-identical. Substeps-10 arms REBASE the old pair on current code
# (bench gained --n-substeps today; weeks of merges since 26508258).
# Falsifiability, written BEFORE submit, on the ratio
#   R(sub) = t(s8@128,sub) / t(s7@32,sub):
#   CONFIRMS barotropic-comm mechanism: R(5) clearly below R(10)
#     (the delta scales with substep count; then split halo-vs-reduction
#     next with an eta-floor n_iter arm).
#   REFUTES: R(5) ~= R(10) (term is NOT in the scan's comm volume;
#     candidates left: per-message cost growth, O(global) leak).
#   Also read: t(sub=5)/t(sub=10) per cell — if ~0.5x the whole lane is
#   substep-dominated; if ~1x compute dominates and comm is minor.
# Cost: CPU compute partition, 4 nodes, minutes of compute — no GPU.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/oc_sub2x2_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0

run_arm () {  # subdiv np nodes substeps tag
  local SUBDIV=$1 NP=$2 NODES=$3 SUB=$4 TAG=$5
  echo "--- ocean mpas s${SUBDIV} np${NP} substeps=${SUB} ---"
  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" \
      --ntasks-per-node=32 --distribution=block:cyclic --cpu-bind=cores \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_ocean_mpas_scaling.py \
      --subdivision "$SUBDIV" --mode strong --nlev 10 \
      --partition-method geometric --n-substeps "$SUB" \
      --out "$OUTDIR/${TAG}.jsonl" < /dev/null \
    || { echo "${TAG} FAILED"; rc=1; }
}

run_arm 7  32 1 10 s7np32_sub10
run_arm 8 128 4 10 s8np128_sub10
run_arm 7  32 1  5 s7np32_sub5
run_arm 8 128 4  5 s8np128_sub5

echo "=== RECEIPTS ==="
"$PY" - "$OUTDIR" <<'EOF' || rc=1
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
t = {}
for tag in ("s7np32_sub10", "s8np128_sub10", "s7np32_sub5", "s8np128_sub5"):
    try:
        rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}"); sys.exit(1)
    r = rows[-1]
    ms = r["steady_median_ms"]
    sub = r["metadata"]["extra"].get("n_barotropic_substeps") if "extra" in r.get("metadata", {}) else None
    if sub is None:
        sub = r.get("n_barotropic_substeps")
    ok = math.isfinite(ms) and ms > 0
    if not ok:
        print(f"{tag}: bad steady {ms}"); sys.exit(1)
    t[tag] = ms
    print(f"{tag}: {ms:9.2f} ms (recorded substeps={sub})")
r10 = t["s8np128_sub10"] / t["s7np32_sub10"]
r5 = t["s8np128_sub5"] / t["s7np32_sub5"]
print(f"rank-count ratio R(10) = {r10:.3f}   (2026-07 receipt: 1.65)")
print(f"rank-count ratio R(5)  = {r5:.3f}")
print(f"substep share: s7 {t['s7np32_sub5']/t['s7np32_sub10']:.3f}, "
      f"s8 {t['s8np128_sub5']/t['s8np128_sub10']:.3f}  (0.5=scan-dominated, 1.0=compute-dominated)")
EOF
echo "DONE rc=$rc"; exit $rc
