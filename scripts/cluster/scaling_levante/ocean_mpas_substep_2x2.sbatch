#!/bin/bash -l
#SBATCH --job-name=oc_sub2x2
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=oc_sub2x2.%j.log
# Ocean-MPAS CPU rank-count term — stage + placement discriminators.
#
# Census (2026-08-09): one step = 35 halo epochs + 65 allreduces; the
# 10-substep barotropic scan holds 30 epochs + 60 reductions.
# First 2x2 (job 26819741, THIS launcher's earlier revision):
#   s7@32: 90.05 (sub10) / 71.40 (sub5); s8@128: 132.94 / 105.05
#   DELTA(sub) = t(s8@128) - t(s7@32): D(10)=42.89, D(5)=33.66,
#   D(5)/D(10) = 0.785 -> with D(sub) = D_fix + (sub/10) D_scan:
#   D_scan ~ 18.4 ms (43%), D_fix ~ 24.4 ms (57%). PLAUSIBLE, one run.
#
# Falsifiability on THIS job (codex r1 blocker fixed: the discriminator
# is the absolute DELTA, not the ratio — a proportional scan share in
# both arms holds R constant even when the term is in the scan):
#  (a) replicate pass, arms in REVERSED order: D(10), D(5) within ~15%
#      of the first job's values CONFIRMS the 43/57 split; wild
#      divergence means node/temporal noise dominates and no split may
#      be quoted.
#  (b) NEW placement arm s7@32 SPREAD (8 rpn x 4 nodes, vs packed
#      32 rpn x 1 node): same ranks, same work, only intra-node vs
#      inter-node messaging (job 26819741 confounded node count with
#      rank count: s7@32 was all-intra-node). Per-rank memory bandwidth
#      IMPROVES at 8 rpn, biasing this arm FASTER — so
#      t(spread) >> t(packed) CONFIRMS a fabric/placement term inside
#      the "rank-count" delta; t(spread) <= t(packed) refutes it.
# Cost: CPU compute partition, minutes (prior job: 3m45s total).
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

# Protocol pinned EXPLICITLY (codex: match the 26508258 receipts by
# contract, not by drifting defaults).
COMMON="--mode strong --nlev 10 --precision float64 --partition-method geometric \
  --steps 12 --warmup 2 --block-steps 8 --blocks 2 --probe-steps 3 \
  --halo-refresh auto --barotropic-solver explicit_substep"

run_arm () {  # subdiv np nodes rpn substeps tag
  local SUBDIV=$1 NP=$2 NODES=$3 RPN=$4 SUB=$5 TAG=$6
  echo "--- ocean mpas s${SUBDIV} np${NP} rpn${RPN} substeps=${SUB} [${TAG}] ---"
  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" \
      --ntasks-per-node="$RPN" --distribution=block:cyclic --cpu-bind=cores \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_ocean_mpas_scaling.py \
      --subdivision "$SUBDIV" $COMMON --n-substeps "$SUB" \
      --out "$OUTDIR/${TAG}.jsonl" < /dev/null \
    || { echo "${TAG} FAILED"; rc=1; }
}

# Pass 1 (same order as job 26819741)
run_arm 7  32 1 32 10 s7np32_sub10_r1
run_arm 8 128 4 32 10 s8np128_sub10_r1
run_arm 7  32 1 32  5 s7np32_sub5_r1
run_arm 8 128 4 32  5 s8np128_sub5_r1
# Placement arm: s7@32 spread over 4 nodes (8 rpn)
run_arm 7  32 4  8 10 s7np32_spread_sub10
# Pass 2, REVERSED order (temporal-drift control)
run_arm 8 128 4 32  5 s8np128_sub5_r2
run_arm 7  32 1 32  5 s7np32_sub5_r2
run_arm 8 128 4 32 10 s8np128_sub10_r2
run_arm 7  32 1 32 10 s7np32_sub10_r2

echo "=== RECEIPTS ==="
"$PY" - "$OUTDIR" <<'EOF' || rc=1
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
WANT = {  # tag -> (n_ranks, substeps)
  "s7np32_sub10_r1": (32, 10), "s8np128_sub10_r1": (128, 10),
  "s7np32_sub5_r1": (32, 5),  "s8np128_sub5_r1": (128, 5),
  "s7np32_spread_sub10": (32, 10),
  "s8np128_sub5_r2": (128, 5), "s7np32_sub5_r2": (32, 5),
  "s8np128_sub10_r2": (128, 10), "s7np32_sub10_r2": (32, 10),
}
t = {}
for tag, (nr, sub) in WANT.items():
    try:
        rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}"); sys.exit(1)
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows (stale append?)"); sys.exit(1)
    r = rows[0]
    ex = r.get("metadata", {}).get("extra", {})
    ok = (math.isfinite(r["steady_median_ms"]) and r["steady_median_ms"] > 0
          and r.get("metadata", {}).get("n_ranks") == nr
          and ex.get("n_barotropic_substeps") == sub
          and ex.get("barotropic_solver") == "explicit_substep")
    if not ok:
        print(f"{tag}: row fails binding: nranks={r.get('metadata',{}).get('n_ranks')} "
              f"sub={ex.get('n_barotropic_substeps')} solver={ex.get('barotropic_solver')}")
        sys.exit(1)
    t[tag] = r["steady_median_ms"]
    print(f"{tag}: {t[tag]:9.2f} ms")
for p in ("r1", "r2"):
    D10 = t[f"s8np128_sub10_{p}"] - t[f"s7np32_sub10_{p}"]
    D5 = t[f"s8np128_sub5_{p}"] - t[f"s7np32_sub5_{p}"]
    print(f"pass {p}: D(10)={D10:7.2f}  D(5)={D5:7.2f}  D5/D10={D5/D10:.3f} "
          f"(0.5=all-scan, 1.0=none-scan)")
sp = t["s7np32_spread_sub10"]
pk10 = min(t["s7np32_sub10_r1"], t["s7np32_sub10_r2"])
print(f"placement: s7@32 spread(8rpnx4n)={sp:.2f} vs packed(32rpnx1n)={pk10:.2f} "
      f"ratio {sp/pk10:.3f}  (>1.15 = fabric term; <=1.0 = refuted)")
EOF
echo "DONE rc=$rc"; exit $rc
