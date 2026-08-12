#!/bin/bash -l
#SBATCH --job-name=wh_s9_64
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --exclude=l50100
#SBATCH --output=wh_s9_64.%j.log
# ===========================================================================
# WIDE-HALO discriminator part 2 (GPU): production A/B/A2 at s9@64.
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/wh_s9_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

# Geometry overridable at submit: the lever CONFIRMED -14.1% at s9@64
# (job 26880593, 8.13 -> 6.98 ms, drift 4.4%); the open question is
# whether it holds past 100 GPUs, where the per-collective price rises.
WH_NODES="${WH_NODES:-16}"
WH_GPN="${WH_GPN:-4}"
WH_SUBDIV="${WH_SUBDIV:-9}"
WH_NDEV=$((WH_NODES * WH_GPN))

run_arm () {  # wide tag
  local WH=$1 TAG=$2
  echo "=== s9@64 wide_halo=$WH [$TAG] ==="
  LEGOESM_MPAS_WIDE_HALO=$WH LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
    srun --nodes="$WH_NODES" --ntasks="$WH_NDEV" \
      --ntasks-per-node="$WH_GPN" --gpus-per-node="$WH_GPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$WH_NDEV" \
      --subdivision "$WH_SUBDIV" --nlev 26 \
      --steps 12 --warmup 3 --lloyd 0 --partition-method sfc \
      --reorder-for "$WH_NDEV" \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
}
# GPU PARITY GATE first (cheap, 1 node, s6@4): wide step vs the
# single-device serial reference — the same gate the ragged lever
# passed (job 26821453) before its timing arms were trusted.
echo "=== parity gate: s6@4 wide_halo=1 ==="
LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
  srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4 \
    --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    --multicontroller --n-devices 4 --subdivision 6 --nlev 8 \
    --steps 4 --warmup 1 --lloyd 0 --partition-method sfc \
    --halo-strategy ppermute --parity-gate \
    --out "$OUTDIR/parity_s6np4.jsonl" \
  || { echo "WIDE PARITY GATE FAILED — timing arms not run"; exit 5; }

run_arm 0 off_a
run_arm 1 on_b
run_arm 0 off_a2

echo "=== RECEIPTS ==="
set +e
WH_NDEV="$WH_NDEV" "$PY" - "$OUTDIR" <<'EOF'
import json, math, os, sys, pathlib
out = pathlib.Path(sys.argv[1])
WANT = {"off_a": (False, 3), "on_b": (True, 9), "off_a2": (False, 3)}
ND = int(os.environ["WH_NDEV"])
t = {}
for tag, (wide, depth) in WANT.items():
    try:
        rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}"); sys.exit(1)
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows"); sys.exit(1)
    r = rows[0]
    ok = (r.get("wide_halo") == wide and r.get("halo_depth") == depth
          and r.get("halo_strategy_effective") == "ppermute"
          and r.get("n_devices") == ND
          and r.get("platform") in ("gpu", "cuda")
          and r.get("n_processes") == ND
          and math.isfinite(r["steady_median_ms"])
          and r["steady_median_ms"] > 0)
    if not ok:
        print(f"{tag}: binding fail wide={r.get('wide_halo')} "
              f"depth={r.get('halo_depth')} "
              f"eff={r.get('halo_strategy_effective')}"); sys.exit(1)
    t[tag] = r["steady_median_ms"]
    cp = r.get("hlo_collective_permutes")
    print(f"{tag}: {t[tag]:9.2f} ms  (hlo CP: {cp})")
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
