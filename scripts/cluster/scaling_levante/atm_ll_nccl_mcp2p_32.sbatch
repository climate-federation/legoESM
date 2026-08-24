#!/bin/bash -l
#SBATCH --job-name=ll_mcp2p
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=ll_mcp2p.%j.log
# ===========================================================================
# SECOND-LANE RECEIPT for the confirmed NCCL multi-channel-p2p env pair
# (assessment 2026-08-16: s9@64 MPAS wide halo −13..−18 %, jobs
# 26999539/27002564). Question: does the pair transfer to the lat-lon
# packed-exchange lane (panels a/e), whose step runs 13 band collectives?
# LL2304x4608 @32 f32, packed exchange ON (the production-best combo).
#
# PRE-REGISTERED GATE:
#   number: steady_median_ms, base vs mcp2p, palindrome pairs, 65 steps.
#   CONFIRM transfer: mcp2p mean <= base mean - 3% with pair spreads
#     < half the delta. REFUTE: |delta| < 1.5%. Between: INCONCLUSIVE.
#   (Lat-lon has fewer, fatter messages than MPAS's 14 rounds — the
#   per-op share is smaller, so the bar is lower than MPAS's 13%.)
#   Why no cheaper test: the packed-exchange serialisation exists only
#   multi-node; vader (2 GPUs/node) reaches 2 nodes but a 4-rank band
#   split is a different comm graph than 32.
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_LATLON_SPMD_FUSED_HALO=1
export LEGOESM_LATLON_PACKED_EXCHANGE=1
export LEGOESM_XLA_OVERLAP=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll_mcp2p_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_scaling/_jitcache_j${SLURM_JOB_ID}"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
rc=0

MCP2P="NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072"

run_arm () { # tag extra_env...
  local TAG=$1; shift
  echo "=== LL2304x4608 @32 f32 packed [$TAG] env: ${*:-defaults} ==="
  printf '%s\n' "$*" > "$OUTDIR/${TAG}.env"
  env "$@" JAX_ENABLE_X64=0 \
    timeout 900 srun --export=ALL --ntasks=32 --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices 32 --mode strong \
      --n-lat 2304 --n-lon 4608 --nlev 26 \
      --steps 65 --warmup 5 \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 15
}

run_arm warmup_discard
run_arm base_a
run_arm mcp2p_a $MCP2P
run_arm mcp2p_b $MCP2P
run_arm base_b

echo "=== RECEIPTS ==="
"$PY" - "$OUTDIR" <<'EOF'
import glob, json, os, sys
d = sys.argv[1]
EXPECT = sorted(["base_a", "base_b", "mcp2p_a", "mcp2p_b"])
tags = sorted(os.path.basename(f)[:-6]
              for f in glob.glob(os.path.join(d, "*_[ab].jsonl")))
if tags != EXPECT:
    print(f"ARM SET MISMATCH: have {tags} want {EXPECT} — NO VERDICT")
    sys.exit(1)
t = {}
for tag in EXPECT:
    lines = open(os.path.join(d, tag + ".jsonl")).readlines()
    if len(lines) != 1:
        print(f"{tag}: {len(lines)} lines (stale) — NO VERDICT"); sys.exit(1)
    rec = json.loads(lines[0])
    t.setdefault(tag.rsplit("_", 1)[0], []).append(rec["steady_median_ms"])
for env, v in sorted(t.items()):
    print(f"{env:6s} reps {[round(x, 3) for x in v]}  mean {sum(v)/2:7.3f}  "
          f"spread {abs(v[0]-v[1]):.3f} ms")
bm, mm = sum(t["base"]) / 2, sum(t["mcp2p"]) / 2
dpct = (mm - bm) / bm * 100
noise = max(abs(t["base"][0] - t["base"][1]), abs(t["mcp2p"][0] - t["mcp2p"][1]))
if noise >= 0.5 * abs(mm - bm) and abs(dpct) >= 1.5:
    band = "NOISE-BOUND"
elif dpct <= -3.0:
    band = "CONFIRM band (transfers)"
elif abs(dpct) < 1.5:
    band = "REFUTE band (no transfer at 32)"
else:
    band = "INCONCLUSIVE band"
print(f"mcp2p delta vs base: {mm - bm:+.3f} ms ({dpct:+.1f}%) [{band}]")
EOF
echo "DONE rc=$rc"; exit $rc
