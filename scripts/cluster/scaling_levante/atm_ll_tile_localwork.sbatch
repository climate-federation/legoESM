#!/bin/bash -l
#SBATCH --job-name=ll_tilelocal
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --exclude=l50100
#SBATCH --output=ll_tilelocal.%j.log
# ===========================================================================
# IS THE TILE SLOWER BECAUSE OF ITS MESSAGES, OR BECAUSE OF ITS ARITHMETIC?
#
# A tiled lat-lon decomposition moves several times fewer halo rows than
# latitude bands and measured 22% SLOWER at 64 GPUs. The message count is the
# obvious suspect -- the tile sends 28 point-to-point messages a step against
# the band's 13 -- and packing the stage exchange was tried and did NOT cut
# them, because the extra ones are the per-operator east-west wrap rather
# than the stage exchange.
#
# Before anyone rewrites those operators, there is a second suspect with a
# number already attached: the tile's compiled step has 5601 instructions
# against the band's 4315, THIRTY PERCENT more. Thirty percent more work is
# the same order as the twenty-two percent it lost, so the messages may be
# explaining a gap that the arithmetic already explains.
#
# This separates them. LEGOESM_LATLON_HALO_NOCOMM=1 replaces every halo
# exchange with the identity: WRONG ANSWERS, and every other line of the step
# -- the kernels, the local region, the masking, the padding shapes -- is
# unchanged. What is left is the local work.
#
# A PRIORI GATE, written before submit:
#   ARITHMETIC  tile/band with no communication at all is >= 1.15. Then the
#               tile's loss is its local work, rewriting the wrap buys little,
#               and the question becomes why the same physics costs 30% more
#               on a tile.
#   MESSAGES    tile/band with no communication is <= 1.05. Then the local
#               work is not the difference, the messages are, and hoisting
#               the per-operator wrap is worth building.
#   Between the two, both terms matter and the wrap work should be priced
#   against the arithmetic gap before it starts.
#
# Four GPUs on one node. The ratio being measured is a per-device property of
# the compiled program, not a property of the machine size, and this arm is
# cheap enough to run before the expensive one. It does NOT measure the tile's
# real loss, which needs the wire back and 64 GPUs.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/atm_ll_tile_localwork.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/tilelocal_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_scaling/_jitcache_shared"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
rc=0
NDEV="${NDEV:-4}"

run_arm () {  # tag p_lon nocomm
  local TAG=$1 PLON=$2 NC=$3
  echo "=== 128x256, ${NDEV} devices, p_lon=${PLON}, nocomm=${NC} (${TAG}) ==="
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_NOCOMM="$NC" JAX_ENABLE_X64=0 \
    timeout 1200 srun --export=ALL --nodes=1 --ntasks=1 \
      --gpus-per-node="$NDEV" --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --n-devices "$NDEV" --mode strong \
      --n-lat 128 --n-lon 256 --nlev 26 --p-lon "$PLON" \
      --steps 60 --warmup 5 --out "$OUTDIR/${TAG}.jsonl" \
      || { echo "${TAG} FAILED"; rc=1; }
  sleep 10
}

# Discarded, so the first timed arm is not the one that warms the node.
run_arm warmup_discard 1 1
# Palindrome, and both arms at both settings.
run_arm band_nocomm_a 1 1
run_arm tile_nocomm_a 2 1
run_arm tile_nocomm_b 2 1
run_arm band_nocomm_b 1 1
run_arm band_full_a   1 0
run_arm tile_full_a   2 0
run_arm tile_full_b   2 0
run_arm band_full_b   1 0

echo "=== VERDICT ==="
"$PY" - "$OUTDIR" <<'PYEOF'
import json, os, sys

outdir = sys.argv[1]
def read(tag):
    path = os.path.join(outdir, tag + ".jsonl")
    lines = [l for l in open(path).read().strip().split("\n")
             if l.startswith("{")]
    return json.loads(lines[-1])["steady_median_ms"]

want = ["band_nocomm_a", "band_nocomm_b", "tile_nocomm_a", "tile_nocomm_b",
        "band_full_a", "band_full_b", "tile_full_a", "tile_full_b"]
ms = {}
for tag in want:
    try:
        ms[tag] = read(tag)
    except Exception as exc:
        print(f"{tag}: MISSING -> {exc}")
missing = [t for t in want if t not in ms]
if missing:
    print(f"ABORT: arms missing {missing}; no verdict")
    sys.exit(4)
for tag in want:
    print(f"{tag:>15}: {ms[tag]:7.3f} ms")

def pair(a, b):
    return (ms[a] + ms[b]) / 2, abs(ms[a] - ms[b]) / ((ms[a] + ms[b]) / 2)

band_nc, s1 = pair("band_nocomm_a", "band_nocomm_b")
tile_nc, s2 = pair("tile_nocomm_a", "tile_nocomm_b")
band_fu, s3 = pair("band_full_a", "band_full_b")
tile_fu, s4 = pair("tile_full_a", "tile_full_b")
worst = max(s1, s2, s3, s4)
print(f"\nno communication : band {band_nc:.3f}  tile {tile_nc:.3f}  "
      f"ratio {tile_nc / band_nc:.3f}")
print(f"with communication: band {band_fu:.3f}  tile {tile_fu:.3f}  "
      f"ratio {tile_fu / band_fu:.3f}")
print(f"worst arm spread: {worst:.2%}")
print(f"the wire itself  : band {band_fu - band_nc:+.3f} ms  "
      f"tile {tile_fu - tile_nc:+.3f} ms")

if worst >= 0.01:
    print(f"NO VERDICT: arms disagree with themselves by {worst:.2%}")
elif tile_nc / band_nc >= 1.15:
    print("ARITHMETIC: the tile's loss is its local work, not its messages. "
          "Rewriting the east-west wrap buys little; the question is why the "
          "same physics costs a third more on a tile.")
elif tile_nc / band_nc <= 1.05:
    print("MESSAGES: the local work is near enough the same, so the tile's "
          "loss is what it sends. Hoisting the per-operator east-west wrap "
          "is worth building.")
else:
    print("BOTH: neither term is dominant. Price the wrap work against the "
          "arithmetic gap before starting it.")
PYEOF
exit $rc
