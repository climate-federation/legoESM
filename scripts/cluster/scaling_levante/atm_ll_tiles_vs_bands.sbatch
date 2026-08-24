#!/bin/bash -l
#SBATCH --job-name=ll_tiles
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ll_tiles.%j.log
# ===========================================================================
# DOES TILING IN BOTH DIRECTIONS FIX THE LAT-LON COMMUNICATION WALL?
#
# Measured, this lane, lat-lon 2048x4096 with 26 vertical levels:
#   32 GPUs  6.972 ms/step,  9.5% communication
#   64 GPUs  4.728 ms/step, 28.8% communication
#  128 GPUs  3.950 ms/step, 42.2% communication
# and the halo moves 18.5 MB per device per step in 13 messages, counted
# from the compiled program. Doubling the payload costs MORE than the whole
# communication term, so the cost is bytes and not per-operation overhead.
#
# Why it never improves: a device owns a BAND of latitudes spanning the
# whole longitude circle, so its halo is two rows of 4096 columns at any
# device count. Adding devices shrinks the work and not the boundary.
# A tile's boundary shrinks with its area. At 64 devices an 8x8 tiling
# exchanges 1,536 boundary cells per tile against 8,192 for a band, a 5.3x
# cut on a term that is pure bytes.
#
# A PRIORI GATE (before submit), at 64 devices, band arm as the control:
#   CONFIRMS: tiled step <= 4.10 ms, i.e. it recovers at least half of the
#     1.360 ms the band arm spends communicating.
#   REFUTES:  tiled step >= 4.73 ms (no better than bands). The most likely
#     reason, documented in the tiled pad body, is its pole fold: it gathers
#     the whole longitude circle, and in this SPMD body BOTH branches of the
#     pole selection execute, so every tile pays it, not just polar ones.
#     That would swamp a 5.3x boundary cut. The follow-up in that case is
#     the antipodal-exchange fold, which is already described in the code.
#   Between: report the split rather than a verdict.
#
# The no-communication arms are what make the verdict attributable: they
# give each lane's communication term directly, so a tiled arm that is
# faster for some OTHER reason cannot be credited to the boundary cut.
#
# Both lanes run the SAME model, resolution and vertical levels; the only
# difference is how the globe is divided among devices.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/atm_ll_tiles_vs_bands.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=1
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false --xla_gpu_enable_nccl_comm_splitting=false"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll_tiles_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
rc=0

NLAT="${NLAT:-2048}"; NLON=$((2 * NLAT)); GPN=4
NDEV="${NDEV:-64}"
PLON_TILED="${PLON_TILED:-8}"
NODES=$(( NDEV / GPN ))

run_arm () {  # p_lon nocomm tag
  local PL=$1 NC=$2 TAG=$3
  echo "=== LL${NLAT}@${NDEV} p_lon=$PL nocomm=$NC [$TAG] $(date +%T) ==="
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_NOCOMM=$NC JAX_ENABLE_X64=0 \
    timeout 1800 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$GPN" --gpus-per-node="$GPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --mode strong \
      --n-lat "$NLAT" --n-lon "$NLON" --nlev 26 --p-lon "$PL" \
      --steps 60 --warmup 10 \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 10
}

# Discarded warm-up arm, then interleaved so the node ramp cannot alias
# into the comparison.
run_arm 1 0 warmup_discard
run_arm 1 0 band_a
run_arm "$PLON_TILED" 0 tile_a
run_arm 1 1 band_nc_a
run_arm "$PLON_TILED" 1 tile_nc_a
run_arm 1 0 band_b
run_arm "$PLON_TILED" 0 tile_b
run_arm 1 1 band_nc_b
run_arm "$PLON_TILED" 1 tile_nc_b

echo "=== RECEIPTS ==="
if [ "$rc" -ne 0 ]; then
  echo "ABORT: an arm failed (rc=$rc); no verdict"
  exit "$rc"
fi
"$PY" - "$OUTDIR" <<'EOF'
import json, os, sys

outdir = sys.argv[1]
TAGS = ("band_a", "band_b", "tile_a", "tile_b",
        "band_nc_a", "band_nc_b", "tile_nc_a", "tile_nc_b")


def med(tag):
    path = os.path.join(outdir, tag + ".jsonl")
    lines = [l for l in open(path).read().strip().split("\n")
             if l.startswith("{")]
    rec = json.loads(lines[-1])
    return rec["steady_median_ms"], rec["metadata"]["extra"]["decomposition"]


vals, decomp = {}, {}
missing = []
for t in TAGS:
    try:
        vals[t], decomp[t] = med(t)
    except Exception as exc:
        missing.append(f"{t} ({exc})")
if missing:
    print(f"ABORT: arms missing {missing}; no verdict")
    sys.exit(4)

# Binding check: an arm labelled tiled must actually have run tiled, or the
# comparison is band against band.
for t in TAGS:
    want_tiles = t.startswith("tile")
    got_tiles = decomp[t].startswith("tiles_")
    if want_tiles != got_tiles:
        print(f"ABORT: {t} ran decomposition {decomp[t]!r}; the arms are not "
              f"what they are labelled")
        sys.exit(5)
print(f"band arms ran {decomp['band_a']!r}, tiled arms ran {decomp['tile_a']!r}")

band = (vals["band_a"] + vals["band_b"]) / 2
tile = (vals["tile_a"] + vals["tile_b"]) / 2
band_nc = (vals["band_nc_a"] + vals["band_nc_b"]) / 2
tile_nc = (vals["tile_nc_a"] + vals["tile_nc_b"]) / 2
spread = max(abs(vals["band_a"] - vals["band_b"]) / band,
             abs(vals["tile_a"] - vals["tile_b"]) / tile)

print(f"{'lane':>6} {'step':>8} {'local':>8} {'comm':>8} {'comm%':>7}")
for name, full, nc in (("bands", band, band_nc), ("tiles", tile, tile_nc)):
    print(f"{name:>6} {full:8.3f} {nc:8.3f} {full-nc:8.3f} "
          f"{100*(full-nc)/full:6.1f}%")
print(f"worst arm spread {100*spread:.1f}%   tiles/bands {tile/band:.3f}")
print(f"communication cut: {band-band_nc:.3f} -> {tile-tile_nc:.3f} ms")

if abs(tile - band) < 0.5 * spread * band:
    print("INCONCLUSIVE: the difference is inside the arm spread")
    sys.exit(3)
if tile <= band - 0.5 * (band - band_nc):
    print("VERDICT: CONFIRMS — tiling recovers at least half the "
          "communication the band lane spends")
elif tile >= band:
    print("VERDICT: REFUTES — tiling is no faster; check whether the pole "
          "fold's whole-circle gather is running on every tile")
    sys.exit(6)
else:
    print("VERDICT: partial — tiling helps but recovers less than half; "
          "report the split, do not claim the lever")
EOF
gate_rc=$?
[ "$gate_rc" -ne 0 ] && rc=$gate_rc
echo "DONE rc=$rc"; exit $rc
