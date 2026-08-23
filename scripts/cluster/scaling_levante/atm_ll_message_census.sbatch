#!/bin/bash -l
#SBATCH --job-name=ll_msgs
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --exclude=l50100
#SBATCH --output=ll_msgs.%j.log
# ===========================================================================
# HOW MANY MESSAGES DOES A TILE SEND, AGAINST A LATITUDE BAND?
#
# At 128 GPUs a latitude band ships FOUR rows of halo for every row it owns:
# the band's halo is two whole circles of longitude however thin the band
# gets. Splitting longitude as well turns the band into a tile whose
# perimeter shrinks with it, and cuts the halo rows six-fold
# (scripts/validate/latlon_halo_surface_ratio.py). The tile was measured once
# at 64 GPUs and LOST, 6.01 against 4.92 ms, and nobody established why.
#
# The candidate explanation, from reading the source: the tile trades fewer
# bytes for far more messages. The packing that makes the band's exchange one
# message per stage is latitude-only; the tile exchanges latitude and
# longitude in separate rounds, every tile runs the pole handling whether or
# not it touches a pole, and each east-west wrap becomes two ring messages
# instead of a local copy. Read off the source, that is 108 messages against
# 13. THAT NUMBER IS NOT MEASURED, which is what this arm is for.
#
# WHY A SEPARATE ARM RATHER THAN THE LAYOUT DUMP: the layout dump runs with
# LEGOESM_LATLON_HALO_NOCOMM=1, which replaces every exchange with the
# identity so that the staging work can be counted on its own. Its compiled
# module therefore contains NO halo messages at all -- a message census taken
# there reads zero and means nothing. This arm leaves the communication in.
#
# A PRIORI GATE, written before submit:
#   CONFIRMS  the tile's module has >= 3x the band's message count. Then the
#             tile's loss is message count, the geometry argument is intact,
#             and packing the longitude exchange the way latitude is packed
#             is the lever -- worth building.
#   REFUTES   the counts are within 50% of each other. Then the messages are
#             not the difference, the source reading was wrong, and the tile's
#             loss needs a different explanation before anyone spends more.
#
# FOUR devices, not two. A "tile" needs a split in BOTH directions, and at two
# devices asking for two longitude splits leaves ONE latitude band -- that is
# not a tile, it is a longitude band, and comparing it against a latitude band
# compares two one-dimensional decompositions in different directions. A first
# version of this arm did exactly that and returned a verdict; the verdict was
# discarded. Four devices give the band arm four latitude bands and the tile
# arm a two-by-two grid, which is the comparison the question is about.
#
# One process, which is enough to make a compiled module: the count of
# messages in the module is a static property of the program, not of how many
# GPUs run it. It does NOT measure time, and no timing claim may be taken from
# this arm.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/atm_ll_message_census.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll_msgs_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR=""   # a cached compile does not re-dump
rc=0

# ONE place names the device count. The reservation asks for it, the launcher
# asks for it, and the model is told it -- and the first version of this arm
# had them disagree, so the model asked for four devices inside a two-GPU
# launch and both arms died before compiling anything.
NDEV="${NDEV:-4}"

run_dump () {  # n_lat n_lon tag p_lon
  local NLAT=$1 NLON=$2 TAG=$3 PLON=$4
  local PLAT=$((NDEV / PLON))
  case "$TAG" in tile*) if [ "$PLAT" -lt 2 ] || [ "$PLON" -lt 2 ]; then
    echo "REFUSING: p_lat=$PLAT p_lon=$PLON is not a tile, it is a band."
    return 1
  fi ;; esac
  mkdir -p "$OUTDIR/hlo_$TAG"
  echo "=== ${NLAT}x${NLON}, ${NDEV} devices, p_lat=${PLAT} p_lon=${PLON} (${TAG}) ==="
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_NOCOMM=0 JAX_ENABLE_X64=0 \
  XLA_FLAGS="--xla_gpu_shard_autotuning=false --xla_dump_to=$OUTDIR/hlo_$TAG" \
    timeout 1200 srun --export=ALL --nodes=1 --ntasks=1 --gpus-per-node="$NDEV" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --n-devices "$NDEV" --mode strong \
      --n-lat "$NLAT" --n-lon "$NLON" --nlev 26 --p-lon "$PLON" \
      --steps 3 --warmup 1 --out "$OUTDIR/probe_$TAG.jsonl" \
      || { echo "DUMP $TAG FAILED"; return 1; }
  "$PY" scripts/bench/census_layout_transposes.py --dump-dir "$OUTDIR/hlo_$TAG" \
    --module-re "jit__scan_run" --out "$OUTDIR/census_$TAG.json"
}

run_dump 128 256 band 1 || rc=1
run_dump 128 256 tile 2 || rc=1
# p_lon=4 on four devices would leave one latitude band; the guard above
# refuses it, which is the point of having the guard.
run_dump 128 256 tile4 4 || echo "(tile4 refused, as designed)"

echo "=== VERDICT ==="
"$PY" - "$OUTDIR" <<'PYEOF'
import json, os, sys

outdir = sys.argv[1]
counts = {}
for tag in ("band", "tile", "tile4"):
    try:
        with open(os.path.join(outdir, f"census_{tag}.json")) as fh:
            c = json.load(fh)
        counts[tag] = c
    except Exception as exc:
        print(f"{tag}: MISSING -> {exc}")

if "band" not in counts or "tile" not in counts:
    print("ABORT: an arm is missing; no verdict")
    sys.exit(4)

for tag, c in counts.items():
    families = ", ".join(f"{n} {f}" for f, n in c["collectives"].items())
    print(f"{tag:>5}: {c['n_collectives']:4d} messages  ({families or 'none'})"
          f"   {c['n_instructions']} instructions")

band, tile = counts["band"]["n_collectives"], counts["tile"]["n_collectives"]
if band == 0:
    # The one way this census lies: an arm with the exchange switched off
    # reads zero and looks like a triumph.
    print("NO VERDICT: the band arm has no messages at all, so the halo was "
          "compiled out. Check that HALO_NOCOMM is unset.")
elif tile >= 3 * band:
    print(f"CONFIRMED: the tile sends {tile / band:.1f}x the band's messages. "
          f"Its loss is message count, not the decomposition, and packing "
          f"the longitude exchange the way latitude is packed is the lever.")
elif tile <= 1.5 * band:
    print(f"REFUTED: {tile / band:.1f}x -- the messages are not the "
          f"difference. The tile's loss needs another explanation before "
          f"anyone spends more on it.")
else:
    print(f"PARTIAL: {tile / band:.1f}x. More messages, but not enough to "
          f"account for a 22% loss on its own.")
PYEOF
exit $rc
