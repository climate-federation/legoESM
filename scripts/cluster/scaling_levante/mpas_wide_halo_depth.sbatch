#!/bin/bash -l
#SBATCH --job-name=mpas_widehalo
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50100
#SBATCH --output=mpas_widehalo.%j.log
# ===========================================================================
# IS THE DEEP GHOST REGION STILL WORTH ITS ONE EXCHANGE AT 128 GPUs?
#
# The wide halo builds a NINE-RING ghost region so that ONE exchange per step
# can serve all three Runge-Kutta stages instead of three. That was measured a
# win at 64 GPUs, when the arithmetic per device was still shrinking as
# devices were added.
#
# It is not shrinking any more. Measured at 128 GPUs by deleting the
# communication (job 27148747): the arithmetic alone is 2.36 ms per device at
# 26 levels, against about 2.0 ms at 64 GPUs. Doubling the machine made the
# per-device arithmetic slightly WORSE, because a nine-ring ghost region is a
# large part of what a device at this size computes on. Half the machine's
# work is arithmetic on other devices' cells.
#
# Turning the wide halo off trades one exchange per step for three and shrinks
# the ghost region threefold. This measures that trade at the size where it
# might now go the other way.
#
# A PRIORI GATE, written before submit:
#   CONFIRMS  the narrow halo is >= 3% faster at 128 GPUs. Then the deep ghost
#             region has stopped paying, the default should be revisited for
#             large runs, and the halo depth joins the schedule as a lever.
#   REFUTES   the narrow halo is slower, or the difference is inside the arm
#             spread. Then the three extra exchanges cost more than the ghost
#             arithmetic they remove, and the wide halo stays right at every
#             size measured.
#   Each arm pair must agree with itself to better than 0.5% or there is no
#   verdict, whichever way the means fall.
#
# Same protocol as the schedule arms otherwise: subdivision 9, lloyd 0, sfc
# partition, reorder-for the device count, 26 levels, coloured schedule, 60
# timed steps after 5 warmup, a discarded warm-up arm, palindrome order.
#
# SUBMIT: NODES=32 sbatch --nodes=32 --export=ALL,NODES=32 \
#           scripts/cluster/scaling_levante/mpas_wide_halo_depth.sbatch
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/merge_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_scaling/_jitcache_shared"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
rc=0

NODES="${NODES:-16}"; GPN="${GPN:-4}"; SUBDIV="${SUBDIV:-9}"
# Per-arm wall limit. Thirty minutes was enough at 64 GPUs and NOT at 128,
# where one arm was killed mid-run and the job returned no verdict at all
# after burning its whole reservation. It scales with the node count for the
# same reason the compile does.
ARM_TIMEOUT="${ARM_TIMEOUT:-$((900 + 60 * NODES))}"
NDEV=$((NODES * GPN))

WIDE="${WIDE:-0}"

run_arm () {  # wide tag
  local W=$1 TAG=$2
  echo "=== s${SUBDIV}@${NDEV} coloured wide_halo=$W [$TAG] ==="
  LEGOESM_MPAS_WIDE_HALO="$W" LEGOESM_MPAS_RAGGED_HALO=0 \
  LEGOESM_MPAS_ANNEAL_COLORING=0 JAX_ENABLE_X64=0 \
    timeout "$ARM_TIMEOUT" srun --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$GPN" --gpus-per-node="$GPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" \
      --subdivision "$SUBDIV" --nlev 26 \
      --steps 60 --warmup 5 --lloyd 0 --partition-method sfc \
      --reorder-for "$NDEV" \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 15
}

# Discarded: the node ramp must not land inside the first timed arm.
run_arm 1 warmup_discard
# Palindrome: off, on, on, off. Any monotone drift in machine state cancels.
run_arm 1 off_a
run_arm 0 on_a
run_arm 0 on_b
run_arm 1 off_b

echo "=== RECEIPTS ==="
NDEV_HINT="$NDEV" "$PY" - "$OUTDIR" <<'PYEOF'
import json, os, sys

outdir = sys.argv[1]
rows = {}
for tag in ("off_a", "off_b", "on_a", "on_b"):
    path = os.path.join(outdir, tag + ".jsonl")
    try:
        lines = [l for l in open(path).read().strip().split("\n")
                 if l.startswith("{")]
        rows[tag] = json.loads(lines[-1])
    except Exception as exc:
        print(f"{tag}: MISSING -> {exc}")

missing = [t for t in ("off_a", "off_b", "on_a", "on_b") if t not in rows]
if missing:
    # A verdict from a partial run is not a verdict.
    print(f"ABORT: arms missing {missing}; no verdict")
    sys.exit(4)

for tag, row in rows.items():
    peak = row.get("peak_bytes_in_use")
    peak_s = f"{peak / 2**20:8.1f} MB" if peak else "     n/a"
    print(f"{tag:>7}: {row['steady_median_ms']:7.3f} ms   peak {peak_s}")

off = (rows["off_a"]["steady_median_ms"] + rows["off_b"]["steady_median_ms"]) / 2
on = (rows["on_a"]["steady_median_ms"] + rows["on_b"]["steady_median_ms"]) / 2
spread_off = abs(rows["off_a"]["steady_median_ms"]
                 - rows["off_b"]["steady_median_ms"]) / off
spread_on = abs(rows["on_a"]["steady_median_ms"]
                - rows["on_b"]["steady_median_ms"]) / on
delta = on - off
print(f"\noff {off:.3f} ms   on {on:.3f} ms   delta {delta:+.3f} ms "
      f"({delta / off:+.2%})")
print(f"arm spreads: off {spread_off:.2%}   on {spread_on:.2%}")

peaks = [rows[t].get("peak_bytes_in_use") for t in
         ("off_a", "off_b", "on_a", "on_b")]
if all(peaks):
    d_peak = ((peaks[2] + peaks[3]) - (peaks[0] + peaks[1])) / 2
    print(f"peak memory delta {d_peak / 2**20:+.2f} MB per device")
else:
    print("peak memory: not reported by this runtime")

worst_spread = max(spread_off, spread_on)
if worst_spread >= 0.005:
    print(f"NO VERDICT: arm spread {worst_spread:.2%} is at or above the "
          f"0.5% the gate requires; the arms disagree with themselves.")
else:
    import os as _os
    bar = 0.03
    if delta <= -bar * off:
        print(f"CONFIRMED: the NARROW halo is faster at this size, past "
              f"the {bar:.1%} bar. The deep ghost region has stopped paying "
              f"for itself; the default deserves revisiting for large runs. "
              f"Adoption is the user's call.")
    elif abs(delta) < worst_spread * off:
        print("REFUTED: inside the arm spread. The three extra exchanges "
              "cost about what the ghost arithmetic they remove costs, so "
              "the wide halo stays right at this size too.")
    else:
        print(f"WEAK: {delta / off:+.2%}, real but under the "
              f"{bar:.1%} bar. Not worth a default move on its own.")
    if delta < -0.20 * off:
        print("WARNING: the gain exceeds the whole wire term, so the knob is "
              "moving something it was not built to move. Do not bank it.")
PYEOF
exit $rc
