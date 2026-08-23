#!/bin/bash -l
#SBATCH --job-name=mpas_anneal
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50100
#SBATCH --output=mpas_anneal.%j.log
# ===========================================================================
# DOES A BETTER HALO ROUND SCHEDULE MAKE THE STEP FASTER?
#
# Every exchange in a coloured halo round ships that round's LARGEST halo,
# because the sharded step needs one shape across devices. So a device with a
# small halo in a round still sends the round maximum, and that padding is
# pure waste on the wire.
#
# MEASURED OFFLINE, on the schedule the model actually builds (the probe calls
# the model's own colouring, it does not re-derive it):
#     64 devices : 3,468,261 real rows,  5,285,958 shipped -> 34.4% padding
#    128 devices : 4,914,922 real rows, 16,138,664 shipped -> 69.5% padding
# The padding gets much worse with device count, which is part of why this
# lane flattens above a hundred GPUs.
#
# LEGOESM_MPAS_ANNEAL_COLORING=N spends N million moves of annealed search on
# the same move -- give one exchange a different round -- but allowed to climb
# out of the basin the model's downhill search stops in, and allowed ONE extra
# round. Offline that takes the padding to 25.9% at 64 devices and 39.1% at
# 128, i.e. it halves the rows on the wire at 128. Two million moves is about
# five seconds of host time and captures the whole win; a hundred million only
# reach 25.6%.
#
# A PRIORI GATE, written before submit. Rows on the wire fall 11.5% at 64
# devices and 50.0% at 128. Wire time is roughly a third of the step, and the
# earlier split puts two thirds of it in payload, so:
#   CONFIRMS  at 128: >= 5% faster. At 64: >= 1.5% faster. Both need each arm
#             pair to agree with itself to better than 0.5%.
#   REFUTES   the difference is inside the arm spread, or slower. Then wire
#             time is not proportional to rows -- the messages are latency
#             bound, not bandwidth bound -- and the bytes axis is closed on
#             this lane regardless of how much padding is left.
#   A gain above 20% at 128 would exceed the whole wire term and means the
#   knob is moving something else. Report it, do not bank it.
#
# The schedule is bit-identical: regrouping rounds moves no data, and the
# 4-device gate asserts the step output does not change.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_anneal_coloring.sbatch
#   128 GPUs: NODES=32 sbatch --nodes=32 --export=ALL,NODES=32 <same script>
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

ANNEAL_MOVES="${ANNEAL_MOVES:-2}"

run_arm () {  # anneal tag
  local ANN=$1 TAG=$2
  echo "=== s${SUBDIV}@${NDEV} wide=1 coloured anneal=$ANN [$TAG] ==="
  LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=0 \
  LEGOESM_MPAS_ANNEAL_COLORING="$ANN" JAX_ENABLE_X64=0 \
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
run_arm 0 warmup_discard
# Palindrome: off, on, on, off. Any monotone drift in machine state cancels.
run_arm 0 off_a
run_arm "$ANNEAL_MOVES" on_a
run_arm "$ANNEAL_MOVES" on_b
run_arm 0 off_b

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
    bar = 0.05 if int(_os.environ.get("NDEV_HINT", "64")) >= 128 else 0.015
    if delta <= -bar * off:
        print(f"CONFIRMED: the annealed halo schedule is faster, past the "
              f"{bar:.1%} bar set before the run. Wire time does track rows "
              f"on this lane. Adoption is the user's call; the switch stays "
              f"off by default.")
    elif abs(delta) < worst_spread * off:
        print("REFUTED: the difference is inside the arm spread. Wire time "
              "is not proportional to rows here -- the exchange is latency "
              "bound, not bandwidth bound -- so the bytes axis is closed on "
              "this lane whatever the padding is.")
    else:
        print(f"WEAK: {delta / off:+.2%}, real but under the "
              f"{bar:.1%} bar. Not worth a default move on its own.")
    if delta < -0.20 * off:
        print("WARNING: the gain exceeds the whole wire term, so the knob is "
              "moving something it was not built to move. Do not bank it.")
PYEOF
exit $rc
