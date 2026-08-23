#!/bin/bash -l
#SBATCH --job-name=mpas_merge
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50100
#SBATCH --output=mpas_merge.%j.log
# ===========================================================================
# DOES WRITING THE HALO ONCE INSTEAD OF THIRTEEN TIMES MAKE THE STEP FASTER?
#
# The icosahedral halo exchange runs thirteen coloured rounds at 64 GPUs and
# writes each round's received rows into the local buffers as that round
# lands: thirteen scatters per entity class per step. The rows a device
# receives are disjoint across rounds, so nothing forces the writes to be
# issued separately. LEGOESM_MPAS_HALO_MERGE_SCATTER=1 defers them and issues
# ONE scatter per entity class, with every index unique (each round owns its
# own padding row) so the scatter carries the no-duplicates promise.
#
# WHAT IT IS AIMED AT, measured, not fitted: the no-stage arm -- which keeps
# the kernel, the local region and the masking but performs no halo staging
# at all -- runs 0.310 ms faster than the no-comm arm at this exact
# configuration. That 0.310 ms of a 5.760 ms step is the whole gather +
# concatenate + scatter budget. The scatters are one of those three.
#
# A PRIORI GATE, written before submit:
#   CONFIRMS  merged is >= 1.0 % faster (>= 0.058 ms) AND each arm pair's
#             spread is < 0.5 %. Then the merge is a real win, it is worth
#             asking the user to adopt, and roughly a third to two thirds of
#             the staging budget was in the scatter count.
#   REFUTES   the difference is inside the arm spread, or merged is slower.
#             Then the thirteen scatters were already being fused or hidden,
#             the staging budget is in the gather and the concatenate, and
#             the next lever on this lane is not the write side.
#   A change larger than 0.310 ms would mean the knob is moving something
#   other than the staging it was built to move -- report it, do not bank it.
#
# SECOND NUMBER IN THE SAME ARMS: peak device memory. Holding thirteen rounds
# of received tiles alive to the end of the fill, then concatenating them,
# costs roughly twice the received halo payload in extra live bytes. Both
# reviewers asked for it to be measured rather than argued, so every row now
# carries the runtime's peak-bytes figure and the receipt prints the delta.
#
# Protocol matches the ballast/staging arms byte for byte except the knob:
# subdivision 9, lloyd 0, sfc partition, reorder-for 64, f32, 26 levels,
# wide halo on, coloured (NOT ragged) schedule -- ragged has no merge path
# and is off by default above 32 devices anyway -- 60 timed steps after 5
# warmup, a discarded warm-up arm, and arms in palindrome order so a drift
# in machine state cannot masquerade as an effect.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_merge_scatter_64.sbatch
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

run_arm () {  # merge tag
  local MRG=$1 TAG=$2
  echo "=== s${SUBDIV}@${NDEV} wide=1 coloured merge_scatter=$MRG [$TAG] ==="
  LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=0 \
  LEGOESM_MPAS_HALO_MERGE_SCATTER="$MRG" JAX_ENABLE_X64=0 \
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
run_arm 1 on_a
run_arm 1 on_b
run_arm 0 off_b

echo "=== RECEIPTS ==="
"$PY" - "$OUTDIR" <<'PYEOF'
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
elif delta <= -0.01 * off:
    print("CONFIRMED: merging the halo scatters is faster. The staging "
          "budget really is partly in the write side. Adoption is the "
          "user's call; the switch stays off by default.")
elif abs(delta) < worst_spread * off:
    print("REFUTED: the difference is inside the arm spread. The thirteen "
          "scatters were already fused or hidden; the staging budget is in "
          "the gather and the concatenate, not the writes.")
else:
    print("REFUTED: merging the scatters is not faster.")
if abs(delta) > 0.310:
    print("WARNING: the change exceeds the entire measured staging budget "
          "of 0.310 ms, so the knob is moving something it was not built "
          "to move. Do not bank this number until that is explained.")
PYEOF
exit $rc
